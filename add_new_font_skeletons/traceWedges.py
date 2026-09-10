"""PROTOTYPE: propose a skeleton for a glyph automatically -- no transfer, no annotation.

Pipeline per sign: detect wedge heads (morphological opening), fit a triangle to each head
(cv2.minEnclosingTriangle) for the three head keypoints, pick the apex corner (the one with ink
continuing beyond it), then extend the tail from the apex along the ink -- re-centering on the
stroke's ridge each step -- until the ink ends or another head begins. Two tails that run toward
each other along the same line (a shared tail) are truncated to meet halfway.

  python add_new_font_skeletons/traceWedges.py Esagil 0x1230b 0x12038 ...
      Writes a proposal image per sign to fontTargets/<font>_proposals/<hex>.png.
      With no hex codes given, processes every sign in fontTargets/<font>/.

Run from the repo root. This is a feasibility prototype: it draws proposals for review, it does
not yet write skeleton CSVs.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from detectWedges import load_processed_glyph, glyph_path, C_RADIUS  # noqa: E402


def find_heads(ink, min_rel_size=0.5):
    """Opening blobs -> per-head (triangle corners, centroid, label map). The triangle is fitted
    only to the blob region around its widest point, so a surviving piece of tail cannot stretch
    it; blobs much narrower than the sign's dominant head (tail-end bulbs) are rejected."""
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    r = max(3, round(C_RADIUS * float(np.median(dist[ink > 0]))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    opened = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(opened)
    raw = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < r * r:
            continue
        blob = labels == i
        blob_dist = np.where(blob, dist, 0)
        peak = np.unravel_index(np.argmax(blob_dist), blob_dist.shape)
        raw.append((i, peak, blob_dist[peak]))
    if not raw:
        return [], labels, dist
    r_max = max(rh for _, _, rh in raw)
    heads = []
    ys, xs = np.mgrid[0:ink.shape[0], 0:ink.shape[1]]
    for i, (py, px), rh in raw:
        if rh < min_rel_size * r_max:
            continue  # much narrower than the dominant heads: likely a tail-end bulb
        near = (labels == i) & ((ys - py) ** 2 + (xs - px) ** 2 <= (2.2 * rh) ** 2)
        pts = np.argwhere(near)[:, ::-1].astype(np.float32)  # x, y
        _, tri = cv2.minEnclosingTriangle(pts.reshape(-1, 1, 2))
        heads.append({"corners": tri.reshape(3, 2), "centroid": np.array([px, py], float), "id": i})
    return heads, labels, dist


def ink_beyond(ink, corner, direction, probe=25):
    """How much ink lies along `direction` beyond this corner (to find the apex)."""
    steps = np.arange(3, probe)
    pts = corner[None, :] + steps[:, None] * direction[None, :]
    pts = np.round(pts).astype(int)
    valid = (pts[:, 0] >= 0) & (pts[:, 0] < ink.shape[1]) & (pts[:, 1] >= 0) & (pts[:, 1] < ink.shape[0])
    return ink[pts[valid, 1], pts[valid, 0]].sum()


def trace_tail(ink, dist, labels, head, step=2.0, halfwidth=6, max_steps=400):
    """Walk from the apex along the stroke ridge until the ink ends or another head begins."""
    corners, centroid = head["corners"], head["centroid"]
    dirs = [(c - centroid) / (np.linalg.norm(c - centroid) + 1e-6) for c in corners]
    scores = [ink_beyond(ink, c, d) for c, d in zip(corners, dirs)]
    apex_i = int(np.argmax(scores))
    apex, u = corners[apex_i], dirs[apex_i]

    pos = apex.copy()
    perp = np.array([-u[1], u[0]])
    for _ in range(max_steps):
        nxt = pos + u * step
        # re-center on the ridge of the distance transform, perpendicular to travel
        cands = nxt[None, :] + np.arange(-halfwidth, halfwidth + 1)[:, None] * perp[None, :]
        ci = np.round(cands).astype(int)
        ok = (ci[:, 0] >= 0) & (ci[:, 0] < ink.shape[1]) & (ci[:, 1] >= 0) & (ci[:, 1] < ink.shape[0])
        if not ok.any():
            break
        vals = np.where(ok, dist[ci[:, 1].clip(0, ink.shape[0] - 1), ci[:, 0].clip(0, ink.shape[1] - 1)], -1)
        best = int(np.argmax(vals))
        if vals[best] < 1.0:      # ran out of ink: the stroke ends here
            break
        newpos = cands[best]
        li = labels[int(round(newpos[1])), int(round(newpos[0]))]
        if li != 0 and li != head["id"]:
            break                 # ran into another wedge's head
        v = newpos - pos
        nv = np.linalg.norm(v)
        if nv > 1e-6:             # let the direction bend slightly with the stroke
            u = 0.85 * u + 0.15 * (v / nv)
            u /= np.linalg.norm(u)
            perp = np.array([-u[1], u[0]])
        pos = newpos
    return apex_i, pos


def meet_halfway(strokes, angle_cos=-0.9, lateral=12.0):
    """User's rule: two wedges whose tails run toward each other along one line share that line;
    truncate both tails to the midpoint between the two apexes."""
    for i in range(len(strokes)):
        for j in range(i + 1, len(strokes)):
            a, b = strokes[i], strokes[j]
            ua = (a["tail"] - a["apex"]); ub = (b["tail"] - b["apex"])
            na, nb = np.linalg.norm(ua), np.linalg.norm(ub)
            if na < 8 or nb < 8:
                continue
            ua, ub = ua / na, ub / nb
            if np.dot(ua, ub) > angle_cos:
                continue          # not opposing
            off = b["apex"] - a["apex"]
            if abs(np.cross(ua, off)) > lateral:
                continue          # not on the same line
            d = np.dot(off, ua)
            if d < 0 or np.dot(a["tail"] - a["apex"], ua) < d * 0.5:
                continue          # b is not ahead of a, or a's tail stops before halfway anyway
            mid = a["apex"] + ua * (d / 2)
            a["tail"], b["tail"] = mid.copy(), mid.copy()


def propose(gray):
    ink = np.uint8(gray < 128)
    heads, labels, dist = find_heads(ink)
    strokes = []
    for h in heads:
        apex_i, tail = trace_tail(ink, dist, labels, h)
        strokes.append({"corners": h["corners"], "apex": h["corners"][apex_i], "tail": tail})
    meet_halfway(strokes)
    return strokes


def draw(gray, strokes):
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    colors = [(60, 60, 230), (230, 130, 40), (40, 160, 40), (160, 40, 160),
              (30, 160, 220), (20, 90, 140), (140, 140, 20), (200, 40, 100)]
    for k, s in enumerate(strokes):
        c = colors[k % len(colors)]
        tri = np.round(s["corners"]).astype(int)
        cv2.polylines(img, [tri], True, c, 2)
        for p in tri:
            cv2.circle(img, tuple(p), 4, c, -1)
        cv2.line(img, tuple(np.round(s["apex"]).astype(int)), tuple(np.round(s["tail"]).astype(int)), c, 2)
        cv2.circle(img, tuple(np.round(s["tail"]).astype(int)), 5, c, 2)
    return img


def main(font, hexes):
    out_dir = Path("fontTargets") / f"{font}_proposals"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not hexes:
        hexes = sorted(t.stem for t in Path("fontTargets", font).glob("0x*.png"))
        print(f"No signs given: proposing wedges for all {len(hexes)} signs in fontTargets/{font}")
    for hx in hexes:
        try:
            gray = load_processed_glyph(glyph_path(font, hx))
            strokes = propose(gray)
        except (ValueError, FileNotFoundError) as e:
            print(f"{hx}: skipped ({e})")
            continue
        cv2.imwrite(str(out_dir / f"{hx}.png"), draw(gray, strokes))
        print(f"{hx}: {len(strokes)} wedges proposed -> {out_dir}/{hx}.png")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
