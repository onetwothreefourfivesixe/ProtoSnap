"""Line-shaped crops: each line cut out between two seams that run through the space between lines, around the signs.

  python -m line_segmentation.crops.seam_crops --faces tablet_data/ebl_tablets/faces/faces.csv --pred tablet_data/ebl_tablets/predictions
      --method unet [--split test] [--outer 0.7] [--margin 0.5] [--fill clay|white|none] --out tablet_data/ebl_tablets/line_crops_seam

Rectangular gap-bounded crops (make_crops.py) stop at a straight boundary halfway between two lines, so a tall sign
that reaches into the gap is cut, and widening the rectangle takes in the neighbour's signs. Here the boundary between
two neighbouring lines is a **seam**: the left-to-right path through the face, in the deskewed frame the crops are cut
in, that crosses the least writing. Writing is measured as the smoothed image gradient (wedges are sharp light/dark
edges, the clay between lines is smooth). The seam is found by dynamic programming column by column, may move up or
down one row per column, is kept at least 0.2 pitch from both line centres, and pays a small penalty for leaving the
midpoint so it stays central where the clay is blank. It therefore bends around a tall sign of either line instead of
cutting through it.

Line i's region is everything between its upper and lower seam, from its predicted start to its end with --margin
pitch added at each end; the first line's upper boundary and the last line's lower boundary are its centre line moved
--outer pitch out (no neighbour there). The crop image is the region's bounding box with every pixel outside the region
filled (--fill clay: the median colour of the region; white; none = keep the pixels, i.e. a rectangle with a mask).
Writes <out>/<face>/<face>__L<nn>.jpg, the region mask as <face>__L<nn>_mask.png, and <out>/crops.csv (bounding box in
the deskewed face, skew, mask path, and the two seams as 'x:y' lists every 16 px) in the same layout as make_crops.py.
"""
import argparse
import csv
import os

import cv2
import numpy as np

from line_segmentation.heuristic_v1.profile_lines import rotate
from line_segmentation.paths import EBL


def line_models(seg_csv, M, W):
    """Segments (face pixels) -> per line (y at each column of the deskewed face, x0, x1), sorted top to bottom."""
    L = []
    for r in csv.DictReader(open(seg_csv)):
        p0 = M @ np.array([float(r['ax0']), float(r['ay0']), 1.0]); p1 = M @ np.array([float(r['ax1']), float(r['ay1']), 1.0])
        if p1[0] < p0[0]:
            p0, p1 = p1, p0
        k = (p1[1] - p0[1]) / max(1e-6, p1[0] - p0[0])
        xs = np.arange(W)
        L.append(dict(y=p0[1] + k * (xs - p0[0]), x0=p0[0], x1=p1[0], yc=(p0[1] + p1[1]) / 2))
    return sorted(L, key=lambda d: d['yc'])


def energy_map(rot_gray, pitch):
    g = cv2.GaussianBlur(rot_gray.astype(np.float32), (0, 0), max(1.0, pitch / 30))
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    mag = cv2.GaussianBlur(mag, (0, 0), max(1.0, pitch / 20))
    return np.clip(mag / (np.percentile(mag, 95) + 1e-6), 0, 1)


def seam(E, ya, yb, x_lo, x_hi, pitch, s, lam=0.3, keep=0.2):
    """Least-energy path between line centres ya(x) < yb(x), for columns x_lo..x_hi, on a grid downsampled by s.
    Returns y (full resolution) for every column of the face; outside [x_lo, x_hi] the midpoint of the two lines."""
    W = E.shape[1]; mid = (ya + yb) / 2; out = mid.copy()
    xs = np.arange(int(x_lo), int(x_hi) + 1, s)
    if len(xs) < 2:
        return out
    Es = E[::s, :]                                        # rows downsampled; columns sampled at xs
    Hs = Es.shape[0]; rows = np.arange(Hs) * s
    INF = 1e9; cost = np.full(Hs, INF); back = np.zeros((len(xs), Hs), np.int32)
    for k, x in enumerate(xs):
        lo, hi = ya[x] + keep * pitch, yb[x] - keep * pitch
        if hi < lo:
            lo = hi = mid[x]
        allowed = (rows >= lo - s) & (rows <= hi + s)
        local = Es[:, x] + lam * np.abs(rows - mid[x]) / pitch
        local = np.where(allowed, local, INF)
        if k == 0:
            cost = local.copy()
            continue
        prev = np.stack([np.r_[INF, cost[:-1]], cost, np.r_[cost[1:], INF]])     # from row-1, row, row+1
        arg = prev.argmin(0); back[k] = np.arange(Hs) + arg - 1
        cost = prev[arg, np.arange(Hs)] + local
    r = int(cost.argmin()); path = np.zeros(len(xs), np.int32)
    for k in range(len(xs) - 1, -1, -1):
        path[k] = r; r = back[k, r] if k > 0 else r
    ys = rows[path].astype(float)
    cols = np.arange(int(x_lo), int(x_hi) + 1)
    out[cols] = np.interp(cols, xs, ys)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--faces', required=True); ap.add_argument('--pred', required=True); ap.add_argument('--method', default='unet')
    ap.add_argument('--split', choices=['all', 'test', 'tune'], default='all'); ap.add_argument('--out', required=True)
    ap.add_argument('--outer', type=float, default=0.7, help='first/last line: boundary this many pitches beyond the centre line')
    ap.add_argument('--margin', type=float, default=0.5, help='added beyond each line end, in pitches')
    ap.add_argument('--fill', choices=['clay', 'white', 'none'], default='clay')
    ap.add_argument('--limit', type=int, default=None); ap.add_argument('--quality', type=int, default=92)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    faces = list(csv.DictReader(open(a.faces)))
    if a.split != 'all':
        keep = {r['face'] for r in csv.DictReader(open(os.path.join(EBL, 'split.csv'))) if r['split'] == a.split}
        faces = [f for f in faces if f['face'] in keep]
    runs = {r['face']: r for r in csv.DictReader(open(os.path.join(a.pred, 'faces_run.csv'))) if r['method'] == a.method}
    rows, n_faces = [], 0
    for f in faces:
        r = runs.get(f['face']); sp = os.path.join(a.pred, a.method + '_segments', f['face'] + '.csv')
        lp = os.path.join(a.pred, a.method, f['face'] + '.csv')
        if r is None or not os.path.exists(sp) or not os.path.exists(lp):
            continue
        img = cv2.imread(f['file'])
        if img is None:
            continue
        H, W = img.shape[:2]; skew = float(r['skew_deg'])
        rot = np.clip(np.dstack([rotate(img[:, :, c].astype(np.float32), skew) for c in range(3)]), 0, 255).astype(np.uint8)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), skew, 1.0)
        L = line_models(sp, M, W)
        rowsL = sorted(csv.DictReader(open(lp)), key=lambda q: float(q['y_deskewed']))
        if not L or len(L) != len(rowsL):
            continue
        ys = np.array([d['yc'] for d in L]); pitch = float(np.median(np.diff(ys))) if len(ys) > 1 else float(r['pitch_px'] or 50)
        if not np.isfinite(pitch) or pitch <= 2:
            pitch = float(f['pitch_px'] or 50)
        s = max(1, int(round(pitch / 16)))
        E = energy_map(cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY), pitch)
        m = a.margin * pitch
        ext = [(max(0.0, float(q['x0']) - m), min(W - 1.0, float(q['x1']) + m)) for q in rowsL]
        seams = []
        for i in range(len(L) - 1):
            lo = max(0.0, min(ext[i][0], ext[i + 1][0])); hi = min(W - 1.0, max(ext[i][1], ext[i + 1][1]))
            seams.append(seam(E, L[i]['y'], L[i + 1]['y'], lo, hi, pitch, s))
        d = os.path.join(a.out, f['face']); os.makedirs(d, exist_ok=True); n_faces += 1
        cols = np.arange(W)
        for i, q in enumerate(rowsL):
            up = seams[i - 1] if i > 0 else L[i]['y'] - a.outer * pitch
            dn = seams[i] if i < len(L) - 1 else L[i]['y'] + a.outer * pitch
            x0, x1 = int(ext[i][0]), int(ext[i][1])
            up_c, dn_c = np.clip(up[x0:x1 + 1], 0, H - 1), np.clip(dn[x0:x1 + 1], 0, H - 1)
            Y0, Y1 = int(np.floor(up_c.min())), int(np.ceil(dn_c.max())) + 1
            if x1 - x0 < 4 or Y1 - Y0 < 4:
                continue
            yy = np.arange(Y0, Y1)[:, None]
            mask = ((yy >= up_c[None, :]) & (yy <= dn_c[None, :])).astype(np.uint8) * 255
            crop = rot[Y0:Y1, x0:x1 + 1].copy()
            if a.fill != 'none':
                fill = np.median(crop[mask > 0], axis=0) if a.fill == 'clay' and (mask > 0).any() else np.array([255, 255, 255])
                crop[mask == 0] = fill.astype(np.uint8)
            name = f"{f['face']}__L{i + 1:02d}"
            cv2.imwrite(os.path.join(d, name + '.jpg'), crop, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
            cv2.imwrite(os.path.join(d, name + '_mask.png'), mask)
            samp = range(0, x1 - x0 + 1, 16)
            rows.append(dict(crop=os.path.join(d, name + '.jpg'), mask=os.path.join(d, name + '_mask.png'), face=f['face'], tablet=f.get('tablet', ''),
                             script=f.get('script', ''), side=f['side'], line_no=i + 1, n_lines=len(L), y_center=round(float(q['y_deskewed']), 1),
                             x0=x0, y0=Y0, x1=x1 + 1, y1=Y1, pitch_px=round(pitch, 1), skew_deg=skew,
                             seam_up=' '.join(f'{x0 + k}:{up_c[k]:.0f}' for k in samp), seam_down=' '.join(f'{x0 + k}:{dn_c[k]:.0f}' for k in samp)))
        if a.limit and n_faces >= a.limit:
            break
    with open(os.path.join(a.out, 'crops.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    h = np.median([(r['y1'] - r['y0']) / r['pitch_px'] for r in rows])
    print(f'{len(rows)} line-shaped crops from {n_faces} faces; median bounding-box height {h:.2f} pitch -> {a.out}')


if __name__ == '__main__':
    main()
