"""How far each truth line's signs reach above and below it, from the eBL sign boxes: the target of the line model's
height channels (run 4).

  python -m line_segmentation.data.truth.line_heights [--pct 90] [--sweep]

For every line of every eBL face, each sign box gives two signed distances, measured straight up and down from the
truth line at the box's centre: to the box's top edge and to its bottom edge. `up` and `down` are the --pct percentile
of those over the line's boxes (not the maximum, so one sign box assigned to the wrong line does not set the height).
Each is capped at 1.2 line spacings. Lines with fewer than 3 boxes have no reliable height and get valid = 0.
Writes tablet_data/ebl_tablets/faces/line_heights.csv (face, line_index, up, down, n_boxes, valid; face pixels), which
travels with the faces, so training needs no annotation files.

--sweep scores crops around the truth lines at several percentiles against fixed heights, on single-column faces of the
tune split with coverage >= 0.9, measured as in eval/height_calibration.py, and writes nothing. It is the ceiling for
the height channels (perfect lines, perfect heights) and the basis for choosing --pct.
"""
import argparse
import collections
import csv
import os

import numpy as np

from line_segmentation.eval.box_containment import face_boxes
from line_segmentation.eval.height_calibration import score, seg_line
from line_segmentation.paths import EBL

MIN_BOXES = 3
CAP = 1.2


def box_reach(T, B):
    """T: truth lines by line_index (seg_line dicts); B: face_boxes. Returns {line_index: (ups, downs)} in pixels."""
    out = collections.defaultdict(lambda: ([], []))
    for li, x0, y0, x1, y1 in B:
        t = T.get(li)
        if t is None:
            continue
        yc = t['k'] * (x0 + x1) / 2 + t['b']
        out[li][0].append(yc - y0); out[li][1].append(y1 - yc)
    return out


def load_face(f, gt):
    lp = os.path.join(EBL, 'faces', 'lines', f['face'] + '.csv')
    if not os.path.exists(lp) or f['tablet'] not in gt:
        return None
    T = {int(r['line_index']): seg_line(float(r['ax0']), float(r['ay0']), float(r['ax1']), float(r['ay1'])) for r in csv.DictReader(open(lp))}
    if not T:
        return None
    ys = np.sort([t['yc'] for t in T.values()])
    g = np.diff(ys); g = g[g > 0]
    pitch = float(f['pitch_px']) if f['pitch_px'] else (float(np.median(g)) if len(g) else float('nan'))
    return T, face_boxes(f, gt[f['tablet']]), pitch


def heights(reach, pct, pitch):
    """(up, down, n, valid) for one line."""
    u, d = reach; n = len(u)
    if n == 0:
        return float('nan'), float('nan'), 0, 0
    up, dn = float(np.percentile(u, pct)), float(np.percentile(d, pct))
    if pitch == pitch:
        up, dn = min(up, CAP * pitch), min(dn, CAP * pitch)
    return up, dn, n, int(n >= MIN_BOXES and up > 0 and dn > 0)


def sweep(faces, gt):
    split = {r['face']: r['split'] for r in csv.DictReader(open(os.path.join(EBL, 'split.csv')))}
    acc = collections.defaultdict(list); n_faces = 0
    for f in faces:
        if split.get(f['face']) != 'tune' or not f['coverage'] or float(f['coverage']) < 0.9 or f['n_columns'] != '1':
            continue
        L = load_face(f, gt)
        if L is None or len(L[0]) < 2 or not L[1] or L[2] != L[2]:
            continue
        T, B, pitch = L; n_faces += 1
        boxes = np.array([b[1:] for b in B]); box_line = np.array([b[0] for b in B]); R = box_reach(T, B)
        lines = list(T.values())
        for h in (0.5, 0.65, 0.7, 0.75):
            acc[f'fixed +-{h}'] += score([(t, h * pitch, h * pitch, t['x0'] - 0.5 * pitch, t['x1'] + 0.5 * pitch) for t in lines], lines, boxes, box_line)
        for pct in (50, 75, 90, 100):
            regs = []
            for li, t in T.items():
                up, dn, _, valid = heights(R[li], pct, pitch) if li in R else (0, 0, 0, 0)
                if not valid:                                       # a real model would fall back on the face's usual height
                    up = dn = 0.65 * pitch
                regs.append((t, up, dn, t['x0'] - 0.5 * pitch, t['x1'] + 0.5 * pitch))
            acc[f'per line, pct {pct}'] += score(regs, lines, boxes, box_line)
    print(f'{n_faces} tune faces')
    for k, r in acc.items():
        r = np.array(r, float)
        print(f'{k:20s} lines {len(r):5d} | own median {np.median(r[:, 0]):.0%}, >=95% {np.mean(r[:, 0] >= 0.95):.0%} | foreign median {np.median(r[:, 1]):.0%} | half {np.mean(r[:, 2]):.1%}')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--pct', type=float, default=90); ap.add_argument('--sweep', action='store_true')
    a = ap.parse_args()
    faces = list(csv.DictReader(open(os.path.join(EBL, 'faces', 'faces.csv'))))
    gt = {t['tablet']: t for t in csv.DictReader(open(os.path.join(EBL, 'ground_truth', 'tablets.csv')))}
    if a.sweep:
        return sweep(faces, gt)
    rows = []
    for f in faces:
        L = load_face(f, gt)
        if L is None:
            continue
        T, B, pitch = L; R = box_reach(T, B)
        for li in T:
            up, dn, n, valid = heights(R[li], a.pct, pitch) if li in R else (float('nan'), float('nan'), 0, 0)
            rows.append(dict(face=f['face'], line_index=li, up=round(up, 1), down=round(dn, 1), n_boxes=n, valid=valid))
    p = os.path.join(EBL, 'faces', 'line_heights.csv')
    with open(p, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    v = [r for r in rows if r['valid']]
    print(f"{len(rows)} lines, {len(v)} with a height; median up {np.median([r['up'] for r in v]):.0f} px, down {np.median([r['down'] for r in v]):.0f} px -> {p}")


if __name__ == '__main__':
    main()
