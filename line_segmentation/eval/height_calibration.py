"""Crop height calibrated on the eBL sign boxes: how much of its own line a crop keeps against how much of the other
lines it takes in, per script, for a range of crop heights.

  python -m line_cropping.eval.height_calibration [--min-cov 0.9] [--out tablet_data/ebl_tablets/linemodel/height_calibration]

Two sweeps, both on single-column faces with sign-annotation coverage >= --min-cov (so neighbouring lines are annotated):
  truth   crops centred on the truth lines (every such face, tune and test): the line's centre line +-h x pitch,
          h = 0.40 ... 0.95. Shows the geometry of the writing, independent of any model.
  model   crops around the line model's lines (run 2, test faces, tablet_data/ebl_tablets/predictions): from the midpoint
          to the line above to the midpoint to the line below (what gap-bounded crops do), extended by e x pitch into the
          neighbours, e = 0 ... 0.5, without the band-edge cap of make_crops --overlap; the first and last line get half a
          pitch plus e. This is the deployable setting.
A crop runs along its (straight, tilted) line from the line's start to its end plus half a pitch. Overlaps are exact
per box and column: for each sign box, the crop's vertical interval is intersected with the box's in every column.
Each truth line is given the crop holding most of its box area. Measures, as in box_containment.py:
  own      share of a line's box area inside its crop       foreign  share of a crop's box area that is other lines'
  half     the crop holds at least half of another line's boxes
eBL boxes are loose (about 1.2 line spacings tall), so 100% own at 0% foreign is unreachable; the curves are for choosing
a setting. Writes <out>/sweep.csv, summary.json and calibration.png.
"""
import argparse
import collections
import csv
import json
import os

import numpy as np

from line_cropping.eval.box_containment import face_boxes
from line_cropping.paths import EBL

STEP = 2


def seg_line(ax0, ay0, ax1, ay1):
    if ax1 < ax0:
        ax0, ay0, ax1, ay1 = ax1, ay1, ax0, ay0
    k = (ay1 - ay0) / max(1e-6, ax1 - ax0)
    return dict(k=k, b=ay0 - k * ax0, x0=ax0, x1=ax1, yc=(ay0 + ay1) / 2)


def overlaps(regions, boxes):
    """regions: [(line, up, dn, x0, x1)] with line y(x) = k x + b and the crop rows y(x)-up .. y(x)+dn;
    boxes: array (n, 4) x0 y0 x1 y1. Returns matrix (n_regions, n_boxes) of overlap areas."""
    out = np.zeros((len(regions), len(boxes)))
    for j, (bx0, by0, bx1, by1) in enumerate(boxes):
        for i, (ln, up, dn, rx0, rx1) in enumerate(regions):
            lo, hi = max(bx0, rx0), min(bx1, rx1)
            if hi <= lo:
                continue
            xs = np.arange(lo, hi, STEP) + STEP / 2
            yc = ln['k'] * xs + ln['b']
            ov = np.clip(np.minimum(by1, yc + dn) - np.maximum(by0, yc - up), 0, None)
            out[i, j] = ov.sum() * STEP
    return out


def score(regions, tlines, boxes, box_line):
    """Per truth line: own share, foreign share of its crop, half-neighbour flag."""
    O = overlaps(regions, boxes)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    rows = []
    lines = sorted(set(box_line))
    A = {li: area[box_line == li].sum() for li in lines}
    per_crop_line = {li: O[:, box_line == li].sum(1) for li in lines}      # overlap of each crop with each line
    for li in lines:
        if A[li] <= 0:
            continue
        c = int(np.argmax(per_crop_line[li])); own = per_crop_line[li][c]
        foreign = sum(per_crop_line[lj][c] for lj in lines if lj != li)
        half = any(per_crop_line[lj][c] / A[lj] >= 0.5 for lj in lines if lj != li and A[lj] > 0)
        rows.append((own / A[li], foreign / (own + foreign) if own + foreign else 0.0, half))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--faces', default=os.path.join(EBL, 'faces', 'faces.csv'))
    ap.add_argument('--pred', default=os.path.join(EBL, 'predictions')); ap.add_argument('--method', default='unet')
    ap.add_argument('--min-cov', type=float, default=0.9)
    ap.add_argument('--out', default=os.path.join(EBL, 'linemodel', 'height_calibration'))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    gt = {t['tablet']: t for t in csv.DictReader(open(os.path.join(EBL, 'ground_truth', 'tablets.csv')))}
    split = {r['face']: r['split'] for r in csv.DictReader(open(os.path.join(EBL, 'split.csv')))}
    faces = [f for f in csv.DictReader(open(a.faces)) if f['coverage'] and float(f['coverage']) >= a.min_cov and f['n_columns'] == '1' and f['tablet'] in gt]
    H_TRUTH = np.round(np.arange(0.40, 0.951, 0.05), 2); E_MODEL = np.round(np.arange(0.0, 0.501, 0.05), 2)
    acc = collections.defaultdict(list)          # (sweep, param, script) -> [(own, foreign, half)]
    n_faces = collections.Counter()
    for f in faces:
        lp = os.path.join(EBL, 'faces', 'lines', f['face'] + '.csv')
        if not os.path.exists(lp):
            continue
        T = sorted([dict(seg_line(float(r['ax0']), float(r['ay0']), float(r['ax1']), float(r['ay1'])), li=int(r['line_index'])) for r in csv.DictReader(open(lp))],
                   key=lambda d: d['yc'])
        B = face_boxes(f, gt[f['tablet']])
        if len(T) < 2 or not B:
            continue
        boxes = np.array([b[1:] for b in B]); box_line = np.array([b[0] for b in B])
        pitch = float(np.median(np.diff([t['yc'] for t in T])))
        if pitch <= 2:
            continue
        sc = f['script']; n_faces[('truth', sc)] += 1
        for h in H_TRUTH:
            regs = [(t, h * pitch, h * pitch, t['x0'] - 0.5 * pitch, t['x1'] + 0.5 * pitch) for t in T]
            acc[('truth', h, sc)] += score(regs, T, boxes, box_line)
        # the model's lines (test faces only)
        sp = os.path.join(a.pred, a.method + '_segments', f['face'] + '.csv')
        if split.get(f['face']) != 'test' or not os.path.exists(sp):
            continue
        P = sorted([seg_line(float(r['ax0']), float(r['ay0']), float(r['ax1']), float(r['ay1'])) for r in csv.DictReader(open(sp))], key=lambda d: d['yc'])
        if len(P) < 2:
            continue
        pp = float(np.median(np.diff([p['yc'] for p in P]))); n_faces[('model', sc)] += 1
        for e in E_MODEL:
            regs = []
            for i, p in enumerate(P):
                up = (p['yc'] - P[i - 1]['yc']) / 2 if i > 0 else pp / 2
                dn = (P[i + 1]['yc'] - p['yc']) / 2 if i < len(P) - 1 else pp / 2
                regs.append((p, up + e * pp, dn + e * pp, p['x0'] - 0.5 * pp, p['x1'] + 0.5 * pp))
            acc[('model', e, sc)] += score(regs, T, boxes, box_line)
    rows = []
    for (sweep, param, sc), R in sorted(acc.items()):
        R = np.array(R, dtype=float)
        rows.append(dict(sweep=sweep, param=param, script=sc, lines=len(R), own_median=np.median(R[:, 0]), own_ge_95=np.mean(R[:, 0] >= 0.95),
                         own_ge_90=np.mean(R[:, 0] >= 0.90), foreign_median=np.median(R[:, 1]), half=np.mean(R[:, 2])))
    for sweep, params in (('truth', H_TRUTH), ('model', E_MODEL)):
        for param in params:
            R = np.array(sum((acc[(sweep, param, sc)] for sc in ('NA', 'NB', 'OA', 'OB')), []), dtype=float)
            if len(R):
                rows.append(dict(sweep=sweep, param=param, script='all', lines=len(R), own_median=np.median(R[:, 0]), own_ge_95=np.mean(R[:, 0] >= 0.95),
                                 own_ge_90=np.mean(R[:, 0] >= 0.90), foreign_median=np.median(R[:, 1]), half=np.mean(R[:, 2])))
    with open(os.path.join(a.out, 'sweep.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        w.writerows([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows])
    json.dump(dict(faces={f'{k[0]}_{k[1]}': v for k, v in n_faces.items()}), open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
    for sweep in ('truth', 'model'):
        print(f'== {sweep} ({"half-height h" if sweep == "truth" else "extension e beyond the midpoint"}), faces: ' +
              ', '.join(f'{sc} {n_faces[(sweep, sc)]}' for sc in ('NA', 'NB', 'OA', 'OB')))
        for r in [r for r in rows if r['sweep'] == sweep and r['script'] == 'all']:
            print(f"  {r['param']:.2f}: own median {r['own_median']:.0%}, >=90% {r['own_ge_90']:.0%}, >=95% {r['own_ge_95']:.0%} | foreign median {r['foreign_median']:.0%} | half {r['half']:.1%}  (lines {r['lines']})")
    print('->', a.out)


if __name__ == '__main__':
    main()
