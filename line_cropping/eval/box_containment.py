"""How well crops hold their own line's signs and keep out other lines' signs, measured with the eBL sign boxes.

  python -m line_cropping.eval.box_containment --sets rect=tablet_data/ebl_tablets/line_crops_unet/crops.csv \\
      seam=tablet_data/ebl_tablets/line_crops_seam/crops.csv [--split test] [--min-cov 0.9] [--out <dir>]

For every face of --split with sign-annotation coverage >= --min-cov that every crop set covers, the eBL sign boxes are
mapped into the deskewed face (the frame each crop set was cut in) and grouped by transliteration line. A crop set is
either rectangles (crops.csv x0..y1), rectangles along a line's own tilt (line_angle_deg filled, make_crops.py), or
line-shaped crops with a mask (crops.csv 'mask' column, seam_crops.py). Areas are counted on a pixel grid (every second pixel). Each truth line is given the crop that holds most of its box area.
  own_in_crop     share of the line's box area inside its crop (1 = every sign whole)
  foreign_share   share of the box area inside a crop that belongs to other lines (0 = none of the neighbours' signs)
  half_neighbour  the crop holds at least half of some other line's box area
  crop_area       crop area / (line length x pitch), how much image each line keeps
eBL boxes are loose (a median 1.2 line spacings tall, so neighbouring lines' boxes overlap), so neither measure can reach
its ideal; they are for comparing crop sets on the same faces.
"""
import argparse
import collections
import csv
import json
import os

import cv2
import numpy as np

from line_cropping.paths import EBL

STEP = 2
SIGN_TYPES = ('HasSign', 'Number', 'Damaged', 'PartiallyBroken', 'UnclearSign')


def face_boxes(f, gt):
    """eBL sign boxes of a face, in face pixels: [(line_index, x0, y0, x1, y1)]."""
    tdir = os.path.join(EBL, f['script'], f['tablet'])
    ann = json.load(open(os.path.join(tdir, 'annotations.json'), encoding='utf-8')).get('annotations', [])
    pw, ph = float(gt['photo_w']), float(gt['photo_h']); cx0, cy0 = float(f['crop_x0']), float(f['crop_y0'])
    W, H = int(f['width']), int(f['height'])
    lp = os.path.join(EBL, 'faces', 'lines', f['face'] + '.csv')
    idx = {int(r['line_index']) for r in csv.DictReader(open(lp))} if os.path.exists(lp) else set()
    out = []
    for an in ann:
        d, g = an.get('data') or {}, an.get('geometry') or {}
        if d.get('type') not in SIGN_TYPES or not d.get('path') or d['path'][0] not in idx:
            continue
        x0 = g['x'] / 100 * pw - cx0; y0 = g['y'] / 100 * ph - cy0; x1 = x0 + g['width'] / 100 * pw; y1 = y0 + g['height'] / 100 * ph
        if 0 <= (x0 + x1) / 2 < W and 0 <= (y0 + y1) / 2 < H:
            out.append((d['path'][0], x0, y0, x1, y1))
    return out


def seam_pts(txt):
    p = np.array([[float(v) for v in q.split(':')] for q in txt.split()])
    return p[:, 0], p[:, 1]


def crop_masks(rows, W, H, grow=0.0):
    """Boolean masks on the STEP grid of the deskewed face, one per crop row. grow > 0 (line-shaped crops only) moves
    both seams outward by grow x pitch, so neighbouring crops share a band of rows, like make_crops --overlap."""
    h, w = (H + STEP - 1) // STEP, (W + STEP - 1) // STEP; out = []
    for r in rows:
        m = np.zeros((h, w), bool); x0, y0, x1, y1 = int(r['x0']), int(r['y0']), int(r['x1']), int(r['y1'])
        if grow > 0 and r.get('seam_up'):
            g = grow * float(r['pitch_px']); cols = np.arange(x0, x1)
            xu, yu = seam_pts(r['seam_up']); xd, yd = seam_pts(r['seam_down'])
            up = np.interp(cols, xu, yu) - g; dn = np.interp(cols, xd, yd) + g
            yy = (np.arange(h) * STEP)[:, None]; cc = cols[::STEP] // STEP
            ok = (cc >= 0) & (cc < w)
            m[:, cc[ok]] = (yy >= up[::STEP][ok][None, :]) & (yy <= dn[::STEP][ok][None, :])
        elif r.get('mask'):
            mk = cv2.imread(r['mask'], cv2.IMREAD_GRAYSCALE) > 0
            ys, xs = np.nonzero(mk)
            ys, xs = (ys + y0) // STEP, (xs + x0) // STEP
            ok = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w); m[ys[ok], xs[ok]] = True
        elif r.get('line_angle_deg'):                   # rectangle in the line's own frame: test the grid points in that frame
            Ml = cv2.getRotationMatrix2D((float(r['line_cx']), float(r['line_cy'])), float(r['line_angle_deg']), 1.0)
            Ms = cv2.getRotationMatrix2D((W / 2, H / 2), float(r['skew_deg']), 1.0)
            A = np.vstack([Ml, [0, 0, 1]]) @ np.vstack([cv2.invertAffineTransform(Ms), [0, 0, 1]])
            yy, xx = np.mgrid[0:h, 0:w] * STEP
            u, v = A[0, 0] * xx + A[0, 1] * yy + A[0, 2], A[1, 0] * xx + A[1, 1] * yy + A[1, 2]
            m = (u >= x0) & (u < x1) & (v >= y0) & (v < y1)
        else:
            m[max(0, y0) // STEP:max(0, y1) // STEP, max(0, x0) // STEP:max(0, x1) // STEP] = True
        out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', nargs='+', required=True, help='name=path/to/crops.csv, or name=path@grow for line-shaped crops widened by grow pitch')
    ap.add_argument('--faces', default=os.path.join(EBL, 'faces', 'faces.csv'))
    ap.add_argument('--split', default='test'); ap.add_argument('--min-cov', type=float, default=0.9)
    ap.add_argument('--out', default=os.path.join(EBL, 'linemodel', 'box_containment'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sets = dict(s.split('=', 1) for s in a.sets)
    grow = {n: float(p.split('@')[1]) if '@' in p else 0.0 for n, p in sets.items()}
    sets = {n: p.split('@')[0] for n, p in sets.items()}
    crops = {}
    for name, p in sets.items():
        d = collections.defaultdict(list)
        for r in csv.DictReader(open(p)):
            d[r['face']].append(r)
        crops[name] = d
    faces = {f['face']: f for f in csv.DictReader(open(a.faces))}
    gt = {t['tablet']: t for t in csv.DictReader(open(os.path.join(EBL, 'ground_truth', 'tablets.csv')))}
    split = {r['face']: r['split'] for r in csv.DictReader(open(os.path.join(EBL, 'split.csv')))}
    common = [fc for fc in crops[next(iter(sets))] if all(fc in crops[n] for n in sets)]
    common = [fc for fc in common if split.get(fc) == a.split and faces[fc]['coverage'] and float(faces[fc]['coverage']) >= a.min_cov
              and faces[fc]['tablet'] in gt and faces[fc]['n_columns'] == '1']
    line_rows = []
    for fc in common:
        f = faces[fc]; W, H = int(f['width']), int(f['height'])
        B = face_boxes(f, gt[f['tablet']])
        if not B:
            continue
        pitch = float(crops[next(iter(sets))][fc][0]['pitch_px'])
        for name in sets:
            rows = crops[name][fc]; skew = float(rows[0]['skew_deg']); M = cv2.getRotationMatrix2D((W / 2, H / 2), skew, 1.0)
            h, w = (H + STEP - 1) // STEP, (W + STEP - 1) // STEP
            lines = collections.defaultdict(lambda: np.zeros((h, w), bool))
            for li, x0, y0, x1, y1 in B:
                P = np.array([[x0, y0, 1], [x1, y0, 1], [x0, y1, 1], [x1, y1, 1]]) @ M.T
                X0, Y0 = int(max(0, P[:, 0].min()) // STEP), int(max(0, P[:, 1].min()) // STEP)
                X1, Y1 = int(min(W, P[:, 0].max()) // STEP), int(min(H, P[:, 1].max()) // STEP)
                lines[li][Y0:Y1, X0:X1] = True
            masks = crop_masks(rows, W, H, grow[name])
            allbox = np.zeros((h, w), bool)
            for m in lines.values():
                allbox |= m
            areas = {li: m.sum() for li, m in lines.items()}
            over = {li: [int((m & cm).sum()) for cm in masks] for li, m in lines.items()}
            for li, m in lines.items():
                if areas[li] == 0:
                    continue
                c = int(np.argmax(over[li])); cm = masks[c]
                own = over[li][c] / areas[li]
                inbox = (allbox & cm).sum(); foreign = ((allbox & ~m) & cm).sum()
                half = any(over[lj][c] / areas[lj] >= 0.5 for lj in lines if lj != li and areas[lj] > 0)
                r = rows[c]; length = max(1.0, float(r['x1']) - float(r['x0']))
                line_rows.append(dict(set=name, face=fc, script=f['script'], line_index=li, crop=os.path.basename(r['crop']),
                                      own_in_crop=round(own, 4), foreign_share=round(foreign / inbox, 4) if inbox else 0.0, half_neighbour=half,
                                      crop_area=round(cm.sum() * STEP * STEP / (length * pitch), 3)))
    with open(os.path.join(a.out, 'lines.csv'), 'w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(line_rows[0].keys())); wr.writeheader(); wr.writerows(line_rows)
    summary = {}
    for name in sets:
        R = [r for r in line_rows if r['set'] == name]
        own = np.array([r['own_in_crop'] for r in R]); fs = np.array([r['foreign_share'] for r in R])
        summary[name] = dict(lines=len(R), own_median=float(np.median(own)), own_ge_95=float(np.mean(own >= 0.95)), own_ge_90=float(np.mean(own >= 0.90)),
                             foreign_median=float(np.median(fs)), foreign_mean=float(fs.mean()), half_neighbour=float(np.mean([r['half_neighbour'] for r in R])),
                             crop_area_median=float(np.median([r['crop_area'] for r in R])),
                             by_script={sc: dict(own_ge_95=float(np.mean([r['own_in_crop'] >= 0.95 for r in R if r['script'] == sc])),
                                                 foreign_median=float(np.median([r['foreign_share'] for r in R if r['script'] == sc])))
                                        for sc in sorted({r['script'] for r in R})})
    json.dump(summary, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
    print(f'{len(common)} faces')
    for name, S in summary.items():
        print(f"{name:14s} lines {S['lines']:5d} | own signs inside: median {S['own_median']:.0%}, >=95% {S['own_ge_95']:.0%}, >=90% {S['own_ge_90']:.0%} | "
              f"other lines' signs in crop: median {S['foreign_median']:.0%} | half a neighbour {S['half_neighbour']:.1%} | image per line {S['crop_area_median']:.2f}")
        print('               ' + ', '.join(f"{sc} >=95% {v['own_ge_95']:.0%} foreign {v['foreign_median']:.0%}" for sc, v in S['by_script'].items()))
    print('->', a.out)


if __name__ == '__main__':
    main()
