"""Side-by-side sample of line crops at today's height and with the calibrated extension, with the eBL sign boxes drawn.

  python -m line_segmentation.report.crop_height_samples [--n 30] [--extend 0.2] [--seed 4] [--crops tablet_data/ebl_tablets/line_crops_unet/crops.csv]
      [--out tablet_data/ebl_tablets/linemodel/crop_height_samples]

Draws --n crops at random (seeded, spread evenly over the scripts) from test faces with sign-annotation coverage >= 0.9,
and cuts each line twice, in the frame it was cut in (deskewed face, or the line's own frame for crops cut along their tilt):
the stored crop, and the same rectangle with its top and bottom moved out by a further --extend line spacings. For both, the share of the line's
own sign-box area inside the crop and the share of the crop's box area that belongs to other lines are computed as in
eval/height_calibration.py. Writes <out>/crops/<crop>__now.jpg and __extend.jpg (plain crops, for use elsewhere),
<out>/sheet_<k>.jpg (ten lines per sheet: stored left, extended right; own signs' boxes green, other lines' orange) and
<out>/samples.csv.
"""
import argparse
import collections
import csv
import os
import random

import cv2
import numpy as np

from line_segmentation.crops.crop_frame import box_in_frame, crop_matrix
from line_segmentation.eval.box_containment import face_boxes
from line_segmentation.paths import EBL


def box_area_in(rect, b):
    x0, y0, x1, y1 = rect
    return max(0.0, min(x1, b[2]) - max(x0, b[0])) * max(0.0, min(y1, b[3]) - max(y0, b[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=30); ap.add_argument('--extend', type=float, default=0.2); ap.add_argument('--seed', type=int, default=4)
    ap.add_argument('--crops', default=os.path.join(EBL, 'line_crops_unet', 'crops.csv'))
    ap.add_argument('--out', default=os.path.join(EBL, 'linemodel', 'crop_height_samples'))
    ap.add_argument('--lines', default=None, help='CSV with face, line_index, crop: show these lines (their crop, with that truth line as "own") instead of a random sample')
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, 'crops'), exist_ok=True)
    faces = {f['face']: f for f in csv.DictReader(open(os.path.join(EBL, 'faces', 'faces.csv')))}
    gt = {t['tablet']: t for t in csv.DictReader(open(os.path.join(EBL, 'ground_truth', 'tablets.csv')))}
    split = {r['face']: r['split'] for r in csv.DictReader(open(os.path.join(EBL, 'split.csv')))}
    by_script = collections.defaultdict(list)
    for r in csv.DictReader(open(a.crops)):
        f = faces.get(r['face'])
        if f and split.get(r['face']) == 'test' and f['coverage'] and float(f['coverage']) >= 0.9 and f['n_columns'] == '1' and f['tablet'] in gt:
            by_script[f['script']].append(r)
    random.seed(a.seed); scripts = sorted(by_script); pick = []; forced = {}
    if a.lines:
        idx = {os.path.basename(r['crop']): r for rs in by_script.values() for r in rs}
        for q in csv.DictReader(open(a.lines)):
            if q['crop'] in idx:
                pick.append(idx[q['crop']]); forced[len(pick) - 1] = int(q['line_index'])
    else:
        for k, sc in enumerate(scripts):
            n = a.n // len(scripts) + (1 if k < a.n % len(scripts) else 0)
            pick += random.sample(by_script[sc], n)
    rows, panels, cache = [], [], {}
    for pi, r in enumerate(pick):
        f = faces[r['face']]; pitch = float(r['pitch_px'])
        if r['face'] not in cache:
            cache = {r['face']: (cv2.imread(f['file']), face_boxes(f, gt[f['tablet']]))}
        img, boxes = cache[r['face']]; H, W = img.shape[:2]
        M = crop_matrix(r, W, H); rot = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        B = [(li, *box_in_frame(M, x0, y0, x1, y1)) for li, x0, y0, x1, y1 in boxes]
        now = (float(r['x0']), float(r['y0']), float(r['x1']), float(r['y1']))
        ext = (now[0], max(0.0, now[1] - a.extend * pitch), now[2], min(float(H), now[3] + a.extend * pitch))
        per_line = collections.Counter()
        for li, *b in B:
            per_line[li] += box_area_in(now, b)
        own_li = forced.get(pi, per_line.most_common(1)[0][0] if per_line else None)
        area = collections.Counter()
        for li, *b in B:
            area[li] += (b[2] - b[0]) * (b[3] - b[1])
        stats = {}
        for name, rect in (('now', now), ('extend', ext)):
            own = sum(box_area_in(rect, b) for li, *b in B if li == own_li); oth = sum(box_area_in(rect, b) for li, *b in B if li != own_li)
            stats[name] = (own / area[own_li] if own_li is not None and area[own_li] else float('nan'), oth / (own + oth) if own + oth else 0.0)
        cid = os.path.splitext(os.path.basename(r['crop']))[0]
        imgs = {}
        for name, rect in (('now', now), ('extend', ext)):
            x0, y0, x1, y1 = (int(round(v)) for v in rect)
            crop = rot[y0:y1, x0:x1].copy(); cv2.imwrite(os.path.join(a.out, 'crops', f'{cid}__{name}.jpg'), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            vis = crop.copy()
            for li, bx0, by0, bx1, by1 in B:
                c = (60, 190, 60) if li == own_li else (0, 140, 255)
                cv2.rectangle(vis, (int(bx0 - x0), int(by0 - y0)), (int(bx1 - x0), int(by1 - y0)), c, max(1, int(pitch / 60)))
            imgs[name] = vis
        rows.append(dict(crop=cid, own_line_index=own_li, script=f['script'], pitch_px=pitch, now_h_px=int(now[3] - now[1]), extend_h_px=int(ext[3] - ext[1]),
                         own_now=round(stats['now'][0], 3), own_extend=round(stats['extend'][0], 3),
                         foreign_now=round(stats['now'][1], 3), foreign_extend=round(stats['extend'][1], 3)))
        panels.append((cid, f['script'], imgs, stats))
    with open(os.path.join(a.out, 'samples.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # sheets: ten lines each, stored left, extended right, at one common width
    WIDTH = 900
    for k in range(0, len(panels), 10):
        blocks = []
        for j, (cid, sc, imgs, st) in enumerate(panels[k:k + 10]):
            pair = []
            for name, label in (('now', 'stored crop'), ('extend', f'extended by {a.extend:g} line spacings')):
                im = imgs[name]; s = WIDTH / im.shape[1]; im = cv2.resize(im, (WIDTH, max(1, int(im.shape[0] * s))))
                bar = np.full((24, WIDTH, 3), 245, np.uint8)
                cv2.putText(bar, f'{label}: own signs {st[name][0]:.0%}, other lines {st[name][1]:.0%}', (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
                pair.append(np.vstack([bar, im]))
            h = max(p.shape[0] for p in pair)
            pair = [cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, 12, cv2.BORDER_CONSTANT, value=(252, 252, 251)) for p in pair]
            head = np.full((28, 2 * (WIDTH + 12), 3), 252, np.uint8)
            cv2.putText(head, f'{cid}  ({sc})' + (f'  - truth line {forced[k + j]}' if (k + j) in forced else ''), (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 1, cv2.LINE_AA)
            blocks.append(np.vstack([head, np.hstack(pair), np.full((14, 2 * (WIDTH + 12), 3), 252, np.uint8)]))
        cv2.imwrite(os.path.join(a.out, f'sheet_{k // 10 + 1}.jpg'), np.vstack(blocks), [cv2.IMWRITE_JPEG_QUALITY, 85])
    own_n = np.array([r['own_now'] for r in rows]); own_e = np.array([r['own_extend'] for r in rows])
    fn = np.array([r['foreign_now'] for r in rows]); fe = np.array([r['foreign_extend'] for r in rows])
    print(f"{len(rows)} crops {dict(collections.Counter(r['script'] for r in rows))}")
    print(f"own signs inside, median: stored {np.nanmedian(own_n):.0%} -> extended {np.nanmedian(own_e):.0%} | lines >= 95%: {np.mean(own_n >= 0.95):.0%} -> {np.mean(own_e >= 0.95):.0%}")
    print(f"other lines' signs in crop, median: stored {np.median(fn):.0%} -> extended {np.median(fe):.0%} | crop height, median: {np.median([r['now_h_px'] / r['pitch_px'] for r in rows]):.2f} -> {np.median([r['extend_h_px'] / r['pitch_px'] for r in rows]):.2f} line spacings")
    print('->', a.out)


if __name__ == '__main__':
    main()
