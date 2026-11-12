"""Line envelopes from ProtoSnap, step 1: choose lines and write one ProtoSnap target per annotated sign.

  python -m line_cropping.protosnap.line_envelope_prep [--export tablet_data/exports/line_crops_60] [--per-script 20]
      [--pad 0.3] [--seed 5] [--out tablet_data/protosnap_inputs/envelope_pilot]

Idea: fit each annotated sign's font skeleton to the photograph with ProtoSnap, then join a line's fitted skeletons
into its wedge envelope (where the writing really is), and compare that envelope with the line crop.

Lines come from the 60-tablet export (line crops of test faces with eBL sign boxes in crop coordinates). A line is
eligible when its script has a ProtoSnap font (OB: Santakku, NA: Assurbanipal), it holds at least 6 annotated signs,
and at least 80% of them have a skeleton in that font, so the envelope covers most of the line. --per-script lines are
drawn at random (seeded), at most two per tablet.

Each target is cut from the face rotated into its crop's frame (crops/crop_frame.py), around the sign's
box with --pad of its width and height added on every side, so a fitted skeleton can reach outside the eBL box and
outside the crop. Writes per script <out>/<script>/signs/<target>.png, samples.csv (the columns run_test.py reads) and
targets.csv (target -> line, sign, and the box, crop and target positions in the crop's frame, for step 2).
"""
import argparse
import collections
import csv
import os
import random

import cv2
import numpy as np

from line_cropping.crops.crop_frame import crop_matrix
from line_cropping.protosnap.protosnap_prep import sign_hex

FONT = {'OB': 'Santakku', 'NA': 'Assurbanipal'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export', default='tablet_data/exports/line_crops_60')
    ap.add_argument('--per-script', type=int, default=20)
    ap.add_argument('--pad', type=float, default=0.3)
    ap.add_argument('--min-signs', type=int, default=6)
    ap.add_argument('--seed', type=int, default=5)
    ap.add_argument('--out', default='tablet_data/protosnap_inputs/envelope_pilot')
    a = ap.parse_args()
    E = a.export
    script_of = {r['tablet']: r['script'] for r in csv.DictReader(open(os.path.join(E, 'tablets.csv')))}
    faces = {r['face']: r for r in csv.DictReader(open(os.path.join(E, 'faces.csv')))}
    crops = {r['crop_id']: r for r in csv.DictReader(open(os.path.join(E, 'line_crops.csv')))}
    signs = collections.defaultdict(list)
    for s in csv.DictReader(open(os.path.join(E, 'signs.csv'))):
        if s['crop_id']:
            signs[s['crop_id']].append(s)
    meta = list(csv.DictReader(open('prototypes/metadata.csv')))
    name_of = {f: {r['hex']: r['name'] for r in meta if r['name'] and r.get(f) == 'True'} for f in FONT.values()}
    skel = {f: {fn.split('_')[0] for fn in os.listdir(f'skeletons/{f}') if fn.endswith('_adf.csv')} for f in FONT.values()}
    # eligible lines
    cand = collections.defaultdict(list)
    for cid, ss in signs.items():
        sc = script_of[crops[cid]['tablet']]
        if sc not in FONT or len(ss) < a.min_signs:
            continue
        f = FONT[sc]
        ok = [s for s in ss if (h := sign_hex(s['sign_name'])) and h in skel[f] and h in name_of[f]]
        if len(ok) >= 0.8 * len(ss):
            cand[sc].append(cid)
    random.seed(a.seed)
    rot_cache = {}
    for sc, f in FONT.items():
        pool = sorted(cand[sc]); random.shuffle(pool)
        pick, per_tab = [], collections.Counter()
        for cid in pool:
            t = crops[cid]['tablet']
            if per_tab[t] < 2:
                pick.append(cid); per_tab[t] += 1
            if len(pick) == a.per_script:
                break
        d = os.path.join(a.out, sc); os.makedirs(os.path.join(d, 'signs'), exist_ok=True)
        samples, targets = [], []
        for cid in pick:
            c = crops[cid]; fc = faces[c['face']]
            if c['face'] not in rot_cache:
                im = cv2.imread(os.path.join(E, fc['image'])); rot_cache = {c['face']: (im, tuple(int(v) for v in np.median(im.reshape(-1, 3), 0)))}
            im, border = rot_cache[c['face']]; H, W = im.shape[:2]
            rot = cv2.warpAffine(im, crop_matrix(c, W, H), (W, H), borderValue=border)
            for s in signs[cid]:
                h = sign_hex(s['sign_name'])
                if not h or h not in skel[f] or h not in name_of[f]:
                    continue
                x0, y0, x1, y1 = float(s['crop_x0']) + int(c['x0']), float(s['crop_y0']) + int(c['y0']), float(s['crop_x1']) + int(c['x0']), float(s['crop_y1']) + int(c['y0'])
                pw, ph = (x1 - x0) * a.pad, (y1 - y0) * a.pad
                tx0, ty0 = int(max(0, x0 - pw)), int(max(0, y0 - ph)); tx1, ty1 = int(min(W, x1 + pw)), int(min(H, y1 + ph))
                if tx1 - tx0 < 8 or ty1 - ty0 < 8:
                    continue
                fn = f"{s['sign_id']}__{h}.png"
                cv2.imwrite(os.path.join(d, 'signs', fn), rot[ty0:ty1, tx0:tx1])
                samples.append(dict(fn=fn, file_path=os.path.join(d, 'signs', fn), abz='', name=name_of[f][h], hex=h, sign=s['sign_name']))
                targets.append(dict(target=fn[:-4], crop_id=cid, face=c['face'], tablet=c['tablet'], script=sc, line_no=c['line_no'], sign_id=s['sign_id'],
                                    sign=s['sign_name'], hex=h, box_x0=round(x0), box_y0=round(y0), box_x1=round(x1), box_y1=round(y1),
                                    target_x0=tx0, target_y0=ty0, target_x1=tx1, target_y1=ty1,
                                    crop_x0=c['x0'], crop_y0=c['y0'], crop_x1=c['x1'], crop_y1=c['y1'], skew_deg=fc['skew_deg'],
                                    line_angle_deg=c.get('line_angle_deg', ''), line_cx=c.get('line_cx', ''), line_cy=c.get('line_cy', '')))
        for name, rows in (('samples.csv', samples), ('targets.csv', targets)):
            with open(os.path.join(d, name), 'w', newline='', encoding='utf-8') as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f'{sc} ({f}): {len(pick)} lines from {len(per_tab)} tablets, {len(samples)} sign targets '
              f'(eligible lines: {len(cand[sc])}) -> {d}')


if __name__ == '__main__':
    main()
