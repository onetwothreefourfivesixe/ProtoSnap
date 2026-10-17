"""Export a self-contained set of tablets with their line crops and sign annotations, for use in another pipeline.

  python -m line_cropping.report.export_tablets [--per-script 15] [--seed 16] [--crops tablet_data/ebl_tablets/line_crops_unet]
      [--out tablet_data/exports/line_crops_60] [--zip]

Selection: eBL tablets whose faces all have line crops in --crops (the line model's crops of the Phase 5 test split,
which the model never trained on), are single-column, have sign-annotation coverage of at least 0.9, and give at least
five crops; --per-script tablets drawn at random (seeded) from each of NA, NB, OA, OB.

Writes, under --out: README.md (columns, coordinate frames, provenance), tablets.csv, faces.csv, line_crops.csv,
signs.csv, faces/<face>.jpg, line_crops/<face>/<crop>.jpg, transliterations/<tablet>.atf; --zip also writes <out>.zip.
"""
import argparse
import collections
import csv
import json
import os
import random
import shutil
import zipfile

import cv2
import numpy as np

from line_cropping.paths import EBL

README = """# Line crops and sign annotations: {n} eBL tablets

Exported {date} from the ProtoSnap line-cropping project (`python -m line_cropping.report.export_tablets`).

## What this is

{n} cuneiform tablets from the Electronic Babylonian Library (eBL), {per} each in four scripts: Neo-Assyrian (NA),
Neo-Babylonian (NB), Old Assyrian (OA) and Old Babylonian (OB). For each tablet: the photographed faces, one image
crop per text line, and the eBL sign annotations (a box and reading for each annotated sign), plus the
transliteration.

The line crops come from the project's learned line model (run 2, a U-Net that finds text lines without a
transliteration) with its default read-out. All {n} tablets are from the held-out test split, so the model never saw
them in training. Every face is single-column and has at least 90% of its transliterated lines covered by sign
boxes; signs that were not annotated by eBL are not in signs.csv.

Counts: {n_faces} faces, {n_crops} line crops, {n_signs} sign boxes ({n_in} of them inside a line crop).

## Files

- `tablets.csv` — one row per tablet: `tablet` (museum number), `script`, `period` and `genres` (eBL),
  `n_faces`, `n_line_crops`, `n_signs`, `transliteration` (path to the ATF file).
- `faces.csv` — one row per face: `face` (id), `tablet`, `side` (obverse, reverse, ...), `image` (path),
  `width`, `height`, `photo_x0`, `photo_y0` (where the face sits in the eBL photograph), `skew_deg` (rotation used for
  the line crops, see frames), `line_pitch_px` (typical distance between lines), `annotation_coverage`.
- `line_crops.csv` — one row per line crop, in reading order within a face: `crop_id`, `tablet`, `face`, `line_no`
  (1 = top line), `image` (path), `x0`, `y0`, `x1`, `y1` (the crop rectangle in the deskewed face frame), `skew_deg`,
  `atf_line` (the transliteration line label, e.g. `3` or `5'`; filled only where the number of crops on the face
  equals the transliteration's line count, otherwise blank), `n_signs` (annotated signs whose centre is in the crop).
- `signs.csv` — one row per annotated sign: `sign_id`, `tablet`, `face`, `sign_name` (eBL sign list), `reading`
  (transliterated value), `type` (HasSign, Number, Damaged, PartiallyBroken, UnclearSign), `ebl_label` (eBL's line
  label, e.g. `o 3` = obverse line 3), `ebl_path` (eBL's position of the sign in the transliteration:
  line, word, sign), `face_x0..face_y1` (box in face image pixels), `deskew_x0..deskew_y1` (box in the deskewed face
  frame), `crop_id` (the line crop containing the box centre, blank if none), `crop_x0..crop_y1` (box relative to that
  crop image).
- `faces/`, `line_crops/` — the images; `transliterations/<tablet>.atf` — eBL transliteration (ATF).

## Coordinate frames

- **Face frame:** pixels of `faces/<face>.jpg`, origin top left.
- **Deskewed face frame:** the face image rotated by `skew_deg` degrees about its centre (OpenCV
  `cv2.getRotationMatrix2D((width / 2, height / 2), skew_deg, 1.0)`, positive = counter-clockwise), same size. The line
  crops are cut from this rotated image, so a crop's rectangle and the `deskew_*` sign boxes are in this frame.
- **Crop frame:** pixels of the crop image = deskewed frame minus (`x0`, `y0`) of the crop.

## Caveats

- Line crops are model output, not ground truth. On test faces like these the model finds about 87 lines in 100 at
  about 88% precision; occasionally a crop holds no annotated sign (`n_signs` = 0) or a line is missed.
- Crops are gap-bounded: each runs from the middle of the gap above its line to the middle of the gap below, with
  half a line pitch of margin at the ends; a tall sign can be clipped at the boundary.
- Sign boxes were drawn by eBL annotators on the eBL photographs; a few boxes overlap or are loose, and a tall box can
  extend past the top or bottom of its crop, so `crop_*` coordinates can be negative or exceed the crop size.

## Provenance and licence

Photographs and sign annotations: the Electronic Babylonian Library (https://www.ebl.lmu.de/), photographs by the
holding museums (British Museum images CC BY-NC-SA 4.0; others vary). Sign annotations are also published as the eBL
sign dataset (Lewenstein et al. 2026, Journal of Open Humanities Data, https://doi.org/10.5334/johd.503; CC BY-NC
4.0). For research use; check the holding institution's terms before redistributing the images.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-script', type=int, default=15)
    ap.add_argument('--seed', type=int, default=16)
    ap.add_argument('--crops', default=os.path.join(EBL, 'line_crops_unet'))
    ap.add_argument('--out', default='tablet_data/exports/line_crops_60')
    ap.add_argument('--zip', action='store_true')
    a = ap.parse_args()
    faces = {f['face']: f for f in csv.DictReader(open(os.path.join(EBL, 'faces', 'faces.csv')))}
    gt = {t['tablet']: t for t in csv.DictReader(open(os.path.join(EBL, 'ground_truth', 'tablets.csv')))}
    crops = collections.defaultdict(list)
    for r in csv.DictReader(open(os.path.join(a.crops, 'crops.csv'))):
        crops[r['face']].append(r)
    by_tab = collections.defaultdict(list)
    for fc in crops:
        by_tab[faces[fc]['tablet']].append(faces[fc])
    cand = collections.defaultdict(list)
    for t, fs in by_tab.items():
        if (all(f['n_columns'] == '1' and f['coverage'] and float(f['coverage']) >= 0.9 for f in fs)
                and sum(len(crops[f['face']]) for f in fs) >= 5 and t in gt):
            cand[fs[0]['script']].append(t)
    random.seed(a.seed)
    pick = []
    for sc in ('NA', 'NB', 'OA', 'OB'):
        pool = sorted(cand[sc]); random.shuffle(pool); pick += pool[:a.per_script]
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    for d in ('faces', 'line_crops', 'transliterations'):
        os.makedirs(os.path.join(a.out, d))
    T, F, C, S = [], [], [], []
    for tab in pick:
        sc = by_tab[tab][0]['script']
        tdir = os.path.join(EBL, sc, tab)
        frag = json.load(open(os.path.join(tdir, 'fragment.json'), encoding='utf-8'))
        ann = json.load(open(os.path.join(tdir, 'annotations.json'), encoding='utf-8')).get('annotations', [])
        pw, ph = float(gt[tab]['photo_w']), float(gt[tab]['photo_h'])
        atf_path = os.path.join('transliterations', tab + '.atf')
        open(os.path.join(a.out, atf_path), 'w', encoding='utf-8').write(frag.get('atf') or '')
        n_sign_tab = n_crop_tab = 0
        for f in sorted(by_tab[tab], key=lambda f: f['face']):
            fid = f['face']; W, H = int(f['width']), int(f['height'])
            cx0, cy0 = float(f['crop_x0']), float(f['crop_y0'])
            cs = sorted(crops[fid], key=lambda r: int(r['line_no']))
            skew = float(cs[0]['skew_deg'])
            img_dst = os.path.join('faces', fid + '.jpg'); shutil.copy2(f['file'], os.path.join(a.out, img_dst))
            F.append(dict(face=fid, tablet=tab, side=f['side'], image=img_dst, width=W, height=H, photo_x0=int(cx0), photo_y0=int(cy0),
                          skew_deg=skew, line_pitch_px=f['pitch_px'], annotation_coverage=f['coverage']))
            # line indices of this face (eBL path[0]) decide which signs belong to it
            lp = os.path.join(EBL, 'faces', 'lines', fid + '.csv')
            line_idx = {int(r['line_index']) for r in csv.DictReader(open(lp))} if os.path.exists(lp) else set()
            M = cv2.getRotationMatrix2D((W / 2, H / 2), skew, 1.0)
            os.makedirs(os.path.join(a.out, 'line_crops', fid), exist_ok=True)
            crop_rows = []
            for r in cs:
                cid = f"{fid}__L{int(r['line_no']):02d}"
                dst = os.path.join('line_crops', fid, cid + '.jpg'); shutil.copy2(r['crop'], os.path.join(a.out, dst))
                crop_rows.append(dict(crop_id=cid, tablet=tab, face=fid, line_no=int(r['line_no']), image=dst, x0=int(r['x0']), y0=int(r['y0']),
                                      x1=int(r['x1']), y1=int(r['y1']), skew_deg=skew, atf_line=r['atf_line'], n_signs=0))
            for k, an in enumerate(ann):
                d, g = an.get('data') or {}, an.get('geometry') or {}
                if d.get('type') not in ('HasSign', 'Number', 'Damaged', 'PartiallyBroken', 'UnclearSign') or not d.get('path'):
                    continue
                x0 = g['x'] / 100 * pw - cx0; y0 = g['y'] / 100 * ph - cy0
                x1 = x0 + g['width'] / 100 * pw; y1 = y0 + g['height'] / 100 * ph
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                if d['path'][0] not in line_idx or not (0 <= mx < W and 0 <= my < H):
                    continue
                P = np.array([[x0, y0, 1], [x1, y0, 1], [x0, y1, 1], [x1, y1, 1]]) @ M.T
                dx0, dy0, dx1, dy1 = P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max()
                dcx, dcy = (dx0 + dx1) / 2, (dy0 + dy1) / 2
                hit = next((c for c in crop_rows if c['x0'] <= dcx < c['x1'] and c['y0'] <= dcy < c['y1']), None)
                if hit: hit['n_signs'] += 1
                S.append(dict(sign_id=f'{fid}__s{k:04d}', tablet=tab, face=fid, sign_name=d.get('signName', ''), reading=d.get('value', ''), type=d['type'],
                              ebl_label=(an.get('croppedSign') or {}).get('label', ''), ebl_path='.'.join(map(str, d['path'])),
                              face_x0=round(x0), face_y0=round(y0), face_x1=round(x1), face_y1=round(y1),
                              deskew_x0=round(dx0), deskew_y0=round(dy0), deskew_x1=round(dx1), deskew_y1=round(dy1),
                              crop_id=hit['crop_id'] if hit else '',
                              crop_x0=round(dx0 - hit['x0']) if hit else '', crop_y0=round(dy0 - hit['y0']) if hit else '',
                              crop_x1=round(dx1 - hit['x0']) if hit else '', crop_y1=round(dy1 - hit['y0']) if hit else ''))
                n_sign_tab += 1
            C += crop_rows; n_crop_tab += len(crop_rows)
        T.append(dict(tablet=tab, script=sc, period=(frag.get('script') or {}).get('period', ''),
                      genres=';'.join('/'.join(g.get('category', [])) for g in frag.get('genres') or []),
                      n_faces=len(by_tab[tab]), n_line_crops=n_crop_tab, n_signs=n_sign_tab, transliteration=atf_path))
    for name, rows in (('tablets.csv', T), ('faces.csv', F), ('line_crops.csv', C), ('signs.csv', S)):
        with open(os.path.join(a.out, name), 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    import datetime
    open(os.path.join(a.out, 'README.md'), 'w', encoding='utf-8').write(README.format(
        n=len(T), per=a.per_script, date=datetime.date.today().isoformat(), n_faces=len(F), n_crops=len(C), n_signs=len(S),
        n_in=sum(1 for s in S if s['crop_id'])))
    print(f"{len(T)} tablets {dict(collections.Counter(t['script'] for t in T))}, {len(F)} faces, {len(C)} line crops, "
          f"{len(S)} signs ({sum(1 for s in S if s['crop_id'])} inside a crop; crops with no sign: {sum(1 for c in C if c['n_signs'] == 0)}) -> {a.out}")
    if a.zip:
        z = a.out.rstrip('/') + '.zip'
        with zipfile.ZipFile(z, 'w', zipfile.ZIP_STORED) as zf:
            for dp, _, fs in os.walk(a.out):
                for fn in fs:
                    p = os.path.join(dp, fn); zf.write(p, os.path.relpath(p, os.path.dirname(a.out.rstrip('/'))))
        print('->', z, f'{os.path.getsize(z) / 1e6:.0f} MB')


if __name__ == '__main__':
    main()
