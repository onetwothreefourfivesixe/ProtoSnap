"""A data set of line crops made by the line model, tied to their face, with the signs inside each crop in reading order.

  python -m line_cropping.report.export_model_line_signs [--crops tablet_data/ebl_tablets/line_crops_unet]
      [--out tablet_data/exports/line_signs_run6] [--limit-faces N]

The crops are model output (crops/make_crops.py on the default model); the signs are eBL's boxes. A sign goes to the crop that holds its
box centre, in that crop's own frame; where crops overlap, to the one whose middle is closest. So a merged or missed line puts signs on the
wrong crop, and each crop records how it compares with eBL's lines. Run truth/full_line_annotations.py first: its levels go into the tables.
Writes README.md, faces.csv, lines.csv, signs.csv, crops_without_signs.csv, signs_outside_crops.csv and lines/<face>/<crop>.jpg under --out.
"""
import argparse
import collections
import csv
import datetime
import json
import os
import shutil

import cv2
import numpy as np

from line_cropping.crops.crop_frame import box_in_frame, crop_matrix
from line_cropping.paths import EBL, LINE_WEIGHTS
from line_cropping.report.export_line_signs import atf_texts, find_text
from line_cropping.truth.ebl_ground_truth import SIGN_TYPES
from line_cropping.truth.full_line_annotations import visible_signs

README = """# Line crops from the line model, with annotated signs

This set holds {n_lines} line crops with {n_signs} sign boxes, from {n_faces} test faces of {n_tablets} eBL tablets. It was exported on {date} with
`python -m line_cropping.report.export_model_line_signs`, using the model `{model}`.

The crops are model output, not annotations. The signs are eBL's boxes, and each is placed in a crop when its box centre falls inside it. All faces are from the
line model's test split, so the model never trained on them. Another {n_empty} crops hold no annotated sign; they are listed in `crops_without_signs.csv`. In addition, {n_out}
signs fall in no crop; they are listed in `signs_outside_crops.csv`.

## Files

- `faces.csv` has one row per face. Its columns are `face`, `tablet`, `script`, `side`, `object`, `image` (the face photograph, relative to the repository), `width`, `height`, `pitch_px`, `coverage`, `tablet_level` and `n_lines`.
- `lines.csv` has one row per crop. Its columns are `line_id`, `face`, `tablet`, `script`, `side`, `object`, `line_no` (the model's order, with 1 at the top), `image`, `x0`, `y0`, `x1`, `y1`, `skew_deg`, `line_angle_deg`, `line_cx`, `line_cy`, `pitch_px`,
  `n_signs`, `ebl_line_index`, `ebl_label`, `ebl_column`, `atf_text`, `signs_visible`, `sign_share`, `main_line_share`, `n_ebl_lines`, `line_kept`, `order_ok`, `flags` and `tablet_level`.
  The `ebl_*` and `atf_text` columns describe eBL's main line in the crop and its transliteration. The `sign_share` column is an estimate: the number of signs of the main line in the crop divided by the visible signs in `atf_text`, capped at 1.
  The `main_line_share` column is the share of the crop's signs that are on the main line, and `line_kept` is the share of the main line's signs on this face that are in this crop.
- `signs.csv` has one row per sign, in left-to-right order within its crop. Its columns are `sign_id`, `line_id`, `face`, `tablet`, `pos`, `sign_name`, `reading`, `type`, `ebl_label`, `ebl_line_index`, `on_main_line`, `ebl_path`,
  `face_x0..face_y1` (the box in the face photograph) and `crop_x0..crop_y1` (the bounds of the box in the crop image).
- The folder `lines/<face>/` holds the crops, one `<line_id>.jpg` per crop.

## Coordinate frames

The line frame is the face rotated with `cv2.getRotationMatrix2D((line_cx, line_cy), line_angle_deg, 1.0)`, at the same size. Where `line_angle_deg` is blank, the face is rotated with
`cv2.getRotationMatrix2D((width / 2, height / 2), skew_deg, 1.0)` instead. A crop's pixels are that frame minus the crop's (`x0`, `y0`).

## Choosing lines

The `flags` column lists any of these problems:

- `few_signs`: the crop has fewer than 3 signs.
- `no_atf`: no transliteration was found for the main line.
- `mixed_lines`: fewer than 75% of the crop's signs are on one eBL line, which often means a merged line or an error in eBL's line assignment.
- `split_line`: the crop holds fewer than 75% of its main line's signs on this face.
- `order`: the signs, read left to right, do not follow eBL's order.

A crop with none of these flags and a `sign_share` of at least 0.8 is a good default, and there are {n_clean} such crops here.

## Caveats

- The crops run 0.2 line spacings past the middle of the gaps and overlap. A crop image therefore also shows parts of its neighbours, whose signs are not listed for it.
- eBL's boxes are loose and sometimes partial, so a crop with few signs may simply be a partly annotated line.

## Licence

The photographs and sign annotations come from the Electronic Babylonian Library (https://www.ebl.lmu.de/), and the photographs were taken by the holding museums. British Museum images are licensed CC BY-NC-SA 4.0, and
the licences of others vary. The sign annotations are also published as the eBL sign dataset (Lewenstein et al. 2026, https://doi.org/10.5334/johd.503; CC BY-NC 4.0). The set is for research use. Check the holding institution's terms before redistributing the images.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--crops', default=os.path.join(EBL, 'line_crops_unet')); ap.add_argument('--out', default='tablet_data/exports/line_signs_run6'); ap.add_argument('--limit-faces', type=int, default=None)
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out, 'lines.csv')):
        raise SystemExit(f'{a.out} already holds an export; remove it or choose another --out')
    faces = {f['face']: f for f in csv.DictReader(open(os.path.join(EBL, 'faces', 'faces.csv')))}
    gt = {t['tablet']: t for t in csv.DictReader(open(os.path.join(EBL, 'ground_truth', 'tablets.csv')))}
    level = {r['tablet']: r['level'] for r in csv.DictReader(open(os.path.join(EBL, 'ground_truth', 'full_line_annotations.csv'), encoding='utf-8')) if r['source'] == 'eBL sign boxes'}
    split = {r['face']: r['split'] for r in csv.DictReader(open(os.path.join(EBL, 'split.csv')))}
    by_face = collections.defaultdict(list)
    for r in csv.DictReader(open(os.path.join(a.crops, 'crops.csv'))):
        by_face[r['face']].append(r)
    names = sorted(by_face)
    if a.limit_faces: names = names[::max(1, len(names) // a.limit_faces)][:a.limit_faces]
    F, L, S, E, O = [], [], [], [], []; skipped = collections.Counter()
    for fi, name in enumerate(names):
        f = faces[name]; tab = f['tablet']
        if split.get(name) != 'test':
            skipped['not a test face'] += 1; continue
        if tab not in gt:
            skipped['no eBL annotations'] += 1; continue
        d = os.path.join(EBL, f['script'], tab); W, H = int(f['width']), int(f['height'])
        ann = json.load(open(os.path.join(d, 'annotations.json'), encoding='utf-8')).get('annotations', [])
        texts = atf_texts(json.load(open(os.path.join(d, 'fragment.json'), encoding='utf-8')).get('atf'))
        lines = {int(r['line_index']): r for r in csv.DictReader(open(os.path.join(EBL, 'faces', 'lines', name + '.csv')))}
        pw, ph = float(gt[tab]['photo_w']), float(gt[tab]['photo_h']); cx0, cy0 = float(f['crop_x0']), float(f['crop_y0'])
        cs = sorted(by_face[name], key=lambda r: int(r['line_no'])); Ms = [crop_matrix(c, W, H) for c in cs]; held = [[] for _ in cs]; line_total = collections.Counter()
        for an in ann:
            dt, g = an.get('data') or {}, an.get('geometry') or {}
            if dt.get('type') not in SIGN_TYPES or not dt.get('path') or dt['path'][0] not in lines:
                continue
            x0 = g['x'] / 100 * pw - cx0; y0 = g['y'] / 100 * ph - cy0; x1 = x0 + g['width'] / 100 * pw; y1 = y0 + g['height'] / 100 * ph
            if not (0 <= (x0 + x1) / 2 < W and 0 <= (y0 + y1) / 2 < H):
                continue
            line_total[dt['path'][0]] += 1; best, hit, box = float('inf'), None, None
            for i, (c, M) in enumerate(zip(cs, Ms)):
                q = box_in_frame(M, x0, y0, x1, y1); bx, by = (q[0] + q[2]) / 2, (q[1] + q[3]) / 2; gap = abs(by - (float(c['y0']) + float(c['y1'])) / 2)
                if float(c['x0']) <= bx < float(c['x1']) and float(c['y0']) <= by < float(c['y1']) and gap < best:
                    best, hit, box = gap, i, q
            sign = dict(tablet=tab, face=name, sign_name=dt.get('signName', ''), reading=dt.get('value', ''), type=dt['type'], ebl_label=(an.get('croppedSign') or {}).get('label', '').strip(), ebl_line_index=dt['path'][0],
                        ebl_path='.'.join(map(str, dt['path'])), face_x0=round(x0), face_y0=round(y0), face_x1=round(x1), face_y1=round(y1))
            if hit is None: O.append(sign)
            else: held[hit].append((sign, box, dt['path']))
        n_face = 0; os.makedirs(os.path.join(a.out, 'lines', name), exist_ok=True)
        for c, items in zip(cs, held):
            line_id = os.path.splitext(os.path.basename(c['crop']))[0]
            if not items:
                E.append(dict(line_id=line_id, face=name, tablet=tab, script=f['script'], line_no=c['line_no'], source_image=c['crop'])); continue
            items.sort(key=lambda t: (t[1][0] + t[1][2]) / 2); n_face += 1
            count = collections.Counter(t[0]['ebl_line_index'] for t in items); main, n_main = count.most_common(1)[0]; r = lines[main]
            col = '' if r['column'] == 'single' else r['column']; text = find_text(texts, '' if f['object'] == 'tablet' else f['object'], f['side'], col, r['line_label'])
            vis = len(visible_signs(text)) if text else 0; share = round(min(1.0, n_main / vis), 3) if vis else ''
            paths = [tuple(t[2][1:]) for t in items if t[0]['ebl_line_index'] == main]; order = float(np.mean([p < q for p, q in zip(paths, paths[1:])])) if len(paths) > 1 else 1.0
            main_share, kept = n_main / len(items), n_main / line_total[main]
            flags = [n for n, on in (('few_signs', len(items) < 3), ('no_atf', not text), ('mixed_lines', main_share < 0.75), ('split_line', kept < 0.75), ('order', order < 0.8)) if on]
            rel = os.path.join('lines', name, line_id + '.jpg'); shutil.copy2(c['crop'], os.path.join(a.out, rel))
            L.append(dict(line_id=line_id, face=name, tablet=tab, script=f['script'], side=f['side'], object=f['object'], line_no=c['line_no'], image=rel, x0=c['x0'], y0=c['y0'], x1=c['x1'], y1=c['y1'], skew_deg=c['skew_deg'],
                          line_angle_deg=c['line_angle_deg'], line_cx=c['line_cx'], line_cy=c['line_cy'], pitch_px=c['pitch_px'], n_signs=len(items), ebl_line_index=main, ebl_label=r['line_label'], ebl_column=col, atf_text=text,
                          signs_visible=vis or '', sign_share=share, main_line_share=round(main_share, 3), n_ebl_lines=len(count), line_kept=round(kept, 3), order_ok=round(order, 2), flags=' '.join(flags), tablet_level=level.get(tab, '')))
            for k, (sg, q, _) in enumerate(items):
                S.append(dict(sign_id=f'{line_id}__s{k + 1:03d}', line_id=line_id, face=name, tablet=tab, pos=k + 1, sign_name=sg['sign_name'], reading=sg['reading'], type=sg['type'], ebl_label=sg['ebl_label'], ebl_line_index=sg['ebl_line_index'],
                              on_main_line=sg['ebl_line_index'] == main, ebl_path=sg['ebl_path'], face_x0=sg['face_x0'], face_y0=sg['face_y0'], face_x1=sg['face_x1'], face_y1=sg['face_y1'],
                              crop_x0=round(q[0] - int(c['x0'])), crop_y0=round(q[1] - int(c['y0'])), crop_x1=round(q[2] - int(c['x0'])), crop_y1=round(q[3] - int(c['y0']))))
        if n_face:
            F.append(dict(face=name, tablet=tab, script=f['script'], side=f['side'], object=f['object'], image=f['file'], width=W, height=H, pitch_px=f['pitch_px'], coverage=f['coverage'], tablet_level=level.get(tab, ''), n_lines=n_face))
        else:
            os.rmdir(os.path.join(a.out, 'lines', name))
        if (fi + 1) % 200 == 0: print(f'{fi + 1}/{len(names)} faces, {len(L)} crops, {len(S)} signs', flush=True)
    for fn, rows in (('faces.csv', F), ('lines.csv', L), ('signs.csv', S), ('crops_without_signs.csv', E), ('signs_outside_crops.csv', O)):
        if rows:
            with open(os.path.join(a.out, fn), 'w', newline='', encoding='utf-8') as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    clean = sum(1 for r in L if not r['flags'] and r['sign_share'] != '' and r['sign_share'] >= 0.8)
    open(os.path.join(a.out, 'README.md'), 'w', encoding='utf-8').write(README.format(n_lines=len(L), n_signs=len(S), n_faces=len(F), n_tablets=len({r['tablet'] for r in F}), n_empty=len(E), n_out=len(O), n_clean=clean,
                                                                                      model=LINE_WEIGHTS, date=datetime.date.today().isoformat()))
    print(f"{len(F)} faces, {len(L)} crops, {len(S)} signs, {clean} crops with no flag and sign_share >= 0.8; {len(E)} crops without signs, {len(O)} signs outside crops; skipped {dict(skipped)} -> {a.out}")


if __name__ == '__main__':
    main()
