"""A data set of annotated text lines: one image per line, tied to its face, with the line's signs in reading order.

  python -m line_segmentation.report.export_line_signs [--out tablet_data/exports/line_signs] [--limit-faces N] [--scripts OB NB ...]
      [--half-height 0.7 --end-margin 0.5] [--model-pred <predictions dir>]

Lines are the eBL truth lines (faces/lines/<face>.csv), so a line's signs are the sign boxes eBL assigned to it, not whatever
falls inside a predicted crop. Run truth/full_line_annotations.py first: its levels go into the tables.

Each line is cut in its own frame: the face rotated about the line's midpoint by the line's angle (crops/crop_frame.py reads
these columns). The angle comes from the truth segment when it rests on at least 3 signs and lies within --max-dev degrees of the
face's usual line angle (the median over its truth rows with at least 3 signs); otherwise that usual angle is used. The rectangle holds all the line's signs plus a margin, leaving out signs more than
one line spacing from the line's median row, which are kept in signs.csv with outlier = True. Rows of one line index and column
closer than 0.35 line spacings are one line (as in evaluate_lines); rows farther apart are separate lines, and each sign goes to
the nearest.

--half-height gives every line the same geometry instead: the rectangle runs --half-height line spacings above and below the median row of
the line's sign-box centres, and from its leftmost to its rightmost box plus --end-margin spacings, whatever the boxes' heights. --model-pred
(a line model's predictions, linemodel/infer.py) adds a check: how far the nearest predicted line lies from each eBL line, flagged when it is
more than 0.35 spacings or there is none. Writes README.md, faces.csv, lines.csv, signs.csv and lines/<face>/<line_id>.jpg under --out.
"""
import argparse
import collections
import csv
import json
import os
import re

import cv2
import numpy as np

from line_segmentation.crops.crop_frame import box_in_frame
from line_segmentation.paths import EBL
from line_segmentation.data.truth.ebl_ground_truth import SIGN_TYPES, parse_label
from line_segmentation.data.truth.full_line_annotations import ATF_OBJECTS, ATF_SIDES, ROMAN, visible_signs

MARGIN_X, MARGIN_Y, OUTLIER, ATF_MATCH = 0.25, 0.1, 1.0, 0.5

README = """# Annotated lines of signs

This set holds {n_lines} text lines with {n_signs} sign boxes, from {n_faces} faces of {n_tablets} eBL tablets. It was exported on {date} with
`python -m line_segmentation.report.export_line_signs`.

Each line has one image, is tied to the face it came from, and lists its signs in reading order. The lines are eBL's: eBL's annotators
assigned the sign boxes to transliteration lines, so these lines are annotations, not model output.

{geometry}

## Files

- `faces.csv` has one row per face. Its columns are `face`, `tablet`, `script`, `side`, `object`, `image` (the face photograph, relative to the repository),
  `width`, `height`, `skew_deg` (the face's usual line angle in degrees), `pitch_px` (the line spacing), `coverage`, `n_columns`, `split`, `tablet_level` and `n_lines`.
- `lines.csv` has one row per line. Its columns are `line_id`, `face`, `tablet`, `script`, `side`, `object`, `column`, `line_index`, `line_label`, `atf_text` (the
  transliteration of that line, blank if it was not found), `image`, `x0`, `y0`, `x1` and `y1` (the rectangle in the line frame), `skew_deg`, `line_angle_deg`,
  `line_cx` and `line_cy` (the line frame), `angle_source`, `pitch_px`, `model_offset_pitch`, `n_signs`, `signs_visible`, `sign_share`, `atf_match`, `n_outliers`, `order_ok`, `flags`, `split`
  and `tablet_level`. The `model_offset_pitch` column gives the distance to the nearest line found by the line model and is blank if there is none or the check was not run.
- `signs.csv` has one row per sign, in reading order. Its columns are `sign_id`, `line_id`, `face`, `tablet`, `pos` (1 is the first sign), `sign_name`, `reading`, `type`,
  `ebl_label`, `ebl_path` (line, word and sign), `face_x0..face_y1` (the box in the face photograph), `crop_x0..crop_y1` (the bounds of the box in the line image) and `outlier`.
- The folder `lines/<face>/` holds the line images, one `<line_id>.jpg` per line.

## Coordinate frames

The line frame is the face rotated with `cv2.getRotationMatrix2D((line_cx, line_cy), line_angle_deg, 1.0)`, at the same size. A line image's pixels are that frame minus the line's (`x0`, `y0`).

## Choosing lines

- The `tablet_level` column takes the values `full`, `all_lines`, `edges_missing` and `partial`. A `full` tablet has boxes on every transliterated line, with at least 80% of its visible signs boxed.
  Lines of a `partial` tablet can themselves be well annotated, so `sign_share` is the better guide for a single line.
- The `sign_share` column is the number of boxes on the line divided by the number of visible signs in `atf_text`, capped at 1, and it is blank without a transliteration. It is only an estimate, because the signs are counted by splitting the text.
  It counts boxes and does not check that they match the text, so a line whose `atf_text` belongs to another line can still score 1.
- The `atf_match` column is the share of the line's box readings that occur in `atf_text` (broken parts included, each word of the text used once), and it is blank
  without a transliteration. It is near 1 for most lines and near 0 where eBL labelled a line's boxes with a neighbouring line's number, so `atf_text` is the wrong line's.
- The `flags` column lists any of these problems:
  - `few_signs`: the line has fewer than 3 signs.
  - `no_atf`: no transliteration line was found.
  - `atf_mismatch`: fewer than half the box readings occur in `atf_text`, which usually means it is another line's transliteration.
  - `angle_off`: the truth segment is more than 5 degrees from the face's usual line angle, which often means a sign box was assigned to the wrong line.
  - `order`: the signs do not read left to right.
  - `outlier`: a sign lies far from the line.
  - `clipped`: the rectangle reaches the edge of the photograph.
  - `model_disagrees`: the line model found no line within 0.35 spacings, which often means a misplaced box or a damaged line. The model was trained on about half these faces, so it agrees more often there.

  A line with none of these flags and a `sign_share` of at least 0.8 is a good default, and there are {n_clean} such lines here.

## Caveats

- Neo-Assyrian is thinly annotated, and most of its tablets have boxes on only a few lines.
- eBL boxes are loose, and neighbouring lines' boxes overlap, so a line image also shows parts of its neighbours.
- The lines come from eBL's labels and are straight. They are occasionally wrong where a box was assigned to the wrong line.
- The `split` column is the line model's split, so a face marked `tune` may have been used to train it. Multi-column faces have no split.

## Licence

The photographs and sign annotations come from the Electronic Babylonian Library (https://www.ebl.lmu.de/), and the photographs were taken by the holding museums. British Museum images
are licensed CC BY-NC-SA 4.0, and the licences of others vary. The sign annotations are also published as the eBL sign dataset (Lewenstein et al. 2026, https://doi.org/10.5334/johd.503; CC BY-NC 4.0).
The set is for research use. Check the holding institution's terms before redistributing the images.
"""


def atf_texts(atf):
    """{(object, side, column, line): transliteration} of an ATF, as truth/full_line_annotations.atf_lines keys them."""
    out = {}; obj = side = col = ''
    for l in (atf or '').splitlines():
        t = l.strip()
        if t.startswith('@'):
            w = t[1:].split(); tok = w[0].lower() if w else ''
            if tok in ATF_SIDES: side = tok; col = ''
            elif tok in ATF_OBJECTS: obj = '' if tok == 'tablet' else tok; side = col = ''
            elif tok == 'column' and len(w) > 1 and w[1].isdigit() and int(w[1]) < len(ROMAN): col = ROMAN[int(w[1])]
            continue
        m = re.match(r"^(\d+[a-z]?'?)\.\s*(.*)$", t)
        if m:
            out[(obj, side, col, m.group(1))] = m.group(2)
    return out


def norm_reading(t):
    """A reading as compared with the transliteration: lower case, without brackets, damage and correction marks."""
    return re.sub(r'[\[\]#!?<>*&()⸢⸣|]', '', t).lower()


def atf_match(text, readings):
    """Share of a line's box readings that occur in its transliteration, broken parts included (each token used once).
    Low values mean eBL's label for the boxes points to another line's text."""
    body = re.sub(r'#tr\..*$', '', text); body = re.sub(r'\(?\$[^$]*\$\)?|%\S+', ' ', body).replace('...', ' ')
    toks = collections.Counter(norm_reading(t) for t in re.split(r'[\s\-.:{}+]+', body) if norm_reading(t))
    got = collections.Counter(norm_reading(r) for r in readings)
    return round(sum((toks & got).values()) / sum(got.values()), 3) if got else ''


def find_text(texts, obj, side, col, label):
    if (obj, side, col, label) in texts:
        return texts[(obj, side, col, label)]
    # column left out in one record, or a face with no side heading in the ATF (@fragment, plain numbered lines)
    same = [v for k, v in texts.items() if k[3] == label and k[0] == obj and (k[1] == side or (side == 'face' and k[1] == ''))]
    return same[0] if len(same) == 1 else ''


def segments(rows, skew, W, H, pitch):
    """Truth rows of a face -> lines; rows of one index and column within 0.35 pitch are one line."""
    M = cv2.getRotationMatrix2D((W / 2, H / 2), skew, 1.0); groups = collections.defaultdict(list)
    for r in rows:
        groups[(int(r['line_index']), r['column'])].append(r)
    out = []
    for (li, col), rs in groups.items():
        ys = [(M @ np.array([float(r['ax0']), float(r['ay0']), 1]))[1] / 2 + (M @ np.array([float(r['ax1']), float(r['ay1']), 1]))[1] / 2 for r in rs]
        order = np.argsort(ys); parts = [[order[0]]]
        for i in order[1:]:
            if pitch == pitch and ys[i] - ys[parts[-1][-1]] < 0.35 * pitch: parts[-1].append(i)
            else: parts.append([i])
        for p in parts:
            pts = np.array([[float(rs[i]['ax0']), float(rs[i]['ay0'])] for i in p] + [[float(rs[i]['ax1']), float(rs[i]['ay1'])] for i in p])
            a, b = pts[pts[:, 0].argmin()], pts[pts[:, 0].argmax()]
            out.append(dict(line_index=li, column=col, label=rs[p[0]]['line_label'], a=a, b=b, yc=float(np.mean([ys[i] for i in p])), signs=[]))
    out.sort(key=lambda s: (s['column'], s['yc']))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='tablet_data/exports/line_signs'); ap.add_argument('--limit-faces', type=int, default=None)
    ap.add_argument('--half-height', type=float, default=None); ap.add_argument('--end-margin', type=float, default=0.5); ap.add_argument('--model-pred', default=None)
    ap.add_argument('--scripts', nargs='*', default=None); ap.add_argument('--max-dev', type=float, default=10.0); ap.add_argument('--quality', type=int, default=92)
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out, 'lines.csv')):
        raise SystemExit(f'{a.out} already holds an export; remove it or choose another --out')
    faces = list(csv.DictReader(open(os.path.join(EBL, 'faces', 'faces.csv'))))
    if a.scripts: faces = [f for f in faces if f['script'] in a.scripts]
    if a.limit_faces:
        step = max(1, len(faces) // a.limit_faces); faces = faces[::step][:a.limit_faces]
    gt = {t['tablet']: t for t in csv.DictReader(open(os.path.join(EBL, 'ground_truth', 'tablets.csv')))}
    level = {r['tablet']: r['level'] for r in csv.DictReader(open(os.path.join(EBL, 'ground_truth', 'full_line_annotations.csv'), encoding='utf-8')) if r['source'] == 'eBL sign boxes'}
    split = {r['face']: r['split'] for r in csv.DictReader(open(os.path.join(EBL, 'split.csv')))}
    F, L, S = [], [], []; tablet_cache = {}; skipped = collections.Counter()
    for fi, f in enumerate(faces):
        lp = os.path.join(EBL, 'faces', 'lines', f['face'] + '.csv'); tab = f['tablet']
        if not os.path.exists(lp) or tab not in gt:
            skipped['no truth'] += 1; continue
        if tablet_cache.get('tab') != tab:
            d = os.path.join(EBL, f['script'], tab)
            ann = json.load(open(os.path.join(d, 'annotations.json'), encoding='utf-8')).get('annotations', [])
            tablet_cache = dict(tab=tab, ann=ann, texts=atf_texts(json.load(open(os.path.join(d, 'fragment.json'), encoding='utf-8')).get('atf')))
        img = cv2.imread(f['file'])
        if img is None:
            skipped['no image'] += 1; continue
        H, W = img.shape[:2]; pw, ph = float(gt[tab]['photo_w']), float(gt[tab]['photo_h']); cx0, cy0 = float(f['crop_x0']), float(f['crop_y0'])
        rows = list(csv.DictReader(open(lp)))
        angs = [np.degrees(np.arctan2(float(r['ay1']) - float(r['ay0']), float(r['ax1']) - float(r['ax0']))) for r in rows if int(r['n_signs']) >= 3]
        skew = round(float(np.median(angs)), 3) if angs else -float(f['skew_deg'] or 0)       # faces.csv stores minus the line angle
        obj = '' if f['object'] == 'tablet' else f['object']
        boxes = []
        for an in tablet_cache['ann']:
            dt, g = an.get('data') or {}, an.get('geometry') or {}
            if dt.get('type') not in SIGN_TYPES or not dt.get('path'):
                continue
            x0 = g['x'] / 100 * pw - cx0; y0 = g['y'] / 100 * ph - cy0; x1 = x0 + g['width'] / 100 * pw; y1 = y0 + g['height'] / 100 * ph
            if 0 <= (x0 + x1) / 2 < W and 0 <= (y0 + y1) / 2 < H:
                boxes.append((dt, an, (x0, y0, x1, y1)))
        hs = [b[2][3] - b[2][1] for b in boxes]; pitch = float(f['pitch_px']) if f['pitch_px'] else (float(np.median(hs)) if hs else float('nan'))
        segs = segments(rows, skew, W, H, pitch)
        pred = []
        if a.model_pred and os.path.exists(os.path.join(a.model_pred, 'unet_segments', f['face'] + '.csv')):
            pred = [np.array([[float(q['ax0']), float(q['ay0']), 1.0], [float(q['ax1']), float(q['ay1']), 1.0]]) for q in csv.DictReader(open(os.path.join(a.model_pred, 'unet_segments', f['face'] + '.csv')))]
        by_idx = collections.defaultdict(list)
        for s in segs: by_idx[s['line_index']].append(s)
        for dt, an, b in boxes:
            cand = by_idx.get(dt['path'][0])
            if not cand:
                continue
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2                    # nearest row of this line index
            def dist(s):
                k = (s['b'][1] - s['a'][1]) / max(1e-6, s['b'][0] - s['a'][0]); return abs(s['a'][1] + k * (cx - s['a'][0]) - cy)
            min(cand, key=dist)['signs'].append((dt, an, b))
        n_face = 0; os.makedirs(os.path.join(a.out, 'lines', f['face']), exist_ok=True)
        for s in segs:
            if not s['signs']:
                skipped['line without signs'] += 1; continue
            cx, cy = float((s['a'][0] + s['b'][0]) / 2), float((s['a'][1] + s['b'][1]) / 2)
            ang = float(np.degrees(np.arctan2(s['b'][1] - s['a'][1], s['b'][0] - s['a'][0]))); dev = abs(((ang - skew) + 90) % 180 - 90)
            src = 'segment' if len(s['signs']) >= 3 and dev <= a.max_dev else 'face'; use = ang if src == 'segment' else skew
            M = cv2.getRotationMatrix2D((cx, cy), use, 1.0); fb = [box_in_frame(M, *b) for _, _, b in s['signs']]
            rowc = np.array([(q[1] + q[3]) / 2 for q in fb]); med = float(np.median(rowc))
            lim = OUTLIER * pitch if pitch == pitch else float('inf'); out = np.abs(rowc - med) > lim
            keep = [q for q, o in zip(fb, out) if not o] or fb; m = pitch if pitch == pitch else float(np.median([q[3] - q[1] for q in fb]))
            if a.half_height:                                                   # the same geometry for every line
                mid = float(np.median([(q[1] + q[3]) / 2 for q in keep]))
                rx0, rx1 = min(q[0] for q in keep) - a.end_margin * m, max(q[2] for q in keep) + a.end_margin * m
                ry0, ry1 = mid - a.half_height * m, mid + a.half_height * m
            else:
                mid = med
                rx0, rx1 = min(q[0] for q in keep) - MARGIN_X * m, max(q[2] for q in keep) + MARGIN_X * m
                ry0, ry1 = min(q[1] for q in keep) - MARGIN_Y * m, max(q[3] for q in keep) + MARGIN_Y * m
            clipped = rx0 < 0 or ry0 < 0 or rx1 > W or ry1 > H
            X0, Y0, X1, Y1 = int(max(0, rx0)), int(max(0, ry0)), int(min(W, rx1)), int(min(H, ry1))
            moff = ''
            if a.model_pred:                                                    # nearest model line at this line's middle, in the line frame
                xc = (rx0 + rx1) / 2; best = float('inf')
                for P in pred:
                    (u0, u1), (v0, v1) = P @ M[0], P @ M[1]
                    if min(u0, u1) <= xc <= max(u0, u1) and u1 != u0:
                        best = min(best, abs(v0 + (v1 - v0) * (xc - u0) / (u1 - u0) - mid))
                moff = round(best / m, 3) if best < float('inf') else ''
            if X1 - X0 < 4 or Y1 - Y0 < 4:
                skipped['empty rectangle'] += 1; continue
            n_face += 1; line_id = f"{f['face']}__L{n_face:03d}"; rel = os.path.join('lines', f['face'], line_id + '.jpg')
            cv2.imwrite(os.path.join(a.out, rel), cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))[Y0:Y1, X0:X1], [cv2.IMWRITE_JPEG_QUALITY, a.quality])
            items = sorted(zip(s['signs'], fb, out), key=lambda t: tuple(t[0][0]['path'][1:]) if len(t[0][0]['path']) > 1 else ())
            xs = [(q[0] + q[2]) / 2 for _, q, o in items if not o]; order = float(np.mean(np.diff(xs) > 0)) if len(xs) > 1 else 1.0
            col = '' if s['column'] == 'single' else s['column']; text = find_text(tablet_cache['texts'], obj, f['side'], col, s['label'])
            vis = len(visible_signs(text)) if text else 0; share = round(min(1.0, len(items) / vis), 3) if vis else ''
            match = atf_match(text, [it[0][0].get('value') or it[0][0].get('signName', '') for it in items]) if text else ''
            flags = [n for n, on in (('few_signs', len(items) < 3), ('no_atf', not text), ('atf_mismatch', match != '' and match < ATF_MATCH),('angle_off', len(items) >= 3 and dev > 5), ('order', order < 0.8), ('outlier', bool(out.any())), ('clipped', clipped),
                                                 ('model_disagrees', bool(a.model_pred) and (moff == '' or moff > 0.35))) if on]
            L.append(dict(line_id=line_id, face=f['face'], tablet=tab, script=f['script'], side=f['side'], object=f['object'], column=col, line_index=s['line_index'], line_label=s['label'], atf_text=text, image=rel,
                          x0=X0, y0=Y0, x1=X1, y1=Y1, skew_deg=skew, line_angle_deg=round(use, 3), line_cx=round(cx, 2), line_cy=round(cy, 2), angle_source=src, pitch_px=f['pitch_px'], model_offset_pitch=moff, n_signs=len(items),
                          signs_visible=vis or '', sign_share=share, atf_match=match, n_outliers=int(out.sum()), order_ok=round(order, 2), flags=' '.join(flags), split=split.get(f['face'], ''), tablet_level=level.get(tab, '')))
            for k, ((dt, an, b), q, o) in enumerate(items):
                S.append(dict(sign_id=f'{line_id}__s{k + 1:03d}', line_id=line_id, face=f['face'], tablet=tab, pos=k + 1, sign_name=dt.get('signName', ''), reading=dt.get('value', ''), type=dt['type'],
                              ebl_label=(an.get('croppedSign') or {}).get('label', '').strip(), ebl_path='.'.join(map(str, dt['path'])),
                              face_x0=round(b[0]), face_y0=round(b[1]), face_x1=round(b[2]), face_y1=round(b[3]),
                              crop_x0=round(q[0] - X0), crop_y0=round(q[1] - Y0), crop_x1=round(q[2] - X0), crop_y1=round(q[3] - Y0), outlier=bool(o)))
        if n_face:
            F.append(dict(face=f['face'], tablet=tab, script=f['script'], side=f['side'], object=f['object'], image=f['file'], width=W, height=H, skew_deg=skew, pitch_px=f['pitch_px'], coverage=f['coverage'],
                          n_columns=f['n_columns'], split=split.get(f['face'], ''), tablet_level=level.get(tab, ''), n_lines=n_face))
        if (fi + 1) % 200 == 0: print(f'{fi + 1}/{len(faces)} faces, {len(L)} lines, {len(S)} signs', flush=True)
    for name, rows in (('faces.csv', F), ('lines.csv', L), ('signs.csv', S)):
        with open(os.path.join(a.out, name), 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    clean = sum(1 for r in L if not r['flags'] and r['sign_share'] != '' and r['sign_share'] >= 0.8)
    import datetime
    geometry = (f"Every line has the same geometry. Its image runs {a.half_height:g} line spacings above and below the middle of its signs, which is the median row of their box centres. "
                f"It runs from the line's first sign to its last plus {a.end_margin:g} spacings at each end, along the line's own angle. Boxes taller than the image reach past its edge, "
                "so their `crop_*` coordinates can be negative or exceed the image size."
                if a.half_height else "Each image is cut around the line's own sign boxes with a small margin, so image heights vary with the boxes.")
    open(os.path.join(a.out, 'README.md'), 'w', encoding='utf-8').write(README.format(geometry=geometry, n_lines=len(L), n_signs=len(S), n_faces=len(F), n_tablets=len({r['tablet'] for r in F}), n_clean=clean, date=datetime.date.today().isoformat()))
    print(f"{len(F)} faces, {len(L)} lines, {len(S)} signs, {clean} lines with no flag and sign_share >= 0.8; skipped {dict(skipped)} -> {a.out}")


if __name__ == '__main__':
    main()
