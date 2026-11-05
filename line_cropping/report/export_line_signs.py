"""A data set of annotated text lines: one image per line, tied to its face, with the line's signs in reading order.

  python -m line_cropping.report.export_line_signs [--out tablet_data/exports/line_signs] [--limit-faces N] [--scripts OB NB ...]

Lines are the eBL truth lines (faces/lines/<face>.csv), so a line's signs are the sign boxes eBL assigned to it, not whatever
falls inside a predicted crop. Run truth/full_line_annotations.py first: its levels go into the tables.

Each line is cut in its own frame: the face rotated about the line's midpoint by the line's angle (crops/crop_frame.py reads
these columns). The angle comes from the truth segment when it rests on at least 3 signs and lies within --max-dev degrees of the
face's usual line angle (the median over its truth rows with at least 3 signs); otherwise that usual angle is used. The rectangle holds all the line's signs plus a margin, leaving out signs more than
one line spacing from the line's median row, which are kept in signs.csv with outlier = True. Rows of one line index and column
closer than 0.35 line spacings are one line (as in evaluate_lines); rows farther apart are separate lines, and each sign goes to
the nearest. Writes README.md, faces.csv, lines.csv, signs.csv and lines/<face>/<line_id>.jpg under --out.
"""
import argparse
import collections
import csv
import json
import os
import re

import cv2
import numpy as np

from line_cropping.crops.crop_frame import box_in_frame
from line_cropping.paths import EBL
from line_cropping.truth.ebl_ground_truth import SIGN_TYPES, parse_label
from line_cropping.truth.full_line_annotations import ATF_OBJECTS, ATF_SIDES, ROMAN, visible_signs

MARGIN_X, MARGIN_Y, OUTLIER = 0.25, 0.1, 1.0

README = """# Annotated lines of signs

{n_lines} text lines with {n_signs} sign boxes, from {n_faces} faces of {n_tablets} eBL tablets. Exported {date} with
`python -m line_cropping.report.export_line_signs`.

Each line has one image, is tied to the face it came from, and lists its signs in reading order. Lines are eBL's: the sign boxes
were assigned to transliteration lines by eBL's annotators, so these are annotations, not model output.

## Files

- `faces.csv`: one row per face. `face`, `tablet`, `script`, `side`, `object`, `image` (face photograph, relative to the repository),
  `width`, `height`, `skew_deg` (the face's usual line angle, degrees), `pitch_px` (line spacing), `coverage`, `n_columns`, `split`, `tablet_level`, `n_lines`.
- `lines.csv`: one row per line. `line_id`, `face`, `tablet`, `script`, `side`, `object`, `column`, `line_index`, `line_label`, `atf_text` (the
  transliteration of that line, blank if not found), `image`, `x0`, `y0`, `x1`, `y1` (rectangle in the line frame), `skew_deg`, `line_angle_deg`,
  `line_cx`, `line_cy` (the line frame), `angle_source`, `pitch_px`, `n_signs`, `signs_visible`, `sign_share`, `n_outliers`, `order_ok`, `flags`, `split`, `tablet_level`.
- `signs.csv`: one row per sign, in reading order. `sign_id`, `line_id`, `face`, `tablet`, `pos` (1 = first), `sign_name`, `reading`, `type`,
  `ebl_label`, `ebl_path` (line, word, sign), `face_x0..face_y1` (box in the face photograph), `crop_x0..crop_y1` (bounds of the box in the line image), `outlier`.
- `lines/<face>/<line_id>.jpg`: the line images.

## Coordinate frames

Line frame: the face rotated with `cv2.getRotationMatrix2D((line_cx, line_cy), line_angle_deg, 1.0)`, same size. A line image's pixels are that frame minus (`x0`, `y0`).

## Choosing lines

- `tablet_level`: `full` (every transliterated line has boxes, at least 80% of visible signs boxed), `all_lines`, `edges_missing`, `partial`.
  Lines of a `partial` tablet can themselves be well annotated, so `sign_share` is the better guide for one line.
- `sign_share`: boxes on this line / visible signs in `atf_text`, capped at 1, blank without a transliteration. Only an estimate: the signs are counted by splitting the text.
- `flags`: `few_signs` (under 3), `no_atf` (no transliteration line found), `angle_off` (the truth segment is more than 5 degrees from the face's usual line angle,
  often a sign box assigned to the wrong line), `order` (signs do not read left to right), `outlier` (a sign far from the line).
  A line with none of these and `sign_share` >= 0.8 is a good default: {n_clean} lines here.

## Caveats

- Neo-Assyrian is thinly annotated: most of its tablets have boxes on few lines.
- eBL boxes are loose, and neighbouring lines' boxes overlap, so a line image also shows parts of its neighbours.
- Lines come from eBL's labels, are straight, and are occasionally wrong where a box was assigned to the wrong line.
- `split` is the line model's split: a face marked `tune` may have been used to train it. Multi-column faces have none.

## Licence

Photographs and sign annotations: the Electronic Babylonian Library (https://www.ebl.lmu.de/), photographs by the holding museums (British Museum images
CC BY-NC-SA 4.0; others vary). Sign annotations are also published as the eBL sign dataset (Lewenstein et al. 2026, https://doi.org/10.5334/johd.503; CC BY-NC 4.0).
For research use; check the holding institution's terms before redistributing the images.
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
            X0, Y0 = int(max(0, min(q[0] for q in keep) - MARGIN_X * m)), int(max(0, min(q[1] for q in keep) - MARGIN_Y * m))
            X1, Y1 = int(min(W, max(q[2] for q in keep) + MARGIN_X * m)), int(min(H, max(q[3] for q in keep) + MARGIN_Y * m))
            if X1 - X0 < 4 or Y1 - Y0 < 4:
                skipped['empty rectangle'] += 1; continue
            n_face += 1; line_id = f"{f['face']}__L{n_face:03d}"; rel = os.path.join('lines', f['face'], line_id + '.jpg')
            cv2.imwrite(os.path.join(a.out, rel), cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))[Y0:Y1, X0:X1], [cv2.IMWRITE_JPEG_QUALITY, a.quality])
            items = sorted(zip(s['signs'], fb, out), key=lambda t: tuple(t[0][0]['path'][1:]) if len(t[0][0]['path']) > 1 else ())
            xs = [(q[0] + q[2]) / 2 for _, q, o in items if not o]; order = float(np.mean(np.diff(xs) > 0)) if len(xs) > 1 else 1.0
            col = '' if s['column'] == 'single' else s['column']; text = find_text(tablet_cache['texts'], obj, f['side'], col, s['label'])
            vis = len(visible_signs(text)) if text else 0; share = round(min(1.0, len(items) / vis), 3) if vis else ''
            flags = [n for n, on in (('few_signs', len(items) < 3), ('no_atf', not text), ('angle_off', len(items) >= 3 and dev > 5), ('order', order < 0.8), ('outlier', bool(out.any()))) if on]
            L.append(dict(line_id=line_id, face=f['face'], tablet=tab, script=f['script'], side=f['side'], object=f['object'], column=col, line_index=s['line_index'], line_label=s['label'], atf_text=text, image=rel,
                          x0=X0, y0=Y0, x1=X1, y1=Y1, skew_deg=skew, line_angle_deg=round(use, 3), line_cx=round(cx, 2), line_cy=round(cy, 2), angle_source=src, pitch_px=f['pitch_px'], n_signs=len(items),
                          signs_visible=vis or '', sign_share=share, n_outliers=int(out.sum()), order_ok=round(order, 2), flags=' '.join(flags), split=split.get(f['face'], ''), tablet_level=level.get(tab, '')))
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
    open(os.path.join(a.out, 'README.md'), 'w', encoding='utf-8').write(README.format(n_lines=len(L), n_signs=len(S), n_faces=len(F), n_tablets=len({r['tablet'] for r in F}), n_clean=clean, date=datetime.date.today().isoformat()))
    print(f"{len(F)} faces, {len(L)} lines, {len(S)} signs, {clean} lines with no flag and sign_share >= 0.8; skipped {dict(skipped)} -> {a.out}")


if __name__ == '__main__':
    main()
