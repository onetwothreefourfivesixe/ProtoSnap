"""Which tablets have every line annotated: eBL sign boxes checked against the transliteration, plus the hand-marked lines.

  python -m line_cropping.truth.full_line_annotations [--out tablet_data/ebl_tablets/ground_truth/full_line_annotations.csv]

The `coverage` column of faces.csv only compares two line counts per face. This compares labels instead, per tablet: every
numbered ATF line (object, side, column, line number) is looked up among the labels of the eBL sign boxes. A line whose signs
are all broken away ([x x x]) has nothing to box and is not expected. eBL leaves some boxes (mostly numerals) without a label;
those take the label of the other boxes on their transliteration line (eBL's path[0]). Per eBL tablet:
  lines_expected, lines_annotated, lines_missing     expected lines, those with at least one box, the labels of the rest
  extra_labels                                       box labels with no ATF line (differences between the two records)
  signs_expected, signs_boxed, sign_share            signs outside [...] in the expected lines, and boxes on those lines
  level                                              full: every expected line has a box and sign_share >= --min-signs
                                                     all_lines: every expected line has a box, fewer signs than that
                                                     edges_missing: every expected line has a box except on the edges
                                                     partial: some expected lines on the obverse or reverse have no box
Heidelberg Neo-Assyrian lines drawn by hand (Dencker et al. 2020) and Old Assyrian Line Marker faces with status done are added
as level hand_lines, source says which. These give line positions only, no signs, and list the faces that were marked, which need
not be every face of the tablet.
Also: scripts, faces in faces.csv, single_column (all those faces), split.
"""
import argparse
import collections
import csv
import json
import os
import re

from line_cropping.paths import EBL, OA
from line_cropping.truth.ebl_ground_truth import SIGN_TYPES, parse_label

ATF_SIDES = {'obverse', 'reverse', 'left', 'right', 'top', 'bottom', 'edge'}
ATF_OBJECTS = {'tablet', 'envelope', 'case'}
ROMAN = ['', 'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi', 'xii']


def visible_signs(body):
    """Signs on the clay in an ATF line body: outside [...], split at word and sign separators; x counts."""
    body = re.sub(r'#tr\..*$', '', body); body = re.sub(r'\(?\$[^$]*\$\)?|%\S+', ' ', body); body = re.sub(r'\[[^\]]*\]', ' ', body)      # blanks ($___$), language tags (%sux)
    body = re.sub(r'\[.*$|^[^\[]*\]', ' ', body) if '[' in body or ']' in body else body
    body = re.sub(r'[#!?<>*&()⸢⸣]', '', body).replace('...', ' ')
    return [t for t in re.split(r'[\s\-.:{}+]+', body) if t]


def atf_lines(atf):
    """{(object, side, column, line): number of visible signs} of an ATF."""
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
            k = (obj, side, col, m.group(1)); out[k] = out.get(k, 0) + len(visible_signs(m.group(2)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(EBL, 'ground_truth', 'full_line_annotations.csv')); ap.add_argument('--min-signs', type=float, default=0.8)
    a = ap.parse_args()
    faces = collections.defaultdict(list)
    for f in csv.DictReader(open(os.path.join(EBL, 'faces', 'faces.csv'))):
        faces[f['tablet']].append(f)
    split = {r['face']: r['split'] for r in csv.DictReader(open(os.path.join(EBL, 'split.csv')))}
    rows = []
    for t in csv.DictReader(open(os.path.join(EBL, 'ground_truth', 'tablets.csv'))):
        d = os.path.join(EBL, t['script'], t['tablet'])
        atf = json.load(open(os.path.join(d, 'fragment.json'), encoding='utf-8')).get('atf')
        ann = json.load(open(os.path.join(d, 'annotations.json'), encoding='utf-8')).get('annotations', [])
        L = atf_lines(atf); boxes = collections.Counter(); ann = [an for an in ann if (an.get('data') or {}).get('type') in SIGN_TYPES]
        label_of = collections.defaultdict(collections.Counter)
        for an in ann:
            lab = (an.get('croppedSign') or {}).get('label', '').strip()
            if lab and an['data'].get('path'):
                label_of[an['data']['path'][0]][lab] += 1
        for an in ann:
            lab = (an.get('croppedSign') or {}).get('label', '').strip(); path = an['data'].get('path') or [None]
            if not lab and label_of.get(path[0]):
                lab = label_of[path[0]].most_common(1)[0][0]
            col, side, obj, line = parse_label(lab)
            boxes[(obj if obj != 'tablet' else '', side, col, line)] += 1
        # a transliteration or a box label may leave out the column: compare without it where that is what differs
        if not any(k in L for k in boxes):
            Lc = collections.Counter()
            for k, n in L.items(): Lc[(k[0], k[1], '', k[3])] += n
            L, B = dict(Lc), collections.Counter()
            for k, n in boxes.items(): B[(k[0], k[1], '', k[3])] += n
            boxes = B
        exp = {k: n for k, n in L.items() if n > 0}
        miss = [k for k in exp if not boxes.get(k)]; extra = [k for k in boxes if k not in L]
        got = sum(min(boxes.get(k, 0), n) for k, n in exp.items()); total = sum(exp.values()); share = got / total if total else 0.0
        on_edge = bool(miss) and all(k[1] in ('left', 'right', 'top', 'bottom', 'edge') for k in miss)
        level = 'none' if not exp else 'edges_missing' if on_edge else 'partial' if miss else 'full' if share >= a.min_signs else 'all_lines'
        fs = faces.get(t['tablet'], [])
        rows.append(dict(source='eBL sign boxes', tablet=t['tablet'], script=t['script'], period=t['ebl_period'], level=level, lines_expected=len(exp), lines_annotated=len(exp) - len(miss),
                         signs_expected=total, signs_boxed=got, sign_share=round(share, 3), n_extra_labels=len(extra), n_faces=len(fs),
                         faces=';'.join(f['face'] for f in fs), single_column=bool(fs) and all(f['n_columns'] == '1' for f in fs),
                         split=';'.join(sorted({split.get(f['face'], '') for f in fs} - {''})),
                         lines_missing=';'.join(' '.join(x for x in k if x) for k in miss[:30]), extra_labels=';'.join(' '.join(x for x in k if x) for k in extra[:30])))
    blank = dict.fromkeys(rows[0], '')
    h = collections.defaultdict(lambda: collections.defaultdict(set))
    for part in ('train', 'test'):
        for r in csv.DictReader(open(f'data/heidelberg/line_annotations_{part}.csv')):
            h[r['tablet_CDLI']][r['segm_idx']].add(r['line_idx'])
    for tab, fs in h.items():
        rows.append(dict(blank, source='Heidelberg hand-drawn lines', tablet=tab, script='NA', period='Neo-Assyrian', level='hand_lines', lines_expected=sum(len(v) for v in fs.values()),
                         lines_annotated=sum(len(v) for v in fs.values()), n_faces=len(fs), faces=';'.join(f'{tab}__s{k}' for k in sorted(fs, key=int))))
    o = collections.defaultdict(lambda: collections.defaultdict(set))
    for r in csv.DictReader(open(os.path.join(OA, 'ground_truth', 'labels.csv'))):
        if r['status'] == 'done':
            o[r['face_id'].split('_')[0]][r['face_id']].add(r['line_no'])
    for tab, fs in o.items():
        rows.append(dict(blank, source='Old Assyrian Line Marker', tablet=tab, script='OA', period='Old Assyrian', level='hand_lines', lines_expected=sum(len(v) for v in fs.values()),
                         lines_annotated=sum(len(v) for v in fs.values()), n_faces=len(fs), faces=';'.join(sorted(fs))))
    order = {'full': 0, 'all_lines': 1, 'hand_lines': 2, 'edges_missing': 3, 'partial': 4, 'none': 5}
    rows.sort(key=lambda r: (order[r['level']], r['script'], r['tablet']))
    with open(a.out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    for sc in sorted({r['script'] for r in rows}):
        c = collections.Counter(r['level'] for r in rows if r['script'] == sc); print(sc, dict(c))
    for lv in order:
        R = [r for r in rows if r['level'] == lv]; print(f"{lv}: {len(R)} tablets, {sum(r['lines_expected'] for r in R)} lines")
    print('all', dict(collections.Counter(r['level'] for r in rows)), '->', a.out)


if __name__ == '__main__':
    main()
