"""Phase 1a: derive line-level ground truth from eBL sign annotations.

Reads ebl_tablets/manifest.csv and, for every downloaded tablet, groups the eBL sign boxes into text
lines per (side, object, column). Writes:
  ebl_tablets/ground_truth/lines/<tablet>.csv   one row per line (pixel box, centre, tilt, #signs, flags)
  ebl_tablets/ground_truth/blocks.csv           one row per text block (side/column): pitch, tilt, extent
  ebl_tablets/ground_truth/tablets.csv          one row per tablet
  ebl_tablets/ground_truth/previews/<tablet>.jpg  overlay for visual checking (first --preview tablets)

Usage (from repo root):  python -m line_cropping.truth.ebl_ground_truth [--preview 12] [--force]
"""
import argparse, csv, json, os, re, collections
import numpy as np
from PIL import Image, ImageDraw

ROOT = 'ebl_tablets'
GT = os.path.join(ROOT, 'ground_truth')
SIGN_TYPES = {'HasSign', 'Number', 'Damaged', 'PartiallyBroken', 'UnclearSign', 'CompoundGrapheme'}
ROMAN = {'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi', 'xii'}
SIDES = {'o': 'obverse', 'r': 'reverse', 't.e.': 'top', 'b.e.': 'bottom', 'l.e.': 'left', 'r.e.': 'right', 'e.': 'edge'}
OBJECTS = {'tablet', 'envelope', 'case', 'seal'}
Image.MAX_IMAGE_PIXELS = None


def parse_label(label):
    """'ii o 6\'' -> (column='ii', side='obverse', obj='', line='6\'')."""
    col = side = obj = line = ''
    for tok in label.split():
        if tok in ROMAN: col = tok
        elif tok in SIDES: side = SIDES[tok]
        elif tok in OBJECTS: obj = tok
        elif re.match(r"^\d+[a-z]?'?$", tok): line = tok
    return col, side, obj, line


def box_px(g, W, H):
    return (g['x'] / 100 * W, g['y'] / 100 * H, (g['x'] + g['width']) / 100 * W, (g['y'] + g['height']) / 100 * H)


def side_from_surfaces(cx, cy, surfaces):
    for name, (x0, y0, x1, y1) in surfaces:
        if x0 <= cx <= x1 and y0 <= cy <= y1: return name
    return ''


SIDE_WORDS = {'obverse', 'reverse', 'left', 'right', 'top', 'bottom', 'edge', 'face', 'surface'}
OBJ_WORDS = {'tablet', 'envelope', 'case', 'seal', 'object', 'fragment', 'prism', 'bulla'}
ROMAN_NUM = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi', 'xii']


def ebl_text_lines(atf):
    """eBL text lines as indexed by annotation paths: a physical ATF line ending in '&' continues on the next."""
    out, buf = [], None
    for raw in (atf or '').split('\n'):
        buf = raw if buf is None else buf + ' ' + raw
        if buf.rstrip().endswith('&'): continue
        out.append(buf); buf = None
    if buf is not None: out.append(buf)
    return out


def atf_index_map(atf):
    """For every eBL text-line index: (side, column, object, line_label) read from the @-structure of the
    eBL ATF. Column is the roman numeral of '@column N'; side words like '@reverse', '@left'; objects like
    '@envelope'. line_label is the number of a numbered text line, '' otherwise."""
    side = col = obj = ''; out = []
    for l in ebl_text_lines(atf):
        t = l.strip()
        if t.startswith('@'):
            toks = t[1:].split(); head = toks[0].lower() if toks else ''
            if head in SIDE_WORDS: side = head if head not in ('face', 'surface') else side; col = ''
            elif head == 'column' and len(toks) > 1 and toks[1].isdigit(): col = ROMAN_NUM[int(toks[1]) - 1] if 0 < int(toks[1]) <= 12 else toks[1]
            elif head in OBJ_WORDS: obj = head if head not in ('object', 'fragment') else obj; side = col = ''
            out.append((side, col, obj, ''))
        else:
            m = re.match(r"^(\d+[a-z]?'?)\.", t)
            out.append((side, col, obj, m.group(1) if m else ''))
    return out


def validate_index_map(annotations, imap):
    """Is the ATF index map still valid for these annotations? Checks that labelled signs land on a text
    line with the same line number, and that surface/column markers land on '@' lines. Returns the
    agreement rate (None if nothing could be checked)."""
    ok = tot = 0
    for a in annotations:
        path = a['data'].get('path') or []
        if not path or path[0] >= len(imap): continue
        t = a['data'].get('type')
        if t in ('SurfaceAtLine', 'ColumnAtLine'):
            tot += 1; ok += imap[path[0]][3] == '' and imap[path[0]] != ('', '', '', '')
        elif t in SIGN_TYPES:
            lab = (a.get('croppedSign') or {}).get('label', '')
            _c, _s, _o, ln = parse_label(lab)
            if ln:
                tot += 1; ok += imap[path[0]][3] == ln
    return (ok / tot) if tot else None


def build_lines(annotations, W, H, atf=None):
    surfaces = [(a['data'].get('value', '').lower(), box_px(a['geometry'], W, H))
                for a in annotations if a['data'].get('type') == 'SurfaceAtLine']
    imap = atf_index_map(atf) if atf else []
    agree = validate_index_map(annotations, imap) if imap else None
    use_atf = bool(imap) and (agree is None or agree >= 0.9)      # unverifiable maps are used but flagged
    build_lines.last_map = dict(atf_map_used=use_atf, atf_map_agreement=agree)
    groups = collections.defaultdict(list)
    for a in annotations:
        if a['data'].get('type') not in SIGN_TYPES: continue
        path = a['data'].get('path') or []
        if not path: continue
        col, side, obj, line = parse_label((a.get('croppedSign') or {}).get('label', ''))
        b = box_px(a['geometry'], W, H)
        if use_atf and path[0] < len(imap):
            aside, acol, aobj, aline = imap[path[0]]
            side = side or aside; col = col or acol; obj = obj or aobj; line = line or aline
        if not side: side = side_from_surfaces((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, surfaces) or 'unknown'
        groups[(side, obj, col, int(path[0]))].append((b, line, a['data'].get('signName', '')))
    lines = []
    for (side, obj, col, li), items in groups.items():
        bs = np.array([it[0] for it in items])
        cx = (bs[:, 0] + bs[:, 2]) / 2; cy = (bs[:, 1] + bs[:, 3]) / 2
        tilt = float(np.degrees(np.arctan(np.polyfit(cx, cy, 1)[0]))) if len(items) >= 3 and np.ptp(cx) > 1 else float('nan')
        lab = collections.Counter(it[1] for it in items if it[1]).most_common(1)
        lines.append(dict(_cx=cx, _cy=cy, _bs=bs,side=side, object=obj, column=col or 'single', line_index=li, side_inferred=False, column_inferred=False, line_label=lab[0][0] if lab else '',
                          primed=("'" in lab[0][0]) if lab else '', n_signs=len(items),
                          x0=bs[:, 0].min(), y0=bs[:, 1].min(), x1=bs[:, 2].max(), y1=bs[:, 3].max(),
                          cy=float(np.median(cy)), tilt_deg=tilt, sign_h=float(np.median(bs[:, 3] - bs[:, 1]))))
    _resolve_unknown_sides(lines)
    _resolve_unknown_columns(lines)
    # per-block statistics
    blocks = {}
    for key, L in _by_block(lines).items():
        L.sort(key=lambda l: l['line_index'])
        ys = np.array([l['cy'] for l in L]); idx = np.array([l['line_index'] for l in L])
        d = np.diff(ys); gap = np.diff(idx)
        pitch = float(np.median(d[gap == 1])) if (gap == 1).sum() >= 2 else (float(np.median(d / gap)) if len(d) else float('nan'))
        bx0 = min(l['x0'] for l in L); bx1 = max(l['x1'] for l in L)
        good = [l['tilt_deg'] for l in L if l['tilt_deg'] == l['tilt_deg'] and l['n_signs'] >= 3]
        blk_tilt = float(np.median(good)) if good else 0.0
        for l in L:
            l['block_x0'] = bx0; l['block_x1'] = bx1; l['is_first_line'] = l is L[0]; l['is_last_line'] = l is L[-1]
            l['sparse'] = l['n_signs'] <= 2
            _oriented(l, blk_tilt)
        blocks[key] = dict(side=key[0], object=key[1], column=key[2], n_lines=len(L), pitch_px=pitch,
                           median_tilt_deg=(float(np.nanmedian([l['tilt_deg'] for l in L])) if any(l['tilt_deg'] == l['tilt_deg'] for l in L) else float('nan')),
                           block_x0=bx0, block_y0=min(l['y0'] for l in L), block_x1=bx1, block_y1=max(l['y1'] for l in L),
                           first_line_index=L[0]['line_index'], first_line_label=L[0]['line_label'], first_line_primed=L[0]['primed'],
                           median_sign_h=float(np.median([l['sign_h'] for l in L])))
    return lines, blocks


def _resolve_unknown_sides(lines):
    """Lines whose eBL label had no side token and that fall outside any surface box: assign them to the
    known (side, object) whose text-line index range contains them, since eBL line indices run in text
    order across the whole tablet. Falls back to the nearest range; stays 'unknown' if none is known."""
    ranges = {}
    for l in lines:
        if l['side'] == 'unknown': continue
        k = (l['side'], l['object']); lo, hi = ranges.get(k, (l['line_index'], l['line_index']))
        ranges[k] = (min(lo, l['line_index']), max(hi, l['line_index']))
    if not ranges:
        # nothing on this tablet names a side (typically a one-sided fragment whose eBL edition starts
        # directly with numbered lines): treat all lines as one unspecified face
        for l in lines: l['side'] = 'face'; l['side_inferred'] = True
        return
    for l in lines:
        if l['side'] != 'unknown': continue
        i = l['line_index']
        inside = [k for k, (lo, hi) in ranges.items() if lo <= i <= hi]
        if inside: k = inside[0]
        else: k = min(ranges, key=lambda k: min(abs(i - ranges[k][0]), abs(i - ranges[k][1])))
        l['side'], l['object'] = k; l['side_inferred'] = True


def _resolve_unknown_columns(lines):
    """On a side that has column-labelled lines, a line whose label lacked the numeral (column 'single')
    is assigned to the labelled column whose text-line index range contains it; if no range does, to the
    labelled column whose horizontal extent overlaps the line's own signs most."""
    sides = collections.defaultdict(list)
    for l in lines: sides[(l['side'], l['object'])].append(l)
    for key, L in sides.items():
        cols = {l['column'] for l in L} - {'single'}
        if not cols: continue
        rng = {c: (min(l['line_index'] for l in L if l['column'] == c), max(l['line_index'] for l in L if l['column'] == c)) for c in cols}
        ext = {c: (min(l['x0'] for l in L if l['column'] == c), max(l['x1'] for l in L if l['column'] == c)) for c in cols}
        for l in L:
            if l['column'] != 'single': continue
            inside = [c for c, (lo, hi) in rng.items() if lo <= l['line_index'] <= hi]
            if len(inside) == 1: l['column'] = inside[0]
            else:
                def overlap(c):
                    a0, a1 = ext[c]; return max(0.0, min(a1, l['x1']) - max(a0, l['x0']))
                l['column'] = max(cols, key=overlap)
            l['column_inferred'] = True


def _oriented(l, block_tilt_deg):
    """Oriented line segment: centre line y = a*x + b fitted through the sign centres (block tilt for
    lines with < 3 signs), spanning the line's horizontal extent; thickness from the median sign height."""
    cx, cy, bs = l.pop('_cx'), l.pop('_cy'), l.pop('_bs')
    if len(cx) >= 3 and np.ptp(cx) > 1:
        a, b = np.polyfit(cx, cy, 1)
    else:
        a = np.tan(np.radians(block_tilt_deg)); b = float(np.mean(cy)) - a * float(np.mean(cx))
        l['tilt_deg'] = block_tilt_deg
    x0, x1 = float(bs[:, 0].min()), float(bs[:, 2].max())
    l['ax0'], l['ay0'], l['ax1'], l['ay1'] = x0, a * x0 + b, x1, a * x1 + b
    l['thickness'] = 1.15 * float(np.median(bs[:, 3] - bs[:, 1]))


def _by_block(lines):
    out = collections.defaultdict(list)
    for l in lines: out[(l['side'], l['object'], l['column'])].append(l)
    return out


def preview(photo_path, lines, out_path, max_dim=1800):
    im = Image.open(photo_path).convert('RGB'); W, H = im.size
    s = min(1.0, max_dim / max(W, H)); im = im.resize((int(W * s), int(H * s)))
    dr = ImageDraw.Draw(im)
    palette = [(0, 255, 0), (255, 128, 0), (0, 200, 255), (255, 0, 255), (255, 255, 0), (255, 80, 80)]
    keys = sorted({(l['side'], l['object'], l['column']) for l in lines})
    for l in lines:
        c = palette[keys.index((l['side'], l['object'], l['column'])) % len(palette)]
        dx, dy = l['ax1'] - l['ax0'], l['ay1'] - l['ay0']; n = (dx * dx + dy * dy) ** 0.5 or 1.0
        nx, ny = -dy / n * l['thickness'] / 2, dx / n * l['thickness'] / 2
        quad = [((l['ax0'] + nx) * s, (l['ay0'] + ny) * s), ((l['ax1'] + nx) * s, (l['ay1'] + ny) * s),
                ((l['ax1'] - nx) * s, (l['ay1'] - ny) * s), ((l['ax0'] - nx) * s, (l['ay0'] - ny) * s)]
        w = 5 if l['is_first_line'] else (1 if l['sparse'] else 2)
        dr.polygon(quad, outline=c if not l['sparse'] else tuple(int(v * 0.6) for v in c), width=w)
    for k in keys:  # block extent as a thin dashed-looking outline
        L = [l for l in lines if (l['side'], l['object'], l['column']) == k]
        c = palette[keys.index(k) % len(palette)]
        dr.rectangle([L[0]['block_x0'] * s, min(l['y0'] for l in L) * s, L[0]['block_x1'] * s, max(l['y1'] for l in L) * s], outline=c, width=1)
    for i, k in enumerate(keys):
        dr.text((10, 10 + 18 * i), f'{k[0]} {k[1]} col {k[2]}', fill=palette[i % len(palette)])
    im.save(out_path, quality=80)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--preview', type=int, default=12); ap.add_argument('--force', action='store_true'); ap.add_argument('--tablets', nargs='*', default=None, help='preview only these tablets')
    args = ap.parse_args()
    os.makedirs(os.path.join(GT, 'lines'), exist_ok=True); os.makedirs(os.path.join(GT, 'previews'), exist_ok=True)
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, 'manifest.csv'))) if r['status'] == 'ok']
    tab_rows, blk_rows = [], []
    line_fields = ['tablet', 'script', 'side', 'object', 'column', 'line_index', 'line_label', 'primed', 'is_first_line', 'is_last_line',
                   'side_inferred', 'column_inferred', 'sparse', 'n_signs', 'ax0', 'ay0', 'ax1', 'ay1', 'thickness', 'x0', 'y0', 'x1', 'y1', 'block_x0', 'block_x1', 'cy', 'tilt_deg', 'sign_h']
    n_prev = 0
    for r in rows:
        d = os.path.join(ROOT, r['script'], r['tablet']); ann_p = os.path.join(d, 'annotations.json'); ph = os.path.join(d, 'photo.jpg')
        if not (os.path.exists(ann_p) and os.path.exists(ph)): continue
        out_csv = os.path.join(GT, 'lines', f"{r['tablet']}.csv")
        try: W, H = Image.open(ph).size
        except Exception as e: print('bad photo', r['tablet'], e); continue
        ann = json.load(open(ann_p, encoding='utf-8'))['annotations']
        frag_p = os.path.join(d, 'fragment.json')
        atf = json.load(open(frag_p, encoding='utf-8')).get('atf') if os.path.exists(frag_p) else None
        lines, blocks = build_lines(ann, W, H, atf)
        mapinfo = build_lines.last_map
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=line_fields, extrasaction='ignore'); w.writeheader()
            for l in sorted(lines, key=lambda l: (l['side'], l['object'], l['column'], l['line_index'])):
                w.writerow(dict(tablet=r['tablet'], script=r['script'], **{k: (round(v, 1) if isinstance(v, float) else v) for k, v in l.items()}))
        for k, b in blocks.items():
            blk_rows.append(dict(tablet=r['tablet'], script=r['script'], ebl_period=r['ebl_period'], photo_w=W, photo_h=H,
                                 px_per_line=round(b['pitch_px'], 1) if b['pitch_px'] == b['pitch_px'] else '', **{kk: (round(v, 1) if isinstance(v, float) else v) for kk, v in b.items()}))
        tab_rows.append(dict(tablet=r['tablet'], script=r['script'], ebl_period=r['ebl_period'], photo_w=W, photo_h=H, n_blocks=len(blocks),
                             n_lines=len(lines), n_signs=sum(l['n_signs'] for l in lines), n_unknown_side=sum(l['n_signs'] for l in lines if l['side'] == 'unknown'), n_side_inferred=sum(1 for l in lines if l.get('side_inferred')), n_column_inferred=sum(1 for l in lines if l.get('column_inferred')),
                             atf_lines=r['atf_lines'], atf_map_used=mapinfo['atf_map_used'], atf_map_agreement=round(mapinfo['atf_map_agreement'], 3) if mapinfo['atf_map_agreement'] is not None else ''))
        if (args.tablets and r['tablet'] in args.tablets) or (not args.tablets and n_prev < args.preview):
            preview(ph, lines, os.path.join(GT, 'previews', f"{r['tablet']}.jpg")); n_prev += 1
    for name, data in (('blocks.csv', blk_rows), ('tablets.csv', tab_rows)):
        with open(os.path.join(GT, name), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)
    T = tab_rows; B = blk_rows
    print(f"tablets processed: {len(T)}  (NA {sum(t['script']=='NA' for t in T)}, OB {sum(t['script']=='OB' for t in T)})")
    print(f"text blocks: {len(B)}; lines: {sum(t['n_lines'] for t in T)}; signs: {sum(t['n_signs'] for t in T)}; signs with unknown side: {sum(t['n_unknown_side'] for t in T)}")
    ppl = np.array([b['px_per_line'] for b in B if b['px_per_line'] != ''], dtype=float)
    print(f"line pitch px: median {np.median(ppl):.0f}, 10th pct {np.percentile(ppl,10):.0f}, 90th pct {np.percentile(ppl,90):.0f}")
    print('tablets using the ATF index map for side/column:', sum(1 for t in T if t['atf_map_used']), '; map agreement median:', np.median([t['atf_map_agreement'] for t in T if t['atf_map_agreement'] != '']))
    print('blocks by side:', collections.Counter(b['side'] for b in B).most_common())
    print('multi-column tablets:', sum(1 for t in T if any(b['tablet'] == t['tablet'] and b['column'] != 'single' for b in B)))
    print('first line primed (broken beginning):', sum(1 for b in B if b['first_line_primed'] is True), 'of', len(B), 'blocks')


if __name__ == '__main__':
    main()
