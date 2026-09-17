"""Phase 6a: from line crops + ATF lines + detector boxes to ProtoSnap inputs.

For each crop with an ATF line label (OB by default): tokenise the eBL ATF line into sign names (values are
mapped to signs with a frequency table built from signs_snippets_metadata.json; logograms and determinatives
are taken as sign names), take the detector boxes whose centres fall inside the crop (in the deskewed frame),
sort them left to right, align the ATF sign sequence to the box sequence (Needleman-Wunsch: +2 for equal sign
name, 0 for a different name, -1 per gap), and for every ATF sign matched to a box that has a Santakku
prototype and skeleton, save the box region as a ProtoSnap target image.

Outputs: <out>/signs/<face>__L<nn>__S<kk>__<hex>.png, <out>/samples.csv (fn,file_path,abz,name,hex,sign — the
columns run_test.py reads) and <out>/alignment.csv (per ATF sign: matched box, detector class, truth IoU).
  python -m line_cropping.protosnap.protosnap_prep --script OB --font Santakku --out protosnap_inputs/OB [--limit-crops 300]
"""
import argparse, csv, json, os, re, collections, unicodedata
import numpy as np, cv2
from line_cropping.lines.profile_lines import rotate

SUB = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')


def sign_hex(name):
    n = name.strip('|').translate(SUB).replace('Š', 'SH').replace('š', 'sh').replace('Ṣ', 'S').replace('Ṭ', 'T')
    n = re.sub(r'\s+', ' ', n.replace('×', ' TIMES ').replace('.', ' ')).strip().upper()
    try: return f"0x{ord(unicodedata.lookup('CUNEIFORM SIGN ' + n)):x}"
    except KeyError: return None


def value_map(script):
    d = json.load(open('signs_snippets_metadata.json'))
    m = collections.defaultdict(collections.Counter)
    for x in d:
        if x['script'] == script and x['value'] and x['signName']: m[x['value'].lower()][x['signName']] += 1
    return {v: c.most_common(1)[0][0] for v, c in m.items()}


def atf_line_signs(line, vmap):
    """ATF text line -> list of (token, sign name or None)."""
    body = line.split('.', 1)[1] if re.match(r"^\d+[a-z]?'?\.", line) else line
    body = re.sub(r'#tr\..*$', '', body); body = re.sub(r'\$.*$', '', body)
    body = re.sub(r'[\[\]#!?<>*&]', '', body); body = re.sub(r'\.\.\.', ' ', body)
    out = []
    for w in body.split():
        # determinatives {d}, {ki}, {m}: separate signs
        parts = re.split(r'(\{[^}]*\})', w)
        for p in parts:
            if not p: continue
            if p.startswith('{'):
                v = p.strip('{}').lower(); toks = re.split(r'[-.]', v)
            else: toks = re.split(r'[-.:]', p)
            for t in toks:
                t = t.strip()
                if not t or t in ('x', 'X', '(x)', '...'): 
                    if t: out.append((t, None))
                    continue
                if t.startswith('|') or t.isupper() or re.match(r'^[A-ZŠṢṬ][A-ZŠṢṬ₀-₉0-9]*$', t.translate(SUB)): out.append((t, t))
                else: out.append((t, vmap.get(t.lower())))
    return out


def nw_align(a, b, match=2.0, mismatch=0.0, gap=-1.0):
    n, m = len(a), len(b); S = np.zeros((n + 1, m + 1)); S[:, 0] = np.arange(n + 1) * gap; S[0, :] = np.arange(m + 1) * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            S[i, j] = max(S[i - 1, j - 1] + (match if a[i - 1] == b[j - 1] else mismatch), S[i - 1, j] + gap, S[i, j - 1] + gap)
    i, j, pairs = n, m, []
    while i > 0 and j > 0:
        if S[i, j] == S[i - 1, j - 1] + (match if a[i - 1] == b[j - 1] else mismatch): pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif S[i, j] == S[i - 1, j] + gap: i -= 1
        else: j -= 1
    return dict(pairs)


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0); ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--script', default='OB'); ap.add_argument('--font', default='Santakku'); ap.add_argument('--out', default='protosnap_inputs/OB')
    ap.add_argument('--limit-crops', type=int, default=None); ap.add_argument('--pad', type=float, default=0.15); ap.add_argument('--min-side', type=int, default=40)
    ap.add_argument('--first-only', action='store_true', help='write a target only for the first sign of each line: first ATF token matched to the leftmost detector box')
    a = ap.parse_args(); os.makedirs(os.path.join(a.out, 'signs'), exist_ok=True)
    vmap = value_map(a.script)
    proto = {}
    for r in csv.DictReader(open('prototypes/metadata.csv')):
        if r.get(a.font) == 'True' and r['name']: proto[r['hex']] = r['name']
    skel = {f.split('_')[0] for f in os.listdir(f'skeletons/{a.font}') if f.endswith('_adf.csv')}
    faces = {f['face']: f for f in csv.DictReader(open('ebl_tablets/faces/faces.csv'))}
    crops = [c for c in csv.DictReader(open('ebl_tablets/line_crops/crops.csv')) if c['script'] == a.script and c['atf_aligned'] == 'True' and c['side'] in ('obverse', 'reverse', 'face')]
    if a.limit_crops: crops = crops[:a.limit_crops]
    by_face = collections.defaultdict(list)
    for c in crops: by_face[c['face']].append(c)
    samples, align_rows = [], []; stats = collections.Counter(); frag_cache = {}; rot_cache = {}
    for face, cs in by_face.items():
        f = faces[face]; W, H = int(f['width']), int(f['height']); skew = float(cs[0]['skew_deg']); M = cv2.getRotationMatrix2D((W / 2, H / 2), skew, 1.0)
        frag = json.load(open(os.path.join('ebl_tablets', f['script'], f['tablet'], 'fragment.json'), encoding='utf-8'))
        # ATF lines of this side, in order
        lines_atf = []; cur = None
        for l in (frag.get('atf') or '').splitlines():
            t = l.strip()
            if t.startswith('@'):
                tok = t[1:].split()[0].lower() if t[1:].strip() else ''
                if tok in ('obverse', 'reverse', 'left', 'right', 'top', 'bottom', 'edge', 'face'): cur = tok
                elif tok in ('tablet', 'envelope', 'object', 'fragment'): cur = None
            elif re.match(r"^\d+[a-z]?'?\.", t) and cur == f['side']: lines_atf.append(t)
        det = list(csv.DictReader(open(os.path.join('ebl_tablets', 'detections', face + '.csv'), encoding='utf-8')))
        boxes = []
        for d in det:
            b = [float(d['x0']), float(d['y0']), float(d['x1']), float(d['y1'])]; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            rc = M @ np.array([cx, cy, 1.0]); boxes.append(dict(box=b, rx=rc[0], ry=rc[1], sign=d['sign'], score=float(d['score'])))
        # truth sign boxes (eBL annotations) for alignment scoring
        truth = []
        annp = os.path.join('ebl_tablets', f['script'], f['tablet'], 'annotations.json')
        if os.path.exists(annp):
            A = json.load(open(annp, encoding='utf-8'))['annotations']; PW, PH = None, None
            from PIL import Image; Image.MAX_IMAGE_PIXELS = None
            PW, PH = Image.open(os.path.join('ebl_tablets', f['script'], f['tablet'], 'photo.jpg')).size
            cx0, cy0 = int(f['crop_x0']), int(f['crop_y0'])
            for an in A:
                if an['data'].get('type') not in ('HasSign', 'Number', 'Damaged', 'PartiallyBroken', 'UnclearSign'): continue
                g = an['geometry']; b = [g['x'] / 100 * PW - cx0, g['y'] / 100 * PH - cy0, (g['x'] + g['width']) / 100 * PW - cx0, (g['y'] + g['height']) / 100 * PH - cy0]
                truth.append((b, an['data'].get('signName', '')))
        gray = None
        for c in cs:
            i = int(c['line_no']) - 1
            if i >= len(lines_atf): continue
            signs = atf_line_signs(lines_atf[i], vmap)
            inb = sorted([b for b in boxes if float(c['y0']) <= b['ry'] <= float(c['y1']) and abs(b['ry'] - float(c['y_center'])) < 0.5 * float(c['pitch_px'])], key=lambda b: b['rx'])
            stats['lines'] += 1; stats['atf_signs'] += len(signs); stats['boxes'] += len(inb)
            if not inb or not signs: continue
            pairs = nw_align([s or '?' for _, s in signs], [b['sign'] for b in inb])
            for k, (tok, sname) in enumerate(signs):
                j = pairs.get(k); row = dict(face=face, line_no=c['line_no'], atf_line=c['atf_line'], pos=k + 1, token=tok, sign=sname or '', matched=j is not None)
                if j is not None:
                    b = inb[j]; row.update(det_sign=b['sign'], det_score=b['score'], box=' '.join(f'{v:.0f}' for v in b['box']))
                    tb = [t for t in truth if t[1] == sname] if sname else []
                    row['truth_iou'] = round(max((iou(b['box'], t[0]) for t in tb), default=0.0), 2) if tb else ''
                    row['det_agrees'] = (b['sign'] == sname)
                    hx = sign_hex(sname) if sname else None; row['hex'] = hx or ''; row['is_first'] = (k == 0 and j == 0)
                    if hx and hx in proto and hx in skel and (not a.first_only or row['is_first']):
                        if gray is None:
                            img = cv2.imread(f['file']); gray = img
                        x0, y0, x1, y1 = b['box']; pw, ph = (x1 - x0) * a.pad, (y1 - y0) * a.pad
                        X0, Y0, X1, Y1 = int(max(0, x0 - pw)), int(max(0, y0 - ph)), int(min(W, x1 + pw)), int(min(H, y1 + ph))
                        if X1 - X0 >= a.min_side and Y1 - Y0 >= a.min_side:
                            fn = f"{face}__L{int(c['line_no']):02d}__S{k+1:02d}__{hx}.png"; cv2.imwrite(os.path.join(a.out, 'signs', fn), gray[Y0:Y1, X0:X1])
                            samples.append(dict(fn=fn, file_path=os.path.join(a.out, 'signs', fn), abz='', name=proto[hx], hex=hx, sign=sname)); row['target'] = fn; stats['targets'] += 1
                    elif hx: stats['no_prototype'] += 1
                align_rows.append(row)
    with open(os.path.join(a.out, 'samples.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['fn', 'file_path', 'abz', 'name', 'hex', 'sign']); w.writeheader(); w.writerows(samples)
    with open(os.path.join(a.out, 'alignment.csv'), 'w', newline='', encoding='utf-8') as fh:
        keys = ['face', 'line_no', 'atf_line', 'pos', 'token', 'sign', 'matched', 'det_sign', 'det_score', 'det_agrees', 'box', 'truth_iou', 'hex', 'is_first', 'target']
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore'); w.writeheader(); w.writerows(align_rows)
    A = align_rows; matched = [r for r in A if r['matched']]; withsign = [r for r in A if r['sign']]
    print(f"lines {stats['lines']}, ATF signs {stats['atf_signs']} (with a sign name {len(withsign)}), detector boxes in those lines {stats['boxes']}")
    print(f"ATF signs matched to a box: {len(matched)} ({len(matched)/max(1,len(A)):.0%}); detector class agrees with ATF sign: {np.mean([r['det_agrees'] for r in matched if r['sign']]):.0%}")
    ious = [float(r['truth_iou']) for r in matched if r.get('truth_iou') not in ('', None)]
    print(f"matched box overlaps an eBL truth box of the same sign (IoU>0.5): {np.mean(np.array(ious)>0.5):.0%} (n={len(ious)})")
    print(f"ProtoSnap targets written: {stats['targets']} (signs without a {a.font} prototype/skeleton: {stats['no_prototype']}) -> {a.out}/samples.csv")


if __name__ == '__main__':
    main()
