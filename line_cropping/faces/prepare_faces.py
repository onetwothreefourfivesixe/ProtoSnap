"""Phase 2: face preparation.

  python -m line_cropping.faces.prepare_faces ebl   -> cut each annotated side out of the eBL composite photos
  python -m line_cropping.faces.prepare_faces oa    -> accepted-face list for the Old Assyrian set, edges rotated

eBL output (ebl_tablets/faces/):
  <tablet>__<side>[__<object>].jpg      face crop (block union + margin), unrotated
  lines/<tablet>__<side>[__<object>].csv  Phase 1 line truth in face-crop pixel coordinates
  faces.csv                             one row per face: crop box in the photo, skew (= -median block tilt),
                                        pitch, n_lines, atf lines for that side, coverage, n_columns
OA output (fat-cross_processed/prepared/):
  faces.csv                             accepted faces with path, rotation applied, ATF line count, px/line
  edges/<P>_<left|right>.png            edge strips rotated so text lines run horizontally
"""
import argparse, csv, json, os, re, collections
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

EBL = 'ebl_tablets'
SIDE_ATF = {'obverse': 'obverse', 'reverse': 'reverse', 'top': 'top', 'bottom': 'bottom', 'left': 'left', 'right': 'right', 'edge': 'edge'}


def atf_lines_per_side(atf):
    """Count numbered lines under each @side block of an ATF text (eBL or CDLI flavour)."""
    out = collections.Counter(); cur = None
    for l in atf.splitlines():
        if l.startswith('@'):
            tok = l[1:].strip().split()[0].lower() if l[1:].strip() else ''
            if tok in ('obverse', 'reverse', 'left', 'right', 'top', 'bottom', 'edge'): cur = tok
            elif tok in ('tablet', 'envelope', 'object', 'fragment'): cur = None
        elif re.match(r"^\d+[a-z]?'?\.", l) and cur: out[cur] += 1
    return out


def tablet_components(im, scale=6, min_area_frac=0.002):
    """Segment bright tablet material from the dark studio background.
    Returns a list of component boxes (x0, y0, x1, y1) in full-image pixels, largest first, plus the label
    image and scale so callers can test which component contains a point."""
    import cv2
    W, H = im.size
    small = np.asarray(im.convert('L').resize((max(1, W // scale), max(1, H // scale)), Image.BILINEAR))
    blur = cv2.GaussianBlur(small, (0, 0), 2)
    # background brightness from the image border (museum photos: black studio cloth or white/grey paper)
    b = 3; border = np.concatenate([blur[:b].ravel(), blur[-b:].ravel(), blur[:, :b].ravel(), blur[:, -b:].ravel()])
    bg = float(np.median(border)); otsu, _bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if bg < 128:                                             # dark background: tablet is the bright material
        thr = min(60, int(otsu)) if otsu else 40             # never above 60 so shadows on the tablet stay inside
        bw = (blur > thr).astype(np.uint8) * 255
    else:                                                    # light background: tablet is darker than the paper
        thr = max(int(otsu), int(bg) - 25)
        bw = (blur < thr).astype(np.uint8) * 255
    tablet_components.background = bg
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)                    # fill wedge shadows and cracks (lightly)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, lab, stats, _c = cv2.connectedComponentsWithStats(bw, connectivity=8)
    tablet_components.mask = bw                                       # kept for gap splitting
    comps = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < min_area_frac * small.size: continue
        comps.append((i, (x * scale, y * scale, (x + w) * scale, (y + h) * scale), a))
    comps.sort(key=lambda c: -c[2])
    return comps, lab, scale


def split_at_gaps(mask, box, pts, min_gap=3, empty=0.04):
    """Refine a component box (small-mask coordinates) by cutting it at runs of near-empty columns or rows
    (foreground fraction < `empty`, at least `min_gap` wide) and keeping the piece holding most of `pts`.
    Repeats along both axes until nothing splits. Handles views that touch in the composite photo."""
    x0, y0, x1, y1 = box
    for _ in range(6):
        changed = False
        for axis in (1, 0):                                   # columns first, then rows
            sub = mask[y0:y1, x0:x1] > 0
            prof = sub.mean(axis=0) if axis == 1 else sub.mean(axis=1)
            gaps = prof < empty
            # runs of empty positions strictly inside the box
            runs, i = [], 0
            while i < len(gaps):
                if gaps[i]:
                    j = i
                    while j < len(gaps) and gaps[j]: j += 1
                    if j - i >= min_gap and i > 0 and j < len(gaps): runs.append((i, j))
                    i = j
                else: i += 1
            if not runs: continue
            cuts = [0] + [ (a + b) // 2 for a, b in runs ] + [len(gaps)]
            segs = list(zip(cuts, cuts[1:]))
            def count(seg):
                a, b = seg
                if axis == 1: return sum(1 for (px, py) in pts if x0 + a <= px < x0 + b and y0 <= py < y1)
                return sum(1 for (px, py) in pts if y0 + a <= py < y0 + b and x0 <= px < x1)
            keep = [sg for sg in segs if count(sg) > 0]          # every piece that holds lines of this side
            if not keep: continue                                 # (a side split over several fragments stays whole)
            lo, hi = keep[0][0], keep[-1][1]
            if axis == 1: nx0, nx1 = x0 + lo, x0 + hi; changed |= (nx0, nx1) != (x0, x1); x0, x1 = nx0, nx1
            else: ny0, ny1 = y0 + lo, y0 + hi; changed |= (ny0, ny1) != (y0, y1); y0, y1 = ny0, ny1
        if not changed: break
    # tighten to the foreground actually present
    sub = mask[y0:y1, x0:x1] > 0
    if sub.any():
        cols = np.where(sub.any(axis=0))[0]; rows = np.where(sub.any(axis=1))[0]
        x0, x1, y0, y1 = x0 + cols[0], x0 + cols[-1] + 1, y0 + rows[0], y0 + rows[-1] + 1
    return (x0, y0, x1, y1)


def component_for_lines(lines, comps, lab, scale):
    """Box (full-image px) of the tablet face holding these lines: the component containing most line
    centres, split at gaps between touching views. None if fewer than half the centres fall on a component."""
    pts = [((float(l['ax0']) + float(l['ax1'])) / 2 / scale, (float(l['ay0']) + float(l['ay1'])) / 2 / scale) for l in lines]
    votes = collections.Counter()
    for cx, cy in pts:
        yi, xi = int(min(lab.shape[0] - 1, max(0, cy))), int(min(lab.shape[1] - 1, max(0, cx)))
        votes[int(lab[yi, xi])] += 1
    votes.pop(0, None)
    if not votes or sum(votes.values()) < 0.5 * len(lines): return None
    chosen = {i for i, v in votes.items() if v >= max(2, 0.1 * len(lines))} or {votes.most_common(1)[0][0]}
    boxes = []
    for i, box, a in comps:
        if i in chosen:                                   # a side broken over several fragments = several blobs
            sb = tuple(int(round(b / scale)) for b in box)
            mine = [p for p in pts if lab[int(min(lab.shape[0]-1, max(0, p[1]))), int(min(lab.shape[1]-1, max(0, p[0])))] == i]
            boxes.append(split_at_gaps(tablet_components.mask, sb, mine))
    if not boxes: return None
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes); x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
    return (x0 * scale, y0 * scale, x1 * scale, y1 * scale)


def ebl():
    gt = os.path.join(EBL, 'ground_truth'); out = os.path.join(EBL, 'faces'); os.makedirs(os.path.join(out, 'lines'), exist_ok=True)
    blocks = list(csv.DictReader(open(os.path.join(gt, 'blocks.csv'))))
    by_face = collections.defaultdict(list)
    for b in blocks:
        if b['side'] == 'unknown': continue
        by_face[(b['tablet'], b['script'], b['side'], b['object'])].append(b)
    rows = []; done_photos = {}
    for (tab, script, side, obj), bs in sorted(by_face.items()):
        d = os.path.join(EBL, script, tab); ph = os.path.join(d, 'photo.jpg')
        if not os.path.exists(ph): continue
        lines = [l for l in csv.DictReader(open(os.path.join(gt, 'lines', f'{tab}.csv'))) if l['side'] == side and l['object'] == obj]
        if not lines: continue
        pv = [float(b['pitch_px']) for b in bs if b['pitch_px'] and np.isfinite(float(b['pitch_px']))]
        pitch = float(np.median(pv)) if pv else np.nan
        margin = pitch if pitch == pitch else 1.5 * float(np.median([float(l['thickness']) for l in lines]))
        x0 = min(float(l['ax0']) for l in lines) - margin; x1 = max(float(l['ax1']) for l in lines) + margin
        y0 = min(min(float(l['ay0']), float(l['ay1'])) - float(l['thickness']) for l in lines) - margin
        y1 = max(max(float(l['ay0']), float(l['ay1'])) + float(l['thickness']) for l in lines) + margin
        if tab not in done_photos:            # faces are sorted by tablet: keep only the current photo decoded
            done_photos.clear(); im_ = Image.open(ph); done_photos[tab] = (im_, tablet_components(im_))
        im, (comps, lab, cscale) = done_photos[tab]; W, H = im.size
        lx0, ly0, lx1, ly1 = x0, y0, x1, y1                       # line-based crop (fallback)
        cbox = component_for_lines(lines, comps, lab, cscale)
        method = 'lines'
        if cbox is not None and (cbox[2] - cbox[0]) * (cbox[3] - cbox[1]) > 0.6 * W * H: cbox = None   # whole photo = not a face
        if cbox is not None:
            pad = int(0.5 * margin)
            # the blob must not swallow another side's lines (e.g. obverse and reverse touching): keep it only if
            # the line-based crop is mostly inside it
            x0, y0, x1, y1 = cbox[0] - pad, cbox[1] - pad, cbox[2] + pad, cbox[3] + pad; method = 'blob'
        x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))
        name = f'{tab}__{side}' + (f'__{obj}' if obj else '')
        im.crop((x0, y0, x1, y1)).save(os.path.join(out, name + '.jpg'), quality=92)
        with open(os.path.join(out, 'lines', name + '.csv'), 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['column', 'line_index', 'line_label', 'primed', 'is_first_line', 'sparse', 'n_signs', 'ax0', 'ay0', 'ax1', 'ay1', 'thickness', 'tilt_deg'])
            for l in sorted(lines, key=lambda l: (l['column'], int(l['line_index']))):
                w.writerow([l['column'], l['line_index'], l['line_label'], l['primed'], l['is_first_line'], l['sparse'], l['n_signs'],
                            round(float(l['ax0']) - x0, 1), round(float(l['ay0']) - y0, 1), round(float(l['ax1']) - x0, 1), round(float(l['ay1']) - y0, 1), l['thickness'], l['tilt_deg']])
        frag = json.load(open(os.path.join(d, 'fragment.json'), encoding='utf-8'))
        atf_side = atf_lines_per_side(frag.get('atf') or '').get(SIDE_ATF.get(side, side), 0)
        tilts = [float(b['median_tilt_deg']) for b in bs if b['median_tilt_deg'] not in ('', 'nan')]
        rows.append(dict(face=name, tablet=tab, script=script, side=side, object=obj, file=os.path.join(out, name + '.jpg'), crop_method=method, background=round(tablet_components.background),
                         crop_x0=x0, crop_y0=y0, crop_x1=x1, crop_y1=y1, width=x1 - x0, height=y1 - y0,
                         n_columns=len(bs), n_lines=len(lines), atf_lines_side=atf_side, coverage=round(len(lines) / atf_side, 2) if atf_side else '',
                         pitch_px=round(float(pitch), 1) if pitch == pitch else '', skew_deg=round(-float(np.median(tilts)), 2) if tilts else '',
                         px_per_line=round(float(pitch), 1) if pitch == pitch else ''))
    with open(os.path.join(out, 'faces.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('crop method:', collections.Counter(r['crop_method'] for r in rows).most_common())
    print(f'eBL faces written: {len(rows)} from {len({r["tablet"] for r in rows})} tablets;', collections.Counter(r["side"] for r in rows).most_common())
    cov = np.array([r['coverage'] for r in rows if r['coverage'] != ''], dtype=float)
    print(f'faces with atf count for the side: {len(cov)}; coverage >= 0.8: {(cov >= 0.8).mean():.0%}; multi-column faces: {sum(r["n_columns"] > 1 for r in rows)}')
    ppl = np.array([r['px_per_line'] for r in rows if r['px_per_line'] != ''], dtype=float); print(f'px per line: median {np.median(ppl):.0f}, 10th pct {np.percentile(ppl, 10):.0f}')


def oa(rot_left, rot_right):
    import cv2
    FACES = 'fat-cross_processed/fat-cross_processed'; out = 'fat-cross_processed/prepared'; os.makedirs(os.path.join(out, 'edges'), exist_ok=True)
    qc = {(r['p_number'], r['face']): r for r in csv.DictReader(open('fat-cross_processed/experiments/face_qc.csv'))}
    inv = {r['p_number']: r for r in csv.DictReader(open('fat-cross_processed/inventory.csv'))}
    atf = {}
    for l in open('artifacts_json/transliterations.jsonl', encoding='utf-8'):
        r = json.loads(l)
        if r['p_number'] in inv: atf[r['p_number']] = atf_lines_per_side(r['atf'])
    rows = []; rot = {'left': rot_left, 'right': rot_right}
    for p, r in inv.items():
        for face in r['faces'].split('|'):
            if face not in ('obverse', 'reverse', 'left', 'right'): continue
            src = f'{FACES}/{p}_{face}.png'
            if not os.path.exists(src): continue
            n = atf.get(p, {}).get(face, '')
            if face in ('obverse', 'reverse'):
                q = qc.get((p, face)); flags = q['flags'] if q else 'no_qc'
                if flags: rows.append(dict(face=f'{p}_{face}', p_number=p, side=face, file=src, rotation='', atf_lines=n, px_per_line=q['px_per_line'] if q else '', accepted=False, reason=flags)); continue
                rows.append(dict(face=f'{p}_{face}', p_number=p, side=face, file=src, rotation='', atf_lines=n, px_per_line=q['px_per_line'], accepted=True, reason=''))
            else:
                if not n: rows.append(dict(face=f'{p}_{face}', p_number=p, side=face, file=src, rotation='', atf_lines='', px_per_line='', accepted=False, reason='no_atf_text_on_edge')); continue
                im = cv2.imread(src); im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE if rot[face] == 'cw' else cv2.ROTATE_90_COUNTERCLOCKWISE)
                dst = os.path.join(out, 'edges', f'{p}_{face}.png'); cv2.imwrite(dst, im)
                ppl = round(0.88 * im.shape[0] / n) if n else ''
                ok = min(im.shape[:2]) >= 40 and (ppl == '' or ppl >= 12)
                rows.append(dict(face=f'{p}_{face}', p_number=p, side=face, file=dst, rotation=rot[face], atf_lines=n, px_per_line=ppl, accepted=ok, reason='' if ok else 'too_small'))
    with open(os.path.join(out, 'faces.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    acc = [r for r in rows if r['accepted']]
    print(f'OA faces listed: {len(rows)}; accepted: {len(acc)};', collections.Counter(r['side'] for r in acc).most_common(), '; rejected reasons:', collections.Counter(r['reason'] for r in rows if not r['accepted']).most_common())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('which', choices=['ebl', 'oa'])
    ap.add_argument('--rot-left', default='cw', choices=['cw', 'ccw']); ap.add_argument('--rot-right', default='ccw', choices=['cw', 'ccw'])
    a = ap.parse_args()
    ebl() if a.which == 'ebl' else oa(a.rot_left, a.rot_right)
