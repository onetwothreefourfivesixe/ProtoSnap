"""Phase 4: one image crop per detected line, bounded by the gaps between lines.

Default (`--bounds gap`): each line's vertical extent is the 10th-90th percentile of the detector boxes assigned to
it, and the crop runs from the midpoint of the gap to the line above to the midpoint of the gap to the line below
(first/last line: extent plus 6% of a pitch). Neighbouring lines are therefore not sliced into the crop. `--bounds band`
restores the fixed strip (centre +- (0.5 + pad) x pitch). `--overlap f` lets neighbouring crops share f pitch beyond
each boundary, capped at the edge of the neighbour's band (`--overlap-into g` lets it reach g of that band's height
past the edge, for interlocking lines), so tall signs and slightly slanted lines are not clipped
while no crop takes in half of the next line.

  python -m line_cropping.crops.make_crops --faces tablet_data/ebl_tablets/faces/faces.csv --pred tablet_data/ebl_tablets/predictions --method det_n --out tablet_data/ebl_tablets/line_crops

For each face with a prediction: rotate the face by the run's skew, then cut, for line i at centre y_i, the rows
[y_i - (0.5 + pad) * pitch, y_i + (0.5 + pad) * pitch] (pad 0.5 by default, i.e. a crop two pitches tall: eBL sign boxes are
about as tall as the line pitch, so a shorter crop cuts the wedges of most lines) and the columns [x0 - margin, x1 + margin] of that line
(pitch = median gap between the predicted lines; margin = 0.5 pitch). Crops are saved as
<out>/<face>/<face>__L<nn>.jpg and listed in <out>/crops.csv with their rectangle in the deskewed face, the skew,
the face's offset in the original photo, and the ATF line label when the predicted count equals the ATF count
for that side (else blank, with `atf_aligned` = False).
"""
from line_cropping.paths import EBL
import argparse, csv, os, json, re
import numpy as np, cv2
from line_cropping.lines.profile_lines import rotate


def atf_labels_for_side(atf, side):
    """Numbered line labels under the @<side> block of an eBL/CDLI ATF, in order (e.g. ["1'", "2'", ...]).
    Returns [] when that side is laid out in more than one @column, because a single-column face prediction
    cannot be aligned line-for-line with a multi-column transliteration."""
    out = []; cur = None; cols = 0
    for l in (atf or '').splitlines():
        t = l.strip()
        if t.startswith('@'):
            tok = t[1:].split()[0].lower() if t[1:].strip() else ''
            if tok in ('obverse', 'reverse', 'left', 'right', 'top', 'bottom', 'edge', 'face'): cur = tok
            elif tok in ('tablet', 'envelope', 'object', 'fragment'): cur = None
            elif tok == 'column' and cur == side: cols += 1
        else:
            m = re.match(r"^(\d+[a-z]?'?)\.", t)
            if m and cur == side: out.append(m.group(1))
    return [] if cols > 1 else out


def line_bounds_from_boxes(det_csv, W, H, skew, ys, pitch, pad_frac=0.06):
    """Top/bottom row of each line from the detector boxes assigned to it (deskewed frame), and the crop
    boundaries between consecutive lines: the midpoint of the gap between line i's bottom and line i+1's top
    when there is a gap, else the midpoint of the two line centres. Returns list of (top, bottom) per line."""
    import csv as _csv
    M = cv2.getRotationMatrix2D((W / 2, H / 2), skew, 1.0)
    tops = [[] for _ in ys]; bots = [[] for _ in ys]
    if os.path.exists(det_csv):
        for r in _csv.DictReader(open(det_csv, encoding='utf-8')):
            x0, y0, x1, y1 = float(r['x0']), float(r['y0']), float(r['x1']), float(r['y1'])
            P = M @ np.array([[x0, x1, x0, x1], [y0, y0, y1, y1], [1, 1, 1, 1]])
            ty, by, cy = P[1].min(), P[1].max(), P[1].mean()
            i = int(np.argmin(np.abs(ys - cy)))
            if abs(ys[i] - cy) < 0.6 * pitch: tops[i].append(ty); bots[i].append(by)
    ext_top = np.array([np.percentile(t, 10) if len(t) >= 2 else (ys[i] - 0.45 * pitch) for i, t in enumerate(tops)])
    ext_bot = np.array([np.percentile(b, 90) if len(b) >= 2 else (ys[i] + 0.45 * pitch) for i, b in enumerate(bots)])
    n = len(ys); bounds = []
    for i in range(n):
        if i == 0: top = ext_top[i] - pad_frac * pitch
        else: top = (ext_bot[i - 1] + ext_top[i]) / 2 if ext_bot[i - 1] < ext_top[i] else (ys[i - 1] + ys[i]) / 2
        if i == n - 1: bot = ext_bot[i] + pad_frac * pitch
        else: bot = (ext_bot[i] + ext_top[i + 1]) / 2 if ext_bot[i] < ext_top[i + 1] else (ys[i] + ys[i + 1]) / 2
        bounds.append((top, bot))
    line_bounds_from_boxes.edges = list(zip(ext_top, ext_bot))
    return bounds


def line_bounds_midpoint(ys, pitch):
    """Crop boundaries halfway between consecutive predicted line centres; first/last line get half a pitch.
    Needs no detector boxes, so it suits faces where detections are sparse (the Old Assyrian set)."""
    n = len(ys); out = []
    for i in range(n):
        top = (ys[i - 1] + ys[i]) / 2 if i > 0 else ys[i] - 0.5 * pitch
        bot = (ys[i] + ys[i + 1]) / 2 if i < n - 1 else ys[i] + 0.5 * pitch
        out.append((top, bot))
    return out


def line_bounds_from_segments(seg_csv, W, H, skew, ys, pitch, pad_frac=0.06):
    """Crop boundaries from predicted oriented segments with a thickness (the line model's *_segments files): each
    line's top/bottom = its band edges in the deskewed frame; boundary between lines = midpoint of the gap between
    one band's bottom and the next band's top (centre midpoint if the bands overlap); first/last get pad_frac pitch."""
    import csv as _csv
    if not os.path.exists(seg_csv): return line_bounds_midpoint(ys, pitch)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), skew, 1.0); bands = []
    for r in _csv.DictReader(open(seg_csv)):
        p0 = M @ np.array([float(r['ax0']), float(r['ay0']), 1.0]); p1 = M @ np.array([float(r['ax1']), float(r['ay1']), 1.0]); c = (p0[1] + p1[1]) / 2; t = float(r['thickness']) / 2
        bands.append((c, c - t, c + t))
    if len(bands) != len(ys): return line_bounds_midpoint(ys, pitch)
    bands.sort(); n = len(bands); out = []
    for i in range(n):
        c, top_i, bot_i = bands[i]
        top = top_i - pad_frac * pitch if i == 0 else ((bands[i - 1][2] + top_i) / 2 if bands[i - 1][2] < top_i else (bands[i - 1][0] + c) / 2)
        bot = bot_i + pad_frac * pitch if i == n - 1 else ((bot_i + bands[i + 1][1]) / 2 if bot_i < bands[i + 1][1] else (c + bands[i + 1][0]) / 2)
        out.append((top, bot))
    line_bounds_from_segments.edges = [(b[1], b[2]) for b in bands]
    return out


def apply_overlap(bounds, ys, pitch, overlap, edges=None, into=0.0):
    """Let neighbouring crops share rows: every boundary between two lines moves outward by overlap x pitch in both
    crops (first/last line's outer edge unchanged). The move is capped at the near edge of the neighbour's band
    (edges = (top, bottom) per line: box extents or predicted bands; without them, 0.35 pitch from the neighbour's
    centre) pushed `into` x that band's height further in. into = 0 takes in the gap and the line's own tall signs but
    none of the neighbour's band, so where two bands touch the boundary stays put; into = 0.25 also allows the outer
    quarter of the neighbour's band, for lines whose signs interlock. Clamped below 0.5: never half of the next line."""
    if overlap <= 0 or not bounds: return bounds
    n = len(bounds); d = overlap * pitch; into = min(max(into, 0.0), 0.45); out = []
    if edges is None or len(edges) != n: edges = [(y - 0.35 * pitch, y + 0.35 * pitch) for y in ys]
    for i, (top, bot) in enumerate(bounds):
        if i > 0:
            et, eb = edges[i - 1]; top = min(top, max(top - d, eb - into * (eb - et)))
        if i < n - 1:
            et, eb = edges[i + 1]; bot = max(bot, min(bot + d, et + into * (eb - et)))
        out.append((top, bot))
    return out


def crop_along_segment(img, seg, top_off, bot_off, margin):
    """Cut a line in its own frame: rotate the face about the segment's midpoint by the segment's angle, then take
    rows [mid_y + top_off, mid_y + bot_off] and columns [x0 - margin, x1 + margin] of the rotated image. top_off/
    bot_off are signed offsets from the line centre (negative above). Returns the crop and its rectangle+angle."""
    H, W = img.shape[:2]; cx, cy = (seg['ax0'] + seg['ax1']) / 2, (seg['ay0'] + seg['ay1']) / 2
    ang = np.degrees(np.arctan2(seg['ay1'] - seg['ay0'], seg['ax1'] - seg['ax0'])); L = ((seg['ax1'] - seg['ax0']) ** 2 + (seg['ay1'] - seg['ay0']) ** 2) ** 0.5
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0); rot = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    X0, X1 = int(max(0, cx - L / 2 - margin)), int(min(W, cx + L / 2 + margin)); Y0, Y1 = int(max(0, cy + top_off)), int(min(H, cy + bot_off))
    return rot[Y0:Y1, X0:X1], dict(x0=X0, y0=Y0, x1=X1, y1=Y1, angle=float(ang), cx=float(cx), cy=float(cy))


def trim_to_tablet(rot_img, X0, Y0, X1, Y1, min_share=0.3):
    """Shrink a crop's left and right edges to the first and last column, within the crop's rows, where at least
    min_share of the pixels are tablet rather than studio background. Clay is never trimmed, only background beyond
    the tablet's edge. Background polarity comes from the image border median."""
    g = cv2.cvtColor(rot_img, cv2.COLOR_BGR2GRAY) if rot_img.ndim == 3 else rot_img
    border = np.concatenate([g[:4].ravel(), g[-4:].ravel(), g[:, :4].ravel(), g[:, -4:].ravel()])
    bg = float(np.median(border))
    band = cv2.GaussianBlur(g[Y0:Y1, X0:X1], (0, 0), 2)
    if band.size == 0:
        return X0, X1
    tab = band > min(60, bg + 30) if bg < 128 else band < bg - 25
    on = np.where(tab.mean(axis=0) >= min_share)[0]
    if len(on) == 0:
        return X0, X1
    return X0 + int(on[0]), X0 + int(on[-1]) + 1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--faces', required=True); ap.add_argument('--pred', required=True); ap.add_argument('--method', default='det_n')
    ap.add_argument('--out', required=True); ap.add_argument('--pad', type=float, default=0.5); ap.add_argument('--margin', type=float, default=0.5)
    ap.add_argument('--quality', type=int, default=92); ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--bounds', choices=['gap', 'segments', 'midpoint', 'band'], default='gap', help="gap: from the gap above the line to the gap below, found from detector boxes; segments: from the predicted bands of the line model (<pred>/<method>_segments); midpoint: halfway between consecutive line centres (no boxes needed); band: centre +- (0.5+pad) pitch")
    ap.add_argument('--overlap', type=float, default=0.0, help='neighbouring crops share this many pitches beyond each boundary between lines, capped at the edge of the neighbour\'s band (see --overlap-into) so no crop takes in half of the next line (gap, segments and midpoint bounds; 0 = no overlap)')
    ap.add_argument('--overlap-into', type=float, default=0.0, help='with --overlap: how far past the edge of the neighbour\'s band a crop may reach, as a share of that band\'s height (0 = up to the edge; 0.25 = its outer quarter, for interlocking lines; clamped below 0.5)')
    ap.add_argument('--trim-background', action='store_true', help='after adding the margin, trim each crop\'s left and right edges back to the edge of the clay')
    ap.add_argument('--rotate-per-line', action='store_true', help='with --bounds segments: cut each line in its own frame, rotated by that line\'s predicted angle (curved/warped faces); lines without a segment fall back to the face-level frame')
    ap.add_argument('--split', choices=['all', 'test', 'tune'], default='all', help='restrict to faces of this Phase 5 split (tablet_data/ebl_tablets/split.csv)')
    ap.add_argument('--atf-source', choices=['ebl', 'cdli'], default='ebl', help='where the ATF line labels come from: eBL fragment.json (eBL faces) or the CDLI dump (faces.csv needs a p_number column)')
    ap.add_argument('--det', default=f'{EBL}/detections', help='detector boxes used for gap bounds')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    faces = list(csv.DictReader(open(a.faces)))
    if a.split != 'all' and os.path.exists(f'{EBL}/split.csv'):
        fs = {r['face'] for r in csv.DictReader(open(f'{EBL}/split.csv')) if r['split'] == a.split}; faces = [f for f in faces if f['face'] in fs]
    runs = {r['face']: r for r in csv.DictReader(open(os.path.join(a.pred, 'faces_run.csv'))) if r['method'] == a.method}
    rows = []; n_faces = 0
    for f in faces:
        r = runs.get(f['face']); pp = os.path.join(a.pred, a.method, f['face'] + '.csv')
        if r is None or not os.path.exists(pp): continue
        lines = list(csv.DictReader(open(pp)))
        if not lines: continue
        img = cv2.imread(f['file'])
        if img is None: continue
        H, W = img.shape[:2]; skew = float(r['skew_deg']); rot = rotate(img.astype(np.float32), skew) if img.ndim == 2 else np.dstack([rotate(img[:, :, c].astype(np.float32), skew) for c in range(3)])
        rot = np.clip(rot, 0, 255).astype(np.uint8)
        ys = np.array([float(l['y_deskewed']) for l in lines]); pitch = float(np.median(np.diff(ys))) if len(ys) > 1 else float(r['pitch_px'])
        if not np.isfinite(pitch) or pitch <= 2: pitch = float(r['pitch_px'])
        half = (0.5 + a.pad) * pitch; mx = a.margin * pitch
        segs_by_line = {}
        if a.rotate_per_line and a.bounds == 'segments':
            sp = os.path.join(a.pred, a.method + '_segments', f['face'] + '.csv')
            if os.path.exists(sp):
                M0 = cv2.getRotationMatrix2D((W / 2, H / 2), skew, 1.0); segl = []
                for r in csv.DictReader(open(sp)):
                    sg = {k: float(r[k]) for k in ('ax0', 'ay0', 'ax1', 'ay1', 'thickness')}; p0 = M0 @ np.array([sg['ax0'], sg['ay0'], 1.0]); p1 = M0 @ np.array([sg['ax1'], sg['ay1'], 1.0]); segl.append(((p0[1] + p1[1]) / 2, sg))
                segl.sort(key=lambda t: t[0])
                if len(segl) == len(lines): segs_by_line = {i + 1: sg for i, (_, sg) in enumerate(segl)}
        line_bounds_from_boxes.edges = line_bounds_from_segments.edges = None
        if a.bounds == 'gap': bounds = line_bounds_from_boxes(os.path.join(a.det, f['face'] + '.csv'), W, H, skew, ys, pitch)
        elif a.bounds == 'midpoint': bounds = line_bounds_midpoint(ys, pitch)
        elif a.bounds == 'segments': bounds = line_bounds_from_segments(os.path.join(a.pred, a.method + '_segments', f['face'] + '.csv'), W, H, skew, ys, pitch)
        else: bounds = None
        if bounds is not None and a.overlap > 0:
            edges = {'gap': getattr(line_bounds_from_boxes, 'edges', None), 'segments': getattr(line_bounds_from_segments, 'edges', None)}.get(a.bounds)
            bounds = apply_overlap(bounds, ys, pitch, a.overlap, edges if edges is not None and len(edges) == len(ys) else None, a.overlap_into)
        # ATF alignment: only when the counts agree
        labels = []
        if a.atf_source == 'ebl':
            try:
                frag = json.load(open(os.path.join(EBL, f['script'], f['tablet'], 'fragment.json'), encoding='utf-8'))
                labels = atf_labels_for_side(frag.get('atf'), f['side'])
            except Exception: pass
        else:
            if not hasattr(main, 'cdli'):
                main.cdli = {}
                for l in open('data/cdli/artifacts_json/transliterations.jsonl', encoding='utf-8'):
                    rec = json.loads(l); main.cdli[rec['p_number']] = rec['atf']
            labels = atf_labels_for_side(main.cdli.get(f.get('p_number', ''), ''), f['side'])
        aligned = len(labels) == len(lines) and len(labels) > 0
        d = os.path.join(a.out, f['face']); os.makedirs(d, exist_ok=True); n_faces += 1
        for i, l in enumerate(lines, 1):
            y = float(l['y_deskewed']); x0 = float(l['x0']) - mx; x1 = float(l['x1']) + mx
            if bounds is not None: ylo, yhi = bounds[i - 1]
            else: ylo, yhi = y - half, y + half
            X0, X1 = int(max(0, x0)), int(min(W, x1)); Y0, Y1 = int(max(0, ylo)), int(min(H, yhi))
            if a.trim_background and Y1 - Y0 >= 4 and X1 - X0 >= 4:
                X0, X1 = trim_to_tablet(rot, X0, Y0, X1, Y1)
            if X1 - X0 < 4 or Y1 - Y0 < 4: continue
            name = f"{f['face']}__L{i:02d}.jpg"; line_angle = ''
            sg = segs_by_line.get(i); sg_ang = np.degrees(np.arctan2(sg['ay1'] - sg['ay0'], sg['ax1'] - sg['ax0'])) if sg else None
            if sg is not None and abs(((sg_ang + skew) + 90) % 180 - 90) <= 20:   # per-line frame only within 20 deg of the face skew
                cimg, rect = crop_along_segment(img, segs_by_line[i], ylo - y, yhi - y, mx)
                if cimg.size and cimg.shape[0] >= 4 and cimg.shape[1] >= 4:
                    cv2.imwrite(os.path.join(d, name), cimg, [cv2.IMWRITE_JPEG_QUALITY, a.quality]); X0, Y0, X1, Y1 = rect['x0'], rect['y0'], rect['x1'], rect['y1']; line_angle = round(rect['angle'], 2)
                else: cv2.imwrite(os.path.join(d, name), rot[Y0:Y1, X0:X1], [cv2.IMWRITE_JPEG_QUALITY, a.quality])
            else: cv2.imwrite(os.path.join(d, name), rot[Y0:Y1, X0:X1], [cv2.IMWRITE_JPEG_QUALITY, a.quality])
            rows.append(dict(crop=os.path.join(d, name), face=f['face'], tablet=f.get('tablet', f.get('p_number', '')), script=f.get('script', ''), side=f['side'], line_no=i, n_lines=len(lines),
                             atf_line=labels[i - 1] if aligned else '', atf_aligned=aligned, y_center=round(y, 1), x0=X0, y0=Y0, x1=X1, y1=Y1, pitch_px=round(pitch, 1),
                             skew_deg=skew, line_angle_deg=line_angle, face_w=W, face_h=H, photo_crop_x0=f.get('crop_x0', ''), photo_crop_y0=f.get('crop_y0', ''), n_boxes=l.get('n_boxes', '')))
        if a.limit and n_faces >= a.limit: break
    with open(os.path.join(a.out, 'crops.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    al = sum(1 for r in rows if r['atf_aligned']); print(f'{len(rows)} crops from {n_faces} faces; crops with an ATF line label: {al} ({al/len(rows):.0%}); median crop {np.median([r["x1"]-r["x0"] for r in rows]):.0f} x {np.median([r["y1"]-r["y0"] for r in rows]):.0f} px')


if __name__ == '__main__':
    main()
