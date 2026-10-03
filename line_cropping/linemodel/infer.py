"""Phase 7 inference and read-out: face image -> line segments -> predictions in the pipeline's format.

  .venv/bin/python -m line_cropping.linemodel.infer --weights weights/linemodel/run1/model.pt \
      --faces tablet_data/ebl_tablets/faces/faces.csv --out tablet_data/ebl_tablets/predictions --method unet [--split test]

Read-out (optional end refinement: --end-thr extends line ends along the band, --clip-tablet trims ends over
background): kernel = sigmoid(ch0) > kernel_thr; connected components (area >= min_area px at stride 4) are line
instances; every band pixel (sigmoid(ch1) > band_thr) within max_reach x the median kernel height joins its nearest
kernel component (farther band pixels are dropped); a PCA line through each
component's band pixels gives the segment and its thickness. Predictions are written like the detector route
(faces_run.csv + <method>/<face>.csv with y_deskewed/x0/x1 in the frame rotated by the face's median line angle), so
evaluate_lines.py / phase5_eval.py score them unchanged, plus <method>_segments/<face>.csv with the raw segments.
"""
from line_cropping.paths import EBL
import argparse, csv, os, time
import numpy as np, cv2, torch
from line_cropping.linemodel.model import LineUNet
from line_cropping.linemodel.targets import STRIDE
from line_cropping.lines.lines_from_boxes import force_n


def load_model(weights, dev):
    m = LineUNet(pretrained=False); m.load_state_dict(torch.load(weights, map_location=dev)); return m.to(dev).eval()


@torch.no_grad()
def predict_face(model, path_or_img, dev, long_side=1024, pad=32):
    im = cv2.imread(path_or_img) if isinstance(path_or_img, str) else path_or_img; H, W = im.shape[:2]; s = long_side / max(H, W)
    r = cv2.resize(im, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA); h, w = r.shape[:2]
    ph, pw = (-h) % 32, (-w) % 32; r = cv2.copyMakeBorder(r, 0, ph, 0, pw, cv2.BORDER_REFLECT)
    x = torch.from_numpy(cv2.cvtColor(r, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255)[None].to(dev)
    with torch.autocast(dev, dtype=torch.float16, enabled=dev == 'cuda'): out = model(x).float()[0].cpu().numpy()
    out = out[:, :h // STRIDE + (1 if h % STRIDE else 0), :w // STRIDE + (1 if w % STRIDE else 0)]
    maps = np.concatenate([1 / (1 + np.exp(-out[:2])), out[2:4]]); return maps, s     # (4, h/4, w/4), scale image->resized


def tablet_mask_for(img, maps_shape, scale):
    """Tablet-versus-background mask at the resolution of the model's maps (stride 4 of the image resized by scale).
    Background polarity from the border median: dark studio cloth (< 60) or light paper (< median - 25)."""
    h, w = maps_shape[-2:]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    small = cv2.resize(g, (max(1, int(round(g.shape[1] * scale / STRIDE))), max(1, int(round(g.shape[0] * scale / STRIDE)))), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 1.0)
    b = np.concatenate([small[:2].ravel(), small[-2:].ravel(), small[:, :2].ravel(), small[:, -2:].ravel()]); bg = float(np.median(b))
    m = (small > min(60, bg + 30)) if bg < 128 else (small < bg - 25)
    out = np.zeros((h, w), bool); hh, ww = min(h, m.shape[0]), min(w, m.shape[1]); out[:hh, :ww] = m[:hh, :ww]
    return out


def _probe(arr, p, n, hw, k=5):
    """Max of arr over k points across the line (normal n, half-width hw) at position p; 0 outside the map."""
    h, w = arr.shape; best = 0.0
    for o in np.linspace(-hw, hw, k):
        x, y = int(round(p[0] + n[0] * o)), int(round(p[1] + n[1] * o))
        if 0 <= x < w and 0 <= y < h: best = max(best, float(arr[y, x]))
    return best


def refine_ends(a, b, d, hw, band_prob, tab=None, end_thr=0.0, max_ext=0.0, tab_min=0.3):
    """Move a segment's ends (map pixels) along its direction d. Extension: while the band probability across the line
    stays above end_thr, step outward up to max_ext pixels per end (two consecutive misses stop it). Background trim:
    where tab (tablet mask) is given, step each end inward while less than tab_min of the line's width is tablet."""
    n = np.array([-d[1], d[0]])
    if end_thr > 0 and max_ext > 0:
        for sign in (1, -1):
            e = b if sign == 1 else a; last = e.copy(); miss = 0; t = 1.0
            while t <= max_ext:
                p = e + sign * d * t
                if _probe(band_prob, p, n, hw) > end_thr: last = p; miss = 0
                else:
                    miss += 1
                    if miss >= 2: break
                t += 1.0
            if sign == 1: b = last
            else: a = last
    if tab is not None:
        L = float(np.linalg.norm(b - a))
        for sign in (1, -1):
            t = 0.0
            while t < 0.45 * L:
                p = (b - d * t) if sign == 1 else (a + d * t)
                if _probe(tab.astype(np.float32), p, n, hw) >= 1.0 or _tab_share(tab, p, n, hw) >= tab_min: break
                t += 1.0
            if sign == 1: b = b - d * t
            else: a = a + d * t
    return a, b


def _tab_share(tab, p, n, hw, k=7):
    h, w = tab.shape; hit = 0; tot = 0
    for o in np.linspace(-hw, hw, k):
        x, y = int(round(p[0] + n[0] * o)), int(round(p[1] + n[1] * o)); tot += 1
        if 0 <= x < w and 0 <= y < h and tab[y, x]: hit += 1
    return hit / tot


def read_lines(maps, scale, kernel_thr=0.5, band_thr=0.4, min_area=12, max_reach=1.5, max_dev=45.0, merge_dy=0.5,
               end_thr=0.0, max_ext=3.0, tab=None):
    """maps (4,h,w) at stride 4 of the resized image; returns segments in ORIGINAL face pixels.
    end_thr > 0 extends each line's ends while the band probability stays above it (up to max_ext x the median kernel
    height per end); tab (tablet mask from tablet_mask_for) trims ends that sit over the studio background."""
    kern = (maps[0] > kernel_thr).astype(np.uint8); band = (maps[1] > band_thr).astype(np.uint8) | kern
    n, lab, stats, _ = cv2.connectedComponentsWithStats(kern, connectivity=8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not keep: return []
    remap = np.zeros(n, np.int32)
    for k, i in enumerate(keep, 1): remap[i] = k
    kl = remap[lab]
    dist, nearest = cv2.distanceTransformWithLabels((kl == 0).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(kl); px_label = {}
    for y, x in zip(ys, xs): px_label.setdefault(int(nearest[y, x]), int(kl[y, x]))
    lut = np.zeros(int(nearest.max()) + 1, np.int32)
    for p, l in px_label.items(): lut[p] = l
    # a band pixel joins its nearest kernel only if it is close to it: within max_reach x the median kernel height,
    # so a stray blob far from any kernel is dropped instead of stretching a segment across the face
    kh = np.median([stats[i, cv2.CC_STAT_HEIGHT] for i in keep]); reach = max(3.0, max_reach * kh)
    band_lab = np.where((band > 0) & (dist <= reach), lut[nearest], 0)
    segs = []; f = STRIDE / scale
    for k in range(1, len(keep) + 1):
        yy, xx = np.nonzero(band_lab == k)
        if len(xx) < 6: continue
        P = np.stack([xx, yy], 1).astype(np.float64); c = P.mean(0); U, S, Vt = np.linalg.svd(P - c, full_matrices=False); d = Vt[0]
        if d[0] < 0: d = -d
        t = (P - c) @ d; perp = np.abs((P - c) @ Vt[1]); a, b = c + d * t.min(), c + d * t.max(); th = 2 * (perp.mean() * 2)
        segs.append(dict(ax0=a[0] * f, ay0=a[1] * f, ax1=b[0] * f, ay1=b[1] * f, thickness=max(th * f, 2.0), area=int(len(xx)), score=float(maps[0][kl == k].mean())))
    # text lines on one face share a direction: drop segments more than max_dev degrees off the median angle
    if len(segs) >= 3:
        angs = np.array([np.degrees(np.arctan2(s['ay1'] - s['ay0'], s['ax1'] - s['ax0'])) for s in segs]); med = np.median(angs)
        dev = np.abs((angs - med + 90) % 180 - 90); segs = [s for s, d in zip(segs, dev) if d <= max_dev]
    if merge_dy > 0: segs = merge_fragments(segs, merge_dy)
    # ends are refined only after merging: extended fragments would overlap and no longer be recognised as one line
    if end_thr > 0 or tab is not None:
        for sg in segs:
            a = np.array([sg['ax0'], sg['ay0']]) / f; b = np.array([sg['ax1'], sg['ay1']]) / f
            d = (b - a) / max(np.linalg.norm(b - a), 1e-6)
            a, b = refine_ends(a, b, d, max(1.0, sg['thickness'] / f / 2), maps[1], tab, end_thr, max_ext * kh)
            sg.update(ax0=a[0] * f, ay0=a[1] * f, ax1=b[0] * f, ay1=b[1] * f)
    return segs


def merge_fragments(segs, merge_dy=0.5, max_overlap=0.3):
    """Join pieces of one text line that the kernel map split side by side. Two segments are fragments of the same
    line when, in the frame rotated by the face's median line angle, their centre heights differ by less than
    merge_dy x the median band thickness and their horizontal ranges overlap by less than max_overlap of the shorter.
    Merged segment: from the leftmost endpoint to the rightmost, thickness = max, areas summed."""
    if len(segs) < 2: return segs
    ang = np.radians(np.median([np.degrees(np.arctan2(s['ay1'] - s['ay0'], s['ax1'] - s['ax0'])) for s in segs]))
    c, sn = np.cos(-ang), np.sin(-ang); rot = lambda x, y: (c * x - sn * y, sn * x + c * y)
    th = float(np.median([s['thickness'] for s in segs])); items = []
    for s in segs:
        (u0, v0), (u1, v1) = rot(s['ax0'], s['ay0']), rot(s['ax1'], s['ay1'])
        items.append(dict(seg=s, v=(v0 + v1) / 2, u0=min(u0, u1), u1=max(u0, u1)))
    items.sort(key=lambda it: it['v']); groups = []
    for it in items:
        for g in groups:
            if abs(g[-1]['v'] - it['v']) < merge_dy * th and all(min(it['u1'], o['u1']) - max(it['u0'], o['u0']) < max_overlap * min(it['u1'] - it['u0'], o['u1'] - o['u0']) for o in g):
                g.append(it); break
        else: groups.append([it])
    out = []
    for g in groups:
        if len(g) == 1: out.append(g[0]['seg']); continue
        pts = [(s['seg']['ax0'], s['seg']['ay0']) for s in g] + [(s['seg']['ax1'], s['seg']['ay1']) for s in g]
        us = [rot(x, y)[0] for x, y in pts]; a, b = pts[int(np.argmin(us))], pts[int(np.argmax(us))]
        out.append(dict(ax0=a[0], ay0=a[1], ax1=b[0], ay1=b[1], thickness=max(s['seg']['thickness'] for s in g),
                        area=sum(s['seg']['area'] for s in g), score=float(np.mean([s['seg']['score'] for s in g])), merged=len(g)))
    return out


def segments_to_lines(segs, W, H):
    """Face-level skew (median segment angle) and each segment's centre height in the frame rotated by -skew."""
    if not segs: return 0.0, []
    ang = float(np.median([np.degrees(np.arctan2(s['ay1'] - s['ay0'], s['ax1'] - s['ax0'])) for s in segs]))
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0); rows = []
    for s in segs:
        p0 = M @ np.array([s['ax0'], s['ay0'], 1.0]); p1 = M @ np.array([s['ax1'], s['ay1'], 1.0])
        rows.append(dict(y=(p0[1] + p1[1]) / 2, x0=min(p0[0], p1[0]), x1=max(p0[0], p1[0]), thickness=s['thickness']))
    rows.sort(key=lambda r: r['y']); return ang, rows



def reconcile_count(rows, n, pitch):
    """Optional refinement with the transliteration's line count n: missing lines are interpolated into gaps wider
    than 1.5 pitches (or appended at the median pitch), surplus lines are merged closest-first. Rows are the
    deskewed-frame lines from segments_to_lines; inserted rows span the block's horizontal extent."""
    if not rows or not n or n == len(rows): return rows
    y = np.array([r['y'] for r in rows]); groups = [[i] for i in range(len(rows))]; cents = [float(v) for v in y]
    groups, cents = force_n(groups, cents, y, n, box_h=pitch / 1.3 if pitch else None)
    x0, x1 = min(r['x0'] for r in rows), max(r['x1'] for r in rows); th = float(np.median([r['thickness'] for r in rows])); out = []
    for g, c in sorted(zip(groups, cents), key=lambda t: t[1]):
        if g: out.append(dict(y=c, x0=min(rows[i]['x0'] for i in g), x1=max(rows[i]['x1'] for i in g), thickness=th, inserted=False))
        else: out.append(dict(y=c, x0=x0, x1=x1, thickness=th, inserted=True))
    return out


def score_against_truth(pred_segs, truth_segs, file=None, tol=0.35, W=None, H=None):
    """Line recall/precision: predicted vs truth centre heights in the truth's deskewed frame, match within tol x pitch."""
    if W is None:
        im = cv2.imread(file, cv2.IMREAD_GRAYSCALE); H, W = im.shape
    ang, T = segments_to_lines(truth_segs, W, H)
    if not T: return float('nan'), float('nan')
    ty = np.array([r['y'] for r in T]); pitch = float(np.median(np.diff(ty))) if len(ty) > 1 else T[0]['thickness'] * 1.5
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0); py = np.array([(M @ np.array([(s['ax0'] + s['ax1']) / 2, (s['ay0'] + s['ay1']) / 2, 1.0]))[1] for s in pred_segs])
    if len(py) == 0: return 0.0, float('nan')
    used = set(); hit = 0
    for y in ty:
        d = np.abs(py - y); j = int(np.argmin(d))
        if d[j] < tol * pitch and j not in used: used.add(j); hit += 1
    return hit / len(ty), len(used) / len(py)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--weights', required=True); ap.add_argument('--faces', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--method', default='unet'); ap.add_argument('--split', choices=['all', 'test', 'tune'], default='all'); ap.add_argument('--n-col', default='atf_lines_side')
    ap.add_argument('--long-side', type=int, default=1024); ap.add_argument('--kernel-thr', type=float, default=None, help='default: the training run\'s config.json, else 0.5'); ap.add_argument('--band-thr', type=float, default=None, help='default: config.json, else 0.4'); ap.add_argument('--min-area', type=int, default=None, help='default: config.json, else 12');
    ap.add_argument('--max-reach', type=float, default=None, help='band pixels join a kernel only within this many kernel heights (default: config.json, else 1.5)')
    ap.add_argument('--max-dev', type=float, default=None, help='drop segments more than this many degrees off the face median angle (default: config.json, else 45)')
    ap.add_argument('--merge-dy', type=float, default=None, help='merge side-by-side fragments whose centres differ by less than this x band thickness; 0 disables (default: config.json, else 0.5)')
    ap.add_argument('--end-thr', type=float, default=None, help='extend line ends while band probability stays above this; 0 disables (default: config.json, else 0.2, chosen in crop width Test 2)')
    ap.add_argument('--max-ext', type=float, default=None, help='longest extension per end, in kernel heights (default: config.json, else 3)')
    ap.add_argument('--clip-tablet', type=int, choices=[0, 1], default=None, help='1 = trim line ends that sit over the studio background (default: config.json, else 1)')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--use-n', action='store_true', help='reconcile the line count with the transliteration count (--n-col) where one exists; method name gets a _n suffix unless --method is given explicitly')
    ap.add_argument('--viz', type=int, default=0, help='write prediction overlays for the first N faces to <out>/<method>_viz/ (green = predicted band, lime = predicted kernel, red = read-out segments)')
    a = ap.parse_args(); dev = 'cuda' if torch.cuda.is_available() else 'cpu'; model = load_model(a.weights, dev)
    import json; cfgp = os.path.join(os.path.dirname(a.weights), 'config.json'); cfg = json.load(open(cfgp)) if os.path.exists(cfgp) else {}
    a.kernel_thr = a.kernel_thr if a.kernel_thr is not None else cfg.get('kernel_thr', 0.5); a.band_thr = a.band_thr if a.band_thr is not None else cfg.get('band_thr', 0.4)
    for k, d in (('min_area', 12), ('max_reach', 1.5), ('max_dev', 45.0), ('merge_dy', 0.5), ('end_thr', 0.2), ('max_ext', 3.0), ('clip_tablet', 1)):
        if getattr(a, k) is None: setattr(a, k, cfg.get(k, d))
    print(f'read-out: kernel_thr {a.kernel_thr} band_thr {a.band_thr} min_area {a.min_area} max_reach {a.max_reach} max_dev {a.max_dev} merge_dy {a.merge_dy} end_thr {a.end_thr} max_ext {a.max_ext} clip_tablet {a.clip_tablet}', flush=True)
    if a.use_n and a.method == 'unet': a.method = 'unet_n'
    faces = list(csv.DictReader(open(a.faces)))
    if a.split != 'all' and os.path.exists(f'{EBL}/split.csv'):
        fs = {r['face'] for r in csv.DictReader(open(f'{EBL}/split.csv')) if r['split'] == a.split}; faces = [f for f in faces if f['face'] in fs]
    if a.limit: faces = faces[:a.limit]
    os.makedirs(os.path.join(a.out, a.method), exist_ok=True); os.makedirs(os.path.join(a.out, a.method + '_segments'), exist_ok=True); run = []; t0 = time.time()
    for k, f in enumerate(faces):
        im = cv2.imread(f['file'])
        if im is None: continue
        H, W = im.shape[:2]; ts = time.time(); maps, s = predict_face(model, im, dev, a.long_side); tab = tablet_mask_for(im, maps.shape, s) if a.clip_tablet else None; segs = read_lines(maps, s, a.kernel_thr, a.band_thr, a.min_area, a.max_reach, a.max_dev, a.merge_dy, a.end_thr, a.max_ext, tab)
        ang, rows = segments_to_lines(segs, W, H)
        n_atf = int(float(f[a.n_col])) if f.get(a.n_col) else None
        if a.use_n and n_atf:
            ys0 = [r['y'] for r in rows]; p0 = float(np.median(np.diff(ys0))) if len(ys0) > 1 else (rows[0]['thickness'] * 1.5 if rows else 0)
            rows = reconcile_count(rows, n_atf, p0)
        with open(os.path.join(a.out, a.method + '_segments', f['face'] + '.csv'), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['ax0', 'ay0', 'ax1', 'ay1', 'thickness', 'area', 'score', 'merged'], extrasaction='ignore', restval=1); w.writeheader(); w.writerows([{k2: round(v, 2) if isinstance(v, float) else v for k2, v in s.items()} for s in segs])
        with open(os.path.join(a.out, a.method, f['face'] + '.csv'), 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(['line_no', 'y_deskewed', 'x0', 'x1', 'inserted'])
            for i, r in enumerate(rows, 1): w.writerow([i, round(r['y'], 1), round(r['x0']), round(r['x1']), int(r.get('inserted', False))])
        if k < a.viz:
            vd = os.path.join(a.out, a.method + '_viz'); os.makedirs(vd, exist_ok=True); ov = im.astype(np.float32)
            up = lambda m: cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
            band, kern = up(maps[1]) > a.band_thr, up(maps[0]) > a.kernel_thr
            ov[band] = ov[band] * 0.5 + np.array([0, 200, 0]) * 0.5; ov[kern] = ov[kern] * 0.4 + np.array([0, 230, 230]) * 0.6; ov = ov.astype(np.uint8)
            for sg in segs: cv2.line(ov, (int(sg['ax0']), int(sg['ay0'])), (int(sg['ax1']), int(sg['ay1'])), (0, 0, 255), max(2, H // 500))
            sc = min(1.0, 1400 / max(H, W)); cv2.putText(ov, f"{f['face']}  {len(segs)} lines", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(vd, f['face'] + '.jpg'), cv2.resize(ov, None, fx=sc, fy=sc), [cv2.IMWRITE_JPEG_QUALITY, 82])
        n = int(float(f[a.n_col])) if f.get(a.n_col) else ''
        ys = [r['y'] for r in rows]; pitch = float(np.median(np.diff(ys))) if len(ys) > 1 else (rows[0]['thickness'] * 1.5 if rows else 0)
        run.append(dict(face=f['face'], method=a.method, n_atf=n, n_pred=len(rows), skew_deg=round(ang, 2), pitch_px=round(pitch, 1), body_h=H, seconds=round(time.time() - ts, 2)))
        if (k + 1) % 100 == 0: print(f'{k+1}/{len(faces)} faces, {time.time()-t0:.0f}s', flush=True)
    p = os.path.join(a.out, 'faces_run.csv'); old = [r for r in csv.DictReader(open(p))] if os.path.exists(p) else []; old = [r for r in old if r['method'] != a.method]
    with open(p, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['face', 'method', 'n_atf', 'n_pred', 'skew_deg', 'pitch_px', 'body_h', 'seconds']); w.writeheader(); w.writerows(old + run)
    print(f'{a.method}: {len(run)} faces in {time.time()-t0:.0f}s; median lines per face {np.median([r["n_pred"] for r in run]) if run else 0:.0f} -> {a.out}/{a.method}/')


if __name__ == '__main__':
    main()
