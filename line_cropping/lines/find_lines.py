"""Phase 3a: place text lines on a prepared face.

Three methods, all working on the deskewed row profile (see profile_lines.py):
  free  : peaks of the profile, no prior knowledge (line count estimated from autocorrelation)
  comb  : N lines (N from the transliteration) on a rigid comb: best pitch and offset by total response
  dp    : N lines at free positions by dynamic programming: maximise profile response subject to a quadratic
          penalty on each gap's deviation from the comb's pitch (gaps limited to 0.55-1.5 pitch)

Batch use:  python -m line_cropping.lines.find_lines --faces ebl_tablets/faces/faces.csv --out ebl_tablets/predictions
Writes <out>/<method>/<face>.csv (line_no, y_deskewed, x0, x1) and <out>/faces_run.csv (skew, pitch, timing).
"""
import argparse, csv, os, time
import numpy as np, cv2
from line_cropping.lines.profile_lines import rotate, row_profile, estimate_pitch, free_peaks, comb_fit


def tablet_mask(small):
    """Tablet pixels (bright on dark background or dark on light background), eroded so that the outline
    of the tablet and the background never contribute to the profile."""
    u8 = np.clip(small, 0, 255).astype(np.uint8); b = 3
    border = np.concatenate([u8[:b].ravel(), u8[-b:].ravel(), u8[:, :b].ravel(), u8[:, -b:].ravel()])
    bg = float(np.median(border)); otsu, _ = cv2.threshold(cv2.GaussianBlur(u8, (0, 0), 2), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (u8 > min(60, otsu)) if bg < 128 else (u8 < max(otsu, bg - 25))
    m = m.astype(np.uint8)
    k = max(3, int(0.04 * min(small.shape)) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return m


def masked_row_profile(img, mask, sigma):
    gy = cv2.Sobel(cv2.GaussianBlur(img, (0, 0), 1.5), cv2.CV_32F, 0, 1, ksize=3)
    num = (np.abs(gy) * mask).sum(axis=1); den = mask.sum(axis=1)
    p = np.where(den > 0.2 * mask.shape[1], num / np.maximum(den, 1), 0.0).astype(np.float32)
    return cv2.GaussianBlur(p.reshape(-1, 1), (0, 0), max(0.5, sigma)).ravel()


def deskew_angle(img, top, bottom, left, right):
    """Coarse-to-fine skew search on a downscaled copy (<= 700 px tall): 2-degree steps over +-12, then
    0.5-degree steps around the best. The criterion is the variance of the row profile computed over tablet
    pixels only (mask eroded away from the outline), so the tablet's edge cannot dominate."""
    H, W = img.shape; f = min(1.0, 700.0 / H)
    small = cv2.resize(img, (max(1, int(W * f)), max(1, int(H * f))), interpolation=cv2.INTER_AREA) if f < 1 else img
    h, w = small.shape; mask = tablet_mask(small).astype(np.float32)
    if mask.sum() < 0.05 * mask.size: mask = np.ones_like(mask)
    def score(a):
        rot = rotate(small, a); rm = cv2.warpAffine(mask, cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0), (w, h))
        body = rot[int(top * h):int(bottom * h), int(left * w):int(right * w)]; bm = rm[int(top * h):int(bottom * h), int(left * w):int(right * w)] > 0.5
        return masked_row_profile(body, bm.astype(np.float32), 1.5).var()
    coarse = np.arange(-12, 12.1, 2.0); best = max(coarse, key=score)
    fine = np.arange(best - 1.5, best + 1.51, 0.5); return float(max(fine, key=score))


def _work(args):
    f, methods, n_col, out, skew_col = args
    gray = cv2.imread(f['file'], cv2.IMREAD_GRAYSCALE)
    if gray is None or min(gray.shape) < 40: return []
    # work at <= 1600 px height; predictions are scaled back to face pixels
    H0 = gray.shape[0]; sc = min(1.0, 1600.0 / H0)
    if sc < 1: gray = cv2.resize(gray, (int(gray.shape[1] * sc), int(H0 * sc)), interpolation=cv2.INTER_AREA)
    n = int(float(f[n_col])) if f.get(n_col) not in (None, '') else None
    rows = []; skew = (-float(f[skew_col]) if (skew_col and f.get(skew_col)) else None)
    for m in methods:
        ts = time.time()
        res = find_lines_on_face(gray, n_lines=n, method=m, skew=skew); skew = res['skew_deg']
        with open(os.path.join(out, m, f['face'] + '.csv'), 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(['line_no', 'y_deskewed', 'x0', 'x1'])
            for i, y in enumerate(res['lines_y'], 1): w.writerow([i, round(y / sc, 1), round(res['x0'] / sc), round(res['x1'] / sc)])
        rows.append(dict(face=f['face'], method=m, n_atf=n or '', n_pred=len(res['lines_y']), skew_deg=res['skew_deg'], pitch_px=round(res['pitch_px'] / sc, 1), body_h=round(res['body_h'] / sc), seconds=round(time.time() - ts, 2)))
    return rows


def dp_positions(p, n, pitch0, gap_range=(0.55, 1.5), lam=6.0, step=1):
    """Place n lines at free row positions y_1 < ... < y_n maximising the profile response while keeping
    consecutive gaps near pitch0: minimise sum(-p(y_i)/max p) + lam * sum(((g_i - pitch0)/pitch0)^2) over
    gaps g_i in [gap_range] * pitch0. Chain dynamic programme over row positions (O(n * H * window))."""
    h = len(p); pn = p / (p.max() or 1.0)
    ys = np.arange(0, h, step); unary = -np.interp(ys, np.arange(h), pn)
    gmin, gmax = int(gap_range[0] * pitch0 / step), max(int(gap_range[0] * pitch0 / step) + 1, int(gap_range[1] * pitch0 / step))
    gaps = np.arange(gmin, gmax + 1); pen = lam * ((gaps * step - pitch0) / pitch0) ** 2
    cost = unary.copy(); back = np.zeros((n, len(ys)), dtype=np.int32)
    for i in range(1, n):
        best = np.full(len(ys), np.inf); arg = np.zeros(len(ys), dtype=np.int32)
        for g, pg in zip(gaps, pen):
            if g >= len(ys): break
            cand = np.full(len(ys), np.inf); cand[g:] = cost[:-g] + pg
            better = cand < best; best[better] = cand[better]; arg[better] = (np.arange(len(ys)) - g)[better]
        cost = best + unary; back[i] = arg
    j = int(np.argmin(cost)); out = [0.0] * n
    for i in range(n - 1, -1, -1):
        out[i] = float(ys[j]); j = back[i][j] if i > 0 else j
    return np.array(out)


def find_lines_on_face(gray, n_lines=None, method='dp', top=0.02, bottom=0.98, left=0.05, right=0.95, skew=None):
    img = gray.astype(np.float32); H, W = img.shape
    ang = deskew_angle(img, top, bottom, left, right) if skew is None else skew
    rot = rotate(img, ang)
    y0, y1, x0, x1 = int(top * H), int(bottom * H), int(left * W), int(right * W)
    body = rot[y0:y1, x0:x1]
    if method == 'free' or not n_lines:
        p = row_profile(body, max(1.5, 0.01 * body.shape[0])); pitch = estimate_pitch(p); ys = free_peaks(p, pitch).astype(float)
    else:
        p = row_profile(body, max(1.5, 0.08 * body.shape[0] / n_lines)); pitch, ys = comb_fit(p, n_lines)
        if method == 'dp': ys = dp_positions(p, n_lines, pitch)
    return dict(skew_deg=float(ang), pitch_px=float(pitch), lines_y=(ys + y0).tolist(), x0=x0, x1=x1, body_h=body.shape[0])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--faces', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--methods', nargs='+', default=['free', 'comb', 'dp']); ap.add_argument('--n-col', default='atf_lines_side')
    ap.add_argument('--limit', type=int, default=None); ap.add_argument('--workers', type=int, default=8); ap.add_argument('--skew-col', default=None, help='oracle: take the deskew angle as minus this faces.csv column'); ap.add_argument('--single-column-only', action='store_true')
    a = ap.parse_args()
    faces = list(csv.DictReader(open(a.faces)))
    if a.single_column_only: faces = [f for f in faces if f.get('n_columns', '1') in ('', '1')]
    if a.limit: faces = faces[:a.limit]
    for m in a.methods: os.makedirs(os.path.join(a.out, m), exist_ok=True)
    run_rows = []; t0 = time.time()
    from multiprocessing import Pool
    with Pool(a.workers) as pool:
        for k, rows in enumerate(pool.imap_unordered(_work, [(f, a.methods, a.n_col, a.out, a.skew_col) for f in faces], chunksize=2)):
            run_rows += rows
            if (k + 1) % 100 == 0: print(f'{k+1}/{len(faces)} faces, {time.time()-t0:.0f}s', flush=True)
    with open(os.path.join(a.out, 'faces_run.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(run_rows[0].keys())); w.writeheader(); w.writerows(run_rows)
    print(f'done: {len(faces)} faces x {len(a.methods)} methods in {time.time()-t0:.0f}s -> {a.out}')


if __name__ == '__main__':
    main()
