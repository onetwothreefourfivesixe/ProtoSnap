"""Phase 3b: text lines from detected sign boxes.

For each face: estimate the line tilt from the boxes (median slope of the best-fitting line through each
box's nearest horizontal neighbours), rotate box centres into the deskewed frame (same convention as
profile_lines.rotate, i.e. cv2.getRotationMatrix2D(centre, angle)), then cluster centre heights into lines
with a 1-D gap rule (new line when the gap exceeds `gap_frac` x median box height) followed by a merge of
clusters closer than `merge_frac` x median box height. Optionally, when the transliteration line count N is
known, split or merge clusters until N lines remain (largest gaps split, closest pairs merge).

  python -m line_cropping.lines.lines_from_boxes --faces <faces.csv> --det <detections dir> --out <pred dir> [--use-n]
Output is in the same format as find_lines.py so evaluate_lines.py can score it.
"""
import argparse, csv, os, collections
import numpy as np, cv2


def tilt_from_boxes(cx, cy, h):
    """Median angle (deg) of segments joining each box centre to its nearest neighbour to the right within
    one box height vertically; 0 if too few pairs."""
    angs = []
    order = np.argsort(cx)
    for i in order:
        dx = cx - cx[i]; dy = cy - cy[i]
        cand = np.where((dx > 0.3 * h) & (dx < 6 * h) & (np.abs(dy) < 0.6 * h))[0]
        if len(cand):
            j = cand[np.argmin(dx[cand])]; angs.append(np.degrees(np.arctan2(dy[j], dx[j])))
    return float(np.median(angs)) if len(angs) >= 3 else 0.0


def cluster_rows(y, h, gap_frac=0.6, merge_frac=0.45, weights=None):
    order = np.argsort(y); ys = y[order]; groups = [[order[0]]]
    for i in order[1:]:
        if y[i] - y[groups[-1][-1]] > gap_frac * h: groups.append([i])
        else: groups[-1].append(i)
    cents = [float(np.mean(y[g])) for g in groups]
    # merge clusters that are too close (split lines)
    merged = True
    while merged and len(groups) > 1:
        merged = False
        d = np.diff(cents); k = int(np.argmin(d))
        if d[k] < merge_frac * h:
            groups[k] = groups[k] + groups[k + 1]; del groups[k + 1]; cents = [float(np.mean(y[g])) for g in groups]; merged = True
    return groups, cents


def force_n(groups, cents, y, n, box_h=None):
    """Reach exactly n lines. Missing lines are inserted where consecutive clusters are separated by more than
    ~1.5 pitches (the biggest gap in pitch units is filled first, evenly), then, if still short, unusually tall
    clusters are split. Surplus lines are removed by merging the closest pair."""
    groups = [list(g) for g in groups]; cents = list(cents)
    def pitch():
        d = np.diff(cents); d = d[d > 0]
        return float(np.median(d)) if len(d) else (1.3 * box_h if box_h else 1.0)
    while len(groups) < n:
        pt = pitch()
        # 1) over-merged clusters first: a cluster spanning more than 0.8 pitch holds two lines
        spans = [np.ptp(y[g]) if len(g) > 1 else -1 for g in groups]; k = int(np.argmax(spans))
        if spans[k] > 0.8 * pt:
            g = sorted(groups[k], key=lambda i: y[i]); m = len(g) // 2
            groups[k:k + 1] = [g[:m], g[m:]]; cents[k:k + 1] = [float(np.mean(y[g[:m]])), float(np.mean(y[g[m:]]))]; continue
        d = np.diff(cents) if len(cents) > 1 else np.array([])
        ratio = d / pt if len(d) else np.array([])
        # 2) then missing lines inside gaps of >= 1.5 pitch
        if len(ratio) and ratio.max() >= 1.5:
            k = int(np.argmax(ratio)); m = min(int(round(ratio[k])) - 1, n - len(groups))
            new = [cents[k] + (cents[k + 1] - cents[k]) * (j + 1) / (m + 1) for j in range(m)]
            for j, c in enumerate(new): groups.insert(k + 1 + j, []); cents.insert(k + 1 + j, c)
            continue
        # 3) nothing to split or fill: extend at the end with the median pitch (text continues beyond detections)
        groups.append([]); cents.append(cents[-1] + pt)
    while len(groups) > n:
        d = np.diff(cents); k = int(np.argmin(d)); groups[k] += groups[k + 1]; del groups[k + 1]
        cents[k] = float(np.mean(y[groups[k]])) if groups[k] else cents[k]; del cents[k + 1]
    return groups, cents


def lines_for_face(boxes, W, H, n_lines=None, score_thr=0.3, gap_frac=0.6, merge_frac=0.45):
    b = boxes[boxes[:, 4] >= score_thr] if len(boxes) else boxes
    if len(b) < 2: return None
    cx = (b[:, 0] + b[:, 2]) / 2; cy = (b[:, 1] + b[:, 3]) / 2; h = float(np.median(b[:, 3] - b[:, 1]))
    ang = tilt_from_boxes(cx, cy, h)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
    yr = M[1, 0] * cx + M[1, 1] * cy + M[1, 2]; xr = M[0, 0] * cx + M[0, 1] * cy + M[0, 2]
    groups, cents = cluster_rows(yr, h, gap_frac, merge_frac)
    if n_lines: groups, cents = force_n(groups, cents, yr, n_lines, h)
    lines = []
    gx0, gx1 = float(xr.min() - 0.5 * h), float(xr.max() + 0.5 * h)
    for g, c in sorted(zip(groups, cents), key=lambda t: t[1]):
        if g: lines.append(dict(y=c, x0=float(xr[g].min() - 0.5 * h), x1=float(xr[g].max() + 0.5 * h), n=len(g)))
        else: lines.append(dict(y=c, x0=gx0, x1=gx1, n=0))          # inserted line: span of the whole text block
    return dict(skew_deg=ang, box_h=h, lines=lines, pitch_px=float(np.median(np.diff(cents))) if len(cents) > 1 else h * 1.3)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--faces', required=True); ap.add_argument('--det', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--use-n', action='store_true'); ap.add_argument('--n-col', default='atf_lines_side'); ap.add_argument('--score-thr', type=float, default=None)
    ap.add_argument('--gap-frac', type=float, default=None); ap.add_argument('--merge-frac', type=float, default=None)
    ap.add_argument('--params', default='ebl_tablets/det_params.json', help='JSON with gap_frac/merge_frac/score_thr tuned by phase5_eval (CLI values override)')
    a = ap.parse_args()
    import json; P = json.load(open(a.params)) if os.path.exists(a.params) else {}
    gap = a.gap_frac if a.gap_frac is not None else P.get('gap_frac', 0.6); merge = a.merge_frac if a.merge_frac is not None else P.get('merge_frac', 0.45)
    thr = a.score_thr if a.score_thr is not None else P.get('score_thr', 0.3); print(f'thresholds: gap_frac {gap} merge_frac {merge} score_thr {thr}')
    method = 'det_n' if a.use_n else 'det'; os.makedirs(os.path.join(a.out, method), exist_ok=True)
    faces = list(csv.DictReader(open(a.faces))); run = []
    for f in faces:
        dp = os.path.join(a.det, f['face'] + '.csv')
        if not os.path.exists(dp): continue
        rows = list(csv.DictReader(open(dp, encoding='utf-8')))
        boxes = np.array([[float(r['x0']), float(r['y0']), float(r['x1']), float(r['y1']), float(r['score'])] for r in rows]) if rows else np.zeros((0, 5))
        n = int(float(f[a.n_col])) if (a.use_n and f.get(a.n_col)) else None
        if f.get('width') and f.get('height'): W, H = int(f['width']), int(f['height'])
        else:
            im = cv2.imread(f['file'], cv2.IMREAD_GRAYSCALE); H, W = im.shape
        res = lines_for_face(boxes, W, H, n, thr, gap, merge)
        if res is None: continue
        with open(os.path.join(a.out, method, f['face'] + '.csv'), 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(['line_no', 'y_deskewed', 'x0', 'x1', 'n_boxes'])
            for i, l in enumerate(res['lines'], 1): w.writerow([i, round(l['y'], 1), round(l['x0']), round(l['x1']), l['n']])
        run.append(dict(face=f['face'], method=method, n_atf=n or '', n_pred=len(res['lines']), skew_deg=round(res['skew_deg'], 2), pitch_px=round(res['pitch_px'], 1), body_h=H, seconds=0))
    p = os.path.join(a.out, 'faces_run.csv'); old = [r for r in csv.DictReader(open(p))] if os.path.exists(p) else []
    old = [r for r in old if r['method'] != method]
    with open(p, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['face', 'method', 'n_atf', 'n_pred', 'skew_deg', 'pitch_px', 'body_h', 'seconds']); w.writeheader(); w.writerows(old + run)
    print(f'{method}: {len(run)} faces; median lines per face {np.median([r["n_pred"] for r in run]):.0f}')


if __name__ == '__main__':
    main()
