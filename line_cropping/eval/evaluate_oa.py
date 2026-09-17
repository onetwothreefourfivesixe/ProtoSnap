"""Score line predictions on the Old Assyrian faces against the hand labels (labels.csv).

Labels are rows in the face deskewed by `skew_deg` (the labelling frame). Predictions from the profile methods
share that frame when their skew matches; predictions from other frames (detector) are mapped by rotating the
line's centre point (face centre x, y) back to the face and then into the labelling frame. A predicted line
matches a label if within `tol` x label pitch (median label gap). Reports recall, precision, first-line error,
count match, by method, split and resolution.
  python -m line_cropping.eval.evaluate_oa
"""
import csv, os, collections, argparse
import numpy as np, cv2

def to_frame(y, W, H, skew_from, skew_to):
    """Map a line centre at (W/2, y) in the frame rotated by skew_from to the frame rotated by skew_to."""
    Mf = cv2.getRotationMatrix2D((W / 2, H / 2), skew_from, 1.0); Mt = cv2.getRotationMatrix2D((W / 2, H / 2), skew_to, 1.0)
    Mf3 = np.vstack([Mf, [0, 0, 1]]); p = np.linalg.inv(Mf3) @ np.array([W / 2, y, 1.0])      # back to the face
    q = Mt @ p; return float(q[1])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tol', type=float, default=0.35); ap.add_argument('--only-done', action='store_true', default=True)
    a = ap.parse_args()
    faces = {f['face']: f for f in csv.DictReader(open('fat-cross_processed/prepared/faces_accepted.csv'))}
    sizes = {}
    labels = collections.defaultdict(list); meta = {}
    for r in csv.DictReader(open('fat-cross_processed/ground_truth/labels.csv')):
        if a.only_done and r['status'] != 'done': continue
        labels[r['face_id']].append(float(r['y_face_deskewed'])); meta[r['face_id']] = r
    preds = {'dp': ('fat-cross_processed/predictions_profile', 'dp'), 'comb': ('fat-cross_processed/predictions_profile', 'comb'),
             'det_n': ('fat-cross_processed/predictions', 'det_n'), 'det': ('fat-cross_processed/predictions', 'det')}
    runs = {}
    for name, (d, m) in preds.items():
        for r in csv.DictReader(open(os.path.join(d, 'faces_run.csv'))):
            if r['method'] == m: runs[(name, r['face'])] = r
    rows = []
    for fid, ty in labels.items():
        ty = np.sort(ty); f = faces.get(fid)
        if f is None or len(ty) < 1: continue
        if fid not in sizes:
            im = cv2.imread(f['file'], cv2.IMREAD_GRAYSCALE); sizes[fid] = im.shape[::-1]
        W, H = sizes[fid]; skew_lab = float(meta[fid]['skew_deg'])
        pitch = float(np.median(np.diff(ty))) if len(ty) > 1 else (float(f['px_per_line']) if f['px_per_line'] else 40.0)
        for name, (d, m) in preds.items():
            r = runs.get((name, fid)); p = os.path.join(d, m, fid + '.csv')
            if r is None or not os.path.exists(p): continue
            py = np.array([float(x['y_deskewed']) for x in csv.DictReader(open(p))])
            if len(py) == 0: continue
            sk = float(r['skew_deg'])
            if abs(sk - skew_lab) > 0.01: py = np.array([to_frame(y, W, H, sk, skew_lab) for y in py])
            py = np.sort(py); tol = a.tol * pitch; used = set(); hit = 0
            for y in ty:
                dd = np.abs(py - y); j = int(np.argmin(dd))
                if dd[j] < tol and j not in used: used.add(j); hit += 1
            rows.append(dict(face=fid, method=name, split=meta[fid]['split'], n_truth=len(ty), n_pred=len(py), recall=hit / len(ty), precision=len(used) / len(py),
                             first_err=abs(py[0] - ty[0]) / pitch, count_ok=len(py) == len(ty), px_per_line=pitch))
    with open('fat-cross_processed/ground_truth/evaluation_oa.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    def report(sel, label):
        for m in ('dp', 'comb', 'det_n', 'det'):
            R = [r for r in sel if r['method'] == m]
            if not R: continue
            fe = np.array([r['first_err'] for r in R])
            print(f"  {label:22s} {m:6s} n={len(R):3d} recall {np.mean([r['recall'] for r in R]):.2f}  precision {np.mean([r['precision'] for r in R]):.2f}  first-line err median {np.median(fe):.2f} pitch, <0.5: {(fe<0.5).mean():.0%}  count match {np.mean([r['count_ok'] for r in R]):.0%}")
    print(f'labelled faces used: {len(labels)}; tolerance {a.tol} pitch')
    report(rows, 'all'); 
    for sp in ('tune', 'test'): report([r for r in rows if r['split'] == sp], f'split {sp}')
    for lo, hi, lab in ((0, 30, 'px/line < 30'), (30, 45, 'px/line 30-45'), (45, 999, 'px/line >= 45')): report([r for r in rows if lo <= r['px_per_line'] < hi], lab)

if __name__ == '__main__':
    main()
