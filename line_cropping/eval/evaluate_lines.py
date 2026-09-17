"""Phase 3 evaluation on eBL faces: predicted lines (find_lines.py) vs Phase 1 truth (ebl_tablets/faces/lines).

Truth segments are mapped into the same deskewed frame as the predictions (rotation about the face centre by
the run's skew angle). A predicted line matches a truth line if their centre heights differ by less than
`tol` x truth pitch. Reports recall, precision (only on faces with coverage >= 0.9, since unannotated lines
would otherwise count as false positives), first-line error in pitch units (only where the topmost truth line
is line 1 or 1'), and count agreement, split by script, method and resolution.

  python -m line_cropping.eval.evaluate_lines --faces ebl_tablets/faces/faces.csv --pred ebl_tablets/predictions
"""
import argparse, csv, os, collections
import numpy as np, cv2


def truth_in_frame(lines_csv, W, H, skew):
    M = cv2.getRotationMatrix2D((W / 2, H / 2), skew, 1.0); out = []
    for l in csv.DictReader(open(lines_csv)):
        mx = (float(l['ax0']) + float(l['ax1'])) / 2; my = (float(l['ay0']) + float(l['ay1'])) / 2
        y = M[1, 0] * mx + M[1, 1] * my + M[1, 2]
        out.append((y, l['line_label'], l['column'], l['sparse'] == 'True'))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--faces', required=True); ap.add_argument('--pred', required=True); ap.add_argument('--tol', type=float, default=0.35)
    a = ap.parse_args()
    faces = {f['face']: f for f in csv.DictReader(open(a.faces))}
    runs = list(csv.DictReader(open(os.path.join(a.pred, 'faces_run.csv'))))
    rows = []
    for r in runs:
        f = faces[r['face']]; W, H = int(f['width']), int(f['height'])
        truth = truth_in_frame(os.path.join(os.path.dirname(a.faces), 'lines', r['face'] + '.csv'), W, H, float(r['skew_deg']))
        if len(truth) < 2: continue
        ty = np.array([t[0] for t in truth]); tp = float(f['pitch_px']) if f['pitch_px'] else float(np.median(np.diff(ty)))
        py = np.array([float(x['y_deskewed']) for x in csv.DictReader(open(os.path.join(a.pred, r['method'], r['face'] + '.csv')))])
        if len(py) == 0: continue
        tol = a.tol * tp
        used = set(); hit = 0
        for y in ty:
            d = np.abs(py - y); j = int(np.argmin(d))
            if d[j] < tol and j not in used: used.add(j); hit += 1
        cov = float(f['coverage']) if f['coverage'] else np.nan
        first_ok = truth[0][1] in ("1", "1'")
        rows.append(dict(face=r['face'], script=f['script'], method=r['method'], n_truth=len(ty), n_pred=len(py), n_atf=r['n_atf'],
                         recall=hit / len(ty), precision=(len(used) / len(py)) if cov >= 0.9 else np.nan,
                         first_err=abs(py[0] - ty[0]) / tp if first_ok else np.nan, count_ok=(r['n_atf'] != '' and len(py) == int(r['n_atf'])),
                         px_per_line=tp, coverage=cov))
    with open(os.path.join(a.pred, 'evaluation.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    def report(sel, label):
        for m in sorted({r['method'] for r in sel}):
            R = [r for r in sel if r['method'] == m]
            rec = np.mean([r['recall'] for r in R]); prec = np.nanmean([r['precision'] for r in R]) if any(not np.isnan(r['precision']) for r in R) else np.nan
            fe = np.array([r['first_err'] for r in R if not np.isnan(r['first_err'])])
            print(f"  {label:28s} {m:5s} n={len(R):4d} recall {rec:.2f}  precision {prec:.2f}  first-line err median {np.median(fe) if len(fe) else np.nan:.2f} pitch, <0.5: {(fe<0.5).mean() if len(fe) else np.nan:.0%} (n={len(fe)})  count match {np.mean([r['count_ok'] for r in R]):.0%}")
    print('recall = share of truth lines matched within', a.tol, 'pitch; precision only on faces with >=0.9 coverage')
    report(rows, 'all faces')
    for sc in ('NA', 'OB'): report([r for r in rows if r['script'] == sc], f'script {sc}')
    for lo, hi, lab in ((0, 60, 'px/line < 60'), (60, 120, 'px/line 60-120'), (120, 9999, 'px/line >= 120')):
        report([r for r in rows if lo <= r['px_per_line'] < hi], lab)


if __name__ == '__main__':
    main()
