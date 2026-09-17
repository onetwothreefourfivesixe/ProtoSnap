"""Per-face confidence signals for line predictions WITHOUT ground truth (Old Assyrian set).

For every accepted face with predictions from both methods:
  agreement   share of profile (dp) lines that have a detector-route (det_n) line within 0.35 pitch, both mapped
              into the dp frame (methods fail differently, so agreement is evidence of correctness)
  proximity   share of detector boxes (score >= 0.3) whose centre lies within 0.3 pitch of a predicted line centre
              (random placement would give about 0.6; midpoint crops tile the face, so plain containment is
              always 1 and is not used)
  count_diff  predicted lines minus ATF lines, reported for information; it is 0 by construction when the
              predictions enforce the ATF count (dp, det_n) and is then left out of the combined score
  spacing_cv  coefficient of variation of the gaps between predicted lines (regular spacing expected)
A combined score (mean of agreement, proximity, and 1 - min(1, spacing_cv)) ranks faces from
least to most confident. Writes fat-cross_processed/confidence.csv.
  python -m line_cropping.eval.confidence_oa [--crops fat-cross_processed/line_crops/crops.csv]
"""
import argparse, csv, os, collections
import numpy as np, cv2
from line_cropping.eval.evaluate_oa import to_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--faces', default='fat-cross_processed/prepared/faces_accepted.csv'); ap.add_argument('--profile', default='fat-cross_processed/predictions_profile')
    ap.add_argument('--det-pred', default='fat-cross_processed/predictions'); ap.add_argument('--det', default='fat-cross_processed/detections')
    ap.add_argument('--crops', default='fat-cross_processed/line_crops/crops.csv'); ap.add_argument('--out', default='fat-cross_processed/confidence.csv'); ap.add_argument('--tol', type=float, default=0.35)
    a = ap.parse_args()
    faces = {f['face']: f for f in csv.DictReader(open(a.faces))}
    run_dp = {r['face']: r for r in csv.DictReader(open(os.path.join(a.profile, 'faces_run.csv'))) if r['method'] == 'dp'}
    run_det = {r['face']: r for r in csv.DictReader(open(os.path.join(a.det_pred, 'faces_run.csv'))) if r['method'] == 'det_n'}
    crops = collections.defaultdict(list)
    if os.path.exists(a.crops):
        for r in csv.DictReader(open(a.crops)): crops[r['face']].append((float(r['y0']), float(r['y1'])))
    rows = []
    for fid, rd in run_dp.items():
        f = faces.get(fid); pdp = os.path.join(a.profile, 'dp', fid + '.csv')
        if f is None or not os.path.exists(pdp): continue
        dp = np.array([float(x['y_deskewed']) for x in csv.DictReader(open(pdp))])
        if len(dp) < 2: continue
        im = cv2.imread(f['file'], cv2.IMREAD_GRAYSCALE); H, W = im.shape; skew = float(rd['skew_deg'])
        pitch = float(np.median(np.diff(dp))); gaps = np.diff(dp); spacing_cv = float(np.std(gaps) / np.mean(gaps)) if np.mean(gaps) > 0 else 1.0
        # agreement with det_n
        agreement = np.nan; rdet = run_det.get(fid); pdet = os.path.join(a.det_pred, 'det_n', fid + '.csv')
        if rdet and os.path.exists(pdet):
            det = np.array([float(x['y_deskewed']) for x in csv.DictReader(open(pdet))])
            if len(det):
                sk = float(rdet['skew_deg'])
                if abs(sk - skew) > 0.01: det = np.array([to_frame(y, W, H, sk, skew) for y in det])
                agreement = float(np.mean([np.min(np.abs(det - y)) < a.tol * pitch for y in dp]))
        # detector-box proximity to the predicted line centres
        proximity = np.nan; n_boxes = 0; pd = os.path.join(a.det, fid + '.csv')
        if os.path.exists(pd):
            M = cv2.getRotationMatrix2D((W / 2, H / 2), skew, 1.0); near = 0
            for b in csv.DictReader(open(pd, encoding='utf-8')):
                if float(b['score']) < 0.3: continue
                cx, cy = (float(b['x0']) + float(b['x1'])) / 2, (float(b['y0']) + float(b['y1'])) / 2
                yr = M[1, 0] * cx + M[1, 1] * cy + M[1, 2]; n_boxes += 1
                near += np.min(np.abs(dp - yr)) < 0.3 * pitch
            proximity = near / n_boxes if n_boxes >= 5 else np.nan
        n_atf = int(f['atf_lines']) if f.get('atf_lines') else None; count_diff = (len(dp) - n_atf) if n_atf is not None else np.nan
        parts = [v for v in (agreement, proximity, 1 - min(1.0, spacing_cv)) if v == v]
        rows.append(dict(face=fid, side=f['side'], n_lines=len(dp), atf_lines=n_atf if n_atf is not None else '', count_diff=count_diff if count_diff == count_diff else '',
                         agreement=round(agreement, 3) if agreement == agreement else '', proximity=round(proximity, 3) if proximity == proximity else '', n_boxes=n_boxes,
                         spacing_cv=round(spacing_cv, 3), confidence=round(float(np.mean(parts)), 3) if parts else ''))
    rows.sort(key=lambda r: (r['confidence'] == '', r['confidence']))
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    ag = np.array([r['agreement'] for r in rows if r['agreement'] != ''], float); ct = np.array([r['proximity'] for r in rows if r['proximity'] != ''], float); cf = np.array([r['confidence'] for r in rows if r['confidence'] != ''], float)
    print(f'faces scored: {len(rows)}')
    print(f'agreement dp vs det_n: median {np.median(ag):.2f}; faces with >= 0.8 agreement {np.mean(ag >= 0.8):.0%}, <= 0.5 {np.mean(ag <= 0.5):.0%}')
    print(f'detector boxes within 0.3 pitch of a predicted line centre: median {np.median(ct):.2f} (n={len(ct)} faces with >= 5 boxes)')
    cd = np.array([r['count_diff'] for r in rows if r['count_diff'] != ''], float); print(f'count vs ATF: exact {np.mean(cd == 0):.0%} (0 by construction for count-enforced predictions; not used in the score)')
    print(f'confidence: median {np.median(cf):.2f}; lowest decile below {np.percentile(cf, 10):.2f}; least confident faces: {[r["face"] for r in rows[:6]]}')
    print(f'-> {a.out}')


if __name__ == '__main__':
    main()
