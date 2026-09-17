"""Phase 5: formal evaluation of the detector-route line finder on the eBL set.

1. Stratified tune/test split of the single-column faces (by script and pixels-per-line bucket), seeded;
   written to ebl_tablets/split.csv.
2. Grid search of the clustering thresholds (gap_frac, merge_frac, score_thr) on the TUNE split only.
3. Test-split report for det (no count) and det_n (ATF count) against the plan's criteria:
   recall >= 0.9, first line within 0.5 pitch on >= 90% of faces, count agreement >= 80%.
4. Failure categories among test faces with recall < 0.7.
  python -m line_cropping.eval.phase5_eval
"""
import csv, os, random, collections, itertools, json
import numpy as np
from line_cropping.eval.evaluate_lines import truth_in_frame
from line_cropping.lines.lines_from_boxes import lines_for_face

FACES = 'ebl_tablets/faces/faces.csv'; DET = 'ebl_tablets/detections'; TRUTH = 'ebl_tablets/faces/lines'


def load():
    faces = [f for f in csv.DictReader(open(FACES)) if f['n_columns'] == '1' and os.path.exists(os.path.join(DET, f['face'] + '.csv'))]
    out = []
    for f in faces:
        rows = list(csv.DictReader(open(os.path.join(DET, f['face'] + '.csv'), encoding='utf-8')))
        boxes = np.array([[float(r['x0']), float(r['y0']), float(r['x1']), float(r['y1']), float(r['score'])] for r in rows]) if rows else np.zeros((0, 5))
        out.append((f, boxes))
    return out


def bucket(f):
    p = float(f['pitch_px']) if f['pitch_px'] else 0
    return f['script'] + ('/lo' if p < 60 else '/mid' if p < 120 else '/hi')


def make_split(items, seed=7):
    random.seed(seed); by = collections.defaultdict(list)
    for f, _ in items: by[bucket(f)].append(f['face'])
    split = {}
    for k, ids in by.items():
        random.shuffle(ids)
        for i, fid in enumerate(ids): split[fid] = 'tune' if i % 2 == 0 else 'test'
    with open('ebl_tablets/split.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['face', 'split', 'bucket'])
        for f, _ in items: w.writerow([f['face'], split[f['face']], bucket(f)])
    return split


def score_face(f, boxes, n, gap, merge, thr, tol=0.35):
    W, H = int(f['width']), int(f['height'])
    res = lines_for_face(boxes, W, H, n, thr, gap_frac=gap, merge_frac=merge)
    if res is None: return None
    truth = truth_in_frame(os.path.join(TRUTH, f['face'] + '.csv'), W, H, res['skew_deg'])
    if len(truth) < 2: return None
    ty = np.array([t[0] for t in truth]); tp = float(f['pitch_px']) if f['pitch_px'] else float(np.median(np.diff(ty)))
    py = np.array([l['y'] for l in res['lines']]); used = set(); hit = 0
    for y in ty:
        d = np.abs(py - y); j = int(np.argmin(d))
        if d[j] < tol * tp and j not in used: used.add(j); hit += 1
    cov = float(f['coverage']) if f['coverage'] else np.nan
    return dict(recall=hit / len(ty), precision=(len(used) / len(py)) if cov >= 0.9 else np.nan,
                first_err=abs(py[0] - ty[0]) / tp if truth[0][1] in ("1", "1'") else np.nan,
                count_ok=(f['atf_lines_side'] != '' and len(py) == int(float(f['atf_lines_side']))), n_pred=len(py), n_truth=len(ty), n_boxes=len(boxes), cov=cov, ppl=tp)


def summarise(R):
    R = [r for r in R if r]
    fe = np.array([r['first_err'] for r in R if not np.isnan(r['first_err'])]); pr = [r['precision'] for r in R if not np.isnan(r['precision'])]
    return dict(n=len(R), recall=np.mean([r['recall'] for r in R]), precision=np.mean(pr) if pr else np.nan, first_lt_half=(fe < 0.5).mean() if len(fe) else np.nan,
                count=np.mean([r['count_ok'] for r in R]))


def main():
    items = load(); split = make_split(items)
    tune = [(f, b) for f, b in items if split[f['face']] == 'tune']; test = [(f, b) for f, b in items if split[f['face']] == 'test']
    print(f'single-column faces with detections: {len(items)}; tune {len(tune)}, test {len(test)}')
    n_of = lambda f: int(float(f['atf_lines_side'])) if f['atf_lines_side'] else None
    print('--- grid search on TUNE (det_n, objective = recall + share of faces with first line < 0.5 pitch) ---')
    best = None
    for gap, merge, thr in itertools.product((0.5, 0.6, 0.7, 0.8), (0.35, 0.45, 0.55), (0.3, 0.5)):
        S = summarise([score_face(f, b, n_of(f), gap, merge, thr) for f, b in tune]); obj = S['recall'] + S['first_lt_half']
        print(f"   gap {gap} merge {merge} thr {thr}: recall {S['recall']:.3f} first<0.5 {S['first_lt_half']:.2f} count {S['count']:.2f}")
        if best is None or obj > best[0]: best = (obj, gap, merge, thr)
    _, gap, merge, thr = best; print(f'chosen: gap_frac {gap}, merge_frac {merge}, score_thr {thr}')
    json.dump(dict(gap_frac=gap, merge_frac=merge, score_thr=thr), open('ebl_tablets/det_params.json', 'w'))
    print('--- TEST split ---')
    results = {}
    for name, use_n in (('det_n', True), ('det', False)):
        R = [(f, score_face(f, b, n_of(f) if use_n else None, gap, merge, thr)) for f, b in test]
        results[name] = R; S = summarise([r for _, r in R])
        print(f"  {name:6s} all      n={S['n']:3d} recall {S['recall']:.2f} precision {S['precision']:.2f} first<0.5 {S['first_lt_half']:.0%} count match {S['count']:.0%}")
        for grp in sorted({bucket(f) for f, _ in R}):
            S = summarise([r for f, r in R if bucket(f) == grp]); print(f"  {name:6s} {grp:8s} n={S['n']:3d} recall {S['recall']:.2f} precision {S['precision']:.2f} first<0.5 {S['first_lt_half']:.0%} count match {S['count']:.0%}")
    print('--- plan criteria on TEST, det_n, faces with >= 120 px/line ---')
    hi = [r for f, r in results['det_n'] if r and r['ppl'] >= 120]; S = summarise(hi)
    print(f"  recall {S['recall']:.2f} (target >= 0.90) | first line < 0.5 pitch on {S['first_lt_half']:.0%} of faces (target >= 90%) | count agreement {S['count']:.0%} (target >= 80%)")
    print('--- failure categories, TEST det_n, recall < 0.7 ---')
    bad = [(f, r) for f, r in results['det_n'] if r and r['recall'] < 0.7]; cats = collections.Counter()
    for f, r in bad:
        truth = truth_in_frame(os.path.join(TRUTH, f['face'] + '.csv'), int(f['width']), int(f['height']), 0.0); ty = np.sort([t[0] for t in truth])
        interleaved = len(ty) > 2 and np.median(np.diff(ty)) < 0.5 * r['ppl']
        if interleaved: cats['undeclared second column (truth lines interleave)'] += 1
        elif r['n_boxes'] < 8: cats['too few detections (< 8 boxes)'] += 1
        elif r['cov'] < 0.5: cats['sparse truth (< 50% coverage)'] += 1
        elif r['ppl'] < 60: cats['low resolution (< 60 px/line)'] += 1
        elif abs(r['n_pred'] - r['n_truth']) >= 3: cats['count far off (>= 3 lines)'] += 1
        else: cats['other (placement)'] += 1
    print(f'  faces with recall < 0.7: {len(bad)} of {len(results["det_n"])}')
    for k, v in cats.most_common(): print(f'   {v:3d}  {k}')
    with open('ebl_tablets/phase5_test_results.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['face', 'method', 'bucket', 'recall', 'precision', 'first_err', 'count_ok', 'n_pred', 'n_truth', 'n_boxes', 'coverage', 'px_per_line'])
        for name, R in results.items():
            for f, r in R:
                if r: w.writerow([f['face'], name, bucket(f), round(r['recall'], 3), '' if np.isnan(r['precision']) else round(r['precision'], 3), '' if np.isnan(r['first_err']) else round(r['first_err'], 3), r['count_ok'], r['n_pred'], r['n_truth'], r['n_boxes'], r['cov'], round(r['ppl'], 1)])


if __name__ == '__main__':
    main()
