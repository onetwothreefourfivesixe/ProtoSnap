"""Aggregate every metric table produced for the two data sets and compare them side by side.

Reads only the result tables (no images), so it runs in seconds:
  eBL  : ebl_tablets/{manifest,split,phase5_test_results}.csv, faces/faces.csv, ground_truth/{tablets,blocks}.csv,
         detections/run.csv, predictions/{evaluation,faces_run}.csv, line_crops/crops.csv (+ crop metrics recomputed
         against the line truth), output/phase6_*/summary.csv, protosnap_inputs/*/alignment.csv
  CDLI : fat-cross_processed/{inventory,confidence}.csv, prepared/faces_accepted.csv, experiments/face_qc.csv,
         detections/run.csv, predictions*/faces_run.csv, ground_truth/{labels,evaluation_oa}.csv, line_crops/crops.csv

Writes the individual results next to their data:
  ebl_tablets/metrics/metrics.csv|json and fat-cross_processed/metrics/metrics.csv|json   (metric, value, source)
and the comparison in the docs:
  docs/metrics_comparison.md
  python -m line_cropping.report.aggregate_metrics
"""
import csv, os, json, glob, collections, datetime
import numpy as np

R = lambda p: list(csv.DictReader(open(p, encoding='utf-8'))) if os.path.exists(p) else []
F = lambda rows, k: np.array([float(r[k]) for r in rows if r.get(k) not in (None, '', 'nan')], float)


def med(x): return float(np.median(x)) if len(x) else float('nan')
def share(rows, pred): return float(np.mean([pred(r) for r in rows])) if rows else float('nan')


class Metrics:
    def __init__(self): self.rows = []
    def add(self, section, name, value, source, note=''):
        self.rows.append(dict(section=section, metric=name, value=value, source=source, note=note))
    def get(self, name, default=float('nan')):
        for r in self.rows:
            if r['metric'] == name: return r['value']
        return default


def line_eval_table(rows, method_key='method'):
    """Per-method aggregates from an evaluation table with recall/precision/first_err/count_ok columns."""
    out = {}
    for m in sorted({r[method_key] for r in rows}):
        R_ = [r for r in rows if r[method_key] == m]; pr = F(R_, 'precision'); fe = F(R_, 'first_err')
        out[m] = dict(n=len(R_), recall=float(np.mean(F(R_, 'recall'))), precision=float(np.mean(pr)) if len(pr) else float('nan'),
                      first_lt_half=float(np.mean(fe < 0.5)) if len(fe) else float('nan'), count_match=share(R_, lambda r: r['count_ok'] == 'True'))
    return out


# ------------------------------------------------------------------ eBL
def ebl_metrics():
    M = Metrics(); S = 'ebl_tablets'
    man = R(f'{S}/manifest.csv'); tabs = R(f'{S}/ground_truth/tablets.csv'); blocks = R(f'{S}/ground_truth/blocks.csv'); faces = R(f'{S}/faces/faces.csv')
    M.add('data', 'tablets', len(man), 'manifest.csv'); M.add('data', 'tablets_by_script', dict(collections.Counter(r['script'] for r in man)), 'manifest.csv')
    M.add('data', 'tablets_with_transliteration', sum(1 for r in man if r['atf_lines'] not in ('', '0')), 'manifest.csv', 'eBL ATF')
    M.add('data', 'faces', len(faces), 'faces/faces.csv'); M.add('data', 'faces_by_side', dict(collections.Counter(r['side'] for r in faces)), 'faces/faces.csv')
    ppl = F(faces, 'pitch_px'); M.add('data', 'px_per_line_median', med(ppl), 'faces/faces.csv'); M.add('data', 'faces_under_60_px_per_line', float(np.mean(ppl < 60)), 'faces/faces.csv')
    M.add('data', 'faces_multi_column', sum(1 for f in faces if f['n_columns'] not in ('', '1')), 'faces/faces.csv')
    M.add('truth', 'truth_type', 'eBL sign boxes grouped by line index', 'ground_truth/')
    M.add('truth', 'truth_faces', len({(b['tablet'], b['side'], b['object']) for b in blocks}), 'ground_truth/blocks.csv')
    M.add('truth', 'truth_lines', sum(int(t['n_lines']) for t in tabs), 'ground_truth/tablets.csv'); M.add('truth', 'truth_signs', sum(int(t['n_signs']) for t in tabs), 'ground_truth/tablets.csv')
    cov = F(faces, 'coverage'); M.add('truth', 'coverage_median', med(cov), 'faces/faces.csv', 'annotated lines / transliterated lines per face'); M.add('truth', 'faces_coverage_ge_0.9', float(np.mean(cov >= 0.9)), 'faces/faces.csv')
    det = R(f'{S}/detections/run.csv'); nb = F(det, 'n_boxes'); M.add('detection', 'faces_detected', len(det), 'detections/run.csv'); M.add('detection', 'boxes_per_face_median', med(nb), 'detections/run.csv'); M.add('detection', 'faces_under_5_boxes', float(np.mean(nb < 5)) if len(nb) else float('nan'), 'detections/run.csv')
    # count agreement of the detector alone with the ATF
    runs = R(f'{S}/predictions/faces_run.csv'); fa = {f['face']: f for f in faces}
    d = [int(r['n_pred']) - int(float(fa[r['face']]['atf_lines_side'])) for r in runs if r['method'] == 'det' and r['face'] in fa and fa[r['face']]['atf_lines_side'] not in ('', '0')]
    M.add('detection', 'det_count_within_1_of_atf', float(np.mean(np.abs(d) <= 1)) if d else float('nan'), 'predictions/faces_run.csv', 'detector-only line count vs transliteration count')
    ev = R(f'{S}/predictions/evaluation.csv'); M.add('lines', 'eval_all_faces', line_eval_table(ev), 'predictions/evaluation.csv', 'all scored single-column faces, thresholds tuned on the tune split')
    p5 = R(f'{S}/phase5_test_results.csv'); M.add('lines', 'eval_test_split', line_eval_table(p5), 'phase5_test_results.csv', 'held-out test half')
    hi = [r for r in p5 if r['method'] == 'det_n' and float(r['px_per_line']) >= 120]; t = line_eval_table(hi).get('det_n', {}); M.add('lines', 'plan_criteria_det_n_ge120', t, 'phase5_test_results.csv', 'targets: recall 0.90, first line 90%, count 80%')
    M.add('lines', 'chosen_method', 'det_n (detector boxes + ATF count)', 'det_params.json')
    if os.path.exists(f'{S}/det_params.json'): M.add('lines', 'thresholds', json.load(open(f'{S}/det_params.json')), 'det_params.json')
    crops = R(f'{S}/line_crops/crops.csv')
    if crops:
        h = np.array([int(r['y1']) - int(r['y0']) for r in crops], float); w = np.array([int(r['x1']) - int(r['x0']) for r in crops], float); pp = F(crops, 'pitch_px')
        M.add('crops', 'crop_mode', 'gap-bounded from detector boxes', 'line_crops/crops.csv'); M.add('crops', 'crops', len(crops), 'line_crops/crops.csv'); M.add('crops', 'crop_faces', len({r['face'] for r in crops}), 'line_crops/crops.csv')
        M.add('crops', 'crop_median_w_h_px', [med(w), med(h)], 'line_crops/crops.csv'); M.add('crops', 'crop_height_over_pitch_median', med(h / pp), 'line_crops/crops.csv'); M.add('crops', 'crops_with_atf_label', share(crops, lambda r: r['atf_aligned'] == 'True'), 'line_crops/crops.csv')
        try:
            from line_cropping.eval.evaluate_crops import compute
            cm = compute(f'{S}/faces/faces.csv', f'{S}/line_crops/crops.csv')
            for k in ('centre_in_one', 'centre_in_none', 'centre_in_two', 'crops_with_two_lines', 'band_fully_inside'): M.add('crops', k, cm[k], 'evaluate_crops.compute', 'against eBL line truth')
        except Exception as e: M.add('crops', 'crop_truth_metrics', f'unavailable: {e}', '')
    for run in sorted(glob.glob('output/phase6_*')):
        sm = R(os.path.join(run, 'summary.csv'))
        if sm: sc = F(sm, 'init_score'); M.add('protosnap', os.path.basename(run), dict(targets=len(sm), finished=sum(r['finished'] == 'True' for r in sm), init_score_median=med(sc)), f'{run}/summary.csv', 'curated single-sign test set: 0.75')
    for ali in sorted(glob.glob('protosnap_inputs/*/alignment.csv')):
        A = R(ali); mt = [r for r in A if r['matched'] == 'True']; iou = F(mt, 'truth_iou')
        M.add('protosnap', 'alignment_' + ali.split('/')[1], dict(lines=len({(r['face'], r['line_no']) for r in A}), atf_signs=len(A), matched=share(A, lambda r: r['matched'] == 'True'), det_agrees=share([r for r in mt if r['sign']], lambda r: r['det_agrees'] == 'True'), truth_iou_gt_0_5=float(np.mean(iou > 0.5)) if len(iou) else float('nan')), ali)
    return M


# ------------------------------------------------------------------ CDLI (Old Assyrian)
def cdli_metrics():
    M = Metrics(); S = 'fat-cross_processed'
    inv = R(f'{S}/inventory.csv'); acc = R(f'{S}/prepared/faces_accepted.csv'); qc = R(f'{S}/experiments/face_qc.csv')
    M.add('data', 'tablets', len(inv), 'inventory.csv'); M.add('data', 'tablets_by_script', {'OA': len(inv)}, 'inventory.csv'); M.add('data', 'tablets_with_transliteration', sum(1 for r in inv if r['has_atf'] == 'True'), 'inventory.csv', 'CDLI ATF')
    M.add('data', 'faces', len(acc), 'prepared/faces_accepted.csv', 'accepted by face_qc'); M.add('data', 'faces_by_side', dict(collections.Counter(r['side'] for r in acc)), 'prepared/faces_accepted.csv')
    ppl = F([q for q in qc if not q['flags']], 'px_per_line'); M.add('data', 'px_per_line_median', med(ppl), 'experiments/face_qc.csv'); M.add('data', 'faces_under_60_px_per_line', float(np.mean(ppl < 60)), 'experiments/face_qc.csv')
    M.add('data', 'faces_rejected_by_qc', sum(1 for q in qc if q['flags']), 'experiments/face_qc.csv'); M.add('data', 'faces_multi_column', 'not measured (ATF: 4 tablets use @column)', '')
    lab = R(f'{S}/ground_truth/labels.csv'); done = [r for r in lab if r['status'] == 'done']
    M.add('truth', 'truth_type', 'hand-marked line centres (Line Marker page)', 'ground_truth/labels.csv'); M.add('truth', 'truth_faces', len({r['face_id'] for r in done}), 'ground_truth/labels.csv'); M.add('truth', 'truth_lines', len(done), 'ground_truth/labels.csv')
    M.add('truth', 'truth_lines_hand_placed', share(done, lambda r: r['src'] == 'manual'), 'ground_truth/labels.csv', 'rest left at the profile-fit start position')
    fa = {f['face']: f for f in acc}
    diff = []
    for f in {r['face_id'] for r in done}:
        n = sum(1 for r in done if r['face_id'] == f); a = fa.get(f, {}).get('atf_lines')
        if a: diff.append(n - int(a))
    M.add('truth', 'labelled_faces_with_fewer_lines_than_atf', float(np.mean(np.array(diff) < 0)) if diff else float('nan'), 'ground_truth/labels.csv', 'ATF count is an upper bound here')
    det = R(f'{S}/detections/run.csv'); nb = F(det, 'n_boxes'); M.add('detection', 'faces_detected', len(det), 'detections/run.csv'); M.add('detection', 'boxes_per_face_median', med(nb), 'detections/run.csv'); M.add('detection', 'faces_under_5_boxes', float(np.mean(nb < 5)) if len(nb) else float('nan'), 'detections/run.csv')
    runs = R(f'{S}/predictions/faces_run.csv'); d = [int(r['n_pred']) - int(fa[r['face']]['atf_lines']) for r in runs if r['method'] == 'det' and r['face'] in fa and fa[r['face']]['atf_lines']]
    M.add('detection', 'det_count_within_1_of_atf', float(np.mean(np.abs(d) <= 1)) if d else float('nan'), 'predictions/faces_run.csv', 'detector-only line count vs transliteration count')
    ev = R(f'{S}/ground_truth/evaluation_oa.csv'); M.add('lines', 'eval_all_faces', line_eval_table(ev), 'ground_truth/evaluation_oa.csv', 'all hand-labelled faces')
    M.add('lines', 'eval_test_split', line_eval_table([r for r in ev if r['split'] == 'test']), 'ground_truth/evaluation_oa.csv', 'test half of the labelled sample')
    M.add('lines', 'chosen_method', 'dp (projection-profile dynamic programme + ATF count)', 'README'); M.add('lines', 'thresholds', 'none tuned; ATF count enforced', '')
    crops = R(f'{S}/line_crops/crops.csv')
    if crops:
        h = np.array([int(r['y1']) - int(r['y0']) for r in crops], float); w = np.array([int(r['x1']) - int(r['x0']) for r in crops], float); pp = F(crops, 'pitch_px')
        M.add('crops', 'crop_mode', 'midpoint between predicted line centres (dp)', 'line_crops/crops.csv'); M.add('crops', 'crops', len(crops), 'line_crops/crops.csv'); M.add('crops', 'crop_faces', len({r['face'] for r in crops}), 'line_crops/crops.csv')
        M.add('crops', 'crop_median_w_h_px', [med(w), med(h)], 'line_crops/crops.csv'); M.add('crops', 'crop_height_over_pitch_median', med(h / pp), 'line_crops/crops.csv'); M.add('crops', 'crops_with_atf_label', share(crops, lambda r: r['atf_aligned'] == 'True'), 'line_crops/crops.csv')
        M.add('crops', 'crop_truth_metrics', 'no line truth for the full set; see confidence signals', '')
    conf = R(f'{S}/confidence.csv')
    if conf:
        M.add('confidence', 'faces_scored', len(conf), 'confidence.csv'); M.add('confidence', 'agreement_dp_vs_det_n_median', med(F(conf, 'agreement')), 'confidence.csv', 'share of dp lines with a det_n line within 0.35 pitch')
        M.add('confidence', 'box_proximity_median', med(F(conf, 'proximity')), 'confidence.csv', 'detector boxes within 0.3 pitch of a predicted line'); M.add('confidence', 'confidence_median', med(F(conf, 'confidence')), 'confidence.csv'); M.add('confidence', 'confidence_10th_pct', float(np.percentile(F(conf, 'confidence'), 10)), 'confidence.csv')
    M.add('protosnap', 'runs', 'none (no Old Assyrian font; Santakku would be a stand-in)', '')
    return M


# ------------------------------------------------------------------ outputs
def write_individual(M, folder):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'metrics.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh); w.writerow(['section', 'metric', 'value', 'source', 'note'])
        for r in M.rows: w.writerow([r['section'], r['metric'], json.dumps(r['value']) if isinstance(r['value'], (dict, list)) else r['value'], r['source'], r['note']])
    json.dump(M.rows, open(os.path.join(folder, 'metrics.json'), 'w', encoding='utf-8'), indent=1, default=float)


def fmt(v, kind='num'):
    if isinstance(v, float) and np.isnan(v): return 'n/a'
    if kind == 'pct': return f'{100*v:.0f}%' if isinstance(v, (int, float)) else str(v)
    if kind == 'int': return f'{int(v):,}' if isinstance(v, (int, float)) else str(v)
    if isinstance(v, float): return f'{v:,.0f}' if abs(v - round(v)) < 1e-9 and abs(v) >= 10 else f'{v:.2f}'
    if isinstance(v, list): return ' x '.join(fmt(x, 'int') for x in v)
    if isinstance(v, dict): return ', '.join(f'{k} {v2}' for k, v2 in v.items())
    return str(v)


def write_comparison(E, C, path='docs/metrics_comparison.md'):
    L = ['# Metrics: eBL set versus CDLI Old Assyrian set', '',
         f'Generated by `python -m line_cropping.report.aggregate_metrics` on {datetime.date.today().isoformat()} from the result tables under `ebl_tablets/` and `fat-cross_processed/`. The per-dataset tables with every value and its source file are `ebl_tablets/metrics/metrics.csv` and `fat-cross_processed/metrics/metrics.csv` (not tracked in git). Definitions: a line is matched within 0.35 pitch of a truth centre; precision only on faces with coverage >= 0.9; first-line accuracy is the share of faces whose first line lies within 0.5 pitch of the truth; count match is predicted lines equal to the transliteration count.', '',
         '**Read the comparison with two caveats.** The two sets are scored against different truths: eBL lines come from the eBL sign annotations on all faces, the CDLI lines from 88 hand-labelled faces whose starting positions were the profile fit itself. And the crops are cut differently: from detector-box gaps on eBL, from midpoints between predicted centres on CDLI. Where a number is not comparable it is marked.', '']
    def row(label, key, kind='num', getter=None):
        g = getter or (lambda M: M.get(key)); return f'| {label} | {fmt(g(E), kind)} | {fmt(g(C), kind)} |'
    L += ['## 1. Data and ground truth', '', '| | eBL (NA + OB) | CDLI (Old Assyrian) |', '|---|---|---|',
          row('tablets', 'tablets', 'int'), row('by script', 'tablets_by_script'), row('tablets with a transliteration', 'tablets_with_transliteration', 'int'),
          row('faces in the pipeline', 'faces', 'int'), row('faces by side', 'faces_by_side'), row('median pixels per line', 'px_per_line_median'), row('faces under 60 px per line', 'faces_under_60_px_per_line', 'pct'),
          row('truth', 'truth_type'), row('faces with truth', 'truth_faces', 'int'), row('truth lines', 'truth_lines', 'int'),
          row('annotation coverage per face, median (eBL only)', 'coverage_median'), row('share of truth lines placed by hand (CDLI only)', 'truth_lines_hand_placed', 'pct'),
          row('labelled faces with fewer lines than the ATF', 'labelled_faces_with_fewer_lines_than_atf', 'pct'), '']
    L += ['## 2. Sign detection (eBL Deformable DETR)', '', '| | eBL | CDLI |', '|---|---|---|', row('faces run', 'faces_detected', 'int'), row('boxes per face, median', 'boxes_per_face_median'), row('faces with fewer than 5 boxes', 'faces_under_5_boxes', 'pct'), row('detector-only line count within 1 of the ATF', 'det_count_within_1_of_atf', 'pct'), '']
    L += ['## 3. Line finding by method', '', 'eBL: all scored single-column faces against the eBL line truth. CDLI: the 88 hand-labelled faces. `det_n` = detector boxes clustered, count forced to the ATF; `det` = detector boxes clustered, free count; `dp` = projection-profile dynamic programme with the ATF count; `comb` = rigid comb with the ATF count; `free` = profile peaks, no prior.', '',
          '| method | set | faces | recall | precision | first line < 0.5 pitch | count match |', '|---|---|---|---|---|---|---|']
    for m in ('det_n', 'det', 'dp', 'comb', 'free'):
        for label, M in (('eBL', E), ('CDLI', C)):
            t = M.get('eval_all_faces', {}).get(m)
            if t: L.append(f"| {m} | {label} | {t['n']} | {t['recall']:.2f} | {fmt(t['precision'])} | {fmt(t['first_lt_half'], 'pct')} | {fmt(t['count_match'], 'pct')} |")
    L += ['', f"Chosen method: eBL `{E.get('chosen_method')}`; CDLI `{C.get('chosen_method')}`. The ranking reverses between the sets: the detector finds {fmt(E.get('boxes_per_face_median'))} boxes per face on eBL photographs and {fmt(C.get('boxes_per_face_median'))} on the Old Assyrian ones, so its line grouping starves there while the profile method, which plateaus at {E.get('eval_all_faces', {}).get('dp', {}).get('recall', float('nan')):.2f} recall on eBL, reaches {C.get('eval_all_faces', {}).get('dp', {}).get('recall', float('nan')):.2f} on the labelled Old Assyrian faces.", '']
    t5 = E.get('eval_test_split', {}); tc = C.get('eval_test_split', {})
    L += ['### Held-out halves', '', '| method | set | faces | recall | precision | first line < 0.5 pitch | count match |', '|---|---|---|---|---|---|---|']
    for m in ('det_n', 'det', 'dp', 'comb'):
        for label, T in (('eBL test split', t5), ('CDLI test half of labels', tc)):
            t = T.get(m)
            if t: L.append(f"| {m} | {label} | {t['n']} | {t['recall']:.2f} | {fmt(t['precision'])} | {fmt(t['first_lt_half'], 'pct')} | {fmt(t['count_match'], 'pct')} |")
    pc = E.get('plan_criteria_det_n_ge120', {})
    if pc: L += ['', f"Plan criteria on eBL faces with >= 120 px per line (test split, det_n, n={pc.get('n')}): recall {pc.get('recall', float('nan')):.2f} (target 0.90), first line within 0.5 pitch {fmt(pc.get('first_lt_half'), 'pct')} (target 90%), count match {fmt(pc.get('count_match'), 'pct')} (target 80%). No equivalent target exists for the CDLI set, whose truth sample is too small to gate on.", '']
    L += ['## 4. Line crops', '', '| | eBL | CDLI |', '|---|---|---|', row('crop mode', 'crop_mode'), row('crops', 'crops', 'int'), row('faces with crops', 'crop_faces', 'int'), row('median crop width x height, px', 'crop_median_w_h_px'), row('crop height / pitch, median', 'crop_height_over_pitch_median'), row('crops carrying a transliteration line label', 'crops_with_atf_label', 'pct'),
          row('truth line centres inside exactly one crop', 'centre_in_one', 'pct', lambda M: M.get('centre_in_one')), row('truth line centres inside two crops', 'centre_in_two', 'pct'), row('crops holding the centres of two lines', 'crops_with_two_lines', 'pct'), '',
          'CDLI crop purity cannot be measured against truth for the full set; the no-truth signals stand in: median agreement between the two line finders ' + fmt(C.get('agreement_dp_vs_det_n_median')) + ', median share of detector boxes within 0.3 pitch of a predicted line ' + fmt(C.get('box_proximity_median')) + ', median confidence ' + fmt(C.get('confidence_median')) + ' with the lowest decile below ' + fmt(C.get('confidence_10th_pct')) + '.', '']
    L += ['## 5. ProtoSnap', '', '| run | set | targets | finished | initialisation score, median |', '|---|---|---|---|---|']
    for r in E.rows:
        if r['section'] == 'protosnap' and isinstance(r['value'], dict) and 'targets' in r['value']: v = r['value']; L.append(f"| {r['metric']} | eBL OB | {v['targets']} | {v['finished']} | {v['init_score_median']:.2f} |")
    L.append(f"| — | CDLI | {C.get('runs')} | | |"); L.append('')
    for r in E.rows:
        if r['metric'].startswith('alignment_'): v = r['value']; L.append(f"- Sign-to-box alignment `{r['metric'][10:]}`: {v['lines']:,} lines, {v['atf_signs']:,} ATF signs, {fmt(v['matched'], 'pct')} matched to a box, detector class agrees on {fmt(v['det_agrees'], 'pct')}, matched box equals the eBL truth box (IoU > 0.5) in {fmt(v['truth_iou_gt_0_5'], 'pct')} of checkable cases.")
    L += ['', 'The curated single-sign test set the method was developed on scores 0.75 on the same measure.', '']
    os.makedirs(os.path.dirname(path), exist_ok=True); open(path, 'w', encoding='utf-8').write('\n'.join(L)); return path


def main():
    E = ebl_metrics(); C = cdli_metrics()
    write_individual(E, 'ebl_tablets/metrics'); write_individual(C, 'fat-cross_processed/metrics')
    p = write_comparison(E, C)
    print(f'eBL metrics: {len(E.rows)} rows -> ebl_tablets/metrics/; CDLI metrics: {len(C.rows)} rows -> fat-cross_processed/metrics/; comparison -> {p}')


if __name__ == '__main__':
    main()
