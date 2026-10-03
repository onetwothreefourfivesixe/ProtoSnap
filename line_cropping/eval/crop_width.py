"""Horizontal fit of line crops: how much of each true line lies inside its crop, and how much extra width the crop has.

  python -m line_cropping.eval.crop_width --crops <crops.csv> [--pred <predictions dir> --method unet]
      [--min-cov 0.9] [--worst 0.1] [--render 16] [--out <dir>]

For every truth line (eBL oriented segments, faces with annotation coverage >= --min-cov) whose centre falls inside
exactly one crop, both are compared in the crop's frame (the face rotated by the crop's skew). The true extent is the
segment between the outermost sign centres, widened by half a sign (0.5 x band thickness) on each side to reach the
sign edges. Per line:
  inside  share of the true width inside the crop            (1.0 = nothing cut off)
  inside_clay  the same, counting only the part of the true extent that lies on clay (tablet mask)
  excess  crop width outside the true extent / true width    (0.0 = no extra width)
  left_excess, right_excess   the same split by side, in pitches
  bg_overhang_pitch  how far the crop runs past the tablet into the studio background, in pitches (needs no truth)
Because eBL annotators often box only part of a line, `excess` against the truth is reported both for all lines and for
fully annotated lines (>= --full-signs signs); background overhang is the truth-free measure of genuine over-width.
With --pred, each crop is also matched to its predicted segment (nearest centre) to record whether it was built by
merging fragments, and every line is given a cause label for the worst --worst share by excess (see classify()).
Writes <out>/lines.csv, <out>/summary.json, and <out>/worst/<face>__L<nn>.jpg renders of the worst cases.
Crops cut in per-line frames (--rotate-per-line) are measured in the face frame and so slightly overstate excess.
"""
import argparse
import collections
import csv
import json
import os

import cv2
import numpy as np

from line_cropping.paths import EBL


def load_truth(face, W, H, M):
    p = os.path.join(EBL, 'faces', 'lines', face + '.csv')
    out = []
    if not os.path.exists(p):
        return out
    for l in csv.DictReader(open(p)):
        p0 = M @ np.array([float(l['ax0']), float(l['ay0']), 1.0])
        p1 = M @ np.array([float(l['ax1']), float(l['ay1']), 1.0])
        th = float(l['thickness'])
        out.append(dict(yc=(p0[1] + p1[1]) / 2, x0=min(p0[0], p1[0]) - 0.5 * th, x1=max(p0[0], p1[0]) + 0.5 * th,
                        column=l['column'], n_signs=int(l['n_signs']), label=l['line_label'], seg=l))
    return out


def load_segments(pred, method, face, M):
    p = os.path.join(pred, method + '_segments', face + '.csv')
    out = []
    if not os.path.exists(p):
        return out
    for r in csv.DictReader(open(p)):
        p0 = M @ np.array([float(r['ax0']), float(r['ay0']), 1.0])
        p1 = M @ np.array([float(r['ax1']), float(r['ay1']), 1.0])
        out.append(dict(yc=(p0[1] + p1[1]) / 2, x0=min(p0[0], p1[0]), x1=max(p0[0], p1[0]), merged=int(float(r.get('merged') or 1)),
                        score=float(r.get('score') or 0)))
    return out


def tablet_mask_rows(rot, y0, y1):
    """Per column of rows y0..y1 of the (rotated) face: share of pixels that are tablet, not studio background.
    Background polarity from the face border median; dark background < 60, light background < median - 25."""
    g = cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY)
    b = np.concatenate([g[:4].ravel(), g[-4:].ravel(), g[:, :4].ravel(), g[:, -4:].ravel()])
    bg = float(np.median(b))
    band = cv2.GaussianBlur(g[max(0, y0):max(y0 + 1, y1)], (0, 0), 2)
    tab = band > min(60, bg + 30) if bg < 128 else band < bg - 25
    return tab.mean(axis=0)


def classify(row):
    """Cause label for an over-wide line. Checked in order; the first that applies wins."""
    if row['bg_overhang_pitch'] >= 0.5:
        return 'runs into the background'
    if row['other_column_same_height']:
        return 'reaches into another column'
    if row['merged'] > 1:
        return 'fragments merged'
    if row['truth_n_signs'] <= 2:
        return 'truth line barely annotated'
    if max(row['left_excess_pitch'], row['right_excess_pitch']) > 1.5 and min(row['left_excess_pitch'], row['right_excess_pitch']) < 0.5:
        return 'one-sided overhang'
    return 'wide on both sides'


def measure(crops, faces=os.path.join(EBL, 'faces', 'faces.csv'), pred=None, method='unet', min_cov=0.9, max_cov=1.01,
            split='all', full_signs=5, worst=0.1):
    """Per-line width rows and the summary dict for one crops.csv (see the module docstring). Returns ([], {}) when no
    truth line could be matched to a crop. Used by main(), evaluate_crops and report.compile_sample."""
    faces_csv, crops_csv = faces, crops
    faces = {f['face']: f for f in csv.DictReader(open(faces_csv))}
    split_of = {r['face']: r['split'] for r in csv.DictReader(open(os.path.join(EBL, 'split.csv')))}
    by_face = collections.defaultdict(list)
    for r in csv.DictReader(open(crops_csv)):
        by_face[r['face']].append(r)
    rows = []
    rot_cache = {}
    for face, cs in by_face.items():
        f = faces.get(face)
        if f is None or not f['coverage'] or not (min_cov <= float(f['coverage']) < max_cov):
            continue
        if split != 'all' and split_of.get(face) != split:
            continue
        W, H = int(f['width']), int(f['height'])
        skew = float(cs[0]['skew_deg'])
        M = cv2.getRotationMatrix2D((W / 2, H / 2), skew, 1.0)
        img = cv2.imread(f['file'])
        rot = cv2.warpAffine(img, M, (W, H), borderValue=(0, 0, 0)) if img is not None else None
        truth = load_truth(face, W, H, M)
        segs = load_segments(pred, method, face, M) if pred else []
        for t in truth:
            hit = [c for c in cs if float(c['y0']) <= t['yc'] <= float(c['y1'])]
            if len(hit) != 1:
                continue
            c = hit[0]
            cx0, cx1 = float(c['x0']), float(c['x1'])
            pitch = float(c['pitch_px'])
            tw = t['x1'] - t['x0']
            if tw <= 0 or pitch <= 0:
                continue
            ov = max(0.0, min(cx1, t['x1']) - max(cx0, t['x0']))
            # the true extent restricted to clay: where writing runs onto the tablet's rounded edge, the half-sign
            # widening of the truth lands on background, and a crop that stops at the clay would score as cut
            cx0t, cx1t = t['x0'], t['x1']
            if rot is not None:
                ty0, ty1 = int(t['yc'] - 0.5 * float(t['seg']['thickness'])), int(t['yc'] + 0.5 * float(t['seg']['thickness']))
                lo, hi = int(max(0, t['x0'])), int(min(W, t['x1']))
                if hi - lo > 2:
                    prof_t = tablet_mask_rows(rot, max(0, ty0), min(H, ty1))[lo:hi]
                    on_t = np.where(prof_t > 0.3)[0]
                    if len(on_t):
                        cx0t, cx1t = lo + on_t[0], lo + on_t[-1] + 1
            tw_clay = max(1e-6, cx1t - cx0t)
            inside_clay = max(0.0, min(cx1, cx1t) - max(cx0, cx0t)) / tw_clay
            bg_pitch = 0.0
            if rot is not None:
                prof = tablet_mask_rows(rot, int(float(c['y0'])), int(float(c['y1'])))[int(max(0, cx0)):int(min(W, cx1))]
                on = np.where(prof > 0.3)[0]
                bg_pitch = ((on[0] + (len(prof) - 1 - on[-1])) if len(on) else len(prof)) / pitch
            left_ex = max(0.0, t['x0'] - cx0)
            right_ex = max(0.0, cx1 - t['x1'])
            cy = (float(c['y0']) + float(c['y1'])) / 2
            seg = min(segs, key=lambda s: abs(s['yc'] - cy)) if segs else None
            other = [u for u in truth if u is not t and u['column'] != t['column'] and abs(u['yc'] - t['yc']) < 0.6 * pitch
                     and max(0.0, min(cx1, u['x1']) - max(cx0, u['x0'])) > 0.3 * (u['x1'] - u['x0'])]
            rows.append(dict(face=face, line_no=int(c['line_no']), script=f['script'], crop=c['crop'], pitch=round(pitch, 1),
                             inside=round(ov / tw, 4), inside_clay=round(min(1.0, inside_clay), 4), excess=round((cx1 - cx0 - ov) / tw, 4),
                             left_excess_pitch=round(left_ex / pitch, 3), right_excess_pitch=round(right_ex / pitch, 3),
                             left_cut_pitch=round(max(0.0, cx0 - t['x0']) / pitch, 3), right_cut_pitch=round(max(0.0, t['x1'] - cx1) / pitch, 3),
                             crop_w_pitch=round((cx1 - cx0) / pitch, 2), truth_w_pitch=round(tw / pitch, 2), truth_n_signs=t['n_signs'],
                             merged=seg['merged'] if seg else 1, other_column_same_height=bool(other), bg_overhang_pitch=round(bg_pitch, 3),
                             crop_x0=round(cx0), crop_x1=round(cx1), crop_y0=round(float(c['y0'])), crop_y1=round(float(c['y1'])),
                             truth_x0=round(t['x0']), truth_x1=round(t['x1']), truth_yc=round(t['yc']), skew=skew))
    if not rows:
        return [], {}
    ex = np.array([r['excess'] for r in rows])
    thr = float(np.quantile(ex, 1 - worst))
    for r in rows:
        r['worst'] = r['excess'] >= thr and r['excess'] > 0
        r['cause'] = classify(r) if r['worst'] else ''
    worst_rows = [r for r in rows if r['worst']]
    ins = np.array([r['inside'] for r in rows])
    full = [r for r in rows if r['truth_n_signs'] >= full_signs]
    fi = np.array([r['inside'] for r in full]); fc = np.array([r['inside_clay'] for r in full]); fe = np.array([r['excess'] for r in full]); bgp = np.array([r['bg_overhang_pitch'] for r in rows])
    summary_full = dict(lines=len(full), inside_median=float(np.median(fi)), inside_ge_95=float(np.mean(fi >= 0.95)), inside_clay_median=float(np.median(fc)), inside_clay_ge_95=float(np.mean(fc >= 0.95)), excess_median=float(np.median(fe)),
                        excess_p90=float(np.quantile(fe, 0.9)), excess_gt_50=float(np.mean(fe > 0.5))) if full else {}
    summary_bg = dict(median_pitch=float(np.median(bgp)), p90_pitch=float(np.quantile(bgp, 0.9)), share_ge_half_pitch=float(np.mean(bgp >= 0.5)), share_ge_one_pitch=float(np.mean(bgp >= 1.0)))
    summary = dict(crops=crops_csv, lines=len(rows), fully_annotated=summary_full, background_overhang=summary_bg, inside_median=float(np.median(ins)), inside_ge_95=float(np.mean(ins >= 0.95)),
                   excess_median=float(np.median(ex)), excess_p75=float(np.quantile(ex, 0.75)), excess_p90=float(np.quantile(ex, 0.9)),
                   worst_threshold=thr, worst_n=len(worst_rows), causes=dict(collections.Counter(r['cause'] for r in worst_rows).most_common()),
                   worst_by_script=dict(collections.Counter(r['script'] for r in worst_rows)), all_by_script=dict(collections.Counter(r['script'] for r in rows)),
                   cut_left_median_pitch=float(np.median([r['left_cut_pitch'] for r in rows])), cut_right_median_pitch=float(np.median([r['right_cut_pitch'] for r in rows])))
    return rows, summary


def headline(summary):
    """The standing width metrics, flat, for evaluate_crops / aggregate_metrics / compile_sample. Shares in 0..1.
    kept_whole_clay is the main one: fully annotated lines whose clay part lies at least 95% inside the crop."""
    if not summary:
        return {}
    fa, bg = summary.get('fully_annotated') or {}, summary['background_overhang']
    return dict(width_lines=summary['lines'], width_full_lines=fa.get('lines', 0),
                kept_whole_clay=fa.get('inside_clay_ge_95', float('nan')), kept_whole=fa.get('inside_ge_95', float('nan')),
                excess_median_full=fa.get('excess_median', float('nan')), excess_gt_50_full=fa.get('excess_gt_50', float('nan')),
                cut_left_median_pitch=summary['cut_left_median_pitch'], cut_right_median_pitch=summary['cut_right_median_pitch'],
                bg_overhang_median_pitch=bg['median_pitch'], bg_overhang_ge_1_pitch=bg['share_ge_one_pitch'])


def report_lines(summary, full_signs=5, worst=0.1):
    """The printed summary, one string per line."""
    if not summary:
        return ['no truth line matched a crop']
    summary_full, summary_bg, thr = summary['fully_annotated'], summary['background_overhang'], summary['worst_threshold']
    out = []
    out.append(f"lines {summary['lines']} | true width inside: median {summary['inside_median']:.0%}, >=95% on {summary['inside_ge_95']:.0%} | "
          f"excess: median {summary['excess_median']:.0%}, p75 {summary['excess_p75']:.0%}, p90 {summary['excess_p90']:.0%}")
    if summary_full:
        out.append(f"fully annotated lines (>= {full_signs} signs, n={summary_full['lines']}): inside median {summary_full['inside_median']:.0%}, >=95% on {summary_full['inside_ge_95']:.0%} | on clay: >=95% on {summary_full['inside_clay_ge_95']:.0%} | excess median {summary_full['excess_median']:.0%}, p90 {summary_full['excess_p90']:.0%}, >50% on {summary_full['excess_gt_50']:.0%}")
    out.append(f"crop running past the tablet into background (no truth needed): median {summary_bg['median_pitch']:.2f} pitch, p90 {summary_bg['p90_pitch']:.2f}; >= 0.5 pitch on {summary_bg['share_ge_half_pitch']:.0%}, >= 1 pitch on {summary_bg['share_ge_one_pitch']:.0%}")
    out.append(f"worst {worst:.0%} (excess >= {thr:.0%}): {summary['worst_n']} lines; causes: {summary['causes']}")
    out.append(f"  by script, worst vs all: {summary['worst_by_script']} / {summary['all_by_script']}")
    out.append(f"  cut ends, median in pitches: left {summary['cut_left_median_pitch']:.2f}, right {summary['cut_right_median_pitch']:.2f}")
    return out


def render_worst(rows, out_dir, n=16, faces=os.path.join(EBL, 'faces', 'faces.csv')):
    """Draw the n worst lines by excess on the face, in the crop frame: red = crop, green = true line extent."""
    os.makedirs(os.path.join(out_dir, 'worst'), exist_ok=True)
    faces = {f['face']: f for f in csv.DictReader(open(faces))}
    worst = [r for r in rows if r['worst']]
    # renders: the worst cases, drawn on the face in the crop frame
    for r in sorted(worst, key=lambda r: -r['excess'])[:n]:
        f = faces[r['face']]
        img = cv2.imread(f['file'])
        H, W = img.shape[:2]
        M = cv2.getRotationMatrix2D((W / 2, H / 2), r['skew'], 1.0)
        rot = cv2.warpAffine(img, M, (W, H), borderValue=(0, 0, 0))
        pad = int(1.5 * r['pitch'])
        y0, y1 = max(0, r['crop_y0'] - pad), min(H, r['crop_y1'] + pad)
        view = rot[y0:y1].copy()
        lw = max(2, H // 600)
        cv2.rectangle(view, (r['crop_x0'], r['crop_y0'] - y0), (r['crop_x1'], r['crop_y1'] - y0), (0, 0, 255), lw)
        cv2.line(view, (r['truth_x0'], r['truth_yc'] - y0), (r['truth_x1'], r['truth_yc'] - y0), (0, 220, 0), lw + 1)
        s = min(1.0, 1400 / view.shape[1])
        view = cv2.resize(view, None, fx=s, fy=s)
        bar = np.full((28, view.shape[1], 3), 40, np.uint8)
        cv2.putText(bar, f"{r['face']} L{r['line_no']}  excess {r['excess']:.0%}  inside {r['inside']:.0%}  {r['cause']}", (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, 'worst', f"{r['face']}__L{r['line_no']:02d}.jpg"), np.vstack([bar, view]), [cv2.IMWRITE_JPEG_QUALITY, 82])
    return min(n, len(worst))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--crops', required=True)
    ap.add_argument('--faces', default=os.path.join(EBL, 'faces', 'faces.csv'))
    ap.add_argument('--pred', default=None)
    ap.add_argument('--method', default='unet')
    ap.add_argument('--min-cov', type=float, default=0.9)
    ap.add_argument('--max-cov', type=float, default=1.01, help='upper coverage bound, e.g. < 0.9 selects tune faces the line model never trained on')
    ap.add_argument('--split', choices=['all', 'test', 'tune'], default='all')
    ap.add_argument('--worst', type=float, default=0.1)
    ap.add_argument('--render', type=int, default=16)
    ap.add_argument('--full-signs', type=int, default=5, help='a truth line counts as fully annotated with at least this many signs')
    ap.add_argument('--out', default=os.path.join(EBL, 'linemodel', 'crop_width'))
    a = ap.parse_args()
    rows, summary = measure(a.crops, a.faces, a.pred, a.method, a.min_cov, a.max_cov, a.split, a.full_signs, a.worst)
    if not rows:
        raise SystemExit('no lines matched')
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'lines.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump(summary, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
    print('\n'.join(report_lines(summary, a.full_signs, a.worst)))
    n = render_worst(rows, a.out, a.render, a.faces)
    print(f'-> {a.out}/lines.csv, summary.json, worst/ ({n} renders: red = crop, green = true line extent)')


if __name__ == '__main__':
    main()
