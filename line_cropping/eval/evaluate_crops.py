"""Score line crops against the eBL line truth: a truth line is 'captured' when its centre line (both ends,
mapped into the deskewed frame) lies inside one crop's rows with at least a quarter of the crop's own pad to
spare, and its band (thickness) is inside the crop. Reports captured / missed / split-over-two-crops and the
share of crops that hold no truth line (empty or unannotated). Neighbour intrusion: per crop, the largest share of
another line's truth band inside the crop's rows (>= 50% = half of the neighbouring line sliced in).
Horizontal fit (crop width) comes from eval/crop_width.py on faces with coverage >= 0.9: the share of fully annotated
lines kept whole on clay, extra width, the median cut at each end, and the overhang past the tablet into the
background (crop width tests, docs/line_model.md section 10). It reads the face images; --no-width skips it.
  python -m line_cropping.eval.evaluate_crops --faces tablet_data/ebl_tablets/faces/faces.csv --crops tablet_data/ebl_tablets/line_crops/crops.csv [--split test] [--no-width]
"""
import argparse, csv, os, collections
import numpy as np, cv2
from line_cropping.eval import crop_width


def compute(faces_csv, crops_csv, width=False, subset='all', max_cov=1.01):
    """Crop metrics against the line truth next to faces_csv. Returns a dict of numbers (shares in 0..1).
    width=True adds the crop_width headline measures (reads the face images); subset restricts both to a Phase 5 split
    ('test' or 'tune'); max_cov keeps faces with annotation coverage below it (< 0.9 = tune faces the line model
    never trained on)."""
    split_of = {r['face']: r['split'] for r in csv.DictReader(open(os.path.join(os.path.dirname(os.path.dirname(faces_csv)), 'split.csv')))} if subset != 'all' else {}
    class A: pass
    a = A(); a.faces = faces_csv; a.crops = crops_csv
    faces = {f['face']: f for f in csv.DictReader(open(a.faces))}
    crops = collections.defaultdict(list)
    for r in csv.DictReader(open(a.crops)): crops[r['face']].append(r)
    stats = collections.Counter(); per_face = []; intr = []; crop_hit = collections.Counter(); purity = collections.Counter()
    for face, cs in crops.items():
        if subset != 'all' and split_of.get(face) != subset: continue
        if float(faces[face]['coverage'] or 0) >= max_cov: continue
        f = faces[face]; W, H = int(f['width']), int(f['height']); skew = float(cs[0]['skew_deg'])
        M = cv2.getRotationMatrix2D((W / 2, H / 2), skew, 1.0)
        tp = os.path.join(os.path.dirname(a.faces), 'lines', face + '.csv')
        if not os.path.exists(tp): continue
        cap = miss = split = 0; used = collections.Counter()
        for l in csv.DictReader(open(tp)):
            p0 = M @ np.array([float(l['ax0']), float(l['ay0']), 1.0]); p1 = M @ np.array([float(l['ax1']), float(l['ay1']), 1.0]); t = float(l['thickness']) / 2
            ylo, yhi = min(p0[1], p1[1]) - t, max(p0[1], p1[1]) + t; yc = (p0[1] + p1[1]) / 2
            inside = [c for c in cs if float(c['y0']) <= ylo and yhi <= float(c['y1'])]
            centre_in = [c for c in cs if float(c['y0']) <= yc <= float(c['y1'])]
            if inside: cap += 1; used[inside[0]['crop']] += 1
            elif len(centre_in) >= 1: split += 1
            else: miss += 1
        stats['captured'] += cap; stats['partial'] += split; stats['missed'] += miss
        for c in cs: crop_hit['with_truth' if used[c['crop']] else 'no_truth'] += 1
        # line-purity metrics on truth centres: each truth centre should fall in exactly one crop, and no crop
        # should hold the centres of two lines (that is the "half of the neighbouring line" failure)
        T = []; TB = []
        for l in csv.DictReader(open(tp)):
            p0 = M @ np.array([float(l['ax0']), float(l['ay0']), 1.0]); p1 = M @ np.array([float(l['ax1']), float(l['ay1']), 1.0]); T.append((p0[1] + p1[1]) / 2)
            TB.append((min(p0[1], p1[1]) - float(l['thickness']) / 2, max(p0[1], p1[1]) + float(l['thickness']) / 2))
        # neighbour intrusion: for every crop, the largest share of another truth line's band (one whose centre lies
        # outside the crop) that falls inside the crop's rows; >= 0.5 is the "half of the neighbouring line" failure
        for c in cs:
            y0c, y1c = float(c['y0']), float(c['y1']); worst = 0.0
            for t, (bt, bb) in zip(T, TB):
                if y0c <= t <= y1c or bb <= bt: continue
                worst = max(worst, max(0.0, min(y1c, bb) - max(y0c, bt)) / (bb - bt))
            intr.append(worst)
        for t in T:
            k = sum(1 for c in cs if float(c['y0']) <= t <= float(c['y1'])); purity['centre_in_' + ('0' if k == 0 else '1' if k == 1 else '2+')] += 1
        for c in cs:
            k = sum(1 for t in T if float(c['y0']) <= t <= float(c['y1'])); purity['crops_total'] += 1; purity['crops_with_2+_centres'] += k >= 2
        per_face.append(dict(face=face, n_truth=cap + split + miss, captured=cap, coverage=float(f['coverage'] or 0)))
    tot = max(1, stats['captured'] + stats['partial'] + stats['missed']); hi = [p for p in per_face if p['coverage'] >= 0.9]
    tc = max(1, purity['centre_in_0'] + purity['centre_in_1'] + purity['centre_in_2+']); nc = max(1, sum(crop_hit.values()))
    wm = {}
    if width:
        _, summary = crop_width.measure(crops_csv, faces_csv, split=subset, min_cov=0.0 if max_cov <= 0.9 else 0.9, max_cov=max_cov); wm = crop_width.headline(summary)
    return dict(**wm, truth_lines=tot, band_fully_inside=stats['captured'] / tot, band_cut=stats['partial'] / tot, missed=stats['missed'] / tot,
                faces_hi_coverage=len(hi), band_fully_inside_hi=sum(p['captured'] for p in hi) / max(1, sum(p['n_truth'] for p in hi)),
                centre_in_one=purity['centre_in_1'] / tc, centre_in_none=purity['centre_in_0'] / tc, centre_in_two=purity['centre_in_2+'] / tc,
                crops_with_two_lines=purity['crops_with_2+_centres'] / max(1, purity['crops_total']), crops=nc, crops_with_truth=crop_hit['with_truth'] / nc,
                neighbour_band_median=float(np.median(intr)) if intr else float('nan'), neighbour_band_ge_25=float(np.mean(np.array(intr) >= 0.25)) if intr else float('nan'),
                neighbour_band_ge_50=float(np.mean(np.array(intr) >= 0.5)) if intr else float('nan'))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--faces', required=True); ap.add_argument('--crops', required=True)
    ap.add_argument('--split', choices=['all', 'test', 'tune'], default='all'); ap.add_argument('--max-cov', type=float, default=1.01, help='only faces with annotation coverage below this (0.9 with --split tune = faces the line model never trained on)'); ap.add_argument('--no-width', action='store_true', help='skip the crop width measures (they read the face images)')
    a = ap.parse_args(); m = compute(a.faces, a.crops, width=not a.no_width, subset=a.split, max_cov=a.max_cov)
    print(f"truth lines: {m['truth_lines']}; fully inside one crop: {m['band_fully_inside']:.0%}; centre inside but band cut: {m['band_cut']:.0%}; not in any crop: {m['missed']:.0%}")
    print(f"faces with >=0.9 coverage: {m['faces_hi_coverage']}; their truth lines fully inside a crop: {m['band_fully_inside_hi']:.0%}")
    print(f"truth line centres inside exactly one crop: {m['centre_in_one']:.0%}; in no crop: {m['centre_in_none']:.0%}; in two or more crops (overlap): {m['centre_in_two']:.0%}")
    print(f"crops holding the centres of two or more lines (neighbour sliced in): {m['crops_with_two_lines']:.0%}")
    print(f"neighbour intrusion (share of another line's truth band inside a crop): median {m['neighbour_band_median']:.0%}; crops with >= 25% of a neighbour's band {m['neighbour_band_ge_25']:.0%}, >= 50% (half a neighbour) {m['neighbour_band_ge_50']:.0%}")
    print(f"crops: {m['crops']}; holding a truth line {m['crops_with_truth']:.0%}; holding none {1-m['crops_with_truth']:.0%} (unannotated lines count as none)")
    if m.get('width_lines'):
        print(f"width ({m['width_lines']} lines, {m['width_full_lines']} fully annotated): kept whole on clay {m['kept_whole_clay']:.0%} of fully annotated lines; "
              f"extra width median {m['excess_median_full']:.0%}, more than 50% too wide {m['excess_gt_50_full']:.0%}; median cut left {m['cut_left_median_pitch']:.2f} / right {m['cut_right_median_pitch']:.2f} pitch; "
              f"past the clay median {m['bg_overhang_median_pitch']:.2f} pitch, >= 1 pitch on {m['bg_overhang_ge_1_pitch']:.0%}")


if __name__ == '__main__':
    main()
