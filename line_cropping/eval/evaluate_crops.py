"""Score line crops against the eBL line truth: a truth line is 'captured' when its centre line (both ends,
mapped into the deskewed frame) lies inside one crop's rows with at least a quarter of the crop's own pad to
spare, and its band (thickness) is inside the crop. Reports captured / missed / split-over-two-crops and the
share of crops that hold no truth line (empty or unannotated).
  python -m line_cropping.eval.evaluate_crops --faces ebl_tablets/faces/faces.csv --crops ebl_tablets/line_crops/crops.csv
"""
import argparse, csv, os, collections
import numpy as np, cv2


def compute(faces_csv, crops_csv):
    """Crop metrics against the line truth next to faces_csv. Returns a dict of numbers (shares in 0..1)."""
    class A: pass
    a = A(); a.faces = faces_csv; a.crops = crops_csv
    faces = {f['face']: f for f in csv.DictReader(open(a.faces))}
    crops = collections.defaultdict(list)
    for r in csv.DictReader(open(a.crops)): crops[r['face']].append(r)
    stats = collections.Counter(); per_face = []; crop_hit = collections.Counter(); purity = collections.Counter()
    for face, cs in crops.items():
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
        T = []
        for l in csv.DictReader(open(tp)):
            p0 = M @ np.array([float(l['ax0']), float(l['ay0']), 1.0]); p1 = M @ np.array([float(l['ax1']), float(l['ay1']), 1.0]); T.append((p0[1] + p1[1]) / 2)
        for t in T:
            k = sum(1 for c in cs if float(c['y0']) <= t <= float(c['y1'])); purity['centre_in_' + ('0' if k == 0 else '1' if k == 1 else '2+')] += 1
        for c in cs:
            k = sum(1 for t in T if float(c['y0']) <= t <= float(c['y1'])); purity['crops_total'] += 1; purity['crops_with_2+_centres'] += k >= 2
        per_face.append(dict(face=face, n_truth=cap + split + miss, captured=cap, coverage=float(f['coverage'] or 0)))
    tot = max(1, stats['captured'] + stats['partial'] + stats['missed']); hi = [p for p in per_face if p['coverage'] >= 0.9]
    tc = max(1, purity['centre_in_0'] + purity['centre_in_1'] + purity['centre_in_2+']); nc = max(1, sum(crop_hit.values()))
    return dict(truth_lines=tot, band_fully_inside=stats['captured'] / tot, band_cut=stats['partial'] / tot, missed=stats['missed'] / tot,
                faces_hi_coverage=len(hi), band_fully_inside_hi=sum(p['captured'] for p in hi) / max(1, sum(p['n_truth'] for p in hi)),
                centre_in_one=purity['centre_in_1'] / tc, centre_in_none=purity['centre_in_0'] / tc, centre_in_two=purity['centre_in_2+'] / tc,
                crops_with_two_lines=purity['crops_with_2+_centres'] / max(1, purity['crops_total']), crops=nc, crops_with_truth=crop_hit['with_truth'] / nc)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--faces', required=True); ap.add_argument('--crops', required=True)
    a = ap.parse_args(); m = compute(a.faces, a.crops)
    print(f"truth lines: {m['truth_lines']}; fully inside one crop: {m['band_fully_inside']:.0%}; centre inside but band cut: {m['band_cut']:.0%}; not in any crop: {m['missed']:.0%}")
    print(f"faces with >=0.9 coverage: {m['faces_hi_coverage']}; their truth lines fully inside a crop: {m['band_fully_inside_hi']:.0%}")
    print(f"truth line centres inside exactly one crop: {m['centre_in_one']:.0%}; in no crop: {m['centre_in_none']:.0%}; in two or more crops (overlap): {m['centre_in_two']:.0%}")
    print(f"crops holding the centres of two or more lines (neighbour sliced in): {m['crops_with_two_lines']:.0%}")
    print(f"crops: {m['crops']}; holding a truth line {m['crops_with_truth']:.0%}; holding none {1-m['crops_with_truth']:.0%} (unannotated lines count as none)")


if __name__ == '__main__':
    main()
