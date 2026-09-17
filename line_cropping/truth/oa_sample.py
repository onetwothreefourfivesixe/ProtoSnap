"""Phase 1b: pick a stratified sample of Old Assyrian faces for hand labelling and write automatic
pre-labels (line centres from the ATF-constrained comb fit) plus preview overlays to correct against.

Writes fat-cross_processed/ground_truth/{sample_faces.csv, prelabels.csv, previews/<face>.png}
Usage (repo root): python -m line_cropping.truth.oa_sample [--n 100] [--seed 0]
"""
import argparse, csv, os, random, collections
import numpy as np, cv2
from line_cropping.lines.profile_lines import find_lines, rotate

FACES = 'fat-cross_processed/fat-cross_processed'
OUT = 'fat-cross_processed/ground_truth'


def stratum(r):
    ppl = float(r['px_per_line']); n = int(r['atf_lines'])
    res = 'low' if ppl < 25 else ('mid' if ppl < 40 else 'high')
    dens = '1-6' if n <= 6 else ('7-10' if n <= 10 else ('11-15' if n <= 15 else '16+'))
    return (res, dens, r['face'])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=100); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); random.seed(a.seed)
    qc = [r for r in csv.DictReader(open('fat-cross_processed/experiments/face_qc.csv')) if not r['flags'] and r['atf_lines']]
    strata = collections.defaultdict(list)
    for r in qc: strata[stratum(r)].append(r)
    total = len(qc); picks = []
    for k, rs in sorted(strata.items()):
        q = max(2, round(a.n * len(rs) / total)); picks += random.sample(rs, min(q, len(rs)))
    picks = picks[:a.n] if len(picks) > a.n else picks
    os.makedirs(os.path.join(OUT, 'previews'), exist_ok=True)
    with open(os.path.join(OUT, 'sample_faces.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['face_id', 'file', 'p_number', 'face', 'atf_lines', 'px_per_line', 'stratum', 'split'])
        for i, r in enumerate(picks):
            w.writerow([f"{r['p_number']}_{r['face']}", f"{FACES}/{r['p_number']}_{r['face']}.png", r['p_number'], r['face'], r['atf_lines'], r['px_per_line'], '/'.join(stratum(r)), 'tune' if i % 2 == 0 else 'test'])
    with open(os.path.join(OUT, 'prelabels.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['face_id', 'skew_deg', 'pitch_px', 'line_no', 'y_deskewed', 'x0', 'x1', 'source'])
        for r in picks:
            fid = f"{r['p_number']}_{r['face']}"; img = cv2.imread(f"{FACES}/{fid}.png", cv2.IMREAD_GRAYSCALE)
            res = find_lines(img, n_lines=int(r['atf_lines']))
            for j, y in enumerate(res['lines_y'], 1):
                w.writerow([fid, res['skew_deg'], round(res['pitch_px'], 1), j, round(y, 1), res['body_x0'], res['body_x1'], 'comb_fit'])
            rot = cv2.cvtColor(rotate(img.astype(np.float32), res['skew_deg']).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            for j, y in enumerate(res['lines_y'], 1):
                cv2.line(rot, (res['body_x0'], int(y)), (res['body_x1'], int(y)), (0, 255, 0), 1)
                cv2.putText(rot, str(j), (2, int(y) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            cv2.imwrite(os.path.join(OUT, 'previews', f'{fid}.png'), rot)
    print(f'sample: {len(picks)} faces from {len(strata)} strata (pool {total} unflagged faces with ATF)')
    print('by stratum:', collections.Counter('/'.join(stratum(r)) for r in picks).most_common())
    print(f'wrote {OUT}/sample_faces.csv, prelabels.csv and previews/')


if __name__ == '__main__':
    main()
