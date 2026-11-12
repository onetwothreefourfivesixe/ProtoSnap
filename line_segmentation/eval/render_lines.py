"""Overlay predicted lines (and truth if present) on faces for visual review.
  python -m line_cropping.eval.render_lines --faces <faces.csv> --pred <pred dir> --method det_n --out <dir> [--n 8] [--truth-dir <lines dir>]
"""
import argparse, csv, os, random
import numpy as np, cv2
from line_cropping.lines.profile_lines import rotate


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--faces', required=True); ap.add_argument('--pred', required=True); ap.add_argument('--method', default='det_n')
    ap.add_argument('--out', required=True); ap.add_argument('--n', type=int, default=8); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--truth-dir', default=None); ap.add_argument('--only', nargs='*', default=None, help='render exactly these face ids'); ap.add_argument('--sheet', default=None, help='also stack the overlays into this contact-sheet image')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); random.seed(a.seed); tiles = []
    runs = {r['face']: r for r in csv.DictReader(open(os.path.join(a.pred, 'faces_run.csv'))) if r['method'] == a.method}
    faces = [f for f in csv.DictReader(open(a.faces)) if f['face'] in runs]; random.shuffle(faces)
    if a.only:
        order = {fid: i for i, fid in enumerate(a.only)}; faces = sorted([f for f in faces if f['face'] in order], key=lambda f: order[f['face']]); a.n = len(faces)
    for f in faces[:a.n]:
        g = cv2.imread(f['file'], cv2.IMREAD_GRAYSCALE)
        if g is None: continue
        H, W = g.shape; skew = float(runs[f['face']]['skew_deg']); img = cv2.cvtColor(rotate(g.astype(np.float32), skew).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        if a.truth_dir:
            from line_cropping.eval.evaluate_lines import truth_in_frame
            for t in truth_in_frame(os.path.join(a.truth_dir, f['face'] + '.csv'), W, H, skew): cv2.line(img, (0, int(t[0])), (W, int(t[0])), (0, 255, 0), max(1, H // 500))
        for p in csv.DictReader(open(os.path.join(a.pred, a.method, f['face'] + '.csv'))):
            y = int(float(p['y_deskewed'])); cv2.line(img, (int(p['x0']), y), (int(p['x1']), y), (0, 0, 255), max(1, H // 500))
            cv2.putText(img, p['line_no'], (max(0, int(p['x0']) - 22), y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(img, f"{f['face']}  {a.method}  N={runs[f['face']]['n_pred']} (atf {runs[f['face']]['n_atf']})", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.imwrite(os.path.join(a.out, f['face'] + '.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if a.sheet: s = 1000 / img.shape[1]; tiles += [cv2.resize(img, (1000, max(1, int(img.shape[0] * s)))), np.zeros((8, 1000, 3), np.uint8)]
    if a.sheet and tiles: cv2.imwrite(a.sheet, np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 82])
    print('wrote', min(a.n, len(faces)), 'overlays to', a.out + (f' and {a.sheet}' if a.sheet else ''))


if __name__ == '__main__':
    main()
