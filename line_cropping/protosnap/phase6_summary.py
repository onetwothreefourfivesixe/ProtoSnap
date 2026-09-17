"""Summarise a ProtoSnap batch run on line-crop sign targets.
  python -m line_cropping.protosnap.phase6_summary --run output/phase6_OB --samples protosnap_inputs/OB/samples_run40.csv
Writes <run>/summary.csv (target, sign, init agg_score, files present) and <run>/contact_sheet.jpg made of each
target's final_result_100.png (prototype skeleton snapped onto the photo).
"""
import argparse, csv, os, glob
import numpy as np, cv2


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); ap.add_argument('--samples', required=True); ap.add_argument('--cols', type=int, default=4); ap.add_argument('--image', default='itr99_viz.png', help='which per-target image to tile (itr99_viz.png = photo with the fitted skeleton and losses)')
    a = ap.parse_args()
    samples = {r['fn'][:-4]: r for r in csv.DictReader(open(a.samples, encoding='utf-8'))}
    rows = []; tiles = []
    for d in sorted(glob.glob(os.path.join(a.run, '*_results'))):
        stem = os.path.basename(d)[:-len('_results')]; s = samples.get(stem, {})
        score = ''
        try:
            import torch; score = round(float(torch.load(os.path.join(d, 'agg_score.pt'))), 3)
        except Exception: pass
        final = os.path.join(d, a.image); has_final = os.path.exists(final)
        rows.append(dict(target=stem, sign=s.get('sign', ''), prototype=s.get('name', ''), init_score=score, finished=has_final))
        if has_final:
            im = cv2.imread(final); h, w = im.shape[:2]; sc = 480 / max(h, w); im = cv2.resize(im, (int(w * sc), int(h * sc)))
            can = np.full((im.shape[0] + 30, 490, 3), 30, np.uint8); can[5:5 + im.shape[0], 5:5 + im.shape[1]] = im
            cv2.putText(can, f"{s.get('name','')}  init {score}", (5, can.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1); tiles.append(can)
    with open(os.path.join(a.run, 'summary.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['target', 'sign', 'prototype', 'init_score', 'finished']); w.writeheader(); w.writerows(rows)
    if tiles:
        hh = max(t.shape[0] for t in tiles); tiles = [cv2.copyMakeBorder(t, 0, hh - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30)) for t in tiles]
        while len(tiles) % a.cols: tiles.append(np.full((hh, 490, 3), 30, np.uint8))
        grid = np.vstack([np.hstack(tiles[i:i + a.cols]) for i in range(0, len(tiles), a.cols)])
        cv2.imwrite(os.path.join(a.run, 'contact_sheet.jpg'), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    fin = sum(r['finished'] for r in rows); sc = [r['init_score'] for r in rows if r['init_score'] != '']
    print(f'{len(rows)} targets attempted, {fin} finished; init score median {np.median(sc) if sc else float("nan"):.3f} -> {a.run}/summary.csv, contact_sheet.jpg')


if __name__ == '__main__':
    main()
