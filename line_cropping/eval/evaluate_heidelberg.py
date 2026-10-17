"""Score the line model against hand-drawn text-line annotations (Dencker et al. 2020, Heidelberg Neo-Assyrian set).

  python -m line_cropping.eval.evaluate_heidelberg --weights weights/linemodel/run2/model.pt [--render 6] [--out <dir>]
  python -m line_cropping.eval.evaluate_heidelberg --segments <dir>/<method>_segments --out <dir>    # another model's lines

Data: data/heidelberg/line_annotations_{train,test}.csv (one row per polyline point: line_idx, x, y, segm_idx,
tablet_CDLI; coordinates in the face crop) and tablet_segments_{train,test}.csv (each face's box in the composite
photograph), from github.com/compvis/cuneiform-sign-detection-dataset. Photographs: the Heidelberg images in the eBL
OCR download (data/ebl/ebl_detection_2024/raw/raw-data/heidelberg/heidelberg-imgs/); faces without one are skipped.

Unlike the eBL truth (straight segments fitted through annotated sign boxes), these lines are drawn by hand from end to
end and follow the line's curve, and every line on a face is drawn, so precision is valid on every face. Annotators
drew a line as separate pieces where it crosses a crack; pieces at the same height whose horizontal ranges do not
overlap are merged into one line before scoring.

Per face, in the frame rotated by the median angle of the truth lines: a predicted line matches a truth line when
their centre heights differ by less than --tol x pitch (pitch = median gap between truth lines), one prediction per
truth line, as in evaluate_lines.py. Also: first-line error (topmost lines, in pitches), count match, and for matched
pairs how much of the truth line's horizontal extent the predicted line covers and how far it overshoots each end.
Writes <out>/faces.csv, <out>/lines.csv, <out>/summary.json, face crops in <data>/faces/ (reused for training) and
<out>/renders/ (green = hand-drawn truth, red = prediction).
"""
import argparse
import ast
import csv
import json
import os

import cv2
import numpy as np
import pandas as pd
import torch

from line_cropping.linemodel.infer import load_model, predict_face, read_lines, tablet_mask_for
from line_cropping.paths import DATA_ROOT

ANN = 'data/heidelberg'
IMGS = 'data/ebl/ebl_detection_2024/raw/raw-data/heidelberg/heidelberg-imgs'
OUT_DATA = os.path.join(DATA_ROOT, 'heidelberg')


def load_faces():
    """[(face_id, tablet, segm_idx, split, image path, crop box, [polyline (n,2) in crop coords])]"""
    out = []
    for split in ('train', 'test'):
        lines = pd.read_csv(os.path.join(ANN, f'line_annotations_{split}.csv'))
        segs = pd.read_csv(os.path.join(ANN, f'tablet_segments_{split}.csv'))
        for (tab, si), g in lines.groupby(['tablet_CDLI', 'segm_idx']):
            row = segs[(segs.tablet_CDLI == tab) & (segs.segm_idx == si)]
            if row.empty:
                continue
            polys = [gg[['x', 'y']].to_numpy(float) for _, gg in g.groupby('line_idx', sort=True)]
            out.append(dict(face=f'{tab}__s{si}', tablet=tab, segm=int(si), split=split, image=os.path.join(IMGS, tab + '.jpg'),
                            box=ast.literal_eval(row.iloc[0].bbox), view=row.iloc[0].view_desc, polys=polys))
    return out


def rot(points, M):
    p = np.asarray(points, float)
    return p @ M[:, :2].T + M[:, 2]


def truth_lines(polys, M):
    """Polylines in the rotated frame -> lines with centre height (y at the middle of the x range), x range;
    then merge pieces of one line (same height, no horizontal overlap)."""
    L = []
    for p in polys:
        q = rot(p, M)
        q = q[np.argsort(q[:, 0])]
        if len(q) < 2 or q[-1, 0] - q[0, 0] < 2:
            continue
        xm = (q[0, 0] + q[-1, 0]) / 2
        L.append(dict(y=float(np.interp(xm, q[:, 0], q[:, 1])), x0=float(q[0, 0]), x1=float(q[-1, 0]), pieces=1, pts=[q]))
    if len(L) < 2:
        return L, float('nan')
    L.sort(key=lambda d: d['y'])
    gaps = np.diff([d['y'] for d in L])
    big = gaps[gaps > 0.4 * np.percentile(gaps, 90)]
    pitch = float(np.median(big)) if len(big) else float(np.median(gaps))
    merged = []
    for d in L:
        for m in merged:
            ov = min(m['x1'], d['x1']) - max(m['x0'], d['x0'])
            if abs(m['y'] - d['y']) < 0.35 * pitch and ov < 0.1 * min(m['x1'] - m['x0'], d['x1'] - d['x0']):
                w0, w1 = m['x1'] - m['x0'], d['x1'] - d['x0']
                m.update(y=(m['y'] * w0 + d['y'] * w1) / (w0 + w1), x0=min(m['x0'], d['x0']), x1=max(m['x1'], d['x1']),
                         pieces=m['pieces'] + 1, pts=m['pts'] + d['pts'])
                break
        else:
            merged.append(dict(d))
    merged.sort(key=lambda d: d['y'])
    return merged, pitch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default=None, help='line model checkpoint (predicts on the fly)')
    ap.add_argument('--segments', default=None, help='instead of --weights: folder of <face>.csv segment files (ax0, ay0, ax1, ay1) from another model')
    ap.add_argument('--out', default=None, help='default: tablet_data/heidelberg/eval_<run>')
    ap.add_argument('--tol', type=float, default=0.35)
    ap.add_argument('--render', type=int, default=6)
    ap.add_argument('--long-side', type=int, default=1024)
    a = ap.parse_args()
    if not a.weights and not a.segments:
        raise SystemExit('give --weights or --segments')
    run = os.path.basename(os.path.dirname(a.weights)) if a.weights else os.path.basename(os.path.normpath(a.segments))
    out = a.out or os.path.join(OUT_DATA, f'eval_{run}')
    os.makedirs(os.path.join(out, 'renders'), exist_ok=True)
    os.makedirs(os.path.join(OUT_DATA, 'faces'), exist_ok=True)
    cfgp = os.path.join(os.path.dirname(a.weights), 'config.json') if a.weights else ''
    cfg = json.load(open(cfgp)) if cfgp and os.path.exists(cfgp) else {}
    ro = {k: cfg.get(k, d) for k, d in (('kernel_thr', 0.5), ('band_thr', 0.4), ('min_area', 12), ('max_reach', 1.5), ('max_dev', 45.0),
                                         ('merge_dy', 0.5), ('end_thr', 0.2), ('max_ext', 3.0), ('clip_tablet', 1))}
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(a.weights, dev) if a.weights else None
    faces = load_faces()
    rows, line_rows, skipped, renders = [], [], [], 0
    for f in faces:
        im = cv2.imread(f['image'])
        if im is None:
            skipped.append(f['face'])
            continue
        x0, y0, x1, y1 = f['box']
        crop = im[max(0, y0):y1, max(0, x0):x1]
        cv2.imwrite(os.path.join(OUT_DATA, 'faces', f['face'] + '.jpg'), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        H, W = crop.shape[:2]
        if model is not None:
            maps, s = predict_face(model, crop, dev, a.long_side)
            tab = tablet_mask_for(crop, maps.shape, s) if ro['clip_tablet'] else None
            segs = read_lines(maps, s, ro['kernel_thr'], ro['band_thr'], ro['min_area'], ro['max_reach'], ro['max_dev'], ro['merge_dy'],
                              ro['end_thr'], ro['max_ext'], tab)
        else:
            sp = os.path.join(a.segments, f['face'] + '.csv')
            if not os.path.exists(sp):     # the other model was not run on this face (e.g. test faces only)
                continue
            segs = [{k: float(v) for k, v in r.items() if k in ('ax0', 'ay0', 'ax1', 'ay1')} for r in csv.DictReader(open(sp))]
        # frame: rotate by the median angle of the truth lines (end to end)
        angs = [np.degrees(np.arctan2(p[-1, 1] - p[0, 1], p[-1, 0] - p[0, 0])) for p in f['polys'] if len(p) >= 2]
        ang = float(np.median(angs)) if angs else 0.0
        M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
        T, pitch = truth_lines(f['polys'], M)
        if len(T) < 2 or not np.isfinite(pitch):
            skipped.append(f['face'])
            continue
        P = []
        for sg in segs:
            q = rot([[sg['ax0'], sg['ay0']], [sg['ax1'], sg['ay1']]], M)
            P.append(dict(y=float(q[:, 1].mean()), x0=float(q[:, 0].min()), x1=float(q[:, 0].max())))
        P.sort(key=lambda d: d['y'])
        py = np.array([p['y'] for p in P])
        used, hit = set(), 0
        for t in T:
            m = None
            if len(py):
                j = int(np.argmin(np.abs(py - t['y'])))
                if abs(py[j] - t['y']) < a.tol * pitch and j not in used:
                    used.add(j)
                    hit += 1
                    m = P[j]
            tw = t['x1'] - t['x0']
            line_rows.append(dict(face=f['face'], split=f['split'], truth_y=round(t['y'], 1), truth_w_pitch=round(tw / pitch, 2), pieces=t['pieces'],
                                  matched=m is not None,
                                  covered=round(max(0.0, min(m['x1'], t['x1']) - max(m['x0'], t['x0'])) / tw, 3) if m else '',
                                  overshoot_left_pitch=round(max(0.0, t['x0'] - m['x0']) / pitch, 3) if m else '',
                                  overshoot_right_pitch=round(max(0.0, m['x1'] - t['x1']) / pitch, 3) if m else '',
                                  short_left_pitch=round(max(0.0, m['x0'] - t['x0']) / pitch, 3) if m else '',
                                  short_right_pitch=round(max(0.0, t['x1'] - m['x1']) / pitch, 3) if m else ''))
        rows.append(dict(face=f['face'], tablet=f['tablet'], split=f['split'], view=f['view'], width=W, height=H, pitch_px=round(pitch, 1),
                         n_truth=len(T), n_truth_pieces=len(f['polys']), n_pred=len(P), recall=round(hit / len(T), 4),
                         precision=round(len(used) / len(P), 4) if P else 0.0,
                         first_err=round(abs(P[0]['y'] - T[0]['y']) / pitch, 3) if P else '', count_ok=len(P) == len(T)))
        if renders < a.render:
            vis = crop.copy()
            lw = max(2, H // 350)
            Mi = cv2.invertAffineTransform(M)
            for t in T:
                for q in t['pts']:
                    cv2.polylines(vis, [rot(q, Mi).astype(np.int32)], False, (0, 200, 0), lw + 1)
            for sg in segs:
                cv2.line(vis, (int(sg['ax0']), int(sg['ay0'])), (int(sg['ax1']), int(sg['ay1'])), (0, 0, 255), lw)
            sc = min(1.0, 900 / H)
            cv2.imwrite(os.path.join(out, 'renders', f['face'] + '.jpg'), cv2.resize(vis, None, fx=sc, fy=sc), [cv2.IMWRITE_JPEG_QUALITY, 85])
            renders += 1
    pd.DataFrame(rows).to_csv(os.path.join(out, 'faces.csv'), index=False)
    pd.DataFrame(line_rows).to_csv(os.path.join(out, 'lines.csv'), index=False)
    R, Lr = pd.DataFrame(rows), pd.DataFrame(line_rows)

    def summ(r, l):
        mt = l[l.matched]
        fe = pd.to_numeric(r.first_err, errors='coerce').dropna()
        cov = pd.to_numeric(mt.covered, errors='coerce')
        return dict(faces=len(r), truth_lines=int(r.n_truth.sum()), recall=float(r.recall.mean()), precision=float(r.precision.mean()),
                    first_line_lt_half_pitch=float((fe < 0.5).mean()) if len(fe) else float('nan'), count_match=float(r.count_ok.mean()),
                    covered_median=float(cov.median()), covered_ge_95=float((cov >= 0.95).mean()),
                    short_right_median_pitch=float(pd.to_numeric(mt.short_right_pitch).median()), short_left_median_pitch=float(pd.to_numeric(mt.short_left_pitch).median()),
                    overshoot_right_median_pitch=float(pd.to_numeric(mt.overshoot_right_pitch).median()), overshoot_left_median_pitch=float(pd.to_numeric(mt.overshoot_left_pitch).median()))
    summary = dict(weights=a.weights or a.segments, readout=ro if a.weights else None, skipped_no_image=skipped, all=summ(R, Lr),
                   **{s: summ(R[R.split == s], Lr[Lr.split == s]) for s in ('train', 'test') if (R.split == s).any()})
    json.dump(summary, open(os.path.join(out, 'summary.json'), 'w'), indent=1)
    print(f"{len(rows)} faces scored ({len(skipped)} skipped, no photograph: {', '.join(skipped)})")
    for k in ('all', 'train', 'test'):
        if k not in summary:
            continue
        S = summary[k]
        print(f"  {k:5s} faces {S['faces']:3d} lines {S['truth_lines']:4d} | recall {S['recall']:.2f} precision {S['precision']:.2f} | first line < 0.5 pitch {S['first_line_lt_half_pitch']:.0%} | "
              f"count match {S['count_match']:.0%} | line length covered: median {S['covered_median']:.0%}, >= 95% on {S['covered_ge_95']:.0%} | "
              f"short at ends L/R {S['short_left_median_pitch']:.2f}/{S['short_right_median_pitch']:.2f} pitch, overshoot L/R {S['overshoot_left_median_pitch']:.2f}/{S['overshoot_right_median_pitch']:.2f}")
    print('->', out)


if __name__ == '__main__':
    main()
