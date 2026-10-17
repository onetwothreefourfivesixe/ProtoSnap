"""Baseline: Kraken's baseline segmenter (BLLA), zero-shot and retrained on the line model's training faces.

Kraken (kraken.re) is the open OCR/HTR engine most used in the humanities; its segmenter predicts each text line as a
polyline. It needs its own environment (it pins its own PyTorch): python3 -m venv ~/venvs/kraken && ~/venvs/kraken/bin/pip
install kraken. Two halves, run with different interpreters:

  # 1. export (project venv): PageXML for run 3's training and validation faces + test face lists
  .venv/bin/python -m line_cropping.baselines.kraken_seg export [--out tablet_data/kraken]
  # 2. train (kraken venv): fine-tune the default model with lines annotated as centre lines
  ~/venvs/kraken/bin/ketos segtrain -f page -t tablet_data/kraken/train.lst -e tablet_data/kraken/val.lst \\
      -i ~/venvs/kraken/lib/python3.11/site-packages/kraken/blla.mlmodel --augment \\
      -q early --min-epochs 20 --lag 10 -N 100 -d cuda:0 -o weights/kraken/blla_cuneiform
  # 4. optional (project venv): join side-by-side pieces of one line, as the line model's read-out does
  .venv/bin/python -m line_cropping.baselines.kraken_seg merge --pred <dir> --method kraken_ft
  # 3. predict (kraken venv): writes the line model's prediction format, scored by evaluate_lines / evaluate_heidelberg
  ~/venvs/kraken/bin/python line_cropping/baselines/kraken_seg.py predict --faces tablet_data/kraken/test_ebl.csv \\
      [--model weights/kraken/blla_cuneiform_best.safetensors] --method kraken_ft --out <dir>

Training data are exactly run 3 (seed 0)'s: eBL tune faces with coverage >= 0.9, the Old Assyrian hand labels and the
Heidelberg train faces, with run 3's validation faces as Kraken's validation set. Each truth line is written as a
centre line (Kraken's --centerline mode): eBL and Old Assyrian lines as straight segments, Heidelberg lines as their
drawn polylines; the line polygon is the band of half the line thickness around it (capped at 0.35 pitch). The predict
half imports only kraken, numpy and PIL, so it runs in the kraken environment.
"""
import argparse
import csv
import os
import sys
from xml.sax.saxutils import quoteattr

NS = 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'


def _pts(points):
    return ' '.join(f'{int(round(x))},{int(round(y))}' for x, y in points)


def _band(line, half):
    """Polygon of the band +-half around a polyline (offset perpendicular to the end-to-end direction)."""
    import numpy as np
    p = np.asarray(line, float)
    d = p[-1] - p[0]; d = d / max(np.linalg.norm(d), 1e-6); n = np.array([-d[1], d[0]])
    return [tuple(q - half * n) for q in p] + [tuple(q + half * n) for q in p[::-1]]


def write_page(path, image, W, H, lines):
    rows = [f'<?xml version="1.0" encoding="UTF-8"?>', f'<PcGts xmlns="{NS}">',
            f'  <Page imageFilename={quoteattr(os.path.abspath(image))} imageWidth="{W}" imageHeight="{H}">',
            f'    <TextRegion id="r0" custom="structure {{type:text;}}"><Coords points="0,0 {W - 1},0 {W - 1},{H - 1} 0,{H - 1}"/>']
    for i, (line, half) in enumerate(lines):
        rows.append(f'      <TextLine id="l{i}"><Coords points="{_pts(_band(line, half))}"/><Baseline points="{_pts(line)}"/></TextLine>')
    rows += ['    </TextRegion>', '  </Page>', '</PcGts>']
    open(path, 'w', encoding='utf-8').write('\n'.join(rows) + '\n')


def export(a):
    import collections
    import numpy as np
    import cv2
    from line_cropping.linemodel.dataset import build_index
    from line_cropping.paths import EBL
    idx = build_index(include_heidelberg=True, seed=0)
    for part in ('train', 'val'):
        d = os.path.join(a.out, 'gt', part); os.makedirs(d, exist_ok=True); paths = []
        for it in idx[part]:
            im = cv2.imread(it['file'], cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            H, W = im.shape
            by = collections.defaultdict(list)
            for s in it['segments']:
                by[s.get('chain', id(s))].append(s)
            lines = []
            for L in by.values():
                pts = sorted({(s['ax0'], s['ay0']) for s in L} | {(s['ax1'], s['ay1']) for s in L})
                lines.append((pts, np.median([s['thickness'] for s in L]) / 2))
            ys = sorted(np.mean([p[1] for p in l]) for l, _ in lines); g = np.diff(ys)
            pitch = float(np.median(g)) if len(g) else 2 * max(h for _, h in lines)
            lines = [(l, min(h, 0.35 * pitch)) for l, h in lines]
            p = os.path.join(d, it['face'].replace('/', '_') + '.xml'); write_page(p, it['file'], W, H, lines); paths.append(p)
        open(os.path.join(a.out, f'{part}.lst'), 'w').write('\n'.join(os.path.abspath(p) for p in paths) + '\n')
        print(f'{part}: {len(paths)} PageXML files', collections.Counter(it['source'] for it in idx[part]))
    # test lists: Heidelberg test faces (hand-drawn truth) and eBL test faces
    import pandas as pd
    lines = pd.read_csv('data/heidelberg/line_annotations_test.csv')
    hr = [dict(face=f'{t}__s{s}', file=os.path.join('tablet_data', 'heidelberg', 'faces', f'{t}__s{s}.jpg'), n=g.line_idx.nunique())
          for (t, s), g in lines.groupby(['tablet_CDLI', 'segm_idx'])]
    hr = [r for r in hr if os.path.exists(r['file'])]
    sp = {r['face']: r['split'] for r in csv.DictReader(open(os.path.join(EBL, 'split.csv')))}
    er = [dict(face=f['face'], file=f['file'], n=f.get('atf_lines_side', '')) for f in csv.DictReader(open(os.path.join(EBL, 'faces', 'faces.csv')))
          if sp.get(f['face']) == 'test' and f.get('file')]
    for name, rows in (('test_heidelberg.csv', hr), ('test_ebl.csv', er)):
        with open(os.path.join(a.out, name), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['face', 'file', 'n']); w.writeheader(); w.writerows(rows)
        print(name, len(rows), 'faces')


def predict(a):
    import math
    import time
    import numpy as np
    from PIL import Image
    from kraken.configs import SegmentationInferenceConfig
    from kraken.tasks.segmentation import SegmentationTaskModel
    Image.MAX_IMAGE_PIXELS = None
    model = SegmentationTaskModel.load_model(a.model)     # kraken 7 API: safetensors or CoreML; None = default blla model
    config = SegmentationInferenceConfig(device=a.device)
    os.makedirs(os.path.join(a.out, a.method), exist_ok=True); os.makedirs(os.path.join(a.out, a.method + '_segments'), exist_ok=True)
    faces = list(csv.DictReader(open(a.faces))); rows = []; t0 = time.time()
    for k, f in enumerate(faces):
        try:
            im = Image.open(f['file']); im.load()
        except Exception:
            continue
        W, H = im.size
        try:
            res = model.predict(im, config)
            bls = [np.asarray(l.baseline, float) for l in res.lines if l.baseline is not None and len(l.baseline) >= 2]
        except Exception as e:
            print('failed', f['face'], e); bls = []
        segs = []
        for b in bls:
            b = b[np.argsort(b[:, 0])]
            segs.append(dict(ax0=b[0, 0], ay0=b[0, 1], ax1=b[-1, 0], ay1=b[-1, 1]))
        with open(os.path.join(a.out, a.method + '_segments', f['face'] + '.csv'), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['ax0', 'ay0', 'ax1', 'ay1', 'thickness']); w.writeheader()
            w.writerows([{**{k: round(v, 1) for k, v in s.items()}, 'thickness': ''} for s in segs])
        # line rows in the frame rotated by the median line angle (as linemodel.infer.segments_to_lines)
        ang = float(np.median([math.degrees(math.atan2(s['ay1'] - s['ay0'], s['ax1'] - s['ax0'])) for s in segs])) if segs else 0.0
        al = math.radians(ang); ca, sa = math.cos(al), math.sin(al); cx, cy = W / 2, H / 2
        M = np.array([[ca, sa, (1 - ca) * cx - sa * cy], [-sa, ca, sa * cx + (1 - ca) * cy]])
        L = []
        for s in segs:
            p0 = M @ np.array([s['ax0'], s['ay0'], 1.0]); p1 = M @ np.array([s['ax1'], s['ay1'], 1.0])
            L.append(((p0[1] + p1[1]) / 2, min(p0[0], p1[0]), max(p0[0], p1[0])))
        L.sort()
        with open(os.path.join(a.out, a.method, f['face'] + '.csv'), 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(['line_no', 'y_deskewed', 'x0', 'x1', 'n_boxes'])
            for i, (y, x0, x1) in enumerate(L, 1): w.writerow([i, round(y, 1), round(x0, 1), round(x1, 1), ''])
        rows.append(dict(face=f['face'], method=a.method, n_atf=f.get('n', ''), n_pred=len(segs), skew_deg=round(ang, 2), pitch_px='', body_h='', seconds=''))
        if (k + 1) % 100 == 0: print(f'{k + 1}/{len(faces)} faces, {time.time() - t0:.0f}s', flush=True)
    p = os.path.join(a.out, 'faces_run.csv')
    old = [r for r in csv.DictReader(open(p)) if r['method'] != a.method] if os.path.exists(p) else []
    with open(p, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(old + rows)
    print(f'{a.method}: {len(rows)} faces in {time.time() - t0:.0f}s -> {a.out}')


def merge(a):
    """Apply the line model's fragment merge (linemodel.infer.merge_fragments) to another model's segments: Kraken
    often returns one text line as several side-by-side pieces. Thickness for the merge rule = 0.6 x the face's line
    spacing estimated from the predicted lines. Writes <method>_merged and <method>_merged_segments."""
    import numpy as np
    from line_cropping.linemodel.infer import merge_fragments, segments_to_lines
    from PIL import Image
    src = os.path.join(a.pred, a.method + '_segments'); dst = a.method + '_merged'
    os.makedirs(os.path.join(a.pred, dst), exist_ok=True); os.makedirs(os.path.join(a.pred, dst + '_segments'), exist_ok=True)
    runs = {r['face']: r for r in csv.DictReader(open(os.path.join(a.pred, 'faces_run.csv'))) if r['method'] == a.method}
    files = {r['face']: r['file'] for f in a.faces for r in csv.DictReader(open(f))}
    rows = []
    for face, run in runs.items():
        segs = [{k: float(v) for k, v in r.items() if k in ('ax0', 'ay0', 'ax1', 'ay1')} for r in csv.DictReader(open(os.path.join(src, face + '.csv')))]
        ys = np.sort([(s['ay0'] + s['ay1']) / 2 for s in segs]); g = np.diff(ys); big = g[g > 0.4 * np.percentile(g, 90)] if len(g) > 1 else g
        pitch = float(np.median(big)) if len(big) else 50.0
        for s in segs: s.update(thickness=0.6 * pitch, area=1, score=1.0)
        segs = merge_fragments(segs, 0.5) if len(segs) > 1 else segs
        with open(os.path.join(a.pred, dst + '_segments', face + '.csv'), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['ax0', 'ay0', 'ax1', 'ay1', 'thickness']); w.writeheader()
            w.writerows([{k: round(float(s[k]), 1) for k in ('ax0', 'ay0', 'ax1', 'ay1', 'thickness')} for s in segs])
        W, H = Image.open(files[face]).size
        ang, L = segments_to_lines(segs, W, H)
        with open(os.path.join(a.pred, dst, face + '.csv'), 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(['line_no', 'y_deskewed', 'x0', 'x1', 'n_boxes'])
            for i, r in enumerate(L, 1): w.writerow([i, round(r['y'], 1), round(r['x0'], 1), round(r['x1'], 1), ''])
        rows.append({**run, 'method': dst, 'n_pred': len(segs), 'skew_deg': round(ang, 2)})
    p = os.path.join(a.pred, 'faces_run.csv')
    old = [r for r in csv.DictReader(open(p)) if r['method'] != dst]
    with open(p, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(old + rows)
    print(f'{dst}: {len(rows)} faces -> {a.pred}')


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('export'); e.add_argument('--out', default='tablet_data/kraken')
    p = sub.add_parser('predict'); p.add_argument('--faces', required=True); p.add_argument('--model', default=None, help='default: kraken\'s own blla model')
    p.add_argument('--method', default='kraken'); p.add_argument('--out', required=True); p.add_argument('--device', default='auto', help='kraken 7 device setting (auto picks the GPU)')
    m = sub.add_parser('merge'); m.add_argument('--pred', required=True); m.add_argument('--method', default='kraken_ft')
    m.add_argument('--faces', nargs='+', default=['tablet_data/kraken/test_heidelberg.csv', 'tablet_data/kraken/test_ebl.csv'], help='face lists (for image sizes)')
    a = ap.parse_args()
    {'export': export, 'predict': predict, 'merge': merge}[a.cmd](a)


if __name__ == '__main__':
    main()
