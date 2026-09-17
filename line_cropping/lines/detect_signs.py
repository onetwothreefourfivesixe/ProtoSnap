"""Phase 3b: run the eBL Deformable DETR sign detector on prepared faces.

Must run in the mmdetection environment:  ~/venvs/mmdet/bin/python -m line_cropping.lines.detect_signs --faces <faces.csv> --out <dir>
Writes <out>/<face>.csv with one box per row (x0,y0,x1,y1,score,label,sign) in face pixels, and <out>/run.csv.
Model: external/cuneiform-ocr/configs/detr.py + weights/ebl_detr/detr173/detr-173-classes-10-2025/epoch_1000.pth
(Zenodo 17395154, CC BY 4.0). Inference resizes each face to fit 1333 x 800 (the model's test pipeline).
"""
import argparse, csv, os, sys, time
import numpy as np, cv2

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG = os.path.join(REPO, 'external', 'cuneiform-ocr', 'configs', 'detr.py')
CKPT = os.path.join(REPO, 'weights', 'ebl_detr', 'detr173', 'detr-173-classes-10-2025', 'epoch_1000.pth')


def load_model(device='cuda:0'):
    sys.path.insert(0, os.path.join(REPO, 'external', 'cuneiform-ocr', 'mmdetection'))
    from mmdet.apis import init_detector
    from mmdet.utils import register_all_modules
    register_all_modules()
    return init_detector(CONFIG, CKPT, device=device)


def _nms(boxes, scores, labels, iou=0.5):
    if len(boxes) == 0: return boxes, scores, labels
    idx = cv2.dnn.NMSBoxes([[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes], [float(s) for s in scores], 0.0, iou)
    idx = np.array(idx).ravel().astype(int) if len(idx) else np.array([], dtype=int)
    return boxes[idx], scores[idx], labels[idx]


def detect(model, bgr, score_thr=0.3, tiles=1, overlap=0.2):
    """Run the detector; with tiles > 1 the face is cut into `tiles` overlapping horizontal bands, each run
    separately (the model's pipeline rescales every input to fit 1333 x 800, so bands see the text at a higher
    effective resolution), and the boxes are merged with NMS."""
    from mmdet.apis import inference_detector
    h, w = bgr.shape[:2]
    if tiles <= 1: windows = [(0, h)]
    else:
        band = h / (tiles - (tiles - 1) * overlap); step = band * (1 - overlap)
        windows = [(int(i * step), int(min(h, i * step + band))) for i in range(tiles)]
    B, S, L = [], [], []
    for y0, y1 in windows:
        res = inference_detector(model, cv2.cvtColor(bgr[y0:y1], cv2.COLOR_BGR2RGB)).pred_instances.cpu()
        keep = res.scores.numpy() >= score_thr
        b = res.bboxes.numpy()[keep].copy(); b[:, [1, 3]] += y0
        B.append(b); S.append(res.scores.numpy()[keep]); L.append(res.labels.numpy()[keep])
    boxes, scores, labels = np.concatenate(B), np.concatenate(S), np.concatenate(L)
    return _nms(boxes, scores, labels) if tiles > 1 else (boxes, scores, labels)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--faces', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=None); ap.add_argument('--score-thr', type=float, default=0.3)
    ap.add_argument('--viz', type=int, default=0, help='write an overlay jpg for the first N faces')
    ap.add_argument('--single-column-only', action='store_true'); ap.add_argument('--skip-existing', action='store_true'); ap.add_argument('--tiles', type=int, default=1)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    faces = list(csv.DictReader(open(a.faces)))
    if a.single_column_only: faces = [f for f in faces if f.get('n_columns', '1') in ('', '1')]
    if a.limit: faces = faces[:a.limit]
    model = load_model(); classes = list(model.dataset_meta['classes']); print('model loaded,', len(classes), 'classes', flush=True)
    run = []; t0 = time.time()
    for k, f in enumerate(faces):
        outp = os.path.join(a.out, f['face'] + '.csv')
        if a.skip_existing and os.path.exists(outp): continue
        bgr = cv2.imread(f['file'])
        if bgr is None: continue
        ts = time.time(); boxes, scores, labels = detect(model, bgr, a.score_thr, a.tiles)
        with open(outp, 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh); w.writerow(['x0', 'y0', 'x1', 'y1', 'score', 'label', 'sign'])
            for b, s, l in zip(boxes, scores, labels): w.writerow([round(float(b[0]), 1), round(float(b[1]), 1), round(float(b[2]), 1), round(float(b[3]), 1), round(float(s), 3), int(l), classes[int(l)]])
        run.append(dict(face=f['face'], n_boxes=len(boxes), median_box_h=round(float(np.median(boxes[:, 3] - boxes[:, 1])), 1) if len(boxes) else '', seconds=round(time.time() - ts, 2)))
        if k < a.viz:
            for b, s in zip(boxes, scores): cv2.rectangle(bgr, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0) if s >= 0.5 else (0, 165, 255), 2)
            sc = min(1.0, 1400 / max(bgr.shape[:2])); cv2.imwrite(os.path.join(a.out, f['face'] + '_viz.jpg'), cv2.resize(bgr, None, fx=sc, fy=sc), [cv2.IMWRITE_JPEG_QUALITY, 80])
        if (k + 1) % 50 == 0: print(f'{k+1}/{len(faces)} faces, {time.time()-t0:.0f}s', flush=True)
    with open(os.path.join(a.out, 'run.csv'), 'a', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['face', 'n_boxes', 'median_box_h', 'seconds']); 
        if fh.tell() == 0: w.writeheader()
        w.writerows(run)
    print(f'done {len(run)} faces in {time.time()-t0:.0f}s; boxes per face median {np.median([r["n_boxes"] for r in run]) if run else 0}')


if __name__ == '__main__':
    main()
