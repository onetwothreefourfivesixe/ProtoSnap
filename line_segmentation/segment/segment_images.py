"""Run the line model on any set of tablet images and write the lines it finds.

  python -m line_segmentation.segment.segment_images --images <folder> --out <folder> [options]
  python -m line_segmentation.segment.segment_images --list sides.csv --id-col side_id --file-col file --out <folder> [options]

Steps (documented in line_segmentation/segment/README.md):
  1. Preprocess each image (segment/preprocess.py): 8-bit BGR, alpha flattened onto --background, optional CLAHE.
  2. Optionally split a photograph into tablet views (--views auto): each view is saved as its own image under views/ and
     becomes a side, with coordinates in that view image. Each view gets an estimated role (obverse, reverse, edge_top, ...).
     --edges decides what happens to edges: 'skip' (default) saves them without lines, and 'segment' runs the model on them
     like faces. By default the whole image is one side.
  3. Run the line model (--weights, default paths.LINE_WEIGHTS) and its read-out (linemodel/infer.py) with the settings stored
     in the run's config.json.
  4. Turn each predicted line into a region: the line's band between its neighbours, extended by --extend line spacings and
     moved down by --shift spacings (the calibrated crop rule of make_crops.py), along the line's own angle when that is
     within --max-line-dev degrees of the face's, and running --end-margin spacings past the line's ends.
  5. Write lines.json (one entry per side, every side included even with no lines), summary.csv, and optionally line
     crops (--crops) and overlays (--overlays N).
Coordinates are pixels of the side's image, origin top left: the image as provided, or with --views auto the saved view
image (view_box gives its place in the original). Lines are ordered top to bottom.
Output is the model's raw read-out: no manual corrections, and low-scoring lines are kept with their score.
"""
import argparse
import csv
import datetime
import glob
import hashlib
import json
import os
import subprocess
import time

import cv2
import numpy as np
import torch

from line_segmentation.linemodel.infer import load_model, predict_face, read_lines, tablet_mask_for
from line_segmentation.paths import DATA_ROOT, LINE_WEIGHTS
from line_segmentation.segment.preprocess import BACKGROUNDS, find_views, load_image

IMAGE_EXT = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp')
READOUT_KEYS = dict(kernel_thr=0.5, band_thr=0.4, min_area=12, max_reach=1.5, max_dev=45.0, merge_dy=0.5, end_thr=0.2, max_ext=3.0)


def readout_settings(weights, overrides):
    """The read-out settings of a training run (its config.json), with command-line overrides."""
    cfg_p = os.path.join(os.path.dirname(weights), 'config.json'); cfg = json.load(open(cfg_p)) if os.path.exists(cfg_p) else {}
    r = {k: cfg.get(k, v) if cfg.get(k) is not None else v for k, v in READOUT_KEYS.items()}
    r.update({k: v for k, v in overrides.items() if v is not None}); return r, cfg


def line_regions(segs, W, H, extend, shift, end_margin, max_line_dev):
    """Predicted segments -> lines ordered top to bottom, each with a region polygon (4 corners, image pixels of the view)."""
    if not segs:
        return []
    ang = float(np.median([np.degrees(np.arctan2(s['ay1'] - s['ay0'], s['ax1'] - s['ax0'])) for s in segs]))
    Mf = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0); bands = []
    for s in segs:                                                     # each band in the face frame rotated by the median angle
        p0, p1 = Mf @ np.array([s['ax0'], s['ay0'], 1.0]), Mf @ np.array([s['ax1'], s['ay1'], 1.0]); c = (p0[1] + p1[1]) / 2
        bands.append(dict(seg=s, c=c, top=c - s['thickness'] / 2, bot=c + s['thickness'] / 2))
    bands.sort(key=lambda b: b['c'])
    cs = [b['c'] for b in bands]; pitch = float(np.median(np.diff(cs))) if len(cs) > 1 else bands[0]['bot'] - bands[0]['top']
    out = []
    for i, b in enumerate(bands):                                      # boundaries: middle of the gap to each neighbour
        prev, nxt = (bands[i - 1] if i else None), (bands[i + 1] if i + 1 < len(bands) else None)
        top = b['top'] if prev is None else ((prev['bot'] + b['top']) / 2 if prev['bot'] < b['top'] else (prev['c'] + b['c']) / 2)
        bot = b['bot'] if nxt is None else ((b['bot'] + nxt['top']) / 2 if b['bot'] < nxt['top'] else (b['c'] + nxt['c']) / 2)
        top, bot = top - extend * pitch + shift * pitch, bot + extend * pitch + shift * pitch
        s = b['seg']; cx, cy = (s['ax0'] + s['ax1']) / 2, (s['ay0'] + s['ay1']) / 2
        sa = float(np.degrees(np.arctan2(s['ay1'] - s['ay0'], s['ax1'] - s['ax0']))); L = float(np.hypot(s['ax1'] - s['ax0'], s['ay1'] - s['ay0']))
        use = sa if abs(((sa - ang) + 90) % 180 - 90) <= max_line_dev else ang
        M = cv2.getRotationMatrix2D((cx, cy), use, 1.0)                # the line frame: rotation about the segment's middle
        x0, x1 = cx - L / 2 - end_margin * pitch, cx + L / 2 + end_margin * pitch
        y0, y1 = cy + (top - b['c']), cy + (bot - b['c'])
        Mi = cv2.invertAffineTransform(M)
        poly = (np.array([[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]], float) @ Mi.T)
        out.append(dict(seg=s, poly=poly, angle=use, frame=(M, x0, y0, x1, y1), pitch=pitch))
    return out


def git_version():
    try:
        h = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--', 'line_segmentation'], capture_output=True, text=True).stdout.strip()
        return h + (' with uncommitted changes in line_segmentation/' if dirty else '')
    except Exception:
        return 'unknown'


def training_tablets(weights, faces_csv):
    """Tablets (and their faces) the weights were trained or validated on, from linemodel/list_training_faces.py's table."""
    cfg_p = os.path.join(os.path.dirname(weights), 'config.json')
    if not os.path.exists(faces_csv) or not os.path.exists(cfg_p):
        return None
    rows = list(csv.DictReader(open(faces_csv, encoding='utf-8')))
    col = os.path.basename(os.path.dirname(os.path.abspath(weights)))       # one column per run
    if rows and col not in rows[0]:
        return None
    rows = [r for r in rows if r.get(col)]
    tabs = {}
    for r in rows:
        t = tabs.setdefault(r['tablet'], dict(tablet=r['tablet'], p_number=r['p_number'], source=r['source'], script=r['script'], faces=[]))
        t['faces'].append(dict(face=r['face'], role=r[col]))
    return sorted(tabs.values(), key=lambda t: (t['source'], t['tablet']))


def inputs(a):
    if a.list:
        root = a.root or os.path.dirname(os.path.abspath(a.list))
        return [(r[a.id_col], os.path.join(root, r[a.file_col])) for r in csv.DictReader(open(a.list, encoding='utf-8'))]
    files = sorted(f for f in glob.glob(os.path.join(a.images, '**', '*'), recursive=True) if f.lower().endswith(IMAGE_EXT))
    return [(os.path.splitext(os.path.relpath(f, a.images))[0].replace(os.sep, '__'), f) for f in files]


def main():
    ap = argparse.ArgumentParser(description='Run the line model on a set of tablet images.')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--images', help='folder of images (searched recursively); the side id is the path without extension')
    src.add_argument('--list', help='CSV listing the images; see --id-col, --file-col and --root')
    ap.add_argument('--id-col', default='side_id'); ap.add_argument('--file-col', default='file'); ap.add_argument('--root', default=None, help='folder that --file-col paths are relative to (default: the CSV\'s folder)')
    ap.add_argument('--out', required=True); ap.add_argument('--weights', default=LINE_WEIGHTS)
    ap.add_argument('--views', choices=['whole', 'auto'], default='whole', help='auto splits photographs that show several tablet views and saves each as views/<side>__v<k>.png')
    ap.add_argument('--edges', choices=['skip', 'segment'], default='skip', help='with --views auto: what to do with views labelled as edges')
    ap.add_argument('--background', choices=sorted(BACKGROUNDS), default='black'); ap.add_argument('--clahe', action='store_true'); ap.add_argument('--exif', action='store_true')
    ap.add_argument('--extend', type=float, default=0.2); ap.add_argument('--shift', type=float, default=0.06); ap.add_argument('--end-margin', type=float, default=0.5)
    ap.add_argument('--max-line-dev', type=float, default=20.0)
    for k in READOUT_KEYS: ap.add_argument('--' + k.replace('_', '-'), type=float, default=None, help=f'read-out override (default: the run\'s config.json, else {READOUT_KEYS[k]})')
    ap.add_argument('--crops', action='store_true', help='also write one image per line, cut along the line'); ap.add_argument('--overlays', type=int, default=0, help='draw the lines on the first N images')
    ap.add_argument('--training-faces', default=os.path.join(DATA_ROOT, 'line_model_training_faces.csv'), help='linemodel/list_training_faces.py output, for trained_on')
    ap.add_argument('--name', default='ProtoSnap line model (U-Net, ResNet-34 encoder)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    readout, cfg = readout_settings(a.weights, {k: getattr(a, k) for k in READOUT_KEYS})
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; model = load_model(a.weights, dev)
    items = inputs(a); sides, summary = [], []
    if a.crops: os.makedirs(os.path.join(a.out, 'crops'), exist_ok=True)
    if a.overlays: os.makedirs(os.path.join(a.out, 'overlays'), exist_ok=True)
    if a.views == 'auto': os.makedirs(os.path.join(a.out, 'views'), exist_ok=True)
    t_all = time.time()
    for k, (sid, path) in enumerate(items, 1):
        if not os.path.exists(path):
            sides.append(dict(side_id=sid, error='file not found', lines=[])); summary.append(dict(side_id=sid, view='', n_lines=0, mean_score='', seconds=0, notes='file not found')); continue
        t0 = time.time(); img, notes = load_image(path, a.background, a.exif, a.clahe); H, W = img.shape[:2]
        views = find_views(img, a.views)
        for v in views:
            x0, y0, x1, y1 = v['box']; face = img[y0:y1, x0:x1]; whole = v['id'] == 'whole'; vid = sid if whole else f"{sid}__{v['id']}"
            if not whole:                                               # each view becomes an image of its own, and coordinates refer to it
                vfile = f'views/{vid}.png'; cv2.imwrite(os.path.join(a.out, vfile), face); ox, oy, IW, IH = 0, 0, face.shape[1], face.shape[0]
            else:
                ox, oy, IW, IH = x0, y0, W, H
            role = v.get('role', 'whole'); edge = role.startswith('edge'); skipped = edge and a.edges == 'skip'
            if skipped:
                segs = []
            else:
                maps, s = predict_face(model, face, dev)
                tab = tablet_mask_for(face, maps.shape, s)
                segs = read_lines(maps, s, readout['kernel_thr'], readout['band_thr'], int(readout['min_area']), readout['max_reach'], readout['max_dev'],
                                  readout['merge_dy'], readout['end_thr'], readout['max_ext'], tab)
            regs = line_regions(segs, face.shape[1], face.shape[0], a.extend, a.shift, a.end_margin, a.max_line_dev); lines = []
            for i, r in enumerate(regs, 1):
                poly = r['poly'] + [ox, oy]; sg = r['seg']
                bb = [max(0.0, poly[:, 0].min()), max(0.0, poly[:, 1].min()), min(float(IW), poly[:, 0].max()), min(float(IH), poly[:, 1].max())]
                ln = dict(order=i, polygon=[[round(float(px), 1), round(float(py), 1)] for px, py in poly], bbox=[round(float(c), 1) for c in bb],
                          score=round(float(sg['score']), 4), baseline=[[round(sg['ax0'] + ox, 1), round(sg['ay0'] + oy, 1)], [round(sg['ax1'] + ox, 1), round(sg['ay1'] + oy, 1)]],
                          angle_deg=round(float(r['angle']), 2))
                if a.crops:
                    M, cx0, cy0, cx1, cy1 = r['frame']; rot = cv2.warpAffine(face, M, (face.shape[1], face.shape[0]), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
                    c = rot[int(max(0, cy0)):int(min(face.shape[0], cy1)), int(max(0, cx0)):int(min(face.shape[1], cx1))]
                    if c.size: fn = f'{vid}__L{i:02d}.jpg'; cv2.imwrite(os.path.join(a.out, 'crops', fn), c, [cv2.IMWRITE_JPEG_QUALITY, 92]); ln['crop'] = f'crops/{fn}'
                lines.append(ln)
            entry = dict(side_id=vid, width=IW, height=IH, lines=lines)
            if not whole:
                entry.update(image=vfile, role=role, source_image=sid, source_file=path, view_box=[x0, y0, x1, y1])
                if skipped: entry['skipped'] = 'edge (see --edges)'
            sides.append(entry)
            summary.append(dict(side_id=vid, view=v['id'], role=role, skipped=bool(skipped), n_lines=len(lines), mean_score=round(float(np.mean([l['score'] for l in lines])), 3) if lines else '',
                                seconds=round(time.time() - t0, 2), notes='; '.join(notes)))
            if a.overlays and k <= a.overlays:
                vis = (img if whole else face).copy(); lw = max(2, int(max(IW, IH) / 600))
                for ln in lines:
                    cv2.polylines(vis, [np.round(np.array(ln['polygon'])).astype(np.int32)], True, (0, 200, 255), lw)
                    (bx0, by0), (bx1, by1) = ln['baseline']; cv2.line(vis, (int(bx0), int(by0)), (int(bx1), int(by1)), (0, 0, 255), lw)
                    cv2.putText(vis, f"{ln['order']} {ln['score']:.2f}", (int(ln['bbox'][0]) + 4, int(ln['bbox'][1]) + 6 * lw), cv2.FONT_HERSHEY_SIMPLEX, 0.25 * lw, (255, 80, 0), max(1, lw // 2))
                cv2.imwrite(os.path.join(a.out, 'overlays', f'{vid}.jpg'), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if k % 25 == 0 or k == len(items): print(f'{k}/{len(items)} images, {sum(len(s["lines"]) for s in sides)} lines', flush=True)
    sha = hashlib.sha256(open(a.weights, 'rb').read()).hexdigest()[:16]
    trained = training_tablets(a.weights, a.training_faces)
    result = dict(segmenter=dict(name=a.name, version_or_commit=f'{git_version()}; weights {a.weights} (sha256 {sha}...)',
                                 trained_on=[t['tablet'] for t in trained] if trained is not None else 'unknown: run linemodel/list_training_faces.py',
                                 trained_on_detail=trained, training_split_seed=cfg.get('seed'),
                                 tuning=('Training used the faces in trained_on (train and val roles; val picked the checkpoint). The read-out thresholds were '
                                         'set by hand, and the region rule (--extend, --shift) was calibrated on eBL faces of the line model\'s tune and test '
                                         'splits (docs/line_segmentation/line_model.md, Appendix A.8 and A.9). No other data was used.')),
                  settings=dict(views=a.views, edges=a.edges if a.views == 'auto' else None, background=a.background, clahe=a.clahe, exif=a.exif, long_side=1024, readout=readout, extend=a.extend, shift=a.shift,
                                end_margin=a.end_margin, max_line_dev=a.max_line_dev),
                  created=datetime.datetime.now().isoformat(timespec='seconds'), sides=sides)
    json.dump(result, open(os.path.join(a.out, 'lines.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    with open(os.path.join(a.out, 'summary.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
    print(f"{len(items)} images, {len(sides)} sides, {sum(len(s['lines']) for s in sides)} lines, {sum(1 for s in sides if not s['lines'])} sides with none, "
          f"{time.time() - t_all:.0f} s -> {a.out}/lines.json")


if __name__ == '__main__':
    main()
