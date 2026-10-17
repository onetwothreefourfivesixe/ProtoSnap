"""Baseline: the line segmentation network of Dencker et al. (2020), ported to current PyTorch and retrained.

  python -m line_cropping.baselines.dencker_linenet train [--epochs 101] [--out weights/dencker/linenet_ours.pt]
  python -m line_cropping.baselines.dencker_linenet predict --weights weights/dencker/linenet_ours.pt
         --set heidelberg|ebl [--split test] [--count none|truth] --out <dir>

Dencker, Klinkisch, Maul, Ommer, "Deep learning of cuneiform sign detection with weak supervision using
transliteration alignment", PLOS ONE 15(12): e0243039 (2020); code github.com/CompVis/cuneiform-sign-detection-code
(lib/models/linenet.py, lib/datasets/lines_dataset.py, lib/detection/line_detection.py). Their trained weights
(lineNet_basic_vpub.pth) were unreachable on 14 Oct 2026, so the model is retrained with their recipe on the same data:

- Network: AlexNet-style patch classifier on 227 x 227 grayscale patches (line / no line), turned into a fully
  convolutional net for whole faces (fc6 -> 6 x 6 conv, score -> 1 x 1 conv), as in their LineNetFCN.
- Data: the Heidelberg faces of their train split with hand-drawn lines (data/heidelberg/), each rescaled by the
  segment's scale (signs ~128 px). A pixel is "line" within 32 px of a drawn line; positives are sampled near lines in
  proportion to line length, 10% of patches are background sampled away from lines. Augmentation: rotation +-8 deg,
  random resized crop (scale 0.5-1). SGD lr 0.01, momentum 0.9, weight decay 5e-4, 101 epochs, lr x0.2 at 60 and 80.
- Read-out (their post-processing, simplified): line map -> skeleton -> Hough transform restricted to near-horizontal
  angles and re-focused on the median angle -> peaks -> non-maximum suppression within 0.6 of the expected line
  spacing (128 px at their scale) -> one straight line per peak, cut to the extent of the skeleton along it.
  --count truth keeps the strongest n peaks, n = the transliteration's line count for eBL faces (their setting) and
  the number of drawn lines for Heidelberg faces; --count none keeps every peak above 0.3 of the strongest (no
  transliteration, like our line model). min_peak 0.3 and nms 0.6 were tuned on the authors' Heidelberg train faces.
- Scale at test time: Heidelberg faces use their own scale; eBL faces use 128 / truth line pitch (sign height ~ line
  pitch on the Heidelberg faces, median ratio 1.0). This gives the baseline the right scale for free.
Writes predictions in the line model's format (<out>/<method>_segments/<face>.csv and <out>/<method>/<face>.csv plus
faces_run.csv), so evaluate_lines.py and evaluate_heidelberg.py --segments score them unchanged.
"""
import argparse
import ast
import csv
import os
import random
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from skimage.transform import hough_line, hough_line_peaks

from line_cropping.linemodel.infer import segments_to_lines
from line_cropping.paths import DATA_ROOT, EBL

ANN = 'data/heidelberg'
FACES = os.path.join(DATA_ROOT, 'heidelberg', 'faces')


class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1e-4, beta=0.75):
        super().__init__()
        self.avg = nn.AvgPool2d(local_size, 1, (local_size - 1) // 2)
        self.alpha, self.beta = alpha, beta

    def forward(self, x):
        return x / self.avg(x.pow(2)).mul(self.alpha).add(1.0).pow(self.beta)


def features():
    return nn.Sequential(
        nn.Conv2d(1, 64, 11, 4), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2), LRN(),
        nn.Conv2d(64, 256, 5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2), LRN(),
        nn.Conv2d(256, 384, 3, padding=1), nn.BatchNorm2d(384, affine=False), nn.ReLU(inplace=True),
        nn.Conv2d(384, 384, 3, padding=1), nn.BatchNorm2d(384, affine=False), nn.ReLU(inplace=True),
        nn.Conv2d(384, 256, 3, padding=1), nn.BatchNorm2d(256, affine=False), nn.ReLU(inplace=True),
        nn.MaxPool2d(3, 2))


class LineNet(nn.Module):
    """Patch classifier (227 x 227 -> 2 classes); fcn() returns the fully convolutional version for whole faces."""

    def __init__(self):
        super().__init__()
        self.features = features()
        self.fc6 = nn.Linear(256 * 6 * 6, 512)
        self.score = nn.Linear(512, 2)
        self.drop = nn.Dropout()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.005); nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.score(self.drop(torch.relu(self.fc6(self.features(x).flatten(1)))))

    def fcn_forward(self, x):
        """Same weights applied convolutionally: line probability map (output stride 32)."""
        f = self.features(x)
        w6 = self.fc6.weight.view(512, 256, 6, 6)
        h = torch.relu(nn.functional.conv2d(f, w6, self.fc6.bias))
        s = nn.functional.conv2d(h, self.score.weight.view(2, 512, 1, 1), self.score.bias)
        return torch.softmax(s, 1)[:, 1]


# ------------------------------------------------------------------ data
def heidelberg_faces(split):
    lines = pd.read_csv(os.path.join(ANN, f'line_annotations_{split}.csv'))
    segs = pd.read_csv(os.path.join(ANN, f'tablet_segments_{split}.csv'))
    out = []
    for (tab, si), g in lines.groupby(['tablet_CDLI', 'segm_idx']):
        row = segs[(segs.tablet_CDLI == tab) & (segs.segm_idx == si)]
        f = os.path.join(FACES, f'{tab}__s{si}.jpg')
        if row.empty or not os.path.exists(f) or not np.isfinite(row.iloc[0].scale):
            continue
        polys = [gg[['x', 'y']].to_numpy(float) for _, gg in g.groupby('line_idx')]
        out.append(dict(face=f'{tab}__s{si}', file=f, scale=float(row.iloc[0].scale), polys=polys))
    return out


class LinePatches(torch.utils.data.Dataset):
    def __init__(self, faces, line_height=32, sample_radius=96, expo=0.5, soft_bg=0.1, patch=320, crop=227, length=None):
        self.items, self.lh, self.soft_bg, self.patch, self.crop = [], line_height, soft_bg, patch, crop
        lines = []
        for fi, f in enumerate(faces):
            im = cv2.imread(f['file'], cv2.IMREAD_GRAYSCALE)
            im = cv2.resize(im, None, fx=f['scale'], fy=f['scale'], interpolation=cv2.INTER_LINEAR)
            mask = np.zeros(im.shape, np.uint8)
            for p in f['polys']:
                q = np.rint(p * f['scale']).astype(np.int32)
                cv2.polylines(mask, [q], False, 1, 1)
                x0, y0 = q.min(0) - sample_radius; x1, y1 = q.max(0) + sample_radius
                lines.append((fi, max(0, x0), max(0, y0), min(im.shape[1], x1), min(im.shape[0], y1), q[:, 0].max() - q[:, 0].min()))
            dist = ndi.distance_transform_edt(1 - mask)
            freq = np.clip(1 - dist / sample_radius, 0, None) ** expo
            self.items.append(dict(im=im, line=dist < line_height, freq=freq))
        w = np.array([l[5] for l in lines], float); w = np.maximum(w, 1)
        self.lines, self.p = lines, w / w.sum()
        self.length = length or int(np.ceil(w.sum() / w.min()))

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if random.random() > self.soft_bg:
            fi, x0, y0, x1, y1, _ = self.lines[np.random.choice(len(self.lines), p=self.p)]
            it = self.items[fi]; fr = it['freq'][y0:y1, x0:x1]
            ys, xs = np.where(fr >= random.random())
            if len(xs) == 0: ys, xs = np.where(fr > 0)
            k = random.randrange(len(xs)); x, y = x0 + xs[k], y0 + ys[k]
        else:
            it = self.items[random.randrange(len(self.items))]; H, W = it['im'].shape
            for _ in range(100):
                x, y = random.randrange(W), random.randrange(H)
                if it['freq'][y, x] == 0: break
        target = int(it['line'][y, x])
        h = self.patch // 2
        pad = cv2.copyMakeBorder(it['im'], h, h, h, h, cv2.BORDER_CONSTANT, value=0)
        p = pad[y:y + 2 * h, x:x + 2 * h].astype(np.float32)
        # augmentation as theirs: rotation +-8, centre crop crop+15, random resized crop (scale 0.5-1, ratio 3/4-4/3)
        M = cv2.getRotationMatrix2D((h, h), random.uniform(-8, 8), 1.0)
        p = cv2.warpAffine(p, M, (2 * h, 2 * h))
        c = self.crop + 15; o = h - c // 2; p = p[o:o + c, o:o + c]
        area = c * c * random.uniform(0.5, 1.0); r = np.exp(random.uniform(np.log(3 / 4), np.log(4 / 3)))
        cw, ch = int(min(c, np.sqrt(area * r))), int(min(c, np.sqrt(area / r)))
        x0, y0 = random.randint(0, c - cw), random.randint(0, c - ch)
        p = cv2.resize(p[y0:y0 + ch, x0:x0 + cw], (self.crop, self.crop))
        return torch.from_numpy(p / 255.0 - 0.5)[None].float(), target


def train(a):
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    faces = heidelberg_faces('train')
    ds = LinePatches(faces)
    print(f'{len(faces)} Heidelberg train faces, {len(ds.lines)} lines, {len(ds)} patches per epoch', flush=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True, num_workers=6, drop_last=True, persistent_workers=True)
    m = LineNet().to(dev)
    opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[60, 80], gamma=0.2)
    lossf = nn.CrossEntropyLoss(); t0 = time.time()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    for ep in range(a.epochs):
        m.train(); tot = n = cor = 0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            out = m(x); loss = lossf(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(y); n += len(y); cor += int((out.argmax(1) == y).sum())
        sch.step()
        if ep % 10 == 0 or ep == a.epochs - 1:
            print(f'epoch {ep + 1:3d}: loss {tot / n:.3f}, patch accuracy {cor / n:.3f}, {(time.time() - t0) / 60:.1f} min', flush=True)
            torch.save(m.state_dict(), a.out)
    print('->', a.out)


# ------------------------------------------------------------------ read-out
def line_map(model, gray, scale, dev):
    im = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    ph, pw = max(0, 260 - im.shape[0]), max(0, 260 - im.shape[1])     # the 227-px window needs at least one position
    if ph or pw: im = cv2.copyMakeBorder(im, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
    x = torch.from_numpy(im.astype(np.float32) / 255.0 - 0.5)[None, None].to(dev)
    with torch.no_grad():
        prob = model.fcn_forward(x)[0].cpu().numpy()
    return prob   # stride 32 in the rescaled face; first cell centred at ~ (227 - 1) / 2


def read_lines(prob, scale, n_lines=None, min_peak=0.3, nms=0.6):
    """Their post-processing, simplified: skeleton of the line map -> near-horizontal Hough, re-focused on the median
    angle -> peaks -> merge peaks closer than half the median spacing -> lines cut to the skeleton's extent.
    Returns segments in face pixels."""
    stride, off = 32.0, 113.0
    lbl = prob > 0.5
    if lbl.sum() < 3:
        return []
    sk = skeletonize(lbl)
    th = np.linspace(np.deg2rad(70), np.deg2rad(110), 81)
    h, theta, d = hough_line(sk, theta=th)
    _, angs, _ = hough_line_peaks(h, theta, d, min_distance=1, min_angle=4, num_peaks=60)
    med = float(np.median(angs)) if len(angs) else np.pi / 2
    th = np.linspace(med - np.deg2rad(3), med + np.deg2rad(3), 25)
    h, theta, d = hough_line(sk, theta=th)
    acc, angs, dists = hough_line_peaks(h, theta, d, min_distance=1, min_angle=1, threshold=0.05 * h.max(), num_peaks=200)
    if len(acc) == 0:
        return []
    # faces are rescaled so signs are ~128 px tall and sign height ~ line pitch (median ratio 1.0 on the Heidelberg
    # faces), so the expected line spacing is 128 px = 4 cells of the stride-32 map. Non-maximum suppression: keep the
    # strongest peak, drop peaks within half a spacing of a kept one (a sequential merge would chain across lines)
    spacing = 128.0 / stride
    cand = []
    for i in np.argsort(-acc):
        if all(abs(dists[i] - c[2]) >= nms * spacing for c in cand):
            cand.append((float(acc[i]), float(angs[i]), float(dists[i])))
    if n_lines:
        cand = sorted(cand, reverse=True)[:n_lines]
    else:
        top = max(c[0] for c in cand); cand = [c for c in cand if c[0] >= min_peak * top]
    ys, xs = np.nonzero(sk); segs = []
    for _, ang, dist in cand:
        c, s = np.cos(ang), np.sin(ang)
        near = np.abs(xs * c + ys * s - dist) <= 1.0
        if near.sum() < 2:
            continue
        xa, xb = xs[near].min(), xs[near].max()
        ya, yb = (dist - xa * c) / s, (dist - xb * c) / s
        to_face = lambda u, v: ((u * stride + off) / scale, (v * stride + off) / scale)
        (x0, y0), (x1, y1) = to_face(xa, ya), to_face(xb, yb)
        segs.append(dict(ax0=x0, ay0=y0, ax1=x1, ay1=y1, thickness=0.6 * spacing * stride / scale))
    return segs


def predict(a):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = LineNet(); m.load_state_dict(torch.load(a.weights, map_location=dev)); m.to(dev).eval()
    method = f'dencker_{a.count}'
    os.makedirs(os.path.join(a.out, method), exist_ok=True); os.makedirs(os.path.join(a.out, method + '_segments'), exist_ok=True)
    if a.set == 'heidelberg':
        items = []
        for split in ('train', 'test'):
            if a.split in (split, 'all'):
                items += [dict(face=f['face'], file=f['file'], scale=f['scale'], n=len(f['polys'])) for f in heidelberg_faces(split)]
    else:
        faces = list(csv.DictReader(open(os.path.join(EBL, 'faces', 'faces.csv'))))
        sp = {r['face']: r['split'] for r in csv.DictReader(open(os.path.join(EBL, 'split.csv')))}
        items = []
        for f in faces:
            if a.split != 'all' and sp.get(f['face']) != a.split:
                continue
            if not f.get('pitch_px') or not f.get('file'):
                continue
            n = int(f['atf_lines_side']) if f.get('atf_lines_side') else None   # the transliteration's count, as in their setting
            items.append(dict(face=f['face'], file=f['file'], scale=128.0 / float(f['pitch_px']), n=n))
    rows = []; t0 = time.time()
    for it in items:
        g = cv2.imread(it['file'], cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        sc = float(np.clip(it['scale'], 0.2, 4.0))
        if max(g.shape) * sc > 6000:
            sc = 6000 / max(g.shape)
        segs = read_lines(line_map(m, g, sc, dev), sc, it['n'] if a.count == 'truth' else None, a.min_peak, a.nms)
        H, W = g.shape
        with open(os.path.join(a.out, method + '_segments', it['face'] + '.csv'), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['ax0', 'ay0', 'ax1', 'ay1', 'thickness']); w.writeheader()
            w.writerows([{k: round(v, 1) for k, v in s.items()} for s in segs])
        ang, L = segments_to_lines(segs, W, H)
        with open(os.path.join(a.out, method, it['face'] + '.csv'), 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(['line_no', 'y_deskewed', 'x0', 'x1', 'n_boxes'])
            for i, r in enumerate(L, 1): w.writerow([i, round(r['y'], 1), round(r['x0'], 1), round(r['x1'], 1), ''])
        rows.append(dict(face=it['face'], method=method, n_atf=it['n'] or '', n_pred=len(segs), skew_deg=round(ang, 2), pitch_px='', body_h='', seconds=''))
    p = os.path.join(a.out, 'faces_run.csv')
    old = [r for r in csv.DictReader(open(p)) if r['method'] != method] if os.path.exists(p) else []
    with open(p, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(old + rows)
    print(f'{method}: {len(rows)} faces in {time.time() - t0:.0f}s -> {a.out}')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train'); t.add_argument('--epochs', type=int, default=101); t.add_argument('--out', default='weights/dencker/linenet_ours.pt')
    p = sub.add_parser('predict'); p.add_argument('--weights', default='weights/dencker/linenet_ours.pt'); p.add_argument('--set', choices=['heidelberg', 'ebl'], required=True)
    p.add_argument('--split', default='test', choices=['train', 'test', 'tune', 'all']); p.add_argument('--count', choices=['none', 'truth'], default='none'); p.add_argument('--min-peak', type=float, default=0.3, help='keep Hough peaks at least this share of the strongest (count none)'); p.add_argument('--nms', type=float, default=0.6, help='suppress peaks within this many expected line spacings of a stronger one'); p.add_argument('--out', required=True)
    a = ap.parse_args()
    train(a) if a.cmd == 'train' else predict(a)


if __name__ == '__main__':
    main()
