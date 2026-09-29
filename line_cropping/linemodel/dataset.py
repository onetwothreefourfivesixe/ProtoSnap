"""Phase 7 dataset: which faces train, their truth segments, and augmented training windows.

Index (build_index):
  train  = eBL faces in the Phase 5 TUNE split with coverage >= min_cov (default 0.9)  +  Old Assyrian faces with
           hand labels (status done) -- minus a validation slice
  val    = val_frac of the training tablets (by tablet), used for model selection
  test faces are never included; evaluation runs through infer.py + the Phase 5 evaluators.
Windows: the face is resized so its long side is `long_side` x a random scale in [0.5, 2], a random `win` square is
cut (biased to contain lines), photometric jitter and background polarity swap are applied, and the five target maps
are painted from the transformed segments at stride 4.
"""
import csv, os, random, collections
import numpy as np, cv2, torch
from line_cropping.linemodel.targets import render_targets, read_segments, oa_segments, STRIDE


def build_index(min_cov=0.9, val_frac=0.1, seed=0, include_oa=True):
    items = []
    faces = {f['face']: f for f in csv.DictReader(open('ebl_tablets/faces/faces.csv'))} if os.path.exists('ebl_tablets/faces/faces.csv') else {}
    split = {r['face']: r['split'] for r in csv.DictReader(open('ebl_tablets/split.csv'))} if os.path.exists('ebl_tablets/split.csv') else {}
    for fid, f in faces.items():
        if split.get(fid) != 'tune' or not f['coverage'] or float(f['coverage']) < min_cov: continue
        lp = os.path.join('ebl_tablets/faces/lines', fid + '.csv')
        if not os.path.exists(lp): continue
        items.append(dict(face=fid, tablet=f['tablet'], file=f['file'], source='ebl-' + f['script'], segments=read_segments(lp)))
    if include_oa and os.path.exists('fat-cross_processed/ground_truth/labels.csv'):
        acc = {f['face']: f for f in csv.DictReader(open('fat-cross_processed/prepared/faces_accepted.csv'))}
        by = collections.defaultdict(list)
        for r in csv.DictReader(open('fat-cross_processed/ground_truth/labels.csv')):
            if r['status'] == 'done': by[r['face_id']].append(r)
        for fid, rows in by.items():
            f = acc.get(fid)
            if f is None or len(rows) < 1: continue
            im = cv2.imread(f['file'], cv2.IMREAD_GRAYSCALE)
            if im is None: continue
            H, W = im.shape
            items.append(dict(face=fid, tablet=f['p_number'], file=f['file'], source='oa', segments=oa_segments(rows, W, H, float(rows[0]['skew_deg']))))
    tablets = sorted({it['tablet'] for it in items}); random.seed(seed); random.shuffle(tablets); nval = max(1, int(val_frac * len(tablets))); val_t = set(tablets[:nval])
    return dict(train=[it for it in items if it['tablet'] not in val_t], val=[it for it in items if it['tablet'] in val_t])


def transform_segments(segs, M):
    out = []
    for s in segs:
        p0 = M @ np.array([s['ax0'], s['ay0'], 1.0]); p1 = M @ np.array([s['ax1'], s['ay1'], 1.0]); sc = np.sqrt(abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]))
        out.append(dict(s, ax0=p0[0], ay0=p0[1], ax1=p1[0], ay1=p1[1], thickness=s['thickness'] * sc))
    return out


class LineWindows(torch.utils.data.Dataset):
    def __init__(self, items, win=768, long_side=1024, scale=(0.5, 2.0), rot=4.0, per_face=4, aug=True, shrink=0.4, band_frac=0.7, ignore_gap=1.5):
        self.items, self.win, self.long, self.scale, self.rot, self.per_face, self.aug = items, win, long_side, scale, rot, per_face, aug
        self.shrink, self.band_frac, self.ignore_gap = shrink, band_frac, ignore_gap
        self.cache = {}
    def __len__(self): return len(self.items) * self.per_face
    def _img(self, it):
        if it['file'] not in self.cache:
            im = cv2.imread(it['file']); H, W = im.shape[:2]; s = self.long / max(H, W)
            self.cache[it['file']] = (cv2.resize(im, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA), s)
        return self.cache[it['file']]
    def __getitem__(self, i):
        it = self.items[i % len(self.items)]; im, s0 = self._img(it); H, W = im.shape[:2]
        sc = random.uniform(*self.scale) if self.aug else 1.0; ang = random.uniform(-self.rot, self.rot) if self.aug else 0.0
        # affine: scale about the centre of a randomly chosen line (so windows tend to contain text), rotate, then crop
        segs = transform_segments(it['segments'], np.array([[s0, 0, 0], [0, s0, 0]], float))
        if segs and self.aug:
            s = random.choice(segs); cx, cy = (s['ax0'] + s['ax1']) / 2 + random.uniform(-0.3, 0.3) * self.win / sc, (s['ay0'] + s['ay1']) / 2 + random.uniform(-0.3, 0.3) * self.win / sc
        else: cx, cy = W / 2, H / 2
        M = cv2.getRotationMatrix2D((cx, cy), ang, sc); M[:, 2] += np.array([self.win / 2 - cx, self.win / 2 - cy])
        bg = int(np.median(np.concatenate([im[0], im[-1], im[:, 0], im[:, -1]]).mean(axis=0)))
        win = cv2.warpAffine(im, M, (self.win, self.win), flags=cv2.INTER_LINEAR, borderValue=(bg, bg, bg))
        segs = transform_segments(segs, M)
        if self.aug:
            a = random.uniform(0.7, 1.3); b = random.uniform(-30, 30); win = np.clip(win.astype(np.float32) * a + b, 0, 255)
            if random.random() < 0.15: win = 255 - win                       # light-background museums
            if random.random() < 0.3: win = cv2.cvtColor(cv2.cvtColor(win.astype(np.uint8), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR).astype(np.float32)
            if random.random() < 0.3:
                q = random.randint(40, 90); ok, enc = cv2.imencode('.jpg', win.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, q]); win = cv2.imdecode(enc, 1).astype(np.float32)
            win = win.astype(np.uint8)
        T = render_targets(segs, self.win, self.win, STRIDE, shrink=self.shrink, ignore_gap=self.ignore_gap, band_frac=self.band_frac)
        x = torch.from_numpy(cv2.cvtColor(win, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255)
        return x, torch.from_numpy(T)
