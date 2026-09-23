"""Train a per-script cuneiform sign classifier on the eBL sign snippets (datasets from build_dataset.py).

  .venv/bin/python -m line_cropping.signs.train_classifier --script OB [--epochs 15] [--size 112]

ResNet-34 from the locally cached ImageNet weights; inputs are letterboxed to a square (aspect kept, padded with the
image's border mean) and held in RAM, since reading tens of thousands of small JPEGs from the mounted drive every
epoch is the bottleneck. Class-balanced sampling (weight 1/sqrt(class count)), light affine + photometric
augmentation, label smoothing, AdamW with cosine decay, mixed precision. The checkpoint with the best validation
top-1 is kept. Writes weights/sign_classifier/<script>/{model.pt, classes.json, metrics.json}.
"""
import argparse, csv, json, os, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from torchvision.transforms import v2

DATA = 'ebl_tablets/sign_classifier'; WEIGHTS = 'weights/sign_classifier'
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def letterbox(img, size):
    """Resize keeping aspect, pad to size x size with the mean colour of the image border."""
    h, w = img.shape[:2]; s = size / max(h, w); nh, nw = max(1, round(h * s)), max(1, round(w * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    border = np.concatenate([r[0], r[-1], r[:, 0], r[:, -1]]).mean(axis=0)
    out = np.empty((size, size, 3), np.uint8); out[:] = border.astype(np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2; out[y0:y0 + nh, x0:x0 + nw] = r
    return out


def load_split(script, split, size):
    rows = list(csv.DictReader(open(os.path.join(DATA, script, split + '.csv'), encoding='utf-8')))
    cache = os.path.join(DATA, script, f'cache_{split}_{size}.npy')
    if os.path.exists(cache) and len(np.load(cache, mmap_mode='r')) == len(rows): X = np.load(cache)
    else:
        def rd(r):
            im = cv2.imread(r['path']); return letterbox(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), size) if im is not None else np.zeros((size, size, 3), np.uint8)
        with ThreadPoolExecutor(16) as ex: X = np.stack(list(ex.map(rd, rows)))
        np.save(cache, X)
    return X, np.array([int(r['label']) for r in rows]), rows


class Snips(torch.utils.data.Dataset):
    def __init__(self, X, y, aug):
        self.X, self.y = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous(), torch.from_numpy(y)
        self.aug = v2.Compose([v2.RandomAffine(degrees=8, translate=(0.06, 0.06), scale=(0.88, 1.12), fill=110), v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2), v2.RandomGrayscale(p=0.15)]) if aug else None
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        x = self.X[i]
        if self.aug is not None: x = self.aug(x)
        return x, self.y[i]


@torch.no_grad()
def evaluate(model, X, y, dev, bs=512):
    model.eval(); logits = []
    for i in range(0, len(y), bs):
        x = torch.from_numpy(X[i:i + bs]).permute(0, 3, 1, 2).to(dev).float().div_(255); x = (x - MEAN.to(dev)) / STD.to(dev)
        with torch.autocast('cuda', dtype=torch.float16): logits.append(model(x).float().cpu())
    L = torch.cat(logits); top5 = L.topk(min(5, L.shape[1]), dim=1).indices; yt = torch.from_numpy(y)
    return float((top5[:, 0] == yt).float().mean()), float((top5 == yt[:, None]).any(dim=1).float().mean()), L


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--script', required=True); ap.add_argument('--epochs', type=int, default=15); ap.add_argument('--size', type=int, default=112)
    ap.add_argument('--bs', type=int, default=128); ap.add_argument('--lr', type=float, default=5e-4); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed); dev = 'cuda'
    classes = json.load(open(os.path.join(DATA, a.script, 'classes.json'), encoding='utf-8')); t0 = time.time()
    Xtr, ytr, _ = load_split(a.script, 'train', a.size); Xva, yva, _ = load_split(a.script, 'val', a.size); Xte, yte, rte = load_split(a.script, 'test', a.size)
    print(f'{a.script}: {len(classes)} classes | train {len(ytr)} val {len(yva)} test {len(yte)} | loaded in {time.time()-t0:.0f}s', flush=True)
    cnt = np.bincount(ytr, minlength=len(classes)); w = 1.0 / np.sqrt(cnt[ytr])
    dl = torch.utils.data.DataLoader(Snips(Xtr, ytr, True), batch_size=a.bs, sampler=torch.utils.data.WeightedRandomSampler(w, len(ytr), replacement=True), num_workers=6, drop_last=True, persistent_workers=True)
    model = torchvision.models.resnet34(weights='IMAGENET1K_V1'); model.fc = nn.Linear(model.fc.in_features, len(classes)); model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(dl), pct_start=0.15)
    scaler = torch.cuda.amp.GradScaler(); best = (-1, None); out = os.path.join(WEIGHTS, a.script); os.makedirs(out, exist_ok=True); hist = []
    for ep in range(a.epochs):
        model.train(); tl = 0; n = 0
        for x, y in dl:
            x = x.to(dev, non_blocking=True).float().div_(255); x = (x - MEAN.to(dev)) / STD.to(dev); y = y.to(dev)
            with torch.autocast('cuda', dtype=torch.float16): loss = F.cross_entropy(model(x), y, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); tl += float(loss) * len(y); n += len(y)
        v1, v5, _ = evaluate(model, Xva, yva, dev); hist.append(dict(epoch=ep + 1, train_loss=tl / n, val_top1=v1, val_top5=v5))
        print(f'  epoch {ep+1:2d}: loss {tl/n:.3f} | val top-1 {v1:.3f} top-5 {v5:.3f} | {time.time()-t0:.0f}s', flush=True)
        if v1 > best[0]: best = (v1, ep + 1); torch.save(model.state_dict(), os.path.join(out, 'model.pt'))
    model.load_state_dict(torch.load(os.path.join(out, 'model.pt'))); v1, v5, _ = evaluate(model, Xva, yva, dev); t1, t5, L = evaluate(model, Xte, yte, dev)
    pred = L.argmax(1).numpy(); per = {classes[c]: float((pred[yte == c] == c).mean()) for c in range(len(classes)) if (yte == c).sum() >= 10}
    types = {}
    for ty in sorted({r['type'] for r in rte}):
        m = np.array([r['type'] == ty for r in rte]); types[ty] = dict(n=int(m.sum()), top1=float((pred[m] == yte[m]).mean()))
    json.dump(classes, open(os.path.join(out, 'classes.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    json.dump(dict(script=a.script, arch='resnet34', size=a.size, n_classes=len(classes), best_epoch=best[1], val_top1=v1, val_top5=v5, test_top1=t1, test_top5=t5, test_n=int(len(yte)),
                   test_by_type=types, worst_classes=sorted(per.items(), key=lambda kv: kv[1])[:8], history=hist), open(os.path.join(out, 'metrics.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'{a.script}: best epoch {best[1]} | val top-1 {v1:.3f} top-5 {v5:.3f} | HELD-OUT TEST top-1 {t1:.3f} top-5 {t5:.3f} (n={len(yte)}) | by type {types}')


if __name__ == '__main__':
    main()
