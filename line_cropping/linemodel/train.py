"""Phase 7, step 3: train the line U-Net.

  .venv/bin/python -m line_cropping.linemodel.train [--epochs 40] [--bs 6] [--lr 3e-4] [--win 768] [--dry-run]
      [--no-oa] [--out weights/linemodel/run1]

Trains on the index from dataset.build_index (tune-split eBL faces with coverage >= 0.9 plus labelled Old Assyrian
faces, minus a by-tablet validation slice). AdamW, cosine schedule with warm-up, mixed precision. After every epoch
the model is run on the validation faces with the same read-out as inference (infer.read_lines) and line recall /
precision at 0.35 pitch are computed against the validation truth; the checkpoint with the best recall+precision is
kept as model.pt. --dry-run: 3 optimisation steps and a 3-face validation, to prove the loop runs.
Writes <out>/{model.pt, last.pt, history.csv, config.json} and, when training ends (or every --plot-every epochs),
the graphs of the whole run in <out>/plots/ (loss, loss components, validation recall/precision, checkpoint-selection
score, dashboard.png; see visualize.py).
"""
import argparse, csv, json, os, time
import matplotlib as pb
import numpy as np, torch
from line_cropping.linemodel.dataset import build_index, LineWindows
from line_cropping.linemodel.model import LineUNet, line_loss
from line_cropping.linemodel.infer import predict_face, read_lines, score_against_truth
from line_cropping.linemodel.visualize import plot_history


def validate(model, items, dev, long_side, max_faces=None, kernel_thr=0.5, band_thr=0.4, min_area=12, max_reach=1.5, max_dev=45.0, merge_dy=0.5):
    rec, prec = [], []
    for it in items[:max_faces] if max_faces else items:
        maps, scale = predict_face(model, it['file'], dev, long_side); segs = read_lines(maps, scale, kernel_thr=kernel_thr, band_thr=band_thr, min_area=min_area, max_reach=max_reach, max_dev=max_dev, merge_dy=merge_dy)
        r, p = score_against_truth(segs, it['segments'], it['file']); rec.append(r); prec.append(p)
    return float(np.nanmean(rec)) if rec else float('nan'), float(np.nanmean(prec)) if prec else float('nan')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--bs', type=int, default=6); ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--win', type=int, default=768); ap.add_argument('--long-side', type=int, default=1024); ap.add_argument('--per-face', type=int, default=4); ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--min-cov', type=float, default=0.9); ap.add_argument('--no-oa', action='store_true'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='weights/linemodel/run1'); ap.add_argument('--dry-run', action='store_true'); ap.add_argument('--resume', default=None)
    g = ap.add_argument_group('read-out (validation scoring and checkpoint selection only; saved to config.json and reused by infer.py)')
    g.add_argument('--kernel-thr', type=float, default=0.5, help='kernel probability above which a pixel is a line core')
    g.add_argument('--band-thr', type=float, default=0.4, help='band probability above which a pixel belongs to a line band')
    g.add_argument('--min-area', type=int, default=12, help='smallest kernel component kept, in pixels at stride 4')
    g.add_argument('--max-reach', type=float, default=1.5, help='band pixels join a kernel only within this many kernel heights')
    g.add_argument('--max-dev', type=float, default=45.0, help='drop segments more than this many degrees off the face median angle')
    g.add_argument('--merge-dy', type=float, default=0.5, help='merge side-by-side fragments of one line (x band thickness); 0 disables')
    g = ap.add_argument_group('target shape (changes what the network learns)')
    g.add_argument('--kernel-shrink', type=float, default=0.4, help='kernel width as a fraction of the band width')
    g.add_argument('--band-frac', type=float, default=0.7, help='maximum band width as a fraction of the block pitch')
    g.add_argument('--ignore-gap', type=float, default=1.5, help='gaps wider than this many pitches are excluded from the loss')
    ap.add_argument('--plot-every', type=int, default=0, help='also redraw the training graphs every N epochs while running (0 = only at the end)')
    a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed); dev = 'cuda' if torch.cuda.is_available() else 'cpu'; os.makedirs(a.out, exist_ok=True)
    idx = build_index(min_cov=a.min_cov, include_oa=not a.no_oa, seed=a.seed); tr, va = idx['train'], idx['val']
    if a.dry_run: tr, va = tr[:12], va[:3]
    src = {}
    for it in tr: src[it['source']] = src.get(it['source'], 0) + 1
    print(f'read-out: kernel_thr {a.kernel_thr} band_thr {a.band_thr} min_area {a.min_area} max_reach {a.max_reach} max_dev {a.max_dev} merge_dy {a.merge_dy} | targets: kernel_shrink {a.kernel_shrink} band_frac {a.band_frac} ignore_gap {a.ignore_gap}', flush=True)
    print(f"train faces {len(tr)} {src} ({sum(len(it['segments']) for it in tr)} lines) | val faces {len(va)} | device {dev}", flush=True)
    ds = LineWindows(tr, win=a.win, long_side=a.long_side, per_face=a.per_face, shrink=a.kernel_shrink, band_frac=a.band_frac, ignore_gap=a.ignore_gap)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=0 if a.dry_run else a.workers, drop_last=True, persistent_workers=not a.dry_run and a.workers > 0)
    model = LineUNet().to(dev)
    if a.resume: model.load_state_dict(torch.load(a.resume, map_location=dev)); print('resumed from', a.resume)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4); steps = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(steps, 4), pct_start=0.08); scaler = torch.cuda.amp.GradScaler(enabled=dev == 'cuda')
    json.dump(vars(a), open(os.path.join(a.out, 'config.json'), 'w'), indent=1); hist = []; best = -1; t0 = time.time()
    for ep in range(a.epochs):
        model.train(); tot = 0; n = 0; parts = dict(kernel=0, band=0, orient=0)
        for k, (x, T) in enumerate(dl):
            x, T = x.to(dev, non_blocking=True), T.to(dev, non_blocking=True)
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == 'cuda'):
                out = model(x); loss, pr = line_loss(out.float(), T)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); scaler.step(opt); scaler.update()
            if sched.last_epoch < steps - 1: sched.step()
            tot += float(loss); n += 1
            for kk in parts: parts[kk] += pr[kk]
            if a.dry_run and k >= 2: break
        r, p = validate(model, va, dev, a.long_side, max_faces=3 if a.dry_run else None, kernel_thr=a.kernel_thr, band_thr=a.band_thr, min_area=a.min_area, max_reach=a.max_reach, max_dev=a.max_dev, merge_dy=a.merge_dy)
        row = dict(epoch=ep + 1, loss=tot / max(1, n), **{k: v / max(1, n) for k, v in parts.items()}, val_recall=r, val_precision=p, minutes=(time.time() - t0) / 60); hist.append(row)
        print(f"epoch {ep+1:3d}: loss {row['loss']:.3f} (kernel {row['kernel']:.3f} band {row['band']:.3f} orient {row['orient']:.3f}) | val recall {r:.3f} precision {p:.3f} | {row['minutes']:.1f} min", flush=True)
        torch.save(model.state_dict(), os.path.join(a.out, 'last.pt'))
        if (r + p) > best: best = r + p; torch.save(model.state_dict(), os.path.join(a.out, 'model.pt'))
        with open(os.path.join(a.out, 'history.csv'), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(hist[0].keys())); w.writeheader(); w.writerows(hist)
        if a.plot_every and (ep + 1) % a.plot_every == 0: plot_history(a.out)   # optional progress refresh
        if a.dry_run: break
    files = plot_history(a.out)               # one set of graphs covering the whole run
    print(f'done. best val recall+precision {best:.3f} -> {a.out}/model.pt; graphs: {os.path.join(a.out, "plots")} ({len(files)} files, dashboard.png shows the whole run)')


if __name__ == '__main__':
    main()
