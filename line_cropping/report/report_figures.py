"""Figures for the line model report (docs/line_model/line_model.md) and the model comparison (related_work.md).

  MPLBACKEND=Agg .venv/bin/python -m line_cropping.report.report_figures [--out docs/line_model/img] [--seed 3]

Reads the stored per-face scores (evaluation.csv of each prediction set in tablet_data/ebl_tablets/linemodel/compare/,
and the Heidelberg summaries) and the stored predictions, so the figures always match the tables. Writes:
  runs_by_script.png        run 1, run 2, run 3 (two seeds) per script: recall, precision, first line
  training_curves.png       validation recall and precision per epoch for every run
  runs_examples.jpg         the same test faces read by run 1, run 2 and run 3
  models_by_script.png      Dencker line network, Kraken (retrained) and the line model: one panel per metric (recall,
                            precision, first line), grouped bars per script and for the Heidelberg hand-drawn lines
  models_examples.jpg       the same test faces read by the three models
Example faces are drawn at random (seeded) from test faces with annotation coverage >= 0.9 and 6 to 16 lines (>= 0.8 and
4 to 25 lines where a script has none), not picked by hand.
"""
import argparse
import csv
import json
import os
import random

import cv2
import numpy as np
import pandas as pd

from line_cropping.paths import EBL, DATA_ROOT

CMP = os.path.join(EBL, 'linemodel', 'compare')
HEID = os.path.join(DATA_ROOT, 'heidelberg')
SURF, TXT, TXT2, MUTED, GRID, BASE = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
def read_eval(path):
    # keep_default_na=False: the script code 'NA' (Neo-Assyrian) must not be read as a missing value
    return pd.read_csv(path, keep_default_na=False, na_values=['', 'nan'])


SCRIPTS = [('all', 'All scripts'), ('NA', 'Neo-Assyrian'), ('NB', 'Neo-Babylonian'), ('OA', 'Old Assyrian'), ('OB', 'Old Babylonian'), ('small', 'Small faces (< 60 px/line)')]


def script_stats(eval_csv, method):
    """Per script: mean recall, mean precision (faces with coverage >= 0.9), share of faces with first line < 0.5 pitch."""
    e = read_eval(eval_csv)
    e = e[e.method == method]
    out = {}
    for key, _ in SCRIPTS:
        s = e if key == 'all' else (e[e.px_per_line < 60] if key == 'small' else e[e.script == key])
        fe = s.first_err.dropna()
        out[key] = dict(recall=s.recall.mean(), precision=s.precision.dropna().mean(), first=(fe < 0.5).mean() if len(fe) else np.nan, n=len(s))
    return out


def heid_stats(summary_json):
    s = json.load(open(summary_json))['test']
    return dict(recall=s['recall'], precision=s['precision'], first=s['first_line_lt_half_pitch'])


def style(ax, xlab=True):
    ax.set_facecolor(SURF)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color(BASE)
    ax.tick_params(length=0, colors=TXT2)
    ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True)
    # dot plots need no zero baseline; every value here lies between 40% and 100%
    ax.set_xlim(0.4, 1.0); ax.set_xticks([.4, .5, .6, .7, .8, .9, 1]); ax.set_xticklabels(['40%', '50%', '60%', '70%', '80%', '90%', '100%'] if xlab else [], color=MUTED, fontsize=9)


def dot_panels(path, title, legend, rows, series, metrics, note):
    """rows: [(key, label)]; series: [(name, color, {key: {metric: value or (lo, hi)}})]; one panel per metric."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'text.color': TXT})
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.4 * len(metrics) + 2.2, 0.55 * len(rows) + 2.6), facecolor=SURF, sharey=True)
    n = len(series); off = np.linspace(-0.22, 0.22, n) if n > 1 else [0]
    for ax, (mk, mlabel) in zip(axes, metrics):
        style(ax)
        for j, (name, color, data) in enumerate(series):
            for i, (key, _) in enumerate(rows):
                v = data.get(key, {}).get(mk)
                if v is None or (not isinstance(v, tuple) and not np.isfinite(v)):
                    continue
                y = len(rows) - 1 - i + off[j]
                if isinstance(v, tuple):
                    ax.plot(v, [y, y], color=color, lw=2, solid_capstyle='round', zorder=2)
                    v = float(np.mean(v))
                v = max(v, 0.405)
                ax.scatter([v], [y], s=46, color=color, edgecolor=SURF, linewidth=1.5, zorder=3)
        ax.set_title(mlabel, loc='left', fontsize=11.5, fontweight='bold', color=TXT, pad=10)
    axes[0].set_yticks(range(len(rows))); axes[0].set_yticklabels([lab for _, lab in rows][::-1], color=TXT2)
    axes[0].set_ylim(-0.6, len(rows) - 0.4)
    fig.suptitle(title, x=0.01, ha='left', fontsize=14, fontweight='bold', color=TXT, y=0.99)
    fig.legend(handles=[Line2D([], [], marker='o', ls='', color=c, markersize=8, label=lab) for lab, c in legend],
               loc='upper left', bbox_to_anchor=(0.005, 0.93), ncol=len(legend), frameon=False, fontsize=9.5, labelcolor=TXT)
    fig.text(0.01, 0.01, note, fontsize=8, color=MUTED, va='bottom')
    fig.subplots_adjust(left=0.17, right=0.985, top=0.78, bottom=0.17, wspace=0.12)
    fig.savefig(path, dpi=160, facecolor=SURF); plt.close(fig); print('->', path)


def runs_by_script(out):
    blues = ['#9ec5f2', '#5b9ae6', '#1f5fb4']        # one hue, light -> dark: run 1 -> run 3
    r1 = script_stats(f'{CMP}/run1/evaluation.csv', 'unet'); r2 = script_stats(f'{CMP}/run2/evaluation.csv', 'unet')
    a = script_stats(f'{CMP}/run3_full_s0/evaluation.csv', 'unet'); b = script_stats(f'{CMP}/run3_full_s1/evaluation.csv', 'unet')
    r3 = {k: {m: (a[k][m], b[k][m]) for m in ('recall', 'precision', 'first')} for k in a}
    dot_panels(os.path.join(out, 'runs_by_script.png'), 'Line model runs on the eBL test faces',
               [('run 1 (302 training faces)', blues[0]), ('run 2 (+ eBL Old Assyrian, Neo-Babylonian; 464)', blues[1]),
                ('run 3 (+ hand-drawn Neo-Assyrian lines; 490), bar = range of two seeds', blues[2])],
               SCRIPTS, [('run 1', blues[0], r1), ('run 2', blues[1], r2), ('run 3', blues[2], r3)],
               [('recall', 'Line recall'), ('precision', 'Line precision'), ('first', 'First line within half a pitch')],
               'Same read-out for every run. Recall: share of truth lines found within 0.35 pitch; precision only on faces with >= 90% of signs annotated. '
               'About 760 test faces (NA 160, NB 183, OA 90, OB 327, small 89).')


def training_curves(out):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'text.color': TXT})
    runs = [('run1', 'run 1', '#9ec5f2', '-'), ('run2', 'run 2', '#5b9ae6', '-'), ('run3_full_s0', 'run 3, seed 0', '#1f5fb4', '-'), ('run3_full_s1', 'run 3, seed 1', '#1f5fb4', '--')]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURF)
    for ax, (col, lab) in zip(axes, (('val_recall', 'Validation recall'), ('val_precision', 'Validation precision'))):
        ax.set_facecolor(SURF)
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'): ax.spines[sp].set_color(BASE)
        ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(colors=TXT2, length=0)
        for d, name, c, ls in runs:
            h = pd.read_csv(f'weights/linemodel/{d}/history.csv')
            best = (h.val_recall + h.val_precision).idxmax()
            ax.plot(h.epoch, h[col], color=c, ls=ls, lw=2, label=name)
            ax.scatter([h.epoch[best]], [h[col][best]], s=50, color=c, edgecolor=SURF, linewidth=1.5, zorder=3)
        ax.set_ylim(0.4, 1.0); ax.set_xlabel('epoch', color=TXT2); ax.set_title(lab, loc='left', fontsize=11.5, fontweight='bold')
    axes[0].legend(frameon=False, fontsize=9, loc='lower right')
    fig.text(0.01, 0.01, 'Dot = saved checkpoint (best validation recall + precision). Validation slices differ between runs (they are redrawn as data are added), '
             'so curves compare shape, not level.', fontsize=8, color=MUTED)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.2, wspace=0.18)
    p = os.path.join(out, 'training_curves.png'); fig.savefig(p, dpi=160, facecolor=SURF); plt.close(fig); print('->', p)


def models_by_script(out):
    """One panel per metric, stacked: scripts along x, the three models side by side; Heidelberg set apart."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    kr = script_stats(f'{CMP}/kraken/evaluation.csv', 'kraken_ft_merged'); dk = script_stats(f'{CMP}/dencker/evaluation.csv', 'dencker_none')
    us = script_stats(f'{CMP}/run3_full_s0/evaluation.csv', 'unet')
    kr['heid'] = heid_stats(f'{HEID}/kraken/eval_ft_merged/summary.json'); dk['heid'] = heid_stats(f'{HEID}/dencker/eval_none/summary.json')
    us['heid'] = heid_stats(f'{HEID}/eval_run3_full_s0/summary.json')
    groups = [('all', 'All\nscripts'), ('NA', 'Neo-\nAssyrian'), ('NB', 'Neo-\nBabylonian'), ('OA', 'Old\nAssyrian'), ('OB', 'Old\nBabylonian'),
              ('small', 'Small faces\n(< 60 px/line)'), ('heid', 'Hand-drawn\nNeo-Assyrian lines')]
    models = [('line model (run 3)', '#2a78d6', us, True), ('Kraken, retrained, pieces joined', '#eb6834', kr, True), ('Dencker et al. network, rebuilt', MUTED, dk, False)]
    metrics = [('recall', 'Recall', 'share of the truth lines that were found (within 0.35 line spacing)'),
               ('precision', 'Precision', 'share of the predicted lines that are real lines (faces with >= 90% of signs annotated)'),
               ('first', 'First line placed', 'share of faces whose top predicted line is within half a line spacing of the first truth line')]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'text.color': TXT})
    fig, axes = plt.subplots(3, 1, figsize=(13, 12.5), facecolor=SURF, sharex=True)
    xs = [0, 1, 2, 3, 4, 5, 6.6]               # gap before the Heidelberg group (a different test set)
    off = [-0.27, 0, 0.27]; bw = 0.25           # bar width; 0.02 gap between adjacent bars
    for ax, (mk, mname, mdesc) in zip(axes, metrics):
        ax.set_facecolor(SURF)
        ax.axvspan(6.1, 7.1, color='#f0efec', zorder=0)
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'): ax.spines[sp].set_color(BASE)
        ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0, colors=TXT2)
        for x in xs[:-2]: ax.axvline(x + 0.5, color=GRID, lw=0.6, zorder=0)
        for (name, color, data, labelled), dx in zip(models, off):
            for (key, _), x in zip(groups, xs):
                v = data.get(key, {}).get(mk)
                if v is None or not np.isfinite(v): continue
                ax.bar(x + dx, v, width=bw, color=color, zorder=2)
                ax.text(x + dx, v + 0.015, f'{v:.0%}', ha='center', va='bottom', fontsize=8.5 if labelled else 7.5, color=TXT if labelled else TXT2)
        ax.set_ylim(0, 1.1); ax.set_yticks([0, .25, .5, .75, 1]); ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], color=MUTED, fontsize=9)
        ax.set_title(f'{mname}', loc='left', fontsize=13, fontweight='bold', color=TXT, pad=22)
        ax.text(0, 1.035, mdesc, transform=ax.transAxes, fontsize=9.5, color=TXT2)
    axes[-1].set_xticks(xs); axes[-1].set_xticklabels([g for _, g in groups], color=TXT2, fontsize=9.5)
    axes[0].text(6.6, 1.02, 'independent test set', ha='center', fontsize=8.5, color=TXT2, transform=axes[0].get_xaxis_transform())
    fig.suptitle('Three line models on the same test faces', x=0.01, ha='left', fontsize=15, fontweight='bold', color=TXT, y=0.995)
    fig.legend(handles=[Line2D([], [], color=c, lw=6, label=n) for n, c, _, _ in models], loc='upper left', bbox_to_anchor=(0.005, 0.975),
               ncol=3, frameon=False, fontsize=10, labelcolor=TXT)
    fig.text(0.01, 0.008, 'eBL test faces: about 760 (NA 160, NB 183, OA 90, OB 327; small faces 89). '
             'Hand-drawn lines: the 9 held-out Heidelberg test faces.\nKraken and the line model were trained on the same 490 faces, the Dencker network on 32 Heidelberg faces; '
             'Kraken\'s side-by-side line pieces were joined with the line model\'s own rule.', fontsize=8, color=MUTED, va='bottom')
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.11, hspace=0.42)
    path = os.path.join(out, 'models_by_script.png'); fig.savefig(path, dpi=160, facecolor=SURF); plt.close(fig); print('->', path)


# ------------------------------------------------------------------ example grids
def read_segs(path):
    if not os.path.exists(path):
        return []
    return [np.array([[float(r['ax0']), float(r['ay0'])], [float(r['ax1']), float(r['ay1'])]]) for r in csv.DictReader(open(path))]


def panel(img, truth, pred, label, box=520):
    """Face resized to fit a box x box canvas, then truth (green) and prediction (red) drawn at a fixed width."""
    H, W = img.shape[:2]; s = min(box / W, box / H)
    vis = cv2.resize(img, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    for p in truth: cv2.polylines(vis, [np.round(np.asarray(p) * s).astype(np.int32)], False, (60, 200, 60), 4)
    for p in pred: cv2.polylines(vis, [np.round(np.asarray(p) * s).astype(np.int32)], False, (40, 40, 230), 2)
    canvas = np.full((vis.shape[0], box, 3), 252, np.uint8); x0 = (box - vis.shape[1]) // 2   # height = the face's own
    canvas[:, x0:x0 + vis.shape[1]] = vis
    bar = np.full((28, box, 3), 245, np.uint8)
    cv2.putText(bar, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    return np.vstack([bar, canvas])


def grid(rows, path, headers=None, title=None):
    """rows: list of lists of equal-size panels; headers: one line of text per row."""
    out = []
    if title:
        t = np.full((40, 10, 3), 252, np.uint8); out.append(('title', title))
    blocks = []
    for k, r in enumerate(rows):
        h = max(p.shape[0] for p in r)
        row = np.hstack([cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, 10, cv2.BORDER_CONSTANT, value=(252, 252, 251)) for p in r])
        if headers:
            hb = np.full((34, row.shape[1], 3), 252, np.uint8)
            cv2.putText(hb, headers[k], (4, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (10, 10, 10), 1, cv2.LINE_AA)
            row = np.vstack([hb, row])
        blocks.append(cv2.copyMakeBorder(row, 0, 16, 0, 0, cv2.BORDER_CONSTANT, value=(252, 252, 251)))
    img = np.vstack(blocks)
    if title:
        tb = np.full((44, img.shape[1], 3), 252, np.uint8)
        cv2.putText(tb, title, (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2, cv2.LINE_AA)
        img = np.vstack([tb, img])
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 86]); print('->', path)


def score_label(name, n, r=None, p=None):
    out = f'{name}: {n} lines'
    if r is not None and np.isfinite(r):
        out += f', R {r:.2f}'
    if p is not None and np.isfinite(p):
        out += f' P {p:.2f}'
    return out


def ebl_examples(seed, scripts, need_dirs):
    faces = {f['face']: f for f in csv.DictReader(open(os.path.join(EBL, 'faces', 'faces.csv')))}
    ev0 = read_eval(f'{CMP}/run3_full_s0/evaluation.csv'); ev0 = ev0[ev0.method == 'unet']
    random.seed(seed); pick = []
    for sc in scripts:
        for cov, lo, hi in ((0.9, 6, 16), (0.8, 4, 25)):     # strict first; relaxed where a script has no such face (NA)
            ev = ev0[(ev0.script == sc) & (ev0.coverage >= cov) & ev0.n_truth.between(lo, hi)]
            cand = sorted(f for f in ev.face if all(os.path.exists(os.path.join(d, f + '.csv')) for d in need_dirs))
            if cand:
                break
        pick.append(random.choice(cand))
    return [(f, faces[f]) for f in pick]


def truth_ebl(face):
    return read_segs(os.path.join(EBL, 'faces', 'lines', face + '.csv'))


def runs_examples(out, seed):
    sets = [('run1', 'run 1'), ('run2', 'run 2'), ('run3_full_s0', 'run 3')]
    ev = {d: read_eval(f'{CMP}/{d}/evaluation.csv').query("method == 'unet'").set_index('face') for d, _ in sets}
    rows, heads = [], []
    for face, f in ebl_examples(seed, ['OA', 'NB', 'NA'], [f'{CMP}/{d}/unet_segments' for d, _ in sets]):
        img = cv2.imread(f['file']); t = truth_ebl(face); r = []
        for d, name in sets:
            segs = read_segs(f'{CMP}/{d}/unet_segments/{face}.csv'); e = ev[d].loc[face] if face in ev[d].index else None
            r.append(panel(img, t, segs, score_label(name, len(segs), e.recall if e is not None else None, e.precision if e is not None else None)))
        rows.append(r); heads.append(f"{face.replace('__', ' ')}  ({f['script']}, {len(t)} annotated lines, coverage {float(f['coverage']):.2f})")
    grid(rows, os.path.join(out, 'runs_examples.jpg'), heads, 'Line model runs on the same test faces (green = truth line, red = prediction; R recall, P precision on the face)')


def our_heid_segments(faces, weights):
    """Run the line model on Heidelberg face crops (its Heidelberg scores were computed on the fly, not stored)."""
    import torch
    from line_cropping.linemodel.infer import load_model, predict_face, read_lines, tablet_mask_for
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; m = load_model(weights, dev); out = {}
    for face in faces:
        im = cv2.imread(os.path.join(HEID, 'faces', face + '.jpg')); maps, s = predict_face(m, im, dev)
        segs = read_lines(maps, s, 0.5, 0.4, 12, 1.5, 45.0, 0.5, 0.2, 3.0, tablet_mask_for(im, maps.shape, s))
        out[face] = [np.array([[g['ax0'], g['ay0']], [g['ax1'], g['ay1']]]) for g in segs]
    return out


def models_examples(out, seed):
    cols = [('Dencker', 'dencker_none'), ('Kraken raw', 'kraken_ft'), ('Kraken joined', 'kraken_ft_merged'), ('line model', 'unet')]
    rows, heads = [], []
    ebl_dirs = {'dencker_none': (f'{CMP}/dencker', 'dencker_none'), 'kraken_ft': (f'{CMP}/kraken', 'kraken_ft'),
                'kraken_ft_merged': (f'{CMP}/kraken', 'kraken_ft_merged'), 'unet': (f'{CMP}/run3_full_s0', 'unet')}
    ev = {m: read_eval(f'{d}/evaluation.csv').query(f"method == '{mm}'").set_index('face') for m, (d, mm) in ebl_dirs.items()}
    for face, f in ebl_examples(seed + 1, ['OB', 'NB'], [f'{d}/{mm}_segments' for d, mm in ebl_dirs.values()]):
        img = cv2.imread(f['file']); t = truth_ebl(face); r = []
        for name, m in cols:
            d, mm = ebl_dirs[m]; segs = read_segs(f'{d}/{mm}_segments/{face}.csv'); e = ev[m].loc[face] if face in ev[m].index else None
            r.append(panel(img, t, segs, score_label(name, len(segs), e.recall if e is not None else None, e.precision if e is not None else None)))
        rows.append(r); heads.append(f"eBL test: {face.replace('__', ' ')}  ({f['script']}, {len(t)} annotated lines)")
    ann = pd.read_csv('data/heidelberg/line_annotations_test.csv')
    hfaces = sorted(f for f in {f'{t}__s{s}' for t, s in zip(ann.tablet_CDLI, ann.segm_idx)} if os.path.exists(os.path.join(HEID, 'faces', f + '.jpg')))
    random.seed(seed + 2); tabs = sorted({f.rsplit('__s', 1)[0] for f in hfaces}); hpick = [random.choice([f for f in hfaces if f.startswith(t + '__s')]) for t in random.sample(tabs, 2)]
    ours = our_heid_segments(hpick, 'weights/linemodel/run3_full_s0/model.pt')
    hdirs = {'dencker_none': (f'{HEID}/dencker/dencker_none_segments', f'{HEID}/dencker/eval_none'), 'kraken_ft': (f'{HEID}/kraken/kraken_ft_segments', f'{HEID}/kraken/eval_ft'),
             'kraken_ft_merged': (f'{HEID}/kraken/kraken_ft_merged_segments', f'{HEID}/kraken/eval_ft_merged'), 'unet': (None, f'{HEID}/eval_run3_full_s0')}
    hev = {m: pd.read_csv(f'{e}/faces.csv').set_index('face') for m, (_, e) in hdirs.items()}
    for face in hpick:
        tab, si = face.rsplit('__s', 1); g = ann[(ann.tablet_CDLI == tab) & (ann.segm_idx == int(si))]
        t = [gg[['x', 'y']].to_numpy() for _, gg in g.groupby('line_idx')]
        img = cv2.imread(os.path.join(HEID, 'faces', face + '.jpg')); r = []
        for name, m in cols:
            segs = ours[face] if m == 'unet' else read_segs(os.path.join(hdirs[m][0], face + '.csv'))
            e = hev[m].loc[face] if face in hev[m].index else None
            r.append(panel(img, t, segs, score_label(name, len(segs), e.recall if e is not None else None, e.precision if e is not None else None)))
        rows.append(r); heads.append(f"Heidelberg test (hand-drawn lines): {tab} face {si}  (Neo-Assyrian, {len(t)} drawn lines)")
    grid(rows, os.path.join(out, 'models_examples.jpg'), heads, 'Three models on the same test faces (green = truth line, red = prediction; R recall, P precision on the face)')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='docs/line_model/img'); ap.add_argument('--seed', type=int, default=3)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    runs_by_script(a.out); training_curves(a.out); models_by_script(a.out); runs_examples(a.out, a.seed); models_examples(a.out, a.seed)


if __name__ == '__main__':
    main()
