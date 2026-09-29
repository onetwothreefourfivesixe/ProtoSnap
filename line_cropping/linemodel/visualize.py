"""Plots of the line model's training statistics.

Each plot_* function takes the rows of history.csv (one dict per epoch, as written by train.py) and a file path,
and writes one PNG. plot_history() writes them all for a run folder and is what train.py calls after every epoch,
so the graphs in weights/linemodel/<run>/ stay current while training runs.

  python -m line_cropping.linemodel.visualize --run weights/linemodel/run1     # regenerate plots from history.csv
"""
import argparse, csv, os
import matplotlib
matplotlib.use('Agg')                     # headless: never open a window
import matplotlib.pyplot as plot


def _col(rows, key):
    return [float(r[key]) for r in rows if r.get(key) not in (None, '', 'nan')]


def _epochs(rows, key):
    return [int(float(r['epoch'])) for r in rows if r.get(key) not in (None, '', 'nan')]


def _finish(title, ylabel, filePath, legend=False):
    plot.title(title); plot.xlabel('epoch'); plot.ylabel(ylabel); plot.grid(True, alpha=0.3)
    if legend: plot.legend()
    plot.tight_layout(); plot.savefig(filePath, dpi=110); plot.close()


def plot_loss(epoch_and_loss, filePath):
    """Total training loss per epoch. Accepts history rows or a list of (epoch, loss) pairs."""
    if epoch_and_loss and isinstance(epoch_and_loss[0], dict):
        epochs, loss = _epochs(epoch_and_loss, 'loss'), _col(epoch_and_loss, 'loss')
    else:
        epochs = [pair[0] for pair in epoch_and_loss]; loss = [pair[1] for pair in epoch_and_loss]
    plot.figure(figsize=(6, 4)); plot.plot(epochs, loss, marker='o', ms=3)
    _finish('Training loss for UNet', 'loss', filePath)


def plot_loss_parts(rows, filePath):
    """The three loss components: kernel and band (BCE + Dice each) and orientation (L1, weighted 0.5 in the total)."""
    plot.figure(figsize=(6, 4))
    for key, label in (('kernel', 'kernel (BCE + Dice)'), ('band', 'band (BCE + Dice)'), ('orient', 'orientation (L1)')):
        plot.plot(_epochs(rows, key), _col(rows, key), marker='o', ms=3, label=label)
    _finish('Loss components', 'loss', filePath, legend=True)


def plot_validation(rows, filePath):
    """Validation line recall and precision per epoch (from the real read-out on the validation faces)."""
    plot.figure(figsize=(6, 4))
    plot.plot(_epochs(rows, 'val_recall'), _col(rows, 'val_recall'), marker='o', ms=3, label='recall')
    plot.plot(_epochs(rows, 'val_precision'), _col(rows, 'val_precision'), marker='o', ms=3, label='precision')
    plot.ylim(0, 1.02); _finish('Validation lines matched within 0.35 pitch', 'share of lines', filePath, legend=True)


def plot_precision(rows, filePath):
    plot.figure(figsize=(6, 4)); plot.plot(_epochs(rows, 'val_precision'), _col(rows, 'val_precision'), marker='o', ms=3, color='tab:orange')
    plot.ylim(0, 1.02); _finish('Validation precision', 'precision', filePath)


def plot_recall(rows, filePath):
    plot.figure(figsize=(6, 4)); plot.plot(_epochs(rows, 'val_recall'), _col(rows, 'val_recall'), marker='o', ms=3, color='tab:blue')
    plot.ylim(0, 1.02); _finish('Validation recall', 'recall', filePath)


def plot_selection(rows, filePath):
    """Recall + precision, the quantity that selects the saved checkpoint; the best epoch is marked."""
    e, s = _epochs(rows, 'val_recall'), [r + p for r, p in zip(_col(rows, 'val_recall'), _col(rows, 'val_precision'))]
    plot.figure(figsize=(6, 4)); plot.plot(e, s, marker='o', ms=3, color='tab:green')
    if s:
        k = max(range(len(s)), key=lambda i: s[i]); plot.scatter([e[k]], [s[k]], s=80, facecolors='none', edgecolors='red', label=f'saved: epoch {e[k]} ({s[k]:.3f})'); plot.legend()
    _finish('Checkpoint selection score (recall + precision)', 'recall + precision', filePath)


def plot_dashboard(rows, filePath):
    """All of the above on one page: loss, loss parts, validation recall/precision, selection score."""
    fig, ax = plot.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].plot(_epochs(rows, 'loss'), _col(rows, 'loss'), marker='o', ms=3); ax[0, 0].set_title('training loss')
    for key in ('kernel', 'band', 'orient'): ax[0, 1].plot(_epochs(rows, key), _col(rows, key), marker='o', ms=3, label=key)
    ax[0, 1].set_title('loss components'); ax[0, 1].legend()
    ax[1, 0].plot(_epochs(rows, 'val_recall'), _col(rows, 'val_recall'), marker='o', ms=3, label='recall'); ax[1, 0].plot(_epochs(rows, 'val_precision'), _col(rows, 'val_precision'), marker='o', ms=3, label='precision')
    ax[1, 0].set_ylim(0, 1.02); ax[1, 0].set_title('validation, lines within 0.35 pitch'); ax[1, 0].legend()
    s = [r + p for r, p in zip(_col(rows, 'val_recall'), _col(rows, 'val_precision'))]; e = _epochs(rows, 'val_recall')
    ax[1, 1].plot(e, s, marker='o', ms=3, color='tab:green'); ax[1, 1].set_title('recall + precision (selects model.pt)')
    if s:
        k = max(range(len(s)), key=lambda i: s[i]); ax[1, 1].scatter([e[k]], [s[k]], s=80, facecolors='none', edgecolors='red')
    for a in ax.ravel(): a.set_xlabel('epoch'); a.grid(True, alpha=0.3)
    m = _col(rows, 'minutes'); fig.suptitle(f'{os.path.basename(os.path.dirname(os.path.dirname(filePath)))}: {len(rows)} epochs' + (f', {m[-1]:.0f} min' if m else ''))
    plot.tight_layout(); plot.savefig(filePath, dpi=110); plot.close(fig)


def plot_history(run_dir):
    """Read <run_dir>/history.csv and write every plot into <run_dir>/plots/. Returns the list of files written."""
    hp = os.path.join(run_dir, 'history.csv')
    if not os.path.exists(hp): return []
    rows = list(csv.DictReader(open(hp)))
    if not rows: return []
    out = os.path.join(run_dir, 'plots'); os.makedirs(out, exist_ok=True); written = []
    for fn, f in (('loss.png', plot_loss), ('loss_parts.png', plot_loss_parts), ('validation.png', plot_validation), ('recall.png', plot_recall), ('precision.png', plot_precision), ('selection.png', plot_selection), ('dashboard.png', plot_dashboard)):
        p = os.path.join(out, fn); f(rows, p); written.append(p)
    return written


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); a = ap.parse_args()
    files = plot_history(a.run); print(f'wrote {len(files)} plots to {os.path.join(a.run, "plots")}' if files else f'no history.csv in {a.run}')
