"""Where the pipeline's data lives. Every module builds its data paths from these constants.

DATA_ROOT defaults to tablet_data/ (relative to the repository root, where all commands are run) and can be moved by
setting the environment variable LINECROP_DATA_ROOT, e.g. `LINECROP_DATA_ROOT=/mnt/e/tablets python -m ...`.
Tables written by the pipeline (faces.csv, crops.csv, ...) store image paths that start with DATA_ROOT; after moving
the data, rewrite those with `python -m line_cropping.paths --rewrite OLD_ROOT`.
"""
import os

DATA_ROOT = os.environ.get('LINECROP_DATA_ROOT', 'tablet_data')
EBL = os.path.join(DATA_ROOT, 'ebl_tablets')            # eBL downloads, ground truth, faces, predictions, crops
OA = os.path.join(DATA_ROOT, 'fat-cross_processed')     # CDLI Old Assyrian faces and everything derived from them
APP_SESSIONS = os.path.join(DATA_ROOT, 'app_sessions')  # Line Splitter uploads and results
PROTOSNAP_INPUTS = os.path.join(DATA_ROOT, 'protosnap_inputs')  # line crops prepared as ProtoSnap targets (protosnap_prep)


def rewrite_tables(old_prefixes, new_root=DATA_ROOT, top=DATA_ROOT, dry_run=False):
    """Rewrite stored paths in the CSV tables under `top`: any cell starting with one of old_prefixes (e.g.
    'ebl_tablets/') is re-rooted under new_root. Returns [(file, cells changed)]."""
    import csv
    changed = []
    for dp, _, files in os.walk(top):
        for fn in files:
            if not fn.endswith('.csv'):
                continue
            p = os.path.join(dp, fn)
            with open(p, encoding='utf-8', newline='') as fh:
                text = fh.read()
            if not any(pre in text for pre in old_prefixes):
                continue
            rows = list(csv.reader(text.splitlines()))
            n = 0
            for r in rows:
                for i, cell in enumerate(r):
                    for pre in old_prefixes:
                        if cell.startswith(pre):
                            r[i] = os.path.join(new_root, cell) if not cell.startswith(new_root + '/') else cell
                            n += 1
                            break
            if n and not dry_run:
                tmp = p + '.tmp'
                with open(tmp, 'w', encoding='utf-8', newline='') as fh:
                    csv.writer(fh).writerows(rows)
                os.replace(tmp, p)
            if n:
                changed.append((p, n))
    return changed


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Show the data locations, or re-root stored paths after moving the data.')
    ap.add_argument('--rewrite', nargs='*', default=None, help="old path prefixes to re-root under DATA_ROOT, e.g. 'ebl_tablets/' 'fat-cross_processed/'")
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    print(f'DATA_ROOT {DATA_ROOT}\nEBL       {EBL}\nOA        {OA}\nAPP       {APP_SESSIONS}\nPROTOSNAP {PROTOSNAP_INPUTS}')
    if a.rewrite:
        res = rewrite_tables(a.rewrite, dry_run=a.dry_run)
        for p, n in res:
            print(f'{"would rewrite" if a.dry_run else "rewrote"} {n:6d} cells in {p}')
        print(f'{len(res)} tables' + (' (dry run)' if a.dry_run else ''))
