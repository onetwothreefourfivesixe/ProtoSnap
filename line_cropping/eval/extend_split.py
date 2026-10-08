"""Add faces that are not yet in the Phase 5 split (tablet_data/ebl_tablets/split.csv) without moving any existing face.

  python -m line_cropping.eval.extend_split [--seed 7] [--dry-run]

phase5_eval.make_split rewrites the whole split, which would reshuffle faces between tune and test and leak test faces
into the line model's training set. This script only appends. New faces must be single-column (n_columns == 1), as in
the original split. They are split by tablet, not by face, so the obverse and reverse of one tablet (same photograph,
same scribe) land on the same side: tablets are grouped by script and pixels-per-line bucket of their first face,
shuffled with the seed, and alternated tune/test within each group. Faces of a tablet that already has faces in the
split follow that tablet's existing side (its majority, ties to test).
"""
import argparse
import collections
import csv
import os
import random

from line_cropping.eval.phase5_eval import bucket
from line_cropping.paths import EBL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    path = os.path.join(EBL, 'split.csv')
    old = list(csv.DictReader(open(path))) if os.path.exists(path) else []
    known = {r['face'] for r in old}
    faces = [f for f in csv.DictReader(open(os.path.join(EBL, 'faces', 'faces.csv'))) if f['n_columns'] == '1']
    new = [f for f in faces if f['face'] not in known]
    tab_of = {f['face']: f['tablet'] for f in csv.DictReader(open(os.path.join(EBL, 'faces', 'faces.csv')))}
    prior = collections.defaultdict(collections.Counter)
    for r in old:
        prior[tab_of.get(r['face'], r['face'])][r['split']] += 1
    by_tab = collections.defaultdict(list)
    for f in new:
        by_tab[f['tablet']].append(f)
    side = {}
    groups = collections.defaultdict(list)
    for t, fs in by_tab.items():
        if prior[t]:
            side[t] = 'tune' if prior[t]['tune'] > prior[t]['test'] else 'test'
        else:
            groups[bucket(fs[0])].append(t)
    random.seed(a.seed)
    for k in sorted(groups):
        ts = sorted(groups[k])
        random.shuffle(ts)
        for i, t in enumerate(ts):
            side[t] = 'tune' if i % 2 == 0 else 'test'
    rows = [dict(face=f['face'], split=side[f['tablet']], bucket=bucket(f)) for f in new]
    c = collections.Counter((r['bucket'].split('/')[0], r['split']) for r in rows)
    print(f'{len(old)} faces already in the split (unchanged); {len(rows)} new faces from {len(by_tab)} tablets')
    for sc in sorted({k[0] for k in c}):
        print(f'  {sc}: tune {c[(sc, "tune")]}, test {c[(sc, "test")]}')
    if a.dry_run or not rows:
        return
    with open(path, 'a' if old else 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['face', 'split', 'bucket'])
        if not old:
            w.writeheader()
        w.writerows(rows)
    print('->', path)


if __name__ == '__main__':
    main()
