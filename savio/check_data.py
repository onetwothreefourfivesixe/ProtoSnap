"""Run on Savio from the repo root, before submitting: every file a run 4 training job reads must be here.
A missing face image is skipped silently by build_index, so a bad sync would otherwise train on less data without an error.

  python -m savio.check_data
"""
import collections
import csv
import os
import sys

from line_segmentation.linemodel.dataset import build_index
from line_segmentation.paths import EBL

EXPECTED = {'ebl-OB': 229, 'ebl-OA': 110, 'oa': 88, 'ebl-NB': 74, 'heid-NA': 32, 'ebl-NA': 23}   # 556 faces, 490 train + 66 validation

problems = []
if not os.path.exists(f'{EBL}/faces/line_heights.csv'):
    sys.exit('faces/line_heights.csv is missing: rerun savio/sync_to_savio.sh')
idx = build_index(include_heidelberg=True, include_heights=True)
items = idx['train'] + idx['val']
found = collections.Counter(it['source'] for it in items)
for src, n in EXPECTED.items():
    if found[src] != n:
        problems.append(f'{src}: {found[src]} training faces, expected {n}')
missing = [it['file'] for it in items if not os.path.exists(it['file'])]
if missing:
    problems.append(f'{len(missing)} face images missing, e.g. {missing[0]}')
with_h = sum(any('up' in s for s in it['segments']) for it in items)
faces = list(csv.DictReader(open(f'{EBL}/faces/faces.csv')))
split = {r['face']: r['split'] for r in csv.DictReader(open(f'{EBL}/split.csv'))}
test = [f for f in faces if split.get(f['face']) == 'test']
no_img = [f['face'] for f in test if not os.path.exists(f['file'])]
no_ann = [f['face'] for f in test if not os.path.exists(os.path.join(EBL, f['script'], f['tablet'], 'annotations.json'))]
if no_img:
    problems.append(f'{len(no_img)} test face images missing, e.g. {no_img[0]}')
if no_ann:
    problems.append(f'{len(no_ann)} test faces without annotations.json, e.g. {no_ann[0]}')
print(f"training faces {len(items)} ({len(idx['train'])} train, {len(idx['val'])} validation): {dict(found)}; {with_h} with heights | test faces {len(test)}")
if problems:
    sys.exit('PROBLEMS:\n  ' + '\n  '.join(problems) + '\nrerun savio/sync_to_savio.sh')
print('data OK')
