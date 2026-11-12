"""The faces each line model was trained and validated on, with their CDLI P-numbers.

  python -m line_segmentation.linemodel.list_training_faces [--runs weights/linemodel/run4_s0 ...] [--out <DATA_ROOT>/line_model_training_faces.csv]

Each run's faces come from <run>/training_faces.csv, which train.py writes when training starts. For a run without one,
the training index is rebuilt with the run's settings (config.json: seed, min_cov, no_oa, heidelberg); that is right only
if the faces and split have not changed since the run trained. Faces are listed once, with a column per run giving their
role: train (used to fit the weights), val (used to pick the checkpoint) or blank (not used). segment/segment_images.py
reads the column named after the weights' folder. The P-number comes from:
  eBL faces          the tablet's fragment.json (externalNumbers.cdliNumber), else the museum-number match to the CDLI dump
                     (data/cdli/artifacts_json/protosnap_tablets_to_cdli.csv, one row only when the match is unique)
  Old Assyrian faces the tablet id, which is the P-number
  Heidelberg faces   the tablet id after its 'heid-' prefix
Any face still without one is looked up by its museum number (letters and number of the tablet id, such as VAT08803 or
HS.2079) in the CDLI dump's index (data/cdli/artifacts_json/artifacts_index.csv). When that finds several records, all are
listed, separated by ';'.
"""
import argparse
import collections
import csv
import glob
import json
import os
import re

from line_segmentation.linemodel.dataset import build_index
from line_segmentation.paths import DATA_ROOT, EBL

DEFAULT_RUNS = ['run4_s0', 'run4_s1', 'run5_s0', 'run6_heights_s0', 'run6_noheights_s0', 'run6_heights_s1', 'run6_noheights_s1', 'run7_heights_s1']


def ebl_p_numbers():
    out = {}
    m = collections.defaultdict(set)
    p = 'data/cdli/artifacts_json/protosnap_tablets_to_cdli.csv'
    if os.path.exists(p):
        for r in csv.DictReader(open(p, encoding='utf-8')):
            if r['p_number']: m[r['tabletNumber']].add(r['p_number'])
    for t, ps in m.items():
        if len(ps) == 1: out[t] = (next(iter(ps)), 'museum-number match')
    for f in glob.glob(os.path.join(EBL, '*', '*', 'fragment.json')):
        c = (json.load(open(f, encoding='utf-8')).get('externalNumbers') or {}).get('cdliNumber')
        if c and re.fullmatch(r'P\d{6}', c.strip()):
            out[os.path.basename(os.path.dirname(f))] = (c.strip(), 'eBL fragment.json')
    return out


def museum_lookup(tablets, index='data/cdli/artifacts_json/artifacts_index.csv'):
    """{tablet id: [P-numbers]} for ids like VAT08803, VAT13604Rs or HS.2079, matched to CDLI museum or accession numbers."""
    keys = {}
    for t in tablets:
        m = re.match(r'(?:heid-)?([A-Za-z]+)[ .]?0*(\d+)', t)
        if m: keys[t] = (m.group(1), int(m.group(2)))
    want = collections.defaultdict(list)
    for t, k in keys.items(): want[k].append(t)
    out = collections.defaultdict(list)
    if not want or not os.path.exists(index):
        return out
    for r in csv.DictReader(open(index, encoding='utf-8')):
        for v in (r.get('museum_no') or '', r.get('accession_no') or ''):
            for pre, num in re.findall(r'\b([A-Za-z]+)\s+0*(\d+)\b', v):
                for t in want.get((pre, int(num)), ()):
                    if r['p_number'] not in out[t]: out[t].append(r['p_number'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='*', default=[f'weights/linemodel/{r}' for r in DEFAULT_RUNS])
    ap.add_argument('--out', default=os.path.join(DATA_ROOT, 'line_model_training_faces.csv'))
    a = ap.parse_args()
    pmap = ebl_p_numbers(); faces = {}; roles = collections.defaultdict(dict); counts = {}
    for r in a.runs:
        name = os.path.basename(os.path.normpath(r)); fixed = os.path.join(r, 'training_faces.csv')
        if os.path.exists(fixed):                               # written by train.py when the run started
            items = [dict(x) for x in csv.DictReader(open(fixed, encoding='utf-8'))]
        else:                                                   # older run: rebuild, valid only if the data has not changed since
            c = json.load(open(os.path.join(r, 'config.json')))
            ix = build_index(min_cov=c['min_cov'], seed=c['seed'], include_oa=not c['no_oa'], include_heidelberg=c['heidelberg'])
            items = [dict(it, role=role) for role in ('train', 'val') for it in ix[role]]
            print(f'{name}: no training_faces.csv, rebuilt from the current data (wrong if faces or split changed after training)')
        counts[name] = (sum(x['role'] == 'train' for x in items), sum(x['role'] == 'val' for x in items))
        for it in items:
            faces.setdefault(it['face'], it); roles[it['face']][name] = it['role']
    rows = []
    for fid, it in faces.items():
        src = it['source']; t = it['tablet']
        if src.startswith('ebl'):
            p, how = pmap.get(t, ('', ''))
        elif src == 'oa':
            p, how = (t, 'tablet id') if re.fullmatch(r'P\d{6}', t) else ('', '')
        else:
            q = t.split('heid-', 1)[-1]; p, how = (q, 'tablet id') if re.fullmatch(r'P\d{6}', q) else ('', '')
        rows.append(dict(face=fid, tablet=t, p_number=p, p_number_from=how, source=src.split('-')[0], script=src.split('-')[1] if '-' in src else ('OA' if src == 'oa' else 'NA'),
                         **{k: roles[fid].get(k, '') for k in counts}))
    found = museum_lookup({r['tablet'] for r in rows if not r['p_number']})
    for r in rows:
        if not r['p_number'] and found.get(r['tablet']):
            ps = found[r['tablet']]; r['p_number'] = ';'.join(ps); r['p_number_from'] = 'museum-number lookup' + (f' ({len(ps)} records)' if len(ps) > 1 else '')
    rows.sort(key=lambda r: (r['source'], r['tablet'], r['face']))
    with open(a.out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    for k, (ntr, nva) in counts.items():
        print(f'{k}: {ntr} train, {nva} val faces')
    print(f"{len(rows)} faces, {len({q for r in rows for q in r['p_number'].split(';') if q})} P-numbers, {sum(not r['p_number'] for r in rows)} faces without one -> {a.out}")


if __name__ == '__main__':
    main()
