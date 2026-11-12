"""The faces each line model was trained and validated on, with their CDLI P-numbers.

  python -m line_cropping.linemodel.list_training_faces [--runs weights/linemodel/run4_s0 ...] [--out <DATA_ROOT>/line_model_training_faces.csv]

The training index is rebuilt with each run's settings (config.json: seed, min_cov, no_oa, heidelberg), as train.py builds it, so the
list reflects the data files as they are now. Faces are listed once, with a column per seed giving their role: train (used to fit the
weights), val (used to pick the checkpoint) or blank (not used). The P-number comes from:
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

from line_cropping.linemodel.dataset import build_index
from line_cropping.paths import DATA_ROOT, EBL

DEFAULT_RUNS = ['run4_s0', 'run4_s1', 'run5_s0', 'run6_heights_s0', 'run6_noheights_s0', 'run6_heights_s1', 'run6_noheights_s1']


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
    settings = collections.defaultdict(list)                    # identical settings give identical indexes
    for r in a.runs:
        c = json.load(open(os.path.join(r, 'config.json')))
        settings[(c['seed'], c['min_cov'], not c['no_oa'], c['heidelberg'])].append(os.path.basename(r))
    pmap = ebl_p_numbers(); faces = {}; roles = collections.defaultdict(dict); counts = {}
    for (seed, min_cov, oa, heid), runs in sorted(settings.items()):
        ix = build_index(min_cov=min_cov, seed=seed, include_oa=oa, include_heidelberg=heid)
        key = f'seed{seed}'; counts[key] = (len(ix['train']), len(ix['val']), runs)
        for role in ('train', 'val'):
            for it in ix[role]:
                faces[it['face']] = it; roles[it['face']][key] = role
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
                         **{k: roles[fid].get(k, '') for k in sorted(counts)}))
    found = museum_lookup({r['tablet'] for r in rows if not r['p_number']})
    for r in rows:
        if not r['p_number'] and found.get(r['tablet']):
            ps = found[r['tablet']]; r['p_number'] = ';'.join(ps); r['p_number_from'] = 'museum-number lookup' + (f' ({len(ps)} records)' if len(ps) > 1 else '')
    rows.sort(key=lambda r: (r['source'], r['tablet'], r['face']))
    with open(a.out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    for k, (ntr, nva, runs) in sorted(counts.items()):
        print(f'{k}: {ntr} train, {nva} val faces (runs {", ".join(runs)})')
    print(f"{len(rows)} faces, {len({q for r in rows for q in r['p_number'].split(';') if q})} P-numbers, {sum(not r['p_number'] for r in rows)} faces without one -> {a.out}")


if __name__ == '__main__':
    main()
