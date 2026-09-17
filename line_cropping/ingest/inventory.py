"""Rebuild fat-cross_processed/inventory.csv: one row per tablet with its faces, CDLI metadata and ATF flag."""
import os, re, csv, json, collections

def main():
    ROOT = 'fat-cross_processed/fat-cross_processed'
    faces = collections.defaultdict(dict); sub = collections.defaultdict(list); other = []
    for dp, dn, fn in os.walk(ROOT):
        rel = os.path.relpath(dp, ROOT)
        for f in fn:
            m = re.match(r'^(P\d{6})_(\w+)\.(png|jpg|jpeg)$', f, re.I)
            if not m: other.append(os.path.join(rel, f)); continue
            if rel == '.': faces[m[1]][m[2].lower()] = f
            else: sub[rel].append(f)
    idx = {r['p_number']: r for r in csv.DictReader(open('artifacts_json/artifacts_index.csv', encoding='utf-8'))}
    m2p = collections.defaultdict(set)
    for r in csv.DictReader(open('artifacts_json/protosnap_tablets_to_cdli.csv', encoding='utf-8')):
        if r['p_number']: m2p[r['p_number']].add(r['tabletNumber'])
    lab = {x['tabletNumber'] for x in json.load(open('signs_snippets_metadata.json')) if x['label'] != '[]'}
    with open('fat-cross_processed/inventory.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['p_number', 'faces', 'has_obverse', 'has_reverse', 'in_cdli', 'has_atf', 'period', 'genre', 'museum_no', 'accession_no', 'ebl_tablet'])
        for p in sorted(faces):
            v = faces[p]; r = idx.get(p, {})
            w.writerow([p, '|'.join(sorted(k for k in v if k != 'layout')), 'obverse' in v, 'reverse' in v, p in idx,
                        r.get('has_atf', ''), r.get('period', ''), r.get('genre', ''), r.get('museum_no', ''), r.get('accession_no', ''),
                        '|'.join(sorted(t for t in m2p.get(p, ()) if t in lab))])
    print('tablets:', len(faces), '| with obverse:', sum('obverse' in v for v in faces.values()),
          '| with reverse:', sum('reverse' in v for v in faces.values()), '| in CDLI:', sum(p in idx for p in faces),
          '| with ATF:', sum(idx.get(p, {}).get('has_atf') == 'True' for p in faces),
          '| linked to eBL-annotated tablet:', sum(any(t in lab for t in m2p.get(p, ())) for p in faces))
    print('periods:', collections.Counter(idx[p]['period'][:24] for p in faces if p in idx).most_common(10))
    print('genres:', collections.Counter(idx[p]['genre'][:16] for p in faces if p in idx).most_common(6))
    print('face suffixes:', collections.Counter(k for v in faces.values() for k in v).most_common(10))
    for k, v in sub.items(): print('subfolder', k, len(v), 'files')
    print('non-matching filenames:', len(other), other[:5])



if __name__ == '__main__':
    main()
