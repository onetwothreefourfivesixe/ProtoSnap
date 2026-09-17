"""Download eBL tablet photos, sign annotations and transliterations for the repo's Neo-Assyrian and
Old Babylonian tablets. Resumable: existing files are skipped. Run from the repo root.

Output: ebl_tablets/<NA|OB>/<museum number>/{photo.jpg, annotations.json, fragment.json}
        ebl_tablets/manifest.csv  (one row per tablet: script, eBL period, photo size, #boxes, #atf lines)
"""
import json, csv, os, sys, time, collections, urllib.request, urllib.error
API = 'https://www.ebl.lmu.de/api/fragments/'
OUT = 'ebl_tablets'
SCRIPTS = {'NA', 'OB'}
HEADERS = {'User-Agent': 'ProtoSnap-research (contact: edwardxuming.lin@gmail.com)'}

def get(url, binary=False, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=120) as r:
                data = r.read()
            return data if binary else json.loads(data)
        except urllib.error.HTTPError as e:
            if e.code == 404: return None
            time.sleep(5 * (i + 1))
        except Exception:
            time.sleep(5 * (i + 1))
    return None

def main():
    meta = json.load(open('signs_snippets_metadata.json'))
    tabs = collections.defaultdict(lambda: {'lab': 0, 'script': collections.Counter()})
    for x in meta:
        t = tabs[x['tabletNumber']]; t['lab'] += x['label'] != '[]'; t['script'][x['script']] += 1
    cands = [(t, v['script'].most_common(1)[0][0], v['lab']) for t, v in tabs.items()
             if v['lab'] > 0 and v['script'].most_common(1)[0][0] in SCRIPTS]
    cands.sort(key=lambda c: -c[2])           # densest annotations first
    print(f'{len(cands)} candidate tablets', flush=True)
    man_path = os.path.join(OUT, 'manifest.csv')
    done = {r['tablet'] for r in csv.DictReader(open(man_path))} if os.path.exists(man_path) else set()
    fields = ['tablet', 'script', 'ebl_period', 'ebl_genres', 'photo_bytes', 'n_annotations', 'n_sign_boxes', 'atf_lines', 'status']
    new = not os.path.exists(man_path)
    with open(man_path, 'a', newline='', encoding='utf-8') as mf:
        w = csv.DictWriter(mf, fieldnames=fields)
        if new: w.writeheader()
        t0 = time.time()
        for i, (tab, script, nlab) in enumerate(cands):
            if tab in done: continue
            d = os.path.join(OUT, script, tab); os.makedirs(d, exist_ok=True)
            row = dict(tablet=tab, script=script, ebl_period='', ebl_genres='', photo_bytes=0, n_annotations=0, n_sign_boxes=0, atf_lines=0, status='ok')
            frag = get(API + tab)
            if frag is None: row['status'] = 'no_fragment'
            else:
                keep = {k: frag.get(k) for k in ('museumNumber', 'atf', 'script', 'genres', 'collection', 'museum', 'joins', 'externalNumbers', 'hasPhoto', 'width', 'length', 'thickness')}
                keep['numberOfLines'] = (frag.get('text') or {}).get('numberOfLines')
                json.dump(keep, open(os.path.join(d, 'fragment.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
                row['ebl_period'] = (frag.get('script') or {}).get('period', '')
                row['ebl_genres'] = ';'.join('/'.join(g.get('category', [])) for g in frag.get('genres') or [])
                row['atf_lines'] = sum(1 for l in (frag.get('atf') or '').splitlines() if l[:1].isdigit())
                ann = get(API + tab + '/annotations')
                if ann is not None:
                    json.dump(ann, open(os.path.join(d, 'annotations.json'), 'w', encoding='utf-8'), ensure_ascii=False)
                    A = ann.get('annotations', []); row['n_annotations'] = len(A)
                    row['n_sign_boxes'] = sum(1 for a in A if a['data'].get('type') in ('HasSign', 'Number', 'Damaged', 'PartiallyBroken', 'UnclearSign'))
                pp = os.path.join(d, 'photo.jpg')
                if not os.path.exists(pp):
                    img = get(API + tab + '/photo', binary=True)
                    if img: open(pp, 'wb').write(img)
                    else: row['status'] = 'no_photo'
                row['photo_bytes'] = os.path.getsize(pp) if os.path.exists(pp) else 0
            w.writerow(row); mf.flush()
            time.sleep(0.5)
            if (i + 1) % 25 == 0: print(f'{i+1}/{len(cands)} done, {time.time()-t0:.0f}s', flush=True)
    print('DONE', flush=True)

if __name__ == '__main__':
    main()
