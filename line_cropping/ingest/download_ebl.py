"""Download eBL tablet photos, sign annotations and transliterations. Resumable: tablets already in the manifest are
skipped. Run from the repo root.

  python -m line_cropping.ingest.download_ebl [--scripts NA OB] [--annotations-dump annotations_11.09.2026/annotations_11.09.2026.json]
      [--min-boxes 50] [--limit N] [--dry-run]

Candidates and their script come from data/ebl/signs_snippets_metadata.json (majority script of a tablet's snippets).
Without --annotations-dump: tablets with at least one line-labelled snippet, densest first, and the boxes are fetched
from the API (/annotations). With --annotations-dump (the eBL annotation export, one JSON record per line): tablets
with at least --min-boxes boxes that carry a line path, densest first, and the boxes are taken from the dump, so only
the fragment (transliteration, period) and the photo are fetched. --dry-run lists the candidates and stops.

Output: tablet_data/ebl_tablets/<script>/<museum number>/{photo.jpg, annotations.json, fragment.json}
        tablet_data/ebl_tablets/manifest.csv  (one row per tablet: script, eBL period, photo size, #boxes, #atf lines)
"""
from line_cropping.paths import EBL
import argparse, json, csv, os, sys, time, collections, urllib.parse, urllib.request, urllib.error
API = 'https://www.ebl.lmu.de/api/fragments/'
OUT = EBL
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--scripts', nargs='+', default=['NA', 'OB'], help='snippet-metadata script codes to download, e.g. NA OB OA NB')
    ap.add_argument('--annotations-dump', default=None, help='eBL annotation export (JSON lines); boxes are read from it instead of the API')
    ap.add_argument('--min-boxes', type=int, default=50, help='with --annotations-dump: minimum boxes with a line path')
    ap.add_argument('--limit', type=int, default=None); ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(); scripts = set(a.scripts)
    meta = json.load(open('data/ebl/signs_snippets_metadata.json'))
    tabs = collections.defaultdict(lambda: {'lab': 0, 'script': collections.Counter()})
    for x in meta:
        t = tabs[x['tabletNumber']]; t['lab'] += x['label'] != '[]'; t['script'][x['script']] += 1
    script_of = {t: v['script'].most_common(1)[0][0] for t, v in tabs.items()}
    dump = {}
    if a.annotations_dump:
        for l in open(a.annotations_dump, encoding='utf-8'):
            if not l.strip(): continue
            rec = json.loads(l); n = sum(1 for x in rec.get('annotations', []) if (x.get('data') or {}).get('path'))
            if n >= a.min_boxes and script_of.get(rec['fragmentNumber']) in scripts:
                rec.pop('_id', None); dump[rec['fragmentNumber']] = (n, rec)
        cands = [(t, script_of[t], n) for t, (n, _) in dump.items()]
    else:
        cands = [(t, script_of[t], v['lab']) for t, v in tabs.items() if v['lab'] > 0 and script_of[t] in scripts]
    cands.sort(key=lambda c: -c[2])           # densest annotations first
    print(f'{len(cands)} candidate tablets {dict(collections.Counter(c[1] for c in cands))}', flush=True)
    man_path = os.path.join(OUT, 'manifest.csv')
    done = {r['tablet'] for r in csv.DictReader(open(man_path))} if os.path.exists(man_path) else set()
    fields = ['tablet', 'script', 'ebl_period', 'ebl_genres', 'photo_bytes', 'n_annotations', 'n_sign_boxes', 'atf_lines', 'status']
    new = not os.path.exists(man_path)
    with open(man_path, 'a', newline='', encoding='utf-8') as mf:
        w = csv.DictWriter(mf, fieldnames=fields)
        if new: w.writeheader()
        todo = [c for c in cands if c[0] not in done][:a.limit]
        print(f'{len(todo)} to download ({len(cands) - len(todo)} already in the manifest or over --limit)', flush=True)
        if a.dry_run: return
        t0 = time.time()
        for i, (tab, script, nlab) in enumerate(todo):
            d = os.path.join(OUT, script, tab); os.makedirs(d, exist_ok=True)
            row = dict(tablet=tab, script=script, ebl_period='', ebl_genres='', photo_bytes=0, n_annotations=0, n_sign_boxes=0, atf_lines=0, status='ok')
            frag = get(API + urllib.parse.quote(tab))
            if frag is None: row['status'] = 'no_fragment'
            else:
                keep = {k: frag.get(k) for k in ('museumNumber', 'atf', 'script', 'genres', 'collection', 'museum', 'joins', 'externalNumbers', 'hasPhoto', 'width', 'length', 'thickness')}
                keep['numberOfLines'] = (frag.get('text') or {}).get('numberOfLines')
                json.dump(keep, open(os.path.join(d, 'fragment.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
                row['ebl_period'] = (frag.get('script') or {}).get('period', '')
                row['ebl_genres'] = ';'.join('/'.join(g.get('category', [])) for g in frag.get('genres') or [])
                row['atf_lines'] = sum(1 for l in (frag.get('atf') or '').splitlines() if l[:1].isdigit())
                ann = dump[tab][1] if tab in dump else get(API + urllib.parse.quote(tab) + '/annotations')
                if ann is not None:
                    json.dump(ann, open(os.path.join(d, 'annotations.json'), 'w', encoding='utf-8'), ensure_ascii=False)
                    A = ann.get('annotations', []); row['n_annotations'] = len(A)
                    row['n_sign_boxes'] = sum(1 for a in A if a['data'].get('type') in ('HasSign', 'Number', 'Damaged', 'PartiallyBroken', 'UnclearSign'))
                pp = os.path.join(d, 'photo.jpg')
                if not os.path.exists(pp):
                    img = get(API + urllib.parse.quote(tab) + '/photo', binary=True)
                    if img: open(pp, 'wb').write(img)
                    else: row['status'] = 'no_photo'
                row['photo_bytes'] = os.path.getsize(pp) if os.path.exists(pp) else 0
            w.writerow(row); mf.flush()
            time.sleep(0.5)
            if (i + 1) % 25 == 0: print(f'{i+1}/{len(todo)} done, {time.time()-t0:.0f}s', flush=True)
    print('DONE', flush=True)

if __name__ == '__main__':
    main()
