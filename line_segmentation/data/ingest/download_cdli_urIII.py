"""Download the first N Ur III tablets that have a CDLI photo, walking the FactGrid ID list in order.

The list (FG_Ur3_CDLI_IDs.csv: tablet, tabletLabel, cdliID) does not say which tablets are photographed, so each
P-number is checked with a HEAD request on https://cdli.earth/dl/photo/<P>.jpg. For every tablet with a photo the script
saves the photo, the line art if there is one, and the CDLI artifact JSON (which carries the ATF).

CDLI has retired some P-numbers by merging them into another artifact. Their JSON address answers with a bare redirect, so the
script looks up the artifact page's redirect target and saves that record instead, with `retired_p_number` set to the
listed P-number. In manifest.csv, `cdli_record` then names the record that was saved.

Output (default <DATA_ROOT>/cdli_urIII_fg/):
  photo/<P>.jpg, lineart/<P>_l.jpg
  metadata.json   list of CDLI artifact records, same format as urIII_metadata.json
  manifest.csv    p_number, factgrid, label, cdli_record, designation, museum_no, period, provenience, has_atf, photo, lineart
  probe.csv       p_number, has_photo for every ID checked, so a rerun skips IDs already probed

Reruns resume: probed IDs and downloaded files are reused.

    python -m line_cropping.ingest.download_cdli_urIII                    # first 1000
    python -m line_cropping.ingest.download_cdli_urIII --n 2000 --workers 6
"""
import argparse
import csv
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from line_cropping.paths import DATA_ROOT

BASE = 'https://cdli.earth'
UA = {'User-Agent': 'Mozilla/5.0 (research download script)'}


class Redirected(Exception):
    """CDLI answered with a redirect; .location is its target, or '' when it gives none."""
    def __init__(self, location):
        super().__init__(location); self.location = location


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


_opener = urllib.request.build_opener(_NoRedirect)


def _open(url, method='GET', timeout=60, tries=4, follow=True):
    """urlopen with retries; returns the response, or None on 404. Raises after repeated other failures. With follow=False a
    redirect raises Redirected instead of being followed."""
    err = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA, method=method)
            return urllib.request.urlopen(req, timeout=timeout) if follow else _opener.open(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code in (301, 302, 303, 307, 308) and not follow:
                raise Redirected(e.headers.get('Location') or '')
            err = e
        except Exception as e:  # timeouts, resets
            err = e
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f'{url}: {err}')


def has_photo(p):
    r = _open(f'{BASE}/dl/photo/{p}.jpg', method='HEAD', timeout=30)
    if r is None:
        return False
    with r:
        return 'image' in r.headers.get('Content-Type', '')


def fetch_image(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return True
    r = _open(url)
    if r is None:
        return False
    with r:
        if 'image' not in r.headers.get('Content-Type', ''):
            return False
        data = r.read()
    tmp = dst + '.part'
    with open(tmp, 'wb') as f:
        f.write(data)
    os.replace(tmp, dst)
    return True


def replacement(p):
    """The artifact a retired P-number now points to, from the redirect of its web page, or None."""
    try:
        r = _open(f'{BASE}/artifacts/{int(p[1:])}', method='HEAD', timeout=30, follow=False)
    except Redirected as e:
        m = re.search(r'/artifacts/(\d+)', e.location)
        return f'P{int(m.group(1)):06d}' if m and int(m.group(1)) != int(p[1:]) else None
    if r is not None:
        r.close()
    return None


def fetch_artifact(p, cache_dir):
    dst = os.path.join(cache_dir, f'{p}.json')
    if os.path.exists(dst):
        with open(dst, encoding='utf-8') as f:
            return json.load(f)
    retired_from = None
    for _ in range(3):                                  # a retired artifact may point to another retired one
        try:
            r = _open(f'{BASE}/artifacts/{int(p[1:])}.json', follow=False)
            break
        except Redirected:
            new = replacement(p)
            if new is None:
                print(f'{p}: CDLI redirects its record and gives no replacement; no metadata saved', flush=True)
                return None
            retired_from = retired_from or os.path.basename(dst)[:-5]; p = new
    else:
        return None
    if r is None:
        return None
    with r:
        d = json.load(r)
    d = d[0] if isinstance(d, list) and d else d
    if retired_from and isinstance(d, dict):
        d['retired_p_number'] = retired_from
    with open(dst, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False)
    return d


def read_ids(path):
    seen, rows = set(), []
    with open(path, encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            p = (r.get('cdliID') or '').strip()
            if p.startswith('P') and p not in seen:
                seen.add(p)
                rows.append({'p': p, 'factgrid': r.get('tablet', ''), 'label': r.get('tabletLabel', '')})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--ids', default='FG_Ur3_CDLI_IDs.csv')
    ap.add_argument('--out', default=os.path.join(DATA_ROOT, 'cdli_urIII_fg'))
    ap.add_argument('--n', type=int, default=1000, help='number of tablets with photos to collect')
    ap.add_argument('--workers', type=int, default=4, help='parallel requests (keep low; CDLI is a shared service)')
    ap.add_argument('--no-lineart', action='store_true')
    a = ap.parse_args()

    for sub in ('photo', 'lineart', 'artifacts'):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    ids = read_ids(a.ids)
    print(f'{len(ids)} unique IDs in {a.ids}')

    # 1. probe in list order, in batches, until n photographed tablets are found
    probe_path = os.path.join(a.out, 'probe.csv')
    probed = {}
    if os.path.exists(probe_path):
        with open(probe_path, newline='') as f:
            probed = {r['p_number']: r['has_photo'] == '1' for r in csv.DictReader(f)}
    new_file = not os.path.exists(probe_path)
    picked, i = [], 0
    with open(probe_path, 'a', newline='') as pf, ThreadPoolExecutor(a.workers) as ex:
        pw = csv.writer(pf)
        if new_file:
            pw.writerow(['p_number', 'has_photo'])
        while len(picked) < a.n and i < len(ids):
            batch = ids[i:i + a.workers * 8]
            i += len(batch)
            todo = [r['p'] for r in batch if r['p'] not in probed]
            for p, ok in zip(todo, ex.map(has_photo, todo)):
                probed[p] = ok
                pw.writerow([p, int(ok)])
            pf.flush()
            picked += [r for r in batch if probed[r['p']]]
            print(f'probed {i}/{len(ids)}  with photo {min(len(picked), a.n)}/{a.n}', flush=True)
    picked = picked[:a.n]
    if len(picked) < a.n:
        print(f'only {len(picked)} tablets in the list have photos')

    # 2. download photo, line art and artifact record for each picked tablet
    def work(r):
        p = r['p']
        ph = fetch_image(f'{BASE}/dl/photo/{p}.jpg', os.path.join(a.out, 'photo', f'{p}.jpg'))
        la = False if a.no_lineart else fetch_image(f'{BASE}/dl/lineart/{p}_l.jpg',
                                                    os.path.join(a.out, 'lineart', f'{p}_l.jpg'))
        art = fetch_artifact(p, os.path.join(a.out, 'artifacts'))
        return r, ph, la, art

    rows, arts = [], []
    with ThreadPoolExecutor(a.workers) as ex:
        for k, (r, ph, la, art) in enumerate(ex.map(work, picked), 1):
            p, art = r['p'], art or {}
            if art:
                arts.append(art)
            rec = f"P{int(art['id']):06d}" if art.get('id') else ''
            rows.append({'p_number': p, 'factgrid': r['factgrid'], 'label': r['label'], 'cdli_record': rec,
                         'designation': art.get('designation'), 'museum_no': art.get('museum_no'),
                         'period': (art.get('period') or {}).get('period'),
                         'provenience': (art.get('provenience') or {}).get('provenience'),
                         'has_atf': bool((art.get('inscription') or {}).get('atf')),
                         'photo': f'photo/{p}.jpg' if ph else '', 'lineart': f'lineart/{p}_l.jpg' if la else ''})
            print(f'[{k}/{len(picked)}] {p} photo={"y" if ph else "n"} lineart={"y" if la else "n"} '
                  f'atf={"y" if rows[-1]["has_atf"] else "n"}', flush=True)

    with open(os.path.join(a.out, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(arts, f, ensure_ascii=False, indent=1)
    with open(os.path.join(a.out, 'manifest.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"done: {len(rows)} tablets, photos {sum(bool(r['photo']) for r in rows)}, "
          f"lineart {sum(bool(r['lineart']) for r in rows)}, with ATF {sum(r['has_atf'] for r in rows)} -> {a.out}")


if __name__ == '__main__':
    main()
