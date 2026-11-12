"""List the CDLI P-numbers of the tablets in this project's CDLI and eBL downloads, read from their JSON files.

  python -m line_segmentation.data.ingest.list_p_numbers [--out <DATA_ROOT>/p_numbers.csv]

Sources:
  eBL               ebl_tablets/<script>/<tablet>/fragment.json, field externalNumbers.cdliNumber (tablets without one are counted, not listed)
  eBL_snippets_cdli the CDLI records matched by museum number to the tablets of eBL's sign-snippet metadata
                    (data/ebl/signs_snippets_metadata.json against the CDLI JSON dump, built by ingest/match_museum.py into
                    data/cdli/artifacts_json/protosnap_tablets_to_cdli.csv). A tablet that matched several records lists each,
                    with n_matches > 1.
eBL's annotation dump names tablets by museum number only, so it is not a source here. Ur III tablets are left out on purpose: the
Ur III downloads (cdli_urIII, cdli_urIII_fg) and the sign-snippet tablets of script Ur3; --include-ur3 keeps the snippet ones.
Writes one row per P-number: p_number, its sources, and the eBL museum number and script.
"""
import argparse
import collections
import csv
import glob
import json
import os
import re

from line_segmentation.paths import DATA_ROOT, EBL


def norm(p):
    """'P451784', 'p451784', '451784' -> 'P451784'; anything else -> None."""
    m = re.fullmatch(r'[Pp]?\s*(\d{1,6})', str(p or '').strip())
    return f'P{int(m.group(1)):06d}' if m else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=os.path.join(DATA_ROOT, 'p_numbers.csv')); ap.add_argument('--include-ur3', action='store_true')
    ap.add_argument('--match', default='data/cdli/artifacts_json/protosnap_tablets_to_cdli.csv')
    a = ap.parse_args()
    rows = collections.defaultdict(lambda: dict(sources=set()))
    odd = collections.Counter()

    no_cdli = 0
    for f in sorted(glob.glob(os.path.join(EBL, '*', '*', 'fragment.json'))):
        d = json.load(open(f, encoding='utf-8')); raw = (d.get('externalNumbers') or {}).get('cdliNumber')
        if not raw:
            no_cdli += 1; continue
        for part in re.split(r'[,;/\s]+', raw):
            p = norm(part)
            if not p:
                odd['eBL: unreadable cdliNumber'] += 1; continue
            r = rows[p]; r['sources'].add('eBL'); r['ebl_museum_number'] = os.path.basename(os.path.dirname(f)); r['ebl_script'] = os.path.basename(os.path.dirname(os.path.dirname(f)))

    m = list(csv.DictReader(open(a.match, encoding='utf-8')))
    n_by_tablet = collections.Counter(r['tabletNumber'] for r in m if r['p_number']); skipped_ur3 = set()
    for r in m:
        if r['script'] == 'Ur3' and not a.include_ur3:
            skipped_ur3.add(r['tabletNumber']); continue
        p = norm(r['p_number'])
        if not p:
            continue
        x = rows[p]; x['sources'].add('eBL_snippets_cdli')
        x.setdefault('snippet_tablets', set()).add(r['tabletNumber']); x['snippet_script'] = r['script']
        x['cdli_museum_no'] = r['cdli_museum_no | accession_no']; x['cdli_designation'] = r['designation']; x['cdli_period'] = r['period']
        x['n_matches'] = max(x.get('n_matches', 0), n_by_tablet[r['tabletNumber']])
    unmatched = len({r['tabletNumber'] for r in m} - {r['tabletNumber'] for r in m if r['p_number']} - skipped_ur3)
    # where both sources name a P-number for the same eBL tablet, do they agree?
    via_frag = {r.get('ebl_museum_number'): p for p, r in rows.items() if 'eBL' in r['sources']}
    via_match = collections.defaultdict(set)
    for p, r in rows.items():
        for t in r.get('snippet_tablets', ()): via_match[t].add(p)
    shared = [t for t in via_frag if t in via_match]; agree = sum(via_frag[t] in via_match[t] for t in shared)

    cols = ['p_number', 'sources', 'ebl_museum_number', 'ebl_script', 'snippet_tablets', 'snippet_script', 'n_matches', 'cdli_museum_no', 'cdli_designation', 'cdli_period']
    with open(a.out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for p in sorted(rows):
            r = rows[p]; w.writerow({k: (';'.join(sorted(r[k])) if k in ('sources', 'snippet_tablets') and k in r else p if k == 'p_number' else r.get(k, '')) for k in cols})
    by = collections.Counter(s for r in rows.values() for s in r['sources'])
    both = sum(1 for r in rows.values() if len(r['sources']) > 1)
    print(f"{len(rows)} P-numbers; by source {dict(by)}; in both: {both}; eBL tablets without a CDLI number: {no_cdli}; "
          f"snippet tablets with no CDLI match: {unmatched}; Ur3 snippet tablets left out: {len(skipped_ur3)}; "
          f"tablets named by both sources: {len(shared)}, same P-number: {agree}"
          + (f'; skipped {dict(odd)}' if odd else '') + f' -> {a.out}')


if __name__ == '__main__':
    main()
