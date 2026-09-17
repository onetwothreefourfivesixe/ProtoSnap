"""Quality check of obverse/reverse face crops: size, aspect ratio and pixels per ATF line. Writes face_qc.csv."""
import csv, os, re, json, collections, numpy as np
from PIL import Image

def main():
    ROOT = 'fat-cross_processed/fat-cross_processed'
    inv = {r['p_number']: r for r in csv.DictReader(open('fat-cross_processed/inventory.csv'))}
    def face_lines(atf):
        out = {}; cur = None
        for l in atf.splitlines():
            if l.startswith('@'): cur = l[1:].strip().split()[0]; out.setdefault(cur, 0)
            elif re.match(r"^\d+'?\.", l) and cur: out[cur] += 1
        return out
    atfs = {}
    for l in open('artifacts_json/transliterations.jsonl', encoding='utf-8'):
        r = json.loads(l)
        if r['p_number'] in inv: atfs[r['p_number']] = face_lines(r['atf'])
    rows = []
    for p, r in inv.items():
        for face in ('obverse', 'reverse'):
            fp = f'{ROOT}/{p}_{face}.png'
            if not os.path.exists(fp): continue
            w, h = Image.open(fp).size; n = atfs.get(p, {}).get(face)
            ppl = 0.88 * h / n if n else None
            flags = []
            if min(w, h) < 150: flags.append('tiny')
            if not (0.45 <= h / w <= 4): flags.append('aspect')
            if ppl is not None and ppl < 12: flags.append('too_few_px_per_line')
            rows.append(dict(p_number=p, face=face, width=w, height=h, atf_lines=n or '', px_per_line=round(ppl) if ppl else '', flags='|'.join(flags), mtime=int(os.path.getmtime(fp))))
    with open('fat-cross_processed/experiments/face_qc.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    ppl = np.array([r['px_per_line'] for r in rows if r['px_per_line'] != ''])
    print(f"faces: {len(rows)}; median size {np.median([r['width'] for r in rows]):.0f}x{np.median([r['height'] for r in rows]):.0f}")
    print(f"px per ATF line (n={len(ppl)}): median {np.median(ppl):.0f}, 10th pct {np.percentile(ppl,10):.0f}; under 25: {(ppl<25).mean():.1%}; under 15: {(ppl<15).mean():.1%}")
    c = collections.Counter(f for r in rows for f in r['flags'].split('|') if f); print('flags:', dict(c), '| faces with any flag:', sum(bool(r['flags']) for r in rows))
    print('P250549 obverse now:', [(r['width'], r['height'], r['flags']) for r in rows if r['p_number'] == 'P250549' and r['face'] == 'obverse'])
    import time; recent = sum(1 for r in rows if time.time() - r['mtime'] < 3 * 3600); print('files modified in the last 3 hours:', recent)



if __name__ == '__main__':
    main()
