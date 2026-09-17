import csv, json, re, collections

def main():
    OUT='/mnt/d/github/ProtoSnap/artifacts_json'
    ROMAN={'II':'2','III':'3','IV':'4'}
    def numtok(s):  # '0055-21-001' -> '55-21-1'
        return '-'.join(str(int(x)) if x.isdigit() else x for x in re.split(r'[-]', s))
    def norm(s):
        """Normalize a museum/accession number from either the repo ('BM.37990','Rm-II.481','1881,0204.233')
        or CDLI ('BM 037990','Rm 2, 0481','1881-02-04, 0233') into a common key. Case-sensitive prefix."""
        if not s: return None
        s=s.strip()
        m=re.match(r'^(\d{4}),(\d{2})(\d{2})\.(\d+)$', s)                      # repo BM registration no.
        if m: return ('REG', f'{m[1]}-{m[2]}-{m[3]}', int(m[4]))
        m=re.match(r'^(\d{4})-(\d{2})-(\d{2}),\s*0*(\d+)\b', s)                # CDLI BM registration no.
        if m: return ('REG', f'{m[1]}-{m[2]}-{m[3]}', int(m[4]))
        m=re.match(r'^([A-Za-z]+)-(II|III|IV)\.(\d+)$', s)                     # repo Rm-II.481 / Sp-III.x
        if m: return (m[1], ROMAN[m[2]], int(m[3]))
        m=re.match(r'^([A-Za-z]+) ([2-4]),\s*0*(\d+)\b', s)                     # CDLI Rm 2, 0481
        if m: return (m[1], m[2], int(m[3]))
        m=re.match(r'^([A-Za-z]+)[ .]+0*(\d[\d\-]*)', s)                        # generic PREFIX number
        if m: return (m[1], numtok(m[2]))
        return None
    cdli=collections.defaultdict(list)
    for r in csv.DictReader(open(f'{OUT}/artifacts_index.csv',encoding='utf-8')):
        for field in ('museum_no','accession_no'):
            for part in re.split(r'\s*(?:\(\+\)|[+;])\s*', r.get(field) or ''):
                k=norm(part)
                if k and r not in cdli[k]: cdli[k].append(r)
    snips=json.load(open('/mnt/d/github/ProtoSnap/signs_snippets_metadata.json'))
    tabs=collections.Counter(x['tabletNumber'] for x in snips)
    script={}
    for x in snips: script.setdefault(x['tabletNumber'],collections.Counter())[x['script']]+=1
    rows=[]; matched=0; matched_atf=0; snips_m=0; snips_atf=0; amb=0; pnums_atf=set()
    for t,c in tabs.items():
        k=norm(t); hits=cdli.get(k,[]) if k else []
        sc=script[t].most_common(1)[0][0]
        if hits:
            matched+=1; snips_m+=c
            if len(hits)>1: amb+=1
            if any(h['has_atf']=='True' for h in hits): matched_atf+=1; snips_atf+=c
        for h in hits or [None]:
            if h and h['has_atf']=='True': pnums_atf.add(h['p_number'])
            rows.append([t,c,sc,h['p_number'] if h else '',(h['museum_no']+' | '+(h['accession_no'] or '')) if h else '',h['designation'] if h else '',h['period'] if h else '',h['has_atf'] if h else '',h['atf_lines'] if h else ''])
    rows.sort(key=lambda r:(-r[1],r[0]))
    with open(f'{OUT}/protosnap_tablets_to_cdli.csv','w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['tabletNumber','n_snippets','script','p_number','cdli_museum_no | accession_no','designation','period','has_atf','atf_lines']); w.writerows(rows)
    # ATF subset for the repo's tablets
    n=0
    with open(f'{OUT}/protosnap_tablets_atf.jsonl','w',encoding='utf-8') as f:
        p2t=collections.defaultdict(list)
        for r in rows:
            if r[3]: p2t[r[3]].append(r[0])
        for l in open(f'{OUT}/transliterations.jsonl',encoding='utf-8'):
            rec=json.loads(l)
            if rec['p_number'] in pnums_atf:
                rec['tabletNumbers']=p2t[rec['p_number']]; f.write(json.dumps(rec,ensure_ascii=False)+'\n'); n+=1
    print(f'repo tablets: {len(tabs)}; matched to CDLI: {matched} ({amb} with >1 hit); with ATF: {matched_atf}')
    print(f'snippets: {len(snips)}; on matched tablets: {snips_m}; on tablets with ATF: {snips_atf}; ATF records written: {n}')
    unm=collections.Counter(t.split('.')[0] for t in tabs if not (norm(t) and cdli.get(norm(t))))
    print('unmatched by prefix:',unm.most_common(12))
    print('--- (script, period) for ATF-bearing matches ---')
    pr=collections.Counter((r[2],r[6][:16],r[0].split('.')[0]) for r in rows if r[7]=='True')
    for k,c in pr.most_common(14): print(k,c)



if __name__ == '__main__':
    main()
