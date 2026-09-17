import os, json, sys, time, csv
from multiprocessing import Pool
SRC='artifacts_json/artifacts_json'   # run from the repo root
OUT='artifacts_json'
def proc(name):
    try:
        d=json.load(open(os.path.join(SRC,name),encoding='utf-8'))
    except Exception as ex:
        return ('BAD',name,str(ex))
    ins=d.get('inscription') or {}
    atf=(ins.get('atf') or '').strip()
    pid=d.get('id'); pno='P%06d'%pid if isinstance(pid,int) else name[:-5]
    per=(d.get('period') or {}).get('period','')
    prov=(d.get('provenience') or {}).get('provenience','')
    gen='; '.join((g.get('genre') or {}).get('genre','') for g in d.get('genres') or [])
    lang='; '.join((g.get('language') or {}).get('language','') for g in d.get('languages') or [])
    lines=[l for l in atf.splitlines() if l and l[0] not in '&#@$>'] if atf else []
    row=[pno,pid,d.get('designation'),d.get('museum_no'),d.get('accession_no'),per,prov,gen,lang,bool(atf),len(lines)]
    rec=json.dumps({'p_number':pno,'designation':d.get('designation'),'museum_no':d.get('museum_no'),'accession_no':d.get('accession_no'),'period':per,'atf':atf},ensure_ascii=False) if atf else None
    return ('OK',row,rec)
if __name__=='__main__':
    t0=time.time()
    names=sorted(n for n in os.listdir(SRC) if n.endswith('.json'))
    print(f'listed {len(names)} files in {time.time()-t0:.0f}s',flush=True)
    jl=open(os.path.join(OUT,'transliterations.jsonl'),'w',encoding='utf-8')
    idx=open(os.path.join(OUT,'artifacts_index.csv'),'w',encoding='utf-8',newline='')
    w=csv.writer(idx); w.writerow(['p_number','id','designation','museum_no','accession_no','period','provenience','genre','language','has_atf','atf_lines'])
    n=n_atf=bad=0
    with Pool(16) as p:
        for r in p.imap(proc,names,chunksize=100):
            n+=1
            if r[0]=='BAD': bad+=1; print('BAD',r[1],r[2],file=sys.stderr); continue
            w.writerow(r[1])
            if r[2]: n_atf+=1; jl.write(r[2]+'\n')
            if n%20000==0: print(f'{n} files, {n_atf} with atf, {time.time()-t0:.0f}s',flush=True)
    jl.close(); idx.close()
    print(f'DONE {n} files, {n_atf} with atf, {bad} unreadable, {time.time()-t0:.0f}s',flush=True)
