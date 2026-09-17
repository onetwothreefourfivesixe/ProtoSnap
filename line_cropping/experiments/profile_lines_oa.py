import json,re,csv,collections,os,sys,time
import numpy as np,cv2
from scipy.signal import find_peaks

def main():
    ROOT='fat-cross_processed/fat-cross_processed'
    inv={r['p_number']:r for r in csv.DictReader(open('fat-cross_processed/inventory.csv'))}
    def face_lines(atf):
        out={}; cur=None
        for l in atf.splitlines():
            if l.startswith('@'): cur=l[1:].strip().split()[0]; out.setdefault(cur,0)
            elif re.match(r"^\d+'?\.",l) and cur: out[cur]+=1
        return out
    atfs={}
    for l in open('artifacts_json/transliterations.jsonl',encoding='utf-8'):
        r=json.loads(l)
        if r['p_number'] in inv: atfs[r['p_number']]=face_lines(r['atf'])
    def profile(img,sig):
        gy=cv2.Sobel(cv2.GaussianBlur(img,(0,0),1.5),cv2.CV_32F,0,1,ksize=3)
        p=np.abs(gy).mean(axis=1); return cv2.GaussianBlur(p.reshape(-1,1),(0,0),sig).ravel()
    def count_lines(path):
        im=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        if im is None: return None
        im=im.astype(np.float32); H,W=im.shape
        if H<60 or W<60: return None
        best=None
        for ang in np.arange(-8,8.1,1):
            M=cv2.getRotationMatrix2D((W/2,H/2),ang,1); rot=cv2.warpAffine(im,M,(W,H),borderValue=float(np.median(im)))
            body=rot[int(0.04*H):int(0.92*H), int(0.12*W):int(0.88*W)]   # drop scale bar + curved margins
            p=profile(body,3); v=p.var()
            if best is None or v>best[0]: best=(v,ang,body,p)
        v,ang,body,p=best
        pc=p-p.mean(); ac=np.correlate(pc,pc,'full')[len(pc)-1:]; ac/=max(ac[0],1e-6)
        lo=max(8,int(0.02*body.shape[0])); hi=max(lo+5,int(0.25*body.shape[0]))
        lag=int(np.argmax(ac[lo:hi])+lo)
        peaks,_=find_peaks(p,distance=max(3,int(0.6*lag)),prominence=0.1*(p.max()-p.min()))
        return dict(skew=float(ang),pitch=lag,peaks=len(peaks),h=body.shape[0])
    rows=[]; t0=time.time()
    for i,(p,r) in enumerate(inv.items()):
        if p not in atfs: continue
        for face in ('obverse','reverse'):
            fp=f'{ROOT}/{p}_{face}.png'
            if not os.path.exists(fp) or face not in atfs[p]: continue
            res=count_lines(fp)
            if res: rows.append(dict(p_number=p,face=face,atf_lines=atfs[p][face],**res))
        if i%200==0: print(i,len(rows),f'{time.time()-t0:.0f}s',flush=True)
    with open('fat-cross_processed/experiments/profile_line_counts.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    d=np.array([r['peaks']-r['atf_lines'] for r in rows]); a=np.array([r['atf_lines'] for r in rows])
    print(f'faces scored: {len(rows)}; exact count match: {(d==0).mean():.1%}; within ±1: {(abs(d)<=1).mean():.1%}; within ±2: {(abs(d)<=2).mean():.1%}; median |diff|: {np.median(abs(d)):.0f}; mean signed diff: {d.mean():+.2f}')
    print('by ATF line count bucket (match within ±1):')
    for lo,hi in ((1,6),(7,10),(11,15),(16,25),(26,99)):
        m=(a>=lo)&(a<=hi); 
        if m.any(): print(f'  {lo:2d}-{hi:2d} lines: n={m.sum():4d}  ±1: {(abs(d[m])<=1).mean():.1%}  mean diff {d[m].mean():+.1f}')
    print('skew distribution (deg):',collections.Counter(int(r['skew']) for r in rows).most_common(6))



if __name__ == '__main__':
    main()
