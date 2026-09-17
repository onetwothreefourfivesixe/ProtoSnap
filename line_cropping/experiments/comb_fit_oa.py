import csv,collections,numpy as np,cv2,random
from scipy.signal import find_peaks
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

def main():
    ROOT='fat-cross_processed/fat-cross_processed'
    rows=list(csv.DictReader(open('fat-cross_processed/experiments/profile_line_counts.csv')))
    random.seed(3)
    picks=[]
    for lo,hi in ((4,6),(9,11),(14,16),(19,21),(24,28),(30,60)):
        c=[r for r in rows if lo<=int(r['atf_lines'])<=hi and r['face']=='obverse']
        if c: picks.append(random.choice(c))
    def profile(img,sig):
        gy=cv2.Sobel(cv2.GaussianBlur(img,(0,0),1.5),cv2.CV_32F,0,1,ksize=3)
        p=np.abs(gy).mean(axis=1); return cv2.GaussianBlur(p.reshape(-1,1),(0,0),sig).ravel()
    def deskew(im):
        H,W=im.shape; best=None
        for ang in np.arange(-12,12.1,1):
            M=cv2.getRotationMatrix2D((W/2,H/2),ang,1); rot=cv2.warpAffine(im,M,(W,H),borderValue=float(np.median(im)))
            body=rot[int(0.04*H):int(0.92*H), int(0.15*W):int(0.85*W)]
            p=profile(body,2); v=p.var()
            if best is None or v>best[0]: best=(v,ang,body)
        return best[1],best[2]
    def comb_fit(p,N):
        h=len(p); best=None
        for pitch in np.linspace(0.6*h/N,1.15*h/N,60):
            for off in np.linspace(0,pitch,25,endpoint=False):
                ys=off+pitch*np.arange(N)
                if ys[-1]>=h: continue
                s=np.interp(ys,np.arange(h),p).sum()
                if best is None or s>best[0]: best=(s,pitch,off)
        s,pitch,off=best; ys=off+pitch*np.arange(N)
        # local snap to nearest profile peak within 0.3 pitch
        pk,_=find_peaks(p,distance=max(2,int(0.5*pitch)))
        snapped=[]
        for y in ys:
            if len(pk): j=pk[np.argmin(abs(pk-y))]; y=j if abs(j-y)<0.3*pitch else y
            snapped.append(y)
        return pitch,np.array(snapped)
    fig,axs=plt.subplots(1,len(picks),figsize=(3.2*len(picks),9))
    for ax,r in zip(axs,picks):
        im=cv2.imread(f"{ROOT}/{r['p_number']}_{r['face']}.png",cv2.IMREAD_GRAYSCALE).astype(np.float32)
        ang,body=deskew(im); N=int(r['atf_lines']); p=profile(body,max(1.5,0.08*len(body)/N))
        free,_=find_peaks(p,distance=max(3,int(0.5*len(p)/N)),prominence=0.1*(p.max()-p.min()))
        pitch,ys=comb_fit(p,N)
        ax.imshow(body,cmap='gray')
        for y in free: ax.axhline(y,color='red',lw=0.7,ls='--')
        for y in ys: ax.axhline(y,color='lime',lw=0.9)
        ax.plot(p/p.max()*body.shape[1]*0.3,np.arange(len(p)),color='yellow',lw=0.7); ax.axis('off')
        ax.set_title(f"{r['p_number']} {r['face']}\nATF {N} lines | free {len(free)} | comb pitch {pitch:.0f}px, skew {ang:+.0f}",fontsize=8)
    plt.suptitle('Old Assyrian faces: red dashed = free profile peaks, green = ATF-constrained comb fit (N lines from transliteration)',fontsize=9)
    plt.tight_layout(); plt.savefig('fat-cross_processed/experiments/oa_comb_fit.png',dpi=100)
    print([ (r['p_number'],r['atf_lines']) for r in picks])



if __name__ == '__main__':
    main()
