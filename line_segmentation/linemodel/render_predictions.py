"""Draw line-model predictions (and the truth where it exists) on the face image and on the original tablet photo.

  python -m line_cropping.linemodel.render_predictions --pred tablet_data/ebl_tablets/predictions --method unet --n 10 [--seed 3]
      [--faces <ids...>] [--out tablet_data/ebl_tablets/linemodel/prediction_overlays]
Red = predicted segments (from <pred>/<method>_segments/<face>.csv); green = eBL truth segments. Writes
<out>/faces/<face>.jpg, <out>/photos/<tablet>__<side>.jpg (segments placed in the composite via the face's crop offset)
and <out>/contact_sheet.jpg.
"""
from line_cropping.paths import EBL
import argparse, csv, os, random
import numpy as np, cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


def draw_segs(img, segs, color, width, off=(0, 0), scale=1.0):
    for s in segs:
        p0 = (int((s['ax0'] + off[0]) * scale), int((s['ay0'] + off[1]) * scale)); p1 = (int((s['ax1'] + off[0]) * scale), int((s['ay1'] + off[1]) * scale))
        cv2.line(img, p0, p1, color, width, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--pred', default=f'{EBL}/predictions'); ap.add_argument('--method', default='unet'); ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--seed', type=int, default=3); ap.add_argument('--faces', nargs='*', default=None); ap.add_argument('--out', default=f'{EBL}/linemodel/prediction_overlays')
    a = ap.parse_args(); os.makedirs(os.path.join(a.out, 'faces'), exist_ok=True); os.makedirs(os.path.join(a.out, 'photos'), exist_ok=True)
    faces = {f['face']: f for f in csv.DictReader(open(f'{EBL}/faces/faces.csv'))}
    seg_dir = os.path.join(a.pred, a.method + '_segments'); have = [f[:-4] for f in os.listdir(seg_dir) if f.endswith('.csv') and f[:-4] in faces]
    random.seed(a.seed); pick = a.faces or random.sample(have, min(a.n, len(have))); tiles = []
    for fid in pick:
        f = faces[fid]; img = cv2.imread(f['file']); H, W = img.shape[:2]
        pred = [{k: float(r[k]) for k in ('ax0', 'ay0', 'ax1', 'ay1')} for r in csv.DictReader(open(os.path.join(seg_dir, fid + '.csv')))]
        tp = os.path.join(f'{EBL}/faces/lines', fid + '.csv'); truth = [{k: float(r[k]) for k in ('ax0', 'ay0', 'ax1', 'ay1')} for r in csv.DictReader(open(tp))] if os.path.exists(tp) else []
        w = max(2, H // 500); ov = img.copy(); draw_segs(ov, truth, (0, 200, 0), w); draw_segs(ov, pred, (0, 0, 255), w)
        cv2.putText(ov, f'{fid}: {len(pred)} predicted (red), {len(truth)} truth (green)', (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        s = min(1.0, 1400 / max(H, W)); ov_s = cv2.resize(ov, None, fx=s, fy=s); cv2.imwrite(os.path.join(a.out, 'faces', fid + '.jpg'), ov_s, [cv2.IMWRITE_JPEG_QUALITY, 85])
        t = 1000 / ov_s.shape[1]; tiles.append(cv2.resize(ov_s, (1000, int(ov_s.shape[0] * t))))
        # on the original composite photograph
        ph = os.path.join(EBL, f['script'], f['tablet'], 'photo.jpg')
        if os.path.exists(ph) and f.get('crop_x0'):
            photo = cv2.imread(ph); PH, PW = photo.shape[:2]; ps = min(1.0, 1800 / max(PH, PW)); photo = cv2.resize(photo, None, fx=ps, fy=ps)
            off = (int(f['crop_x0']), int(f['crop_y0'])); pw_ = max(2, int(PH * ps) // 600)
            draw_segs(photo, truth, (0, 200, 0), pw_, off, ps); draw_segs(photo, pred, (0, 0, 255), pw_, off, ps)
            cv2.rectangle(photo, (int(off[0] * ps), int(off[1] * ps)), (int((off[0] + W) * ps), int((off[1] + H) * ps)), (255, 200, 0), 2)
            cv2.putText(photo, f'{fid}', (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(a.out, 'photos', fid + '.jpg'), photo, [cv2.IMWRITE_JPEG_QUALITY, 82])
    sheet = np.vstack(sum([[t_, np.zeros((8, 1000, 3), np.uint8)] for t_ in tiles], [])); cv2.imwrite(os.path.join(a.out, 'contact_sheet.jpg'), sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(f'{len(pick)} faces -> {a.out}/faces, /photos, contact_sheet.jpg:', pick)


if __name__ == '__main__':
    main()
