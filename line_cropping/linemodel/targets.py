"""Phase 7, step 1: line truth as oriented segments and the target maps painted from them.

A truth line is an oriented segment (ax0, ay0) -> (ax1, ay1) with a thickness, in face pixels of the UNROTATED face.
eBL faces have these directly (tablet_data/ebl_tablets/faces/lines/<face>.csv). Old Assyrian labels are line centres in a
deskewed frame; `oa_segments()` turns them into segments in the same format.

Target maps, at 1/`stride` of the image (default stride 4), channels:
  0 kernel   band shrunk to `shrink` (0.4) of its width  -> one connected region per line
  1 band     the line's band, width = min(truth thickness, 0.7 x block pitch) so neighbours never merge
  2,3 orient sin(2*theta), cos(2*theta) of the line direction, painted inside the band (0 elsewhere)
  4 ignore   1 where the truth is unreliable: gaps between consecutive lines wider than `ignore_gap` (1.5) pitches
             and one pitch beyond the first and last line of a block. Ignored pixels carry no loss.

  python -m line_cropping.linemodel.targets --preview 12   # overlays in tablet_data/ebl_tablets/linemodel/target_previews/
"""
from line_cropping.paths import EBL
import argparse, csv, os, collections, random
import numpy as np, cv2

STRIDE = 4


def read_segments(path):
    out = []
    for l in csv.DictReader(open(path, encoding='utf-8')):
        out.append(dict(ax0=float(l['ax0']), ay0=float(l['ay0']), ax1=float(l['ax1']), ay1=float(l['ay1']), thickness=float(l['thickness']),
                        column=l.get('column', 'single'), line_index=int(float(l.get('line_index', 0)))))
    return out


def oa_segments(labels_rows, W, H, skew_deg, x0_frac=0.05, x1_frac=0.95):
    """Old Assyrian hand labels (line centres y in the frame rotated by skew_deg about the face centre) -> segments
    in the unrotated face. Thickness = 0.6 x median gap (or 0.45 x face height / n for a single line)."""
    ys = sorted(float(r['y_face_deskewed']) for r in labels_rows)
    pitch = float(np.median(np.diff(ys))) if len(ys) > 1 else 0.45 * H / max(1, len(ys))
    M = cv2.getRotationMatrix2D((W / 2, H / 2), skew_deg, 1.0); Mi = cv2.invertAffineTransform(M)
    segs = []
    for i, y in enumerate(ys):
        p0 = Mi @ np.array([x0_frac * W, y, 1.0]); p1 = Mi @ np.array([x1_frac * W, y, 1.0])
        segs.append(dict(ax0=p0[0], ay0=p0[1], ax1=p1[0], ay1=p1[1], thickness=0.6 * pitch, column='single', line_index=i))
    return segs


def _band_polygon(s, thickness):
    dx, dy = s['ax1'] - s['ax0'], s['ay1'] - s['ay0']; n = (dx * dx + dy * dy) ** 0.5 or 1.0
    nx, ny = -dy / n * thickness / 2, dx / n * thickness / 2
    return np.array([[s['ax0'] + nx, s['ay0'] + ny], [s['ax1'] + nx, s['ay1'] + ny], [s['ax1'] - nx, s['ay1'] - ny], [s['ax0'] - nx, s['ay0'] - ny]], np.float32)


def render_targets(segs, W, H, stride=STRIDE, shrink=0.4, ignore_gap=1.5, band_frac=0.7):
    """Paint the five target maps for a face of size W x H. Returns float32 array (5, H/stride, W/stride)."""
    h, w = int(np.ceil(H / stride)), int(np.ceil(W / stride)); T = np.zeros((5, h, w), np.float32)
    sc = 1.0 / stride
    # painted widths are bounded by the block's pitch so neighbouring bands never merge: band <= band_frac x pitch,
    # kernel = shrink x band. (eBL sign boxes are as tall as the pitch, so the raw thickness would overlap.)
    pitch_of = {}
    blocks0 = collections.defaultdict(list)
    for s in segs: blocks0[s['column']].append(s)
    for col, L in blocks0.items():
        mids = sorted((s['ay0'] + s['ay1']) / 2 for s in L); g = np.diff(mids)
        pitch_of[col] = float(np.median(g)) if len(g) else float(np.median([s['thickness'] for s in L])) / band_frac
    for s in segs:
        bw = min(s['thickness'], band_frac * pitch_of[s['column']]); s['_band'] = bw
        poly = (_band_polygon(s, bw) * sc).astype(np.int32); kern = (_band_polygon(s, bw * shrink) * sc).astype(np.int32)
        cv2.fillPoly(T[1], [poly], 1.0); cv2.fillPoly(T[0], [kern], 1.0)
        th = np.arctan2(s['ay1'] - s['ay0'], s['ax1'] - s['ax0']); m = np.zeros((h, w), np.uint8); cv2.fillPoly(m, [poly], 1)
        T[2][m > 0] = np.sin(2 * th); T[3][m > 0] = np.cos(2 * th)
    # ignore: wide gaps between consecutive lines of a block, and one pitch beyond the block's first/last line
    blocks = collections.defaultdict(list)
    for s in segs: blocks[s['column']].append(s)
    for L in blocks.values():
        L = sorted(L, key=lambda s: (s['ay0'] + s['ay1']) / 2)
        mids = np.array([(s['ay0'] + s['ay1']) / 2 for s in L]); gaps = np.diff(mids)
        pitch = float(np.median(gaps)) if len(gaps) else float(np.median([s['thickness'] for s in L])) * 1.3
        for i, g in enumerate(gaps):
            if g > ignore_gap * pitch:
                a, b = L[i], L[i + 1]
                poly = np.array([[a['ax0'], a['ay0'] + a['_band'] / 2], [a['ax1'], a['ay1'] + a['_band'] / 2], [b['ax1'], b['ay1'] - b['_band'] / 2], [b['ax0'], b['ay0'] - b['_band'] / 2]], np.float32) * sc
                cv2.fillPoly(T[4], [poly.astype(np.int32)], 1.0)
        for s, sign in ((L[0], -1), (L[-1], 1)):
            off = sign * (s['_band'] / 2 + pitch); poly = np.array([[s['ax0'], s['ay0'] + sign * s['_band'] / 2], [s['ax1'], s['ay1'] + sign * s['_band'] / 2], [s['ax1'], s['ay1'] + off], [s['ax0'], s['ay0'] + off]], np.float32) * sc
            cv2.fillPoly(T[4], [poly.astype(np.int32)], 1.0)
    T[4][T[1] > 0] = 0            # a painted line is never ignored
    return T


def overlay(img, T, stride=STRIDE):
    """Colour overlay for inspection: band green, kernel yellow, ignore red hatch."""
    H, W = img.shape[:2]; up = lambda m: cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    out = img.copy().astype(np.float32); band, kern, ign = up(T[1]), up(T[0]), up(T[4])
    out[band > 0] = out[band > 0] * 0.5 + np.array([0, 200, 0]) * 0.5; out[kern > 0] = out[kern > 0] * 0.4 + np.array([0, 230, 230]) * 0.6
    hatch = ((np.indices((H, W)).sum(axis=0) // 6) % 2 == 0); out[(ign > 0) & hatch] = out[(ign > 0) & hatch] * 0.4 + np.array([0, 0, 255]) * 0.6
    return out.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--preview', type=int, default=12); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=f'{EBL}/linemodel/target_previews'); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    from line_cropping.linemodel.dataset import build_index
    idx = build_index(); random.seed(a.seed); picks = random.sample(idx['train'], min(a.preview, len(idx['train'])))
    for it in picks:
        img = cv2.imread(it['file']); H, W = img.shape[:2]; T = render_targets(it['segments'], W, H)
        ov = overlay(img, T); s = min(1.0, 1400 / max(H, W)); ov = cv2.resize(ov, None, fx=s, fy=s)
        cv2.putText(ov, f"{it['face']}  {it['source']}  {len(it['segments'])} lines", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(a.out, it['face'] + '.jpg'), ov, [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(f'wrote {len(picks)} target previews to {a.out}  (green = band, yellow = kernel, red hatch = ignore)')


if __name__ == '__main__':
    main()
