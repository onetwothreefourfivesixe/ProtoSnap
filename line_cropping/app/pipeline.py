"""Processing behind the Line Splitter app: find tablet views in an uploaded photo, then split a view into line crops.

Unlike faces/prepare_faces.py, view detection here needs no annotations: the photo is thresholded against its
background, connected bright regions are split at empty gaps, and every region large enough to be a tablet face is
offered as a view. Line splitting runs the Phase 7 line model on the chosen view and cuts one crop per line, bounded
by the gaps between predicted bands and optionally rotated to each line's own angle.
"""
import os

import cv2
import numpy as np
from PIL import Image

from line_cropping.crops.make_crops import crop_along_segment
from line_cropping.faces.prepare_faces import tablet_components
from line_cropping.linemodel.infer import predict_face, read_lines, segments_to_lines, tablet_mask_for

Image.MAX_IMAGE_PIXELS = None


def _split_pieces(mask, box, min_gap=3, empty=0.04, depth=0):
    """Split a component box (in mask pixels) at runs of near-empty columns, then rows; return tight piece boxes."""
    x0, y0, x1, y1 = box
    sub = mask[y0:y1, x0:x1] > 0
    if not sub.any():
        return []
    for axis in (1, 0):
        prof = sub.mean(axis=0) if axis == 1 else sub.mean(axis=1)
        gaps = prof < empty
        cuts, i = [], 0
        while i < len(gaps):
            if gaps[i]:
                j = i
                while j < len(gaps) and gaps[j]:
                    j += 1
                if j - i >= min_gap and i > 0 and j < len(gaps):
                    cuts.append((i + j) // 2)
                i = j
            else:
                i += 1
        if cuts and depth < 6:
            edges = [0] + cuts + [len(gaps)]
            pieces = []
            for a, b in zip(edges, edges[1:]):
                nb = (x0 + a, y0, x0 + b, y1) if axis == 1 else (x0, y0 + a, x1, y0 + b)
                pieces += _split_pieces(mask, nb, min_gap, empty, depth + 1)
            return pieces
    cols = np.where(sub.any(axis=0))[0]
    rows = np.where(sub.any(axis=1))[0]
    return [(x0 + cols[0], y0 + rows[0], x0 + cols[-1] + 1, y0 + rows[-1] + 1)]


def detect_views(img_bgr, min_frac=0.01):
    """Candidate tablet views in a photo, as boxes in image pixels, largest first. Falls back to the whole image."""
    H, W = img_bgr.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    scale = max(2, int(round(max(W, H) / 1200)))
    comps, _lab, _ = tablet_components(pil, scale=scale)
    mask = tablet_components.mask
    views = []
    for _i, box, _a in comps:
        sb = tuple(int(round(v / scale)) for v in box)
        for p in _split_pieces(mask, sb):
            bx = (p[0] * scale, p[1] * scale, min(W, p[2] * scale), min(H, p[3] * scale))
            area = (bx[2] - bx[0]) * (bx[3] - bx[1])
            if area >= min_frac * W * H and min(bx[2] - bx[0], bx[3] - bx[1]) >= 60:
                views.append(dict(box=[int(v) for v in bx], area=int(area)))
    views.sort(key=lambda v: -v['area'])
    if not views:
        views = [dict(box=[0, 0, W, H], area=W * H)]
    for k, v in enumerate(views):
        v['id'] = k
    return views


def _line_bands(segs, W, H, ang):
    """Per line, in the face frame rotated by the median angle: centre, band top, band bottom; sorted top to bottom."""
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
    out = []
    for s in segs:
        p0 = M @ np.array([s['ax0'], s['ay0'], 1.0])
        p1 = M @ np.array([s['ax1'], s['ay1'], 1.0])
        c = (p0[1] + p1[1]) / 2
        t = s['thickness'] / 2
        out.append(dict(seg=s, c=c, top=c - t, bot=c + t, x0=min(p0[0], p1[0]), x1=max(p0[0], p1[0])))
    out.sort(key=lambda b: b['c'])
    return out


def _bounds(bands, pitch, pad=0.06):
    """Crop limits between consecutive bands: midpoint of the gap, or of the centres when the bands overlap."""
    res = []
    for i, b in enumerate(bands):
        if i == 0:
            top = b['top'] - pad * pitch
        elif bands[i - 1]['bot'] < b['top']:
            top = (bands[i - 1]['bot'] + b['top']) / 2
        else:
            top = (bands[i - 1]['c'] + b['c']) / 2
        if i == len(bands) - 1:
            bot = b['bot'] + pad * pitch
        elif b['bot'] < bands[i + 1]['top']:
            bot = (b['bot'] + bands[i + 1]['top']) / 2
        else:
            bot = (b['c'] + bands[i + 1]['c']) / 2
        res.append((top, bot))
    return res


def _unrotate(M, pts):
    Mi = cv2.invertAffineTransform(M)
    P = np.hstack([np.asarray(pts, float), np.ones((len(pts), 1))])
    return (Mi @ P.T).T


def split_view(model, dev, face_bgr, out_dir, prefix, rotate_per_line=True, readout=None, max_line_dev=20.0):
    """Run the line model on one face image and write one crop per line into out_dir.
    Returns (lines, info): lines carry the crop file name and the crop outline as a polygon in face pixels."""
    readout = readout or {}
    H, W = face_bgr.shape[:2]
    maps, s = predict_face(model, face_bgr, dev, readout.get('long_side', 1024))
    tab = tablet_mask_for(face_bgr, maps.shape, s) if readout.get('clip_tablet', 1) else None
    segs = read_lines(maps, s, readout.get('kernel_thr', 0.5), readout.get('band_thr', 0.4), readout.get('min_area', 12),
                      readout.get('max_reach', 1.5), readout.get('max_dev', 45.0), readout.get('merge_dy', 0.5),
                      readout.get('end_thr', 0.2), readout.get('max_ext', 3.0), tab)
    if not segs:
        return [], dict(angle=0.0, pitch=0.0)
    ang, rows = segments_to_lines(segs, W, H)
    bands = _line_bands(segs, W, H, ang)
    cs = [b['c'] for b in bands]
    pitch = float(np.median(np.diff(cs))) if len(cs) > 1 else bands[0]['bot'] - bands[0]['top']
    margin = 0.5 * pitch
    Mf = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
    face_rot = cv2.warpAffine(face_bgr, Mf, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    lines = []
    for i, (b, (top, bot)) in enumerate(zip(bands, _bounds(bands, pitch)), 1):
        seg = b['seg']
        seg_ang = float(np.degrees(np.arctan2(seg['ay1'] - seg['ay0'], seg['ax1'] - seg['ax0'])))
        name = f'{prefix}__L{i:02d}.jpg'
        use_line_frame = rotate_per_line and abs(((seg_ang - ang) + 90) % 180 - 90) <= max_line_dev
        if use_line_frame:
            crop, rect = crop_along_segment(face_bgr, seg, top - b['c'], bot - b['c'], margin)
            M = cv2.getRotationMatrix2D((rect['cx'], rect['cy']), rect['angle'], 1.0)
            x0, y0, x1, y1 = rect['x0'], rect['y0'], rect['x1'], rect['y1']
            line_angle = rect['angle']
        else:
            x0 = int(max(0, b['x0'] - margin))
            x1 = int(min(W, b['x1'] + margin))
            y0 = int(max(0, top))
            y1 = int(min(H, bot))
            crop = face_rot[y0:y1, x0:x1]
            M = Mf
            line_angle = ang
        if crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 3:
            continue
        cv2.imwrite(os.path.join(out_dir, name), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        poly = _unrotate(M, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        lines.append(dict(n=i, file=name, poly=poly.tolist(), angle=round(float(line_angle), 2),
                          width=int(crop.shape[1]), height=int(crop.shape[0]), rotated=bool(use_line_frame),
                          score=round(float(seg.get('score', 0.0)), 3)))
    return lines, dict(angle=round(float(ang), 2), pitch=round(pitch, 1))
