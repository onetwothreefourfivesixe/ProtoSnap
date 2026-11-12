"""Image-based line finding on a single tablet face: deskew, row projection profile, free peaks, and
fitting a known number of lines (comb fit). All functions work on a float32 greyscale numpy image."""
import numpy as np, cv2
from scipy.signal import find_peaks


def row_profile(img, sigma):
    """Mean absolute vertical gradient per row, smoothed. Peaks once per line of writing when upright."""
    gy = cv2.Sobel(cv2.GaussianBlur(img, (0, 0), 1.5), cv2.CV_32F, 0, 1, ksize=3)
    p = np.abs(gy).mean(axis=1)
    return cv2.GaussianBlur(p.reshape(-1, 1), (0, 0), max(0.5, sigma)).ravel()


def body_crop(img, top=0.04, bottom=0.92, left=0.12, right=0.88):
    """Trim the scale bar at the bottom and the curved side margins. Returns crop and its (y, x) offset."""
    H, W = img.shape
    y0, y1, x0, x1 = int(top * H), int(bottom * H), int(left * W), int(right * W)
    return img[y0:y1, x0:x1], (y0, x0)


def rotate(img, angle):
    H, W = img.shape
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (W, H), borderValue=float(np.median(img)))


def deskew(img, angles=np.arange(-12, 12.1, 1.0), sigma=2.0):
    """Rotation (degrees) that makes the row profile of the body crop sharpest (largest variance)."""
    best = None
    for a in angles:
        p = row_profile(body_crop(rotate(img, a))[0], sigma); v = p.var()
        if best is None or v > best[0]: best = (v, a)
    return best[1]


def estimate_pitch(p, lo_frac=0.02, hi_frac=0.25):
    """Line pitch (px) from the first autocorrelation peak of the profile."""
    pc = p - p.mean(); ac = np.correlate(pc, pc, 'full')[len(pc) - 1:]; ac /= max(ac[0], 1e-6)
    lo = max(8, int(lo_frac * len(p))); hi = max(lo + 5, int(hi_frac * len(p)))
    return int(np.argmax(ac[lo:hi]) + lo)


def free_peaks(p, pitch):
    peaks, _ = find_peaks(p, distance=max(3, int(0.6 * pitch)), prominence=0.1 * (p.max() - p.min()))
    return peaks


def comb_fit(p, n, pitch_range=(0.6, 1.15), snap=0.3):
    """Place n lines: search pitch and offset for the comb with the largest total profile response,
    then snap each line to the nearest profile peak within `snap` pitches. Returns (pitch, y positions)."""
    h = len(p); best = None
    for pitch in np.linspace(pitch_range[0] * h / n, pitch_range[1] * h / n, 60):
        for off in np.linspace(0, pitch, 25, endpoint=False):
            ys = off + pitch * np.arange(n)
            if ys[-1] >= h: continue
            s = np.interp(ys, np.arange(h), p).sum()
            if best is None or s > best[0]: best = (s, pitch, off)
    _, pitch, off = best; ys = off + pitch * np.arange(n)
    pk, _ = find_peaks(p, distance=max(2, int(0.5 * pitch)))
    out = []
    for y in ys:
        if len(pk):
            j = pk[np.argmin(abs(pk - y))]
            if abs(j - y) < snap * pitch: y = float(j)
        out.append(float(y))
    return float(pitch), np.array(out)


def find_lines(gray, n_lines=None):
    """Full pipeline on one face. Returns dict with skew, crop offset, pitch and line centres (y) in the
    deskewed full-face frame. If n_lines is given the comb fit is used, otherwise free peaks."""
    img = gray.astype(np.float32)
    ang = deskew(img); rot = rotate(img, ang)
    body, (oy, ox) = body_crop(rot)
    if n_lines:
        p = row_profile(body, max(1.5, 0.08 * len(body) / n_lines))
        pitch, ys = comb_fit(p, n_lines); method = 'comb'
    else:
        p = row_profile(body, 3); pitch = estimate_pitch(p); ys = free_peaks(p, pitch).astype(float); method = 'free'
    return dict(skew_deg=float(ang), body_offset=(oy, ox), pitch_px=float(pitch), lines_y=(ys + oy).tolist(), method=method,
                body_x0=ox, body_x1=ox + body.shape[1])
