"""Flag signs whose form likely changed between fonts, to triage skeleton transfers.

The detector estimates how many wedge heads a glyph contains: wedge heads are the wide triangular
parts of the ink, so erasing everything thinner than a disk (morphological opening, with the disk
sized from the sign's own median stroke width) leaves roughly one blob per head. Absolute counts
are rough, but comparing the SAME detector on the source glyph and the new font's glyph of the
same sign cancels most per-sign bias, so a large count difference is a good hint that the sign's
wedge inventory changed between fonts -- exactly the transfers that need hand review.

Validated on the 148 signs that have skeletons in both Santakku and Assurbanipal: flagging
|difference| >= 2 catches 16 of the 20 signs whose true stroke count differs by >= 2, while
flagging 51 of 148 signs overall. Treat the output as a review shortlist, not ground truth, and
keep the transfer fit-quality metrics as a second net for the misses.

  python add_new_font_skeletons/detectWedges.py triage Esagil
      Writes fontTargets/Esagil_triage.csv, biggest suspected mismatches first.
  python add_new_font_skeletons/detectWedges.py calibrate
      Re-runs the Santakku-vs-Assurbanipal validation above (useful after tuning C_RADIUS).

Run from the repo root.
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for src imports
from src.utils import crop_img  # noqa: E402

SOURCE_FONT = "Assurbanipal"
FONT_PERIOD = {"Esagil": "Neo-Babylonian", "Santakku": "Old Babylonian",
               "Assurbanipal": "Neo Assyrian", "Ullikummi": "Middle Hittite"}
IMG_SIZE = 512
PAD = 10
C_RADIUS = 1.5   # opening-disk radius, as a multiple of the sign's median stroke half-width
FLAG_AT = 2      # |count difference| at which a sign is flagged for review


def load_processed_glyph(path):
    """Load a glyph and process it like the pipeline does: crop margins, pad, resize to 512."""
    from PIL import Image, ImageOps
    img = Image.open(path).convert("L")
    img, *_ = crop_img(img, is_pil=True)
    img = ImageOps.expand(img, border=PAD, fill=255)
    return np.array(img.resize((IMG_SIZE, IMG_SIZE)))


def count_wedge_heads(gray):
    ink = np.uint8(gray < 128)
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    if not ink.any():
        return 0
    r = max(3, round(C_RADIUS * float(np.median(dist[ink > 0]))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    opened = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
    n, _, stats, _ = cv2.connectedComponentsWithStats(opened)
    return sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= r * r)


def true_stroke_counts(font):
    return {fn.split("_")[0]: pd.read_csv(f"skeletons/{font}/{fn}")["label"].nunique()
            for fn in os.listdir(f"skeletons/{font}") if fn.endswith("_adf.csv")}


def glyph_path(font, hex_code):
    return f"prototypes/{font}/{hex_code[:5]}xx/{hex_code}.png"


def detect(font, hex_code):
    path = glyph_path(font, hex_code)
    if not os.path.exists(path):
        return None
    try:
        return count_wedge_heads(load_processed_glyph(path))
    except ValueError:  # blank glyph
        return None


def gottstein_counts(period):
    """Expert-curated wedge counts per sign for one period (fetchGottstein.py output), if fetched."""
    path = Path(__file__).parent / "gottstein_counts.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path).dropna(subset=["hex"])
    df = df[df["period"].str.replace("-", " ") == period.replace("-", " ")].drop_duplicates("hex")
    return dict(zip(df["hex"], df["total_wedges"]))


def triage(font):
    truth = true_stroke_counts(SOURCE_FONT)
    expected = gottstein_counts(FONT_PERIOD.get(font, ""))
    rows = []
    for target in sorted(Path("fontTargets", font).glob("0x*.png")):
        hex_code = target.stem
        if hex_code in expected and hex_code in truth:
            # best case: compare the period's documented wedge count with the source skeleton
            diff = int(expected[hex_code]) - truth[hex_code]
            basis = "gottstein"
        else:  # fall back to comparing image-based estimates of both glyphs
            det_src, det_new = detect(SOURCE_FONT, hex_code), detect(font, hex_code)
            if det_src is None or det_new is None:
                continue
            diff = det_new - det_src
            basis = "image"
        rows.append({"hex": hex_code, "source_strokes": truth.get(hex_code),
                     "expected_new": expected.get(hex_code), "diff": diff, "basis": basis})
    df = pd.DataFrame(rows)
    img = df["basis"] == "image"  # stroke-weight bias only affects the image-based rows
    df["diff_centered"] = df["diff"].astype(float)
    if img.any():
        df.loc[img, "diff_centered"] = df.loc[img, "diff"] - df.loc[img, "diff"].median()
    df["flagged"] = df["diff_centered"].abs() >= FLAG_AT
    df = df.reindex(df["diff_centered"].abs().sort_values(ascending=False).index)
    out = Path("fontTargets") / f"{font}_triage.csv"
    df.to_csv(out, index=False)
    print(df.head(25).to_string(index=False))
    print(f"\n{len(df)} signs -> {out}. {df['flagged'].sum()} flagged (|diff| >= {FLAG_AT}): "
          f"review those transfer overlays first; their skeletons most likely need hand edits.")


def calibrate():
    ta, tb = true_stroke_counts(SOURCE_FONT), true_stroke_counts("Santakku")
    rows = []
    for hx in sorted(set(ta) & set(tb)):
        da, db = detect(SOURCE_FONT, hx), detect("Santakku", hx)
        if da is None or db is None:
            continue
        rows.append({"true": tb[hx] - ta[hx], "det": db - da})
    df = pd.DataFrame(rows)
    truth_flag, det_flag = df["true"].abs() >= FLAG_AT, df["det"].abs() >= FLAG_AT
    print(f"{len(df)} signs with skeletons in both fonts; corr {df['true'].corr(df['det']):.2f}")
    print(f"true mismatches {truth_flag.sum()}: caught {(truth_flag & det_flag).sum()}, "
          f"missed {(truth_flag & ~det_flag).sum()}, false alarms {(~truth_flag & det_flag).sum()}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("calibrate", "triage"):
        sys.exit(__doc__)
    calibrate() if sys.argv[1] == "calibrate" else triage(sys.argv[2])
