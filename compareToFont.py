"""Compare ProtoSnap outputs to the Unicode font glyph they started from.

For every run folder under output/<output_folder>/, this measures how much the fitted skeleton
still resembles the prototype glyph (NOT accuracy against the photo):
  * mean_shift_px : average distance each keypoint moved from its position in the font skeleton
  * max_shift_px  : the single largest keypoint move
  * iou           : overlap between the fitted skeleton (drawn thick) and the glyph's ink
Also writes <run>/font_compare.png: glyph | glyph + fitted skeleton | photo + fitted skeleton.

Usage: python compareToFont.py singleSigns_Assurbanipal [--font Assurbanipal]
"""
import json
import re
import sys
import types
import unicodedata
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from src.data import Prototype  # noqa: E402

SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def sign_hex(sign_name):
    """eBL sign name (ŠU₂, DIŠ) -> unicode hex (0x122d9) via the Unicode name 'CUNEIFORM SIGN SHU2'."""
    name = sign_name.strip("|").translate(SUBSCRIPTS).replace("Š", "SH").replace("š", "sh")
    name = re.sub(r"\s+", " ", name.replace("×", " TIMES ")).strip().upper()
    try:
        return f"0x{ord(unicodedata.lookup('CUNEIFORM SIGN ' + name)):x}"
    except KeyError:
        return None

IMG_SIZE = 512
LINE_WIDTH = 14  # roughly the wedge width in the 512px glyph renderings


def draw_skeleton(points, connectivity, width=LINE_WIDTH):
    canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    for i, neighbours in enumerate(connectivity):
        for j in neighbours:
            cv2.line(canvas, tuple(int(v) for v in points[i]), tuple(int(v) for v in points[j]), 255, width)
    return canvas > 0


def overlay(base_rgb, points, connectivity, colour=(255, 0, 0)):
    img = np.array(base_rgb.convert("RGB").resize((IMG_SIZE, IMG_SIZE))).copy()
    for i, neighbours in enumerate(connectivity):
        for j in neighbours:
            cv2.line(img, tuple(int(v) for v in points[i]), tuple(int(v) for v in points[j]), colour, 3)
    for x, y in points:
        cv2.circle(img, (int(x), int(y)), 5, colour, -1)
    return img


def main():
    parser = ArgumentParser()
    parser.add_argument("output_folder", help="name under output/, e.g. singleSigns_Assurbanipal")
    parser.add_argument("--font", default=None, help="Assurbanipal or Santakku (default: guessed from folder name)")
    parser.add_argument("--itr", type=int, default=100, help="which saved iteration to use (0 = initialization only)")
    args = parser.parse_args()
    font = args.font or ("Santakku" if "Santakku" in args.output_folder else "Assurbanipal")

    with open("signs_snippets_metadata.json") as f:
        id_to_sign = {e["_id"]: e["signName"] for e in json.load(f)}
    proto_df = pd.read_csv("prototypes/metadata.csv")
    hex_to_name = dict(zip(proto_df["hex"], proto_df["name"]))

    proto_args = types.SimpleNamespace(df_filename="prototypes/metadata.csv", con_dir=f"skeletons/{font}",
                                       font_dir=f"prototypes/{font}", img_size=IMG_SIZE, line_weighting=0.0)

    rows = []
    for run_dir in sorted(Path("output", args.output_folder).glob("*_results")):
        pts_file = run_dir / f"dst_points_{args.itr}.pt"
        if not pts_file.exists():
            continue
        image_id = run_dir.name[: -len("_results")]
        sign = id_to_sign.get(image_id)
        hex_code = sign_hex(sign) if sign else None
        name = hex_to_name.get(hex_code)
        if name is None:
            print(f"skip {image_id}: cannot resolve sign {sign!r}")
            continue

        proto = Prototype(proto_args, name)
        fitted = torch.load(pts_file).numpy().astype(float)
        fitted[:, 1] = IMG_SIZE - fitted[:, 1]  # save_result() flips y for plotting; undo it
        original = np.array(proto.points, dtype=float)

        shifts = np.linalg.norm(fitted - original, axis=1)
        glyph_ink = np.array(proto.proto.convert("L")) < 128
        fitted_ink = draw_skeleton(fitted, proto.connectivity)
        iou = (glyph_ink & fitted_ink).sum() / (glyph_ink | fitted_ink).sum()

        rows.append({"image": image_id, "sign": sign, "name": name, "hex": hex_code,
                     "mean_shift_px": shifts.mean(), "max_shift_px": shifts.max(), "iou": iou})

        photo = Image.open(f"singleSignImages/{image_id}.jpeg")
        panel = np.hstack([
            np.array(proto.proto.convert("RGB")),
            overlay(proto.proto, fitted, proto.connectivity),
            overlay(photo, fitted, proto.connectivity),
        ])
        Image.fromarray(panel).save(run_dir / "font_compare.png")

    df = pd.DataFrame(rows).sort_values("iou", ascending=False)
    out_csv = Path("output", args.output_folder, "font_compare.csv")
    df.to_csv(out_csv, index=False)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n{len(df)} runs. mean IoU {df.iou.mean():.3f}, mean shift {df.mean_shift_px.mean():.1f}px -> {out_csv}")


if __name__ == "__main__":
    main()
