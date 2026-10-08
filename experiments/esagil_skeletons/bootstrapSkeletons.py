"""Transfer skeletons from an annotated font (Assurbanipal) to a new font of the same signs.

Three steps:
  1. python add_new_font_skeletons/bootstrapSkeletons.py prepare Esagil
       Renders each Esagil glyph the way the repo processes prototypes (crop, pad, resize to 512)
       into fontTargets/Esagil/, and writes test_set/transfer_Esagil.csv listing every sign that has
       both an Esagil glyph and an Assurbanipal skeleton.
  2. python run_test.py --samples_df_path test_set/transfer_Esagil.csv \
         --font_dir prototypes/Assurbanipal --con_dir skeletons/Assurbanipal \
         --output_folder transfer_Esagil --ignore_errors
       This is the normal pipeline: it snaps the Assurbanipal skeleton onto each Esagil glyph.
  3. python add_new_font_skeletons/bootstrapSkeletons.py convert Esagil
       Reads each run's fitted points, maps them back into the raw glyph image's coordinate frame,
       and writes skeletons/Esagil/<hex>_adf.csv (the _con.csv files are copied unchanged, because
       the transfer never changes which points connect to which).

Prerequisite: prototypes/Esagil/0x12xxx/<hex>.png glyph images must exist (rendered from the font).
Afterwards, eyeball output/transfer_Esagil/*/itr99_viz.png and hand-fix signs whose wedge count differs.
"""
import csv
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for src imports

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.utils import crop_img, get_proto_img, get_path_to_proto

SOURCE_FONT = "Assurbanipal"
IMG_SIZE = 512
PAD = 10


class ProtoArgs:
    def __init__(self, font):
        self.font_dir = f"prototypes/{font}"
        self.df_filename = "prototypes/metadata.csv"


def signs_with_source_skeletons():
    return {fn.split("_")[0] for fn in os.listdir(f"skeletons/{SOURCE_FONT}") if fn.endswith("_adf.csv")}


def prepare(font):
    df = pd.read_csv("prototypes/metadata.csv")
    df = df[df["name"].notna()]
    targets_dir = Path("fontTargets") / font
    targets_dir.mkdir(parents=True, exist_ok=True)

    rows, missing = [], 0
    for hex_code in sorted(signs_with_source_skeletons()):
        match = df[df["hex"] == hex_code]
        if match.empty:
            continue
        name = match["name"].values[0]
        glyph_path = get_path_to_proto(ProtoArgs(font), name, df)
        if not os.path.exists(glyph_path):
            missing += 1
            continue
        # process the glyph exactly like the repo processes a prototype (crop, pad, resize),
        # so the fitted points live in a frame we can map back to the raw image in convert()
        try:
            processed, _, _, _, _ = get_proto_img(ProtoArgs(font), name, df)
        except ValueError:  # blank glyph image (font lacks this sign despite the file existing)
            missing += 1
            continue
        target_path = targets_dir / f"{hex_code}.png"
        processed.save(target_path)
        rows.append([f"{hex_code}.png", str(target_path), "", name, hex_code])

    out_csv = Path("test_set") / f"transfer_{font}.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fn", "file_path", "abz", "name", "hex"])
        writer.writerows(rows)
    print(f"{len(rows)} signs ready ({missing} had no {font} glyph). Now run:")
    print(f"  python run_test.py --samples_df_path {out_csv} --font_dir prototypes/{SOURCE_FONT} "
          f"--con_dir skeletons/{SOURCE_FONT} --output_folder transfer_{font} --ignore_errors")


def regroup_order(labels):
    """dst_points_*.pt rows are stacked stroke-group by stroke-group (see GroupedPoints);
    return the original-point index for each stacked row so we can undo that ordering."""
    order = []
    for L in set(labels):  # replicate the repo's iteration over the label set
        order += [i for i, lab in enumerate(labels) if lab == L]
    return order


def convert(font):
    df = pd.read_csv("prototypes/metadata.csv")
    df = df[df["name"].notna()]
    skel_dir = Path("skeletons") / font
    skel_dir.mkdir(parents=True, exist_ok=True)

    done, skipped = 0, 0
    for run_dir in sorted(Path("output", f"transfer_{font}").glob("*_results")):
        hex_code = run_dir.name.split("_")[0]
        pts_file = run_dir / "dst_points_100.pt"
        if not pts_file.exists():
            skipped += 1
            continue
        name = df[df["hex"] == hex_code]["name"].values[0]

        # the source skeleton provides labels and row order
        src_adf = pd.read_csv(f"skeletons/{SOURCE_FONT}/{hex_code}_adf.csv")
        labels = [int(lab.split()[1]) for lab in src_adf["label"]]

        fitted = torch.load(pts_file).numpy().astype(float)
        fitted[:, 1] = IMG_SIZE - fitted[:, 1]  # save_result() flips y for plotting; undo it
        unstacked = np.empty_like(fitted)
        unstacked[regroup_order(labels)] = fitted  # back to the adf row order

        # map from the processed 512-frame back to the raw glyph image's pixel frame,
        # inverting Prototype.global_adjustment: raw = fitted / scale - PAD + crop_start
        glyph = Image.open(get_path_to_proto(ProtoArgs(font), name, df)).convert("L")
        _, cs, _, rs, _ = crop_img(glyph, is_pil=True)
        _, _, _, w, h = get_proto_img(ProtoArgs(font), name, df)  # size after crop+pad
        xs = unstacked[:, 0] * w / IMG_SIZE - PAD + cs
        ys = unstacked[:, 1] * h / IMG_SIZE - PAD + rs

        out = pd.DataFrame({"label": src_adf["label"], "x": np.rint(xs).astype(int), "y": np.rint(ys).astype(int)})
        out.to_csv(skel_dir / f"{hex_code}_adf.csv", index=False)
        shutil.copy(f"skeletons/{SOURCE_FONT}/{hex_code}_con.csv", skel_dir / f"{hex_code}_con.csv")
        done += 1
    print(f"Wrote {done} skeletons to {skel_dir} ({skipped} runs had no result). "
          f"Review overlays in output/transfer_{font}/ before trusting them.")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("prepare", "convert"):
        sys.exit(__doc__)
    {"prepare": prepare, "convert": convert}[sys.argv[1]](sys.argv[2])
