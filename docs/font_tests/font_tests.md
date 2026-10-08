# Font tests: ProtoSnap on photographed single signs

The first part of this fork ran ProtoSnap unchanged on photographs of single signs, using the two fonts the
repository shipped with: Assurbanipal (Neo-Assyrian) and Santakku (Old Babylonian). The question was how far
ProtoSnap's fitted skeleton moves away from the font glyph it starts from. That shows how much the alignment adapts
to the photograph, and how much a sign's handwritten form differs from the font.

## Data

The photographs are the eBL sign crops (`data/ebl/singleSignImages/`, listed in `data/ebl/signs_snippets_metadata.json`).
`experiments/font_tests/testSingleSigns.py` keeps only crops that ProtoSnap can run: the script matches the font
(NA for Assurbanipal, OB for Santakku), the sign name maps to a Unicode code point, and the font has a prototype and
a skeleton for that sign. It then draws a random sample and writes `test_set/singleSigns_<font>.csv` in the format
`run_test.py` reads.

| font | crops listed | runs finished | results |
|---|---|---|---|
| Assurbanipal | 50 | 51 folders, 51 scored | `output/font_tests/singleSigns_Assurbanipal/` |
| Santakku | 17,895 (every usable crop) | 12 | `output/font_tests/singleSigns_Santakku/` |

The Santakku run was stopped after 12 signs.

## Measures

`experiments/font_tests/compareToFont.py` compares each finished run with the font glyph it started from. It does
not measure accuracy against the photograph, for which there is no ground truth.

- **Mean shift:** the average distance, in pixels, that each skeleton keypoint moved from its position in the font
  skeleton.
- **Max shift:** the largest single keypoint move.
- **IoU:** the overlap between the fitted skeleton, drawn with thick strokes, and the glyph's ink.

It writes `font_compare.csv` per run folder and a `font_compare.png` per sign: glyph, glyph with the fitted skeleton,
and photograph with the fitted skeleton.

## Results

| font | signs | mean shift (median) | max shift (median) | IoU (median) |
|---|---|---|---|---|
| Assurbanipal | 51 | 65.9 px (61.6) | 123.4 px (114.4) | 0.136 (0.131) |
| Santakku | 12 | 57.3 px (52.1) | 101.5 px (97.5) | 0.126 (0.121) |

The fitted skeletons end up far from the font: keypoints move about 60 pixels on average and overlap the glyph's ink
by about 13 percent. ProtoSnap does adapt the prototype to the photograph rather than keeping the font's shape. With
no truth for the photographs these numbers cannot say whether the adapted skeleton is right; the `font_compare.png`
panels have to be judged by eye.

## Reproduce

```
python experiments/font_tests/testSingleSigns.py            # set fontDir in the script; asks how many signs to sample
python run_test.py --samples_df_path test_set/singleSigns_Assurbanipal.csv --font_dir prototypes/Assurbanipal \
    --con_dir skeletons/Assurbanipal --output_folder font_tests/singleSigns_Assurbanipal --ignore_errors
python experiments/font_tests/compareToFont.py font_tests/singleSigns_Assurbanipal
```
