# Line crops and sign annotations: 60 eBL tablets

Exported 2026-10-16 from the ProtoSnap line-cropping project (`python -m line_cropping.report.export_tablets`).

## What this is

60 cuneiform tablets from the Electronic Babylonian Library (eBL), 15 each in four scripts: Neo-Assyrian (NA),
Neo-Babylonian (NB), Old Assyrian (OA) and Old Babylonian (OB). For each tablet: the photographed faces, one image
crop per text line, and the eBL sign annotations (a box and reading for each annotated sign), plus the
transliteration.

The line crops come from the project's learned line model (run 2, a U-Net that finds text lines without a
transliteration) with its default read-out. All 60 tablets are from the held-out test split, so the model never saw
them in training. Every face is single-column and has at least 90% of its transliterated lines covered by sign
boxes; signs that were not annotated by eBL are not in signs.csv.

Counts: 113 faces, 1015 line crops, 7889 sign boxes (7613 of them inside a line crop).

## Files

- `tablets.csv` — one row per tablet: `tablet` (museum number), `script`, `period` and `genres` (eBL),
  `n_faces`, `n_line_crops`, `n_signs`, `transliteration` (path to the ATF file).
- `faces.csv` — one row per face: `face` (id), `tablet`, `side` (obverse, reverse, ...), `image` (path),
  `width`, `height`, `photo_x0`, `photo_y0` (where the face sits in the eBL photograph), `skew_deg` (rotation used for
  the line crops, see frames), `line_pitch_px` (typical distance between lines), `annotation_coverage`.
- `line_crops.csv` — one row per line crop, in reading order within a face: `crop_id`, `tablet`, `face`, `line_no`
  (1 = top line), `image` (path), `x0`, `y0`, `x1`, `y1` (the crop rectangle in the deskewed face frame), `skew_deg`,
  `atf_line` (the transliteration line label, e.g. `3` or `5'`; filled only where the number of crops on the face
  equals the transliteration's line count, otherwise blank), `n_signs` (annotated signs whose centre is in the crop).
- `signs.csv` — one row per annotated sign: `sign_id`, `tablet`, `face`, `sign_name` (eBL sign list), `reading`
  (transliterated value), `type` (HasSign, Number, Damaged, PartiallyBroken, UnclearSign), `ebl_label` (eBL's line
  label, e.g. `o 3` = obverse line 3), `ebl_path` (eBL's position of the sign in the transliteration:
  line, word, sign), `face_x0..face_y1` (box in face image pixels), `deskew_x0..deskew_y1` (box in the deskewed face
  frame), `crop_id` (the line crop containing the box centre, blank if none), `crop_x0..crop_y1` (box relative to that
  crop image).
- `faces/`, `line_crops/` — the images; `transliterations/<tablet>.atf` — eBL transliteration (ATF).

## Coordinate frames

- **Face frame:** pixels of `faces/<face>.jpg`, origin top left.
- **Deskewed face frame:** the face image rotated by `skew_deg` degrees about its centre (OpenCV
  `cv2.getRotationMatrix2D((width / 2, height / 2), skew_deg, 1.0)`, positive = counter-clockwise), same size. The line
  crops are cut from this rotated image, so a crop's rectangle and the `deskew_*` sign boxes are in this frame.
- **Crop frame:** pixels of the crop image = deskewed frame minus (`x0`, `y0`) of the crop.

## Caveats

- Line crops are model output, not ground truth. On test faces like these the model finds about 87 lines in 100 at
  about 88% precision; occasionally a crop holds no annotated sign (`n_signs` = 0) or a line is missed.
- Crops are gap-bounded: each runs from the middle of the gap above its line to the middle of the gap below, with
  half a line pitch of margin at the ends; a tall sign can be clipped at the boundary.
- Sign boxes were drawn by eBL annotators on the eBL photographs; a few boxes overlap or are loose, and a tall box can
  extend past the top or bottom of its crop, so `crop_*` coordinates can be negative or exceed the crop size.

## Provenance and licence

Photographs and sign annotations: the Electronic Babylonian Library (https://www.ebl.lmu.de/), photographs by the
holding museums (British Museum images CC BY-NC-SA 4.0; others vary). Sign annotations are also published as the eBL
sign dataset (Lewenstein et al. 2026, Journal of Open Humanities Data, https://doi.org/10.5334/johd.503; CC BY-NC
4.0). For research use; check the holding institution's terms before redistributing the images.
