# line_cropping — implementation of docs/line_cropping_plan.md, phase by phase

Run everything from the repo root with `python -m line_cropping.<stage>.<module>`.

New to the project? Read `docs/pipeline_overview.md` first: it explains the whole pipeline in plain language.

## Layout

| folder | stage | modules |
|---|---|---|
| `ingest/` | get data in | `extract_atf.py`, `match_museum.py` (CDLI dump -> transliterations, museum-number join), `download_ebl.py` (eBL photos + annotations + ATF), `inventory.py`, `face_qc.py` (Old Assyrian fat-cross set) |
| `truth/` | ground truth | `ebl_ground_truth.py` (lines from eBL sign boxes), `oa_sample.py`, `labels_from_db.py`, `labeler/` (source of the Old Assyrian Line Marker page) |
| `faces/` | Phase 2 | `prepare_faces.py` (cut sides from composite photos; Old Assyrian face list and edge rotation) |
| `lines/` | Phase 3 | `profile_lines.py`, `find_lines.py` (profile methods), `detect_signs.py` (eBL detector, mmdet env), `lines_from_boxes.py` (boxes -> lines) |
| `crops/` | Phase 4 | `make_crops.py` (gap-bounded single-line crops) |
| `eval/` | Phase 5 | `evaluate_lines.py`, `evaluate_crops.py`, `evaluate_oa.py`, `phase5_eval.py`, `render_lines.py` |
| `protosnap/` | Phase 6 | `protosnap_prep.py`, `phase6_summary.py` |
| `experiments/` | early trials | `profile_lines_oa.py`, `comb_fit_oa.py` (the first Old Assyrian profile experiments) |

Data lives outside the package and is gitignored: `artifacts_json/` (CDLI dump and derived tables), `ebl_tablets/`
(eBL downloads, ground truth, faces, detections, predictions, line crops, `logs/`), `fat-cross_processed/`
(Old Assyrian faces and everything derived from them), `ebl_detection_2024/` (Zenodo archive), `protosnap_inputs/`
(sign targets), `external/cuneiform-ocr` (eBL detector code), `weights/ebl_detr/`. Notes on the external data sets
are in `docs/`. Runs of ProtoSnap itself land in `output/` as before.

## Phase 1 — ground truth (status: done, awaiting review)

**1a. eBL tablets (Neo-Assyrian, Old Babylonian)** — `truth/ebl_ground_truth.py`

Turns the eBL sign boxes in `ebl_tablets/<NA|OB>/<tablet>/annotations.json` into line-level truth.
Signs are grouped by side, object (tablet/envelope), column and eBL text-line index. Side comes from the
eBL crop label (`o`, `r`, `t.e.`, `b.e.`), else from the eBL surface box containing the sign, else it is
inferred from text order (an unlabelled line takes the side whose line-index range contains it).

Outputs in `ebl_tablets/ground_truth/`:

| file | content |
|---|---|
| `lines/<tablet>.csv` | one row per line. Oriented segment `ax0,ay0 -> ax1,ay1` (centre line fitted through the sign centres, spanning the annotated signs) with `thickness` (1.15 x median sign height); also the axis-aligned union box `x0,y0,x1,y1`, block extent, `cy`, `tilt_deg`, `n_signs`, `sparse` (<= 2 signs, drawn thin in previews), `primed`, `is_first_line`, `side_inferred`, `column_inferred` |
| `blocks.csv` | one row per (tablet, side, object, column): `n_lines`, `pitch_px`, `median_tilt_deg`, block box, first line label, `first_line_primed` |
| `tablets.csv` | one row per tablet with counts and photo size |
| `previews/<tablet>.jpg` | overlay: one colour per block, thick box = first line of the block |

When the eBL transliteration is available, side, column and object are first read from it: eBL annotation
paths index the eBL text lines (a physical ATF line ending in `&` continues on the next), so each sign's line
index falls under a known `@obverse`/`@column N`/`@envelope` marker. The map is validated against the labelled
signs and surface markers (`atf_map_agreement` in `tablets.csv`) and ignored below 0.9 agreement, which
happens when the eBL edition was edited after annotation (e.g. K.4426). Tablets whose edition names no side at
all (one-sided fragments starting at `1'.`) get the single side `face`. Column comes from the label numeral; on sides that have labelled columns, lines without a numeral are
assigned to the column whose text-line index range contains them, else to the column their signs overlap
most (`column_inferred`). Line boxes are the union of the line's annotated sign boxes; `block_x0/x1` is the
extent of the whole block and is only an auxiliary.

Re-run as more tablets download: `python -m line_cropping.truth.ebl_ground_truth --preview 10` (it re-reads
`ebl_tablets/manifest.csv`). Use the oriented segment, not the axis-aligned box, as the line truth: 56% of lines tilt more than 3 degrees
and axis-aligned boxes of neighbouring lines overlapped in 69% of cases versus 24% for the oriented bands.
Coverage is partial: eBL annotators did not box every line, so compare `n_lines` with `atf_lines` per tablet
(`tablets.csv`) and evaluate only on tablets or blocks with high coverage. Known limits: 14 tablets have no side information at all and stay
`unknown`; line boxes only cover annotated signs, so an unannotated line is simply absent from the truth
(check `n_lines` against `atf_lines` in `tablets.csv` to find sparsely annotated tablets).

**1b. Old Assyrian faces** — `truth/oa_sample.py` (uses `lines/profile_lines.py`)

Picks 100 faces from `fat-cross_processed/experiments/face_qc.csv` (unflagged, with ATF), stratified by
resolution (px per line: low <25, mid <40, high), ATF line-count bucket and face, split alternately into
`tune` and `test`. For each face it writes automatic pre-labels from the ATF-constrained comb fit.

Outputs in `fat-cross_processed/ground_truth/`:

| file | content |
|---|---|
| `sample_faces.csv` | the sample: file, ATF line count, px per line, stratum, split |
| `prelabels.csv` | per face and line: `skew_deg`, `pitch_px`, `y_deskewed` (line centre in the deskewed face), body x-extent, `source=comb_fit` |
| `previews/<face>.png` | deskewed face with the pre-labelled lines numbered, to correct against |

Labelling is done in the published page **Old Assyrian Line Marker** (source in `truth/labeler/`; images are the 100
faces deskewed and downscaled to 900 px wide). Click adds a line, drag moves, Delete removes, D marks a face done;
the ATF count is shown next to the marked count. Marks save to the page's database as `labels/<face>` documents.
To use them: pull the `labels` collection with the Artifact tool's `read_db` into
`fat-cross_processed/ground_truth/db/`, then run `python -m line_cropping.truth.labels_from_db` to write
`fat-cross_processed/ground_truth/labels.csv` in face coordinates.

The older path, hand-editing `prelabels.csv`, still works: correcting `y_deskewed` per line (and deleting or adding rows) so that each row is a
real line centre; set `source` to `manual` on corrected rows. The pre-labels are rough on broken
fragments and on faces where the ATF counts lines that are not on this face.

## Phase 2 — face preparation (status: done, awaiting review)

`prepare_faces.py ebl` cuts every annotated side (with its object, e.g. tablet vs envelope) out of the eBL
composite photo and rewrites the line truth in face-crop coordinates. The face outline comes from the image,
not from the annotations: the photo is greyscaled and thresholded against the black studio background, connected
bright blobs are the tablet views, touching views are split at near-empty gaps in the blob's column/row profile,
and every blob that holds a share of the side's Phase 1 lines is taken (a side broken over several fragments
stays whole). `crop_method` in `faces.csv` says `blob` or, where no blob held the lines, `lines` (union of the
line segments plus one pitch of margin). It does not rotate the crop; `skew_deg` (minus the median block tilt) is recorded
for the line finder, which deskews itself.

Outputs in `ebl_tablets/faces/`: `<tablet>__<side>[__<object>].jpg`, `lines/<same>.csv` (oriented segments in
crop coordinates), `faces.csv` (crop box in the photo, size, `n_columns`, `n_lines`, `atf_lines_side` from
eBL's own ATF, `coverage`, `pitch_px`, `skew_deg`). 1,367 faces from 872 tablets (536 obverse, 460 reverse, 248 unspecified `face`, 123 edges); median
122 px per line; 53% of faces have >= 0.8 coverage; 33 faces have more than one column.

`prepare_faces.py oa --rot-left cw --rot-right ccw` writes `fat-cross_processed/prepared/faces.csv`: every
obverse/reverse that passed `face_qc.csv` (accepted=True) plus rejected ones with the reason, and every left or
right edge whose ATF has text on that edge, rotated so lines run horizontally and saved under `prepared/edges/`.
Rotation direction was chosen by eye on P357359 (clockwise for the left edge gives wedge heads at left/top);
verify on a few more before trusting edge crops. 3,290 accepted faces: 1,359 obverse, 1,254 reverse, 660 left,
17 right.

Re-run both after new downloads or QC changes; they are idempotent.

## Phase 3 — line finding (status: method chosen, awaiting review)

Five methods were scored on 980 single-column eBL faces against the Phase 1 truth (`eval/evaluate_lines.py`;
a line counts as found when its centre is within 0.35 pitch; precision only on faces with >= 0.9 coverage;
first-line error only where the topmost truth line is line 1 or 1'):

| method | what it does | recall | precision | first line within 0.5 pitch |
|---|---|---|---|---|
| `free` | profile peaks, no prior | 0.49 | 0.59 | 25% |
| `comb` | N lines on a rigid comb (N from ATF) | 0.60 | 0.58 | 30% |
| `dp` | N lines by dynamic programme on the profile | 0.62 | 0.59 | 30% |
| `det` | eBL Deformable DETR boxes -> tilt -> 1-D clustering | 0.77 | 0.87 | 77% |
| `det_n` | as `det`, then split over-merged clusters, fill gaps, merge to N from the ATF | **0.87** | **0.87** | **89%** |

The profile methods plateau because the deskewed row profile is only weakly in phase with the line bands and
small pitch errors accumulate over a face; an oracle with the true tilt, count and extent still reached only
0.65. The detector localises signs directly, so its lines follow curvature and gaps. On faces with >= 120 px
per line `det_n` reaches recall 0.89 and 92% first-line accuracy; below 60 px per line it drops to 0.71/67%.

**Old Assyrian faces (no truth yet).** The detector finds a median of 22 boxes per face (51 on eBL faces) and
its line count without the ATF agrees with the ATF within one line on only 18% of faces (63% on left edges,
which carry few lines); tiling the face into 2 or 3 bands to raise the effective resolution did not add boxes.
`det_n` therefore leans on the ATF count and on gap filling (missing lines are inserted where clusters are more
than 1.5 pitches apart), and the profile `dp` was also run on all 3,290 accepted faces
(`fat-cross_processed/predictions_profile/`). Which of the two is better on this set cannot be measured until the
Phase 1b sample is hand-labelled; side-by-side overlays for review are in `fat-cross_processed/predictions/previews_det_n/`
and `fat-cross_processed/predictions_profile/previews_dp/`.

**Old Assyrian faces, scored against 88 hand-labelled faces** (`eval/evaluate_oa.py`, labels from the Line Marker page;
785 lines, 49% of them moved or added by hand, the rest left where the profile fit had put them):

| method | recall | precision | first line within 0.5 pitch | recall on hand-placed lines only |
|---|---|---|---|---|
| `dp` (profile, N from ATF) | 0.89 | 0.76 | 69% | 0.78 |
| `comb` | 0.84 | 0.73 | 64% | — |
| `det_n` (detector, N from ATF) | 0.43 | 0.37 | 33% | 0.33 |
| `det` | 0.35 | 0.49 | 34% | — |

So the two sets want different line finders: the detector on the high-resolution eBL photographs, the profile
dynamic programme on the Old Assyrian faces, where the detector finds too few signs. The labels also show that
the ATF line count is not the number of lines visible on the face: on 49% of faces fewer lines were labelled than
the ATF lists (median 3 fewer), which is where `dp`'s missing precision goes. See the count-selection note below.

**Count selection (open).** With the true line count, `dp` reaches recall 0.88 / precision 0.88 / first line within
0.5 pitch 79% on the labelled faces, so the count is the remaining lever. Choosing N by mean profile response
collapses to the ATF count; estimating N from the text extent divided by the autocorrelation pitch (pitch search
bounded by the ATF count, right within 15% on 70% of faces) trades precision for first-line accuracy
(0.81 / 57% against 0.75 / 69%). A better criterion, or a labelled count from the ATF's state lines
(`$ rest broken`, lines that continue on an edge), is Phase 4 work.

Decision: on eBL-style photographs `det_n` is the line finder when a transliteration exists and `det` when it
does not; on the Old Assyrian faces `dp` is the line finder, with the ATF count treated as an upper bound rather
than the exact number of lines.

Files: `lines/find_lines.py` (profile methods), `lines/detect_signs.py` (run in `~/venvs/mmdet`; weights in
`weights/ebl_detr/`, config from `external/cuneiform-ocr/configs/detr.py`), `lines/lines_from_boxes.py` (boxes ->
lines), `eval/evaluate_lines.py`. Predictions: `ebl_tablets/predictions/<method>/<face>.csv`, detections:
`ebl_tablets/detections/<face>.csv`. The detector processes ~6 faces/s on the RTX 4060.

## Scope note (12 Sep 2026)
The Old Assyrian set is parked by decision of the project owner; Phases 4 onward proceed on the eBL set with the
detector route. The Old Assyrian labels, predictions and the count-selection question stay in place for later.

## Phase 4 — crop generation (status: done on the eBL set; gap-bounded crops)

Goal restated by the project owner: a line crop must hold its own line only, with no slices of the lines above
and below. `crops/make_crops.py` (default `--bounds gap`) therefore sets each line's vertical extent from the detector
boxes assigned to it (10th to 90th percentile of box tops and bottoms in the deskewed frame) and cuts from the
midpoint of the gap to the line above to the midpoint of the gap to the line below; where two lines' boxes overlap
the boundary is the midpoint of their centres, and the first and last lines get 6% of a pitch of margin. Columns
span the line's detected extent plus half a pitch each side. Crops go to `ebl_tablets/line_crops/<face>/<face>__L<nn>.jpg`;
`crops.csv` records the rectangle in the deskewed face, the skew, the face's offset in the photo, and the ATF line
label when the predicted count equals the ATF count for a single-column side (`atf_aligned`).

| item | gap-bounded (current) | fixed 2-pitch band (`line_crops_band/`, superseded) |
|---|---|---|
| crops | 17,117 from 1,327 faces | 17,200 |
| median crop | 1,037 x 102 px (height = 1.0 pitch, 10th-90th pct 0.89-1.31) | 1,030 x 198 px |
| truth line centres inside exactly one crop | **99%** | 28% |
| truth line centres inside two or more crops | **0%** | 71% |
| crops holding the centres of two lines (neighbour sliced in) | **2%** | 30% |
| truth line centres in no crop | 1% | 1% |
| crops with an ATF line label | 74% | 75% |

The older "truth band fully inside one crop" metric (83% for the band crops, 5% for the gap crops) is not the right
yardstick here: eBL sign boxes are about as tall as the line pitch and overhang into the neighbouring rows, so a crop
that contains a whole eBL band must contain parts of the neighbours by construction. The gap crops keep wedge tops
and bottoms up to the detected box extents, which is what a single-line crop can do without admitting the next line.
Evaluate with `python -m line_cropping.eval.evaluate_crops --faces ebl_tablets/faces/faces.csv --crops ebl_tablets/line_crops/crops.csv`.
The band set can be deleted once nothing needs it (2.3 GB).

## Phase 5 — evaluation (status: done on the eBL set)

`eval/phase5_eval.py` splits the 1,334 single-column eBL faces with detections into tune and test halves, stratified
by script and pixels-per-line bucket (`ebl_tablets/split.csv`, seed 7), grid-searches the clustering thresholds
on the tune half only (chosen: `gap_frac 0.5`, `merge_frac 0.35`, `score_thr 0.3`, saved to
`ebl_tablets/det_params.json`, which `lines/lines_from_boxes.py` now reads by default), and reports the test half.
Faces with fewer than two truth lines are not scored (487 of 666 test faces are).

Test half, `det_n` (detector boxes + ATF count):

| group | n | recall | precision | first line < 0.5 pitch | count match |
|---|---|---|---|---|---|
| all | 487 | 0.91 | 0.91 | 90% | 85% |
| OB, >= 120 px/line | 229 | 0.90 | 0.91 | 93% | 99% |
| OB, 60-120 | 95 | 0.93 | 0.92 | 95% | 98% |
| NA, >= 120 | 20 | 0.93 | 0.92 | 71% | 70% |
| NA, 60-120 | 114 | 0.94 | 0.93 | 78% | 54% |
| NA, < 60 | 26 | 0.82 | 0.63 | 57% | 54% |

Without the count (`det`): recall 0.87, precision 0.90, first line < 0.5 pitch 88%, count match 25%.

**Plan criteria (section 7 of the plan), test half, faces with >= 120 px/line:** recall 0.90 (target 0.90),
first line within 0.5 pitch on 92% of faces (target 90%), count agreement 97% (target 80%). All three met.
Below 60 px/line the method is reported but not gated, as the plan allows.

Failure categories among the 51 test faces with recall < 0.7: 18 placement errors with no single cause,
10 sparsely annotated faces (< 50% coverage, so the truth itself is thin), 9 count far off, 5 with too few
detections, 5 with an undeclared second column (truth lines interleave), 4 low resolution. Multi-column faces
(33) were excluded from the run and remain out of scope. Per-face results: `ebl_tablets/phase5_test_results.csv`.

## Phase 6 — ProtoSnap connection (status: sign-level trial done; see note)

Note: the project's intent for this phase is clean single-line crops for ProtoSnap, which Phase 4's gap-bounded
crops now provide. The sign-level snapping trial below was a detour beyond that intent; it is kept because its
alignment step (ATF signs to detector boxes) is what a per-line ProtoSnap run will need, but it is not the deliverable.

`protosnap/protosnap_prep.py` turns aligned line crops into ProtoSnap inputs. For each crop with an ATF line label it
tokenises the eBL ATF line into sign names (transliteration values are mapped to signs with a frequency table
built from `signs_snippets_metadata.json`; 962 distinct OB values, 48 of them ambiguous), collects the detector
boxes whose centres fall in the crop, sorts them left to right, aligns the two sequences with Needleman-Wunsch
(+2 same sign name, 0 different, -1 gap), and saves each matched box (15% padding) as a target image for every
sign that has a Santakku prototype and skeleton. `samples.csv` has the columns `run_test.py` reads; `alignment.csv`
records, per ATF sign, the matched box, the detector's class, and the overlap with the eBL truth box of that sign.

First 400 OB line crops:

| item | value |
|---|---|
| ATF tokens / with a sign name | 3,552 / 2,859 |
| detector boxes inside those lines | 2,094 |
| ATF signs matched to a box | 58% |
| detector class equals the ATF sign (on matched boxes) | 52% |
| matched box is the eBL truth box of that sign (IoU > 0.5, where checkable) | 70% |
| ProtoSnap targets written | 1,567 (882 where detector and ATF agree on the sign; 584 of those verified against truth) |

So the detector finds about six boxes for every ten signs the transliteration lists, and the alignment places the
right sign on the right box seven times in ten. The safe subset for ProtoSnap is the 882 targets where the
detector's own class agrees with the ATF, listed in `samples_agree.csv`. A 40-target sample (`samples_run40.csv`)
was run with `run_test.py --font_dir prototypes/Santakku --con_dir skeletons/Santakku --output_folder phase6_OB`:
all 40 completed without error in about 40 minutes on the RTX 4060 (initialisation dominates; the 100 optimisation
steps take ~7 s). `protosnap/phase6_summary.py` writes `output/phase6_OB/summary.csv` and a `contact_sheet.jpg` of the
per-target `itr99_viz.png` renders. The initialisation score (ProtoSnap's own agreement measure) has a median of
0.60 against 0.75 on the repo's curated single-sign test set: the pipeline runs end to end, and roughly half the
snaps look right by eye. The box itself is rarely the problem: 28 of the 40 boxes coincide with the eBL truth box of
that sign (IoU > 0.5) and none contradicts it (12 have no truth to check), and verified boxes score no better than
the rest (median 0.60 both). The weak snaps come from the sign images themselves: 15% padding often admits parts of
the neighbouring sign, and worn or shallow wedges give the diffusion features little to lock onto. Tighter boxes
(no padding, or padding only where the neighbour is far) and the eBL truth boxes as an upper-bound experiment are
the next two things to try.

**First sign of each line (16 Sep 2026).** `protosnap_prep.py --first-only` writes a target only when the first ATF
token of a line is matched to the leftmost detector box in its crop. Over all 6,299 aligned OB lines that holds for
2,387 lines (38%); 1,961 have a Santakku prototype, 1,823 with detector and ATF agreeing on the sign, 1,675 confirmed
by the eBL truth box (`protosnap_inputs/OB_first/`). A 60-target run, one per face (`samples_run.csv`, results in
`output/phase6_OB_first/`), finished with a median initialisation score of 0.54 (40-sign trial 0.60; curated test
set 0.75); 59 of the 60 boxes coincide with the eBL truth box. First signs are harder than average for two reasons
visible in the overlays: they sit at the tablet's left edge, so the padded box often includes background or a broken
margin, and the commonest first signs (DIŠ 16, IGI 12 of 60) are a single vertical wedge or a dense cluster, both
weakly constrained. Note: `run_test.py` must be run with `MPLBACKEND=Agg` in a headless shell; the Tk backend
aborted the first attempt after 21 targets.

`overlay_results.py` projects each fitted skeleton back onto the unrotated face and onto the gap-bounded line crop of
its line (undoing ProtoSnap's 512-px stretch and y flip, then applying the face deskew), writing
`output/phase6_OB/overlays/{faces,lines}/` plus `line_contact_sheet.jpg` and `index.csv`.

## Old Assyrian set without ground truth (experiment, 15 Sep 2026)

To work on the parked Old Assyrian set without labels, three additions stay inside the existing layout:

- `crops/make_crops.py --bounds midpoint --atf-source cdli`: single-line crops cut halfway between consecutive
  predicted line centres, so no detector boxes are needed; ATF line labels come from the CDLI dump via the face's
  P-number. Run on the profile `dp` predictions: 30,472 crops from 3,289 faces, 88% with an ATF line label, median
  601 x 48 px, in `fat-cross_processed/line_crops/`.
- `eval/confidence_oa.py`: per-face confidence without truth from (a) agreement between the two independent line
  finders (`dp` vs `det_n`, within 0.35 pitch), (b) the share of detector boxes within 0.3 pitch of a predicted line
  centre, (c) spacing regularity. Two tempting signals are uninformative here and are reported but not scored: the
  ATF count (enforced, so always matched) and crop containment (midpoint crops tile the face). Result:
  `fat-cross_processed/confidence.csv`, 3,172 faces; median agreement 0.50, median box proximity 0.44, median
  confidence 0.64, lowest decile below 0.45; the two signals correlate at 0.50.
- `eval/render_lines.py --sheet`: contact sheets of any ranked list of faces; the eight least and eight most
  confident faces are in `fat-cross_processed/review/`.

The ranking is a review queue, not a metric: look at the bottom of it first.

## Later phases
None: all six phases have a first implementation; see the status notes in `docs/line_cropping_plan.md`.
