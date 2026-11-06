# line_cropping — implementation of docs/line_cropping/line_cropping_plan.md, phase by phase

Run everything from the repo root with `python -m line_cropping.<stage>.<module>`.

New to the project? Read `docs/line_cropping/pipeline_overview.md` first: it explains the whole pipeline in plain language.

**Two parts of the project share this package.**

- **First iteration (phases 1–6):** This part uses `ingest/`, `truth/`, `faces/`, `lines/`, `crops/`, `eval/`, `signs/`, `protosnap/`, `report/`, `experiments/`. Lines come from the eBL sign detector plus the transliteration's line count. The docs are in `docs/line_cropping/`.
- **Custom line model (phase 7):** This part uses `linemodel/` (a U-Net that predicts text-line bands directly) and `app/` (the Line Splitter web app). The docs are in `docs/line_model/line_model.md`.

Both use the same data root (`paths.py`), the same crop maker (`crops/make_crops.py`) and the same evaluators (`eval/`), so they stay in one package.

**Data location.** All data lives under `tablet_data/` (`tablet_data/ebl_tablets/`, `tablet_data/fat-cross_processed/`,
`tablet_data/app_sessions/`), which is defined once in `line_cropping/paths.py`. Set `LINECROP_DATA_ROOT` to use another folder. After
moving the data, `python -m line_cropping.paths --rewrite <old-prefix>/` re-roots the image paths stored in the tables.

## Layout

| folder | stage | modules |
|---|---|---|
| `ingest/` | get data in | `extract_atf.py`, `match_museum.py` (CDLI dump -> transliterations, museum-number join), `download_ebl.py` (eBL photos + annotations + ATF; `--scripts` picks scripts, `--annotations-dump` reads boxes from the eBL annotation export), `inventory.py`, `face_qc.py` (Old Assyrian fat-cross set) |
| `truth/` | ground truth | `ebl_ground_truth.py` (lines from eBL sign boxes), `oa_sample.py`, `labels_from_db.py`, `labeler/` (source of the Old Assyrian Line Marker page) |
| `faces/` | Phase 2 | `prepare_faces.py` (cut sides from composite photos; Old Assyrian face list and edge rotation) |
| `lines/` | Phase 3 | `profile_lines.py`, `find_lines.py` (profile methods), `detect_signs.py` (eBL detector, mmdet env), `lines_from_boxes.py` (boxes -> lines) |
| `signs/` | sign classifier | `build_dataset.py` (per-script datasets from `data/ebl/signs_snippets_metadata.json`, face-level hold-out), `train_classifier.py` (ResNet-34), `classify.py` (inference) |
| `crops/` | Phase 4 | `make_crops.py` (gap-bounded single-line crops) |
| `eval/` | Phase 5 | `evaluate_lines.py`, `evaluate_crops.py`, `crop_width.py`, `evaluate_oa.py`, `phase5_eval.py`, `evaluate_heidelberg.py` (line model vs hand-drawn Neo-Assyrian lines, Dencker et al. 2020), `extend_split.py` (adds new faces to split.csv without moving existing ones; do not rerun `phase5_eval` for that, it reshuffles the split), `render_lines.py` |
| `linemodel/` | Phase 7 | `targets.py`, `dataset.py`, `model.py` (U-Net, ResNet-34 encoder), `train.py`, `visualize.py`, `infer.py`, `render_predictions.py` |
| `app/` | Phase 7, local web app | `server.py`, `pipeline.py`, `static/index.html`: upload a photograph, split it into line crops, compare with the original |
| `protosnap/` | Phase 6 | `protosnap_prep.py`, `phase6_summary.py` |
| `report/` | reporting | `aggregate_metrics.py` (every metric table of both sets -> `docs/line_cropping/metrics_comparison.md`, per-set `metrics/`), `compile_sample.py` (tracked `output/line_cropping/samples/`: metrics tables + 20 tablets per set with all outputs) |
| `baselines/` | comparisons | `dencker_linenet.py` (Dencker et al. 2020 line network, re-implemented and retrained), `kraken_seg.py` (Kraken's segmenter: PageXML export, prediction in its own venv, fragment join); see docs/line_model/related_work.md |
| `experiments/` | early trials | `profile_lines_oa.py`, `comb_fit_oa.py` (the first Old Assyrian profile experiments) |

Data lives outside the package and is gitignored: `data/cdli/artifacts_json/` (CDLI dump and derived tables), `tablet_data/ebl_tablets/`
(eBL downloads, ground truth, faces, detections, predictions, line crops, `logs/`), `tablet_data/fat-cross_processed/`
(Old Assyrian faces and everything derived from them), `data/ebl/ebl_detection_2024/` (Zenodo archive), `tablet_data/protosnap_inputs/`
(sign targets), `external/cuneiform-ocr` (eBL detector code), `weights/ebl_detr/`. Notes on the external data sets
are in `docs/`. Runs of ProtoSnap itself land in `output/` as before.

## Phase 1 — ground truth (status: done, awaiting review)

**1a. eBL tablets (Neo-Assyrian, Old Babylonian)** — `truth/ebl_ground_truth.py`

This script turns the eBL sign boxes in `tablet_data/ebl_tablets/<NA|OB>/<tablet>/annotations.json` into line-level truth.
Signs are grouped by side, object (tablet/envelope), column and eBL text-line index. Side comes from the
eBL crop label (`o`, `r`, `t.e.`, `b.e.`). Failing that, it comes from the eBL surface box containing the sign, and failing
that, it is inferred from text order (an unlabelled line takes the side whose line-index range contains it).

The outputs are in `tablet_data/ebl_tablets/ground_truth/`:

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
all (one-sided fragments starting at `1'.`) get the single side `face`. Column comes from the label numeral. On sides that have labelled columns, lines without a numeral are
assigned to the column whose text-line index range contains them, or else to the column their signs overlap
most (`column_inferred`). Line boxes are the union of the line's annotated sign boxes. `block_x0/x1` is the
extent of the whole block and is only an auxiliary.

Re-run it as more tablets download with `python -m line_cropping.truth.ebl_ground_truth --preview 10`. It re-reads
`tablet_data/ebl_tablets/manifest.csv`. Use the oriented segment, not the axis-aligned box, as the line truth: 56% of lines tilt more than 3 degrees,
and axis-aligned boxes of neighbouring lines overlapped in 69% of cases versus 24% for the oriented bands.
Coverage is partial: eBL annotators did not box every line, so compare `n_lines` with `atf_lines` per tablet
(`tablets.csv`) and evaluate only on tablets or blocks with high coverage. The known limits are these: 14 tablets have no side information at all and stay
`unknown`. Line boxes only cover annotated signs, so an unannotated line is simply absent from the truth
(check `n_lines` against `atf_lines` in `tablets.csv` to find sparsely annotated tablets).

**1b. Old Assyrian faces** — `truth/oa_sample.py` (uses `lines/profile_lines.py`)

This script picks 100 faces from `tablet_data/fat-cross_processed/experiments/face_qc.csv` (unflagged, with ATF), stratified by
resolution (px per line: low <25, mid <40, high), ATF line-count bucket and face, and splits them alternately into
`tune` and `test`. For each face it writes automatic pre-labels from the ATF-constrained comb fit.

The outputs are in `tablet_data/fat-cross_processed/ground_truth/`:

| file | content |
|---|---|
| `sample_faces.csv` | the sample: file, ATF line count, px per line, stratum, split |
| `prelabels.csv` | per face and line: `skew_deg`, `pitch_px`, `y_deskewed` (line centre in the deskewed face), body x-extent, `source=comb_fit` |
| `previews/<face>.png` | deskewed face with the pre-labelled lines numbered, to correct against |

Labelling is done in the published page **Old Assyrian Line Marker**. Its source is in `truth/labeler/`, and its images are the 100
faces, deskewed and downscaled to 900 px wide. A click adds a line, dragging moves it, Delete removes it, and D marks a face
done. The ATF count is shown next to the marked count. Marks save to the page's database as `labels/<face>` documents.
To use them, pull the `labels` collection with the Artifact tool's `read_db` into
`tablet_data/fat-cross_processed/ground_truth/db/`, then run `python -m line_cropping.truth.labels_from_db` to write
`tablet_data/fat-cross_processed/ground_truth/labels.csv` in face coordinates.

The older path, hand-editing `prelabels.csv`, still works. Correct `y_deskewed` per line (and delete or add rows) so that each row is a
real line centre, and set `source` to `manual` on corrected rows. The pre-labels are rough on broken
fragments and on faces where the ATF counts lines that are not on this face.

## Phase 2 — face preparation (status: done, awaiting review)

`prepare_faces.py ebl` cuts every annotated side (with its object, e.g. tablet vs envelope) out of the eBL
composite photo and rewrites the line truth in face-crop coordinates. The face outline comes from the image,
not from the annotations. The photo is greyscaled and thresholded against the black studio background, and connected
bright blobs are the tablet views. Touching views are split at near-empty gaps in the blob's column/row profile,
and every blob that holds a share of the side's Phase 1 lines is taken (a side broken over several fragments
stays whole). `crop_method` in `faces.csv` says `blob` or, where no blob held the lines, `lines` (the union of the
line segments plus one pitch of margin). It does not rotate the crop. `skew_deg` (minus the median block tilt) is recorded
for the line finder, which deskews itself.

The outputs in `tablet_data/ebl_tablets/faces/` are `<tablet>__<side>[__<object>].jpg`, `lines/<same>.csv` (oriented segments in
crop coordinates) and `faces.csv` (crop box in the photo, size, `n_columns`, `n_lines`, `atf_lines_side` from
eBL's own ATF, `coverage`, `pitch_px`, `skew_deg`). There are 1,367 faces from 872 tablets (536 obverse, 460 reverse, 248 unspecified `face`, 123 edges). The median
is 122 px per line, 53% of faces have >= 0.8 coverage, and 33 faces have more than one column.

`prepare_faces.py oa --rot-left cw --rot-right ccw` writes `tablet_data/fat-cross_processed/prepared/faces.csv`. It lists every
obverse/reverse that passed `face_qc.csv` (accepted=True) plus the rejected ones with the reason, and every left or
right edge whose ATF has text on that edge. The edges are rotated so lines run horizontally and saved under `prepared/edges/`.
The rotation direction was chosen by eye on P357359 (clockwise for the left edge gives wedge heads at left/top).
Verify it on a few more before trusting edge crops. There are 3,290 accepted faces: 1,359 obverse, 1,254 reverse, 660 left
and 17 right.

Re-run both after new downloads or QC changes; they are idempotent.

## Phase 3 — line finding (status: method chosen, awaiting review)

Five methods were scored on 980 single-column eBL faces against the Phase 1 truth with `eval/evaluate_lines.py`.
A line counts as found when its centre is within 0.35 pitch. Precision is measured only on faces with >= 0.9 coverage,
and first-line error only where the topmost truth line is line 1 or 1'.

| method | what it does | recall | precision | first line within 0.5 pitch |
|---|---|---|---|---|
| `free` | profile peaks, no prior | 0.49 | 0.59 | 25% |
| `comb` | N lines on a rigid comb (N from ATF) | 0.60 | 0.58 | 30% |
| `dp` | N lines by dynamic programme on the profile | 0.62 | 0.59 | 30% |
| `det` | eBL Deformable DETR boxes -> tilt -> 1-D clustering | 0.77 | 0.87 | 77% |
| `det_n` | as `det`, then split over-merged clusters, fill gaps, merge to N from the ATF | **0.87** | **0.87** | **89%** |

The profile methods plateau because the deskewed row profile is only weakly in phase with the line bands and
small pitch errors accumulate over a face. An oracle with the true tilt, count and extent still reached only
0.65. The detector localises signs directly, so its lines follow curvature and gaps. On faces with >= 120 px
per line, `det_n` reaches recall 0.89 and 92% first-line accuracy. Below 60 px per line it drops to 0.71 and 67%.

**Old Assyrian faces (no truth yet).** The detector finds a median of 22 boxes per face (51 on eBL faces), and
its line count without the ATF agrees with the ATF within one line on only 18% of faces (63% on left edges,
which carry few lines). Tiling the face into 2 or 3 bands to raise the effective resolution did not add boxes.
`det_n` therefore leans on the ATF count and on gap filling (missing lines are inserted where clusters are more
than 1.5 pitches apart), and the profile `dp` was also run on all 3,290 accepted faces
(`tablet_data/fat-cross_processed/predictions_profile/`). Which of the two is better on this set cannot be measured until the
Phase 1b sample is hand-labelled. Side-by-side overlays for review are in `tablet_data/fat-cross_processed/predictions/previews_det_n/`
and `tablet_data/fat-cross_processed/predictions_profile/previews_dp/`.

**Old Assyrian faces, scored against 88 hand-labelled faces.** The scoring uses `eval/evaluate_oa.py` with labels from the Line Marker page.
The labels cover 785 lines. Of these, 49% were moved or added by hand, and the rest were left where the profile fit had put them.

| method | recall | precision | first line within 0.5 pitch | recall on hand-placed lines only |
|---|---|---|---|---|
| `dp` (profile, N from ATF) | 0.89 | 0.76 | 69% | 0.78 |
| `comb` | 0.84 | 0.73 | 64% | — |
| `det_n` (detector, N from ATF) | 0.43 | 0.37 | 33% | 0.33 |
| `det` | 0.35 | 0.49 | 34% | — |

So the two sets want different line finders. The detector suits the high-resolution eBL photographs, and the profile
dynamic programme suits the Old Assyrian faces, where the detector finds too few signs. The labels also show that
the ATF line count is not the number of lines visible on the face: on 49% of faces fewer lines were labelled than
the ATF lists (a median of 3 fewer), which is where `dp`'s missing precision goes. See the count-selection note below.

**Count selection (open).** With the true line count, `dp` reaches recall 0.88, precision 0.88 and first line within
0.5 pitch on 79% of the labelled faces, so the count is the remaining lever. Choosing N by mean profile response
collapses to the ATF count. Estimating N from the text extent divided by the autocorrelation pitch (with the pitch search
bounded by the ATF count; the estimate is right within 15% on 70% of faces) trades precision for first-line accuracy
(0.81 / 57% against 0.75 / 69%). A better criterion, or a labelled count from the ATF's state lines
(`$ rest broken`, lines that continue on an edge), is Phase 4 work.

The decision is as follows. On eBL-style photographs, `det_n` is the line finder when a transliteration exists, and `det` when it
does not. On the Old Assyrian faces, `dp` is the line finder, with the ATF count treated as an upper bound rather
than the exact number of lines.

The files are `lines/find_lines.py` (profile methods), `lines/detect_signs.py` (run in `~/venvs/mmdet`, with weights in
`weights/ebl_detr/` and the config from `external/cuneiform-ocr/configs/detr.py`), `lines/lines_from_boxes.py` (boxes to
lines) and `eval/evaluate_lines.py`. Predictions go to `tablet_data/ebl_tablets/predictions/<method>/<face>.csv`, and detections go to
`tablet_data/ebl_tablets/detections/<face>.csv`. The detector processes about 6 faces/s on the RTX 4060.

## Scope note (12 Sep 2026)
The Old Assyrian set is parked by decision of the project owner. Phases 4 onward proceed on the eBL set with the
detector route. The Old Assyrian labels, predictions and the count-selection question stay in place for later.

## Phase 4 — crop generation (status: done on the eBL set; gap-bounded crops)

The project owner restated the goal: a line crop must hold its own line only, with no slices of the lines above
and below. `crops/make_crops.py` (default `--bounds gap`) therefore sets each line's vertical extent from the detector
boxes assigned to it (the 10th to 90th percentile of box tops and bottoms in the deskewed frame) and cuts from the
midpoint of the gap to the line above to the midpoint of the gap to the line below. Where two lines' boxes overlap,
the boundary is the midpoint of their centres, and the first and last lines get 6% of a pitch of margin. Columns
span the line's detected extent plus half a pitch each side. Crops go to `tablet_data/ebl_tablets/line_crops/<face>/<face>__L<nn>.jpg`.
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
Evaluate with `python -m line_cropping.eval.evaluate_crops --faces tablet_data/ebl_tablets/faces/faces.csv --crops tablet_data/ebl_tablets/line_crops/crops.csv [--split test]`. Besides the vertical metrics, it reports horizontal fit (crop width, meaning lines kept whole on clay, cut ends and overhang past the tablet) from `eval/crop_width.py`. That module also writes per-line detail and renders of the worst lines. See docs/line_model/line_model.md section 10.
The band set can be deleted once nothing needs it (2.3 GB).

## Phase 5 — evaluation (status: done on the eBL set)

`eval/phase5_eval.py` splits the 1,334 single-column eBL faces with detections into tune and test halves, stratified
by script and pixels-per-line bucket (`tablet_data/ebl_tablets/split.csv`, seed 7). It grid-searches the clustering thresholds
on the tune half only and reports the test half. The chosen thresholds are `gap_frac 0.5`, `merge_frac 0.35` and `score_thr 0.3`. They are saved to
`tablet_data/ebl_tablets/det_params.json`, which `lines/lines_from_boxes.py` now reads by default.
Faces with fewer than two truth lines are not scored (487 of 666 test faces are).

The results on the test half for `det_n` (detector boxes plus the ATF count) are:

| group | n | recall | precision | first line < 0.5 pitch | count match |
|---|---|---|---|---|---|
| all | 487 | 0.91 | 0.91 | 90% | 85% |
| OB, >= 120 px/line | 229 | 0.90 | 0.91 | 93% | 99% |
| OB, 60-120 | 95 | 0.93 | 0.92 | 95% | 98% |
| NA, >= 120 | 20 | 0.93 | 0.92 | 71% | 70% |
| NA, 60-120 | 114 | 0.94 | 0.93 | 78% | 54% |
| NA, < 60 | 26 | 0.82 | 0.63 | 57% | 54% |

Without the count (`det`), recall is 0.87, precision is 0.90, first line < 0.5 pitch is 88%, and count match is 25%.

**Plan criteria (section 7 of the plan), test half, faces with >= 120 px/line:** Recall is 0.90 (target 0.90),
the first line is within 0.5 pitch on 92% of faces (target 90%), and count agreement is 97% (target 80%). All three are met.
Below 60 px/line, the method is reported but not gated, as the plan allows.

The 51 test faces with recall < 0.7 fall into these failure categories: 18 placement errors with no single cause,
10 sparsely annotated faces (< 50% coverage, so the truth itself is thin), 9 with the count far off, 5 with too few
detections, 5 with an undeclared second column (truth lines interleave) and 4 with low resolution. Multi-column faces
(33) were excluded from the run and remain out of scope. Per-face results are in `tablet_data/ebl_tablets/phase5_test_results.csv`.

## Phase 6 — ProtoSnap connection (status: sign-level trial done; see note)

Note: the project's intent for this phase is clean single-line crops for ProtoSnap, which Phase 4's gap-bounded
crops now provide. The sign-level snapping trial below was a detour beyond that intent. It is kept because its
alignment step (ATF signs to detector boxes) is what a per-line ProtoSnap run will need, but it is not the deliverable.

`protosnap/protosnap_prep.py` turns aligned line crops into ProtoSnap inputs. For each crop with an ATF line label, it
tokenises the eBL ATF line into sign names. Transliteration values are mapped to signs with a frequency table
built from `data/ebl/signs_snippets_metadata.json`, and there are 962 distinct OB values, 48 of them ambiguous. It then collects the detector
boxes whose centres fall in the crop, sorts them left to right, and aligns the two sequences with Needleman-Wunsch
(scoring +2 for the same sign name, 0 for a different one and -1 for a gap). Finally, it saves each matched box (with 15% padding) as a target image for every
sign that has a Santakku prototype and skeleton. `samples.csv` has the columns `run_test.py` reads. `alignment.csv`
records, per ATF sign, the matched box, the detector's class, and the overlap with the eBL truth box of that sign.

The results for the first 400 OB line crops are:

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
was run with `run_test.py --font_dir prototypes/Santakku --con_dir skeletons/Santakku --output_folder phase6_OB`.
All 40 completed without error in about 40 minutes on the RTX 4060. Initialisation dominates, and the 100 optimisation
steps take about 7 s. `protosnap/phase6_summary.py` writes `output/line_cropping/phase6_OB/summary.csv` and a `contact_sheet.jpg` of the
per-target `itr99_viz.png` renders. The initialisation score (ProtoSnap's own agreement measure) has a median of
0.60 against 0.75 on the repo's curated single-sign test set: the pipeline runs end to end, and roughly half the
snaps look right by eye. The box itself is rarely the problem. Of the 40 boxes, 28 coincide with the eBL truth box of
that sign (IoU > 0.5), none contradicts it, and 12 have no truth to check. Verified boxes score no better than
the rest (median 0.60 for both). The weak snaps come from the sign images themselves: 15% padding often admits parts of
the neighbouring sign, and worn or shallow wedges give the diffusion features little to lock onto. Tighter boxes
(no padding, or padding only where the neighbour is far) and the eBL truth boxes as an upper-bound experiment are
the next two things to try.

**First sign of each line (16 Sep 2026).** `protosnap_prep.py --first-only` writes a target only when the first ATF
token of a line is matched to the leftmost detector box in its crop. Over all 6,299 aligned OB lines, that holds for
2,387 lines (38%). Of these, 1,961 have a Santakku prototype, 1,823 have detector and ATF agreeing on the sign, and 1,675 are confirmed
by the eBL truth box (`tablet_data/protosnap_inputs/OB_first/`). A 60-target run, one per face (`samples_run.csv`, with results in
`output/line_cropping/phase6_OB_first/`), finished with a median initialisation score of 0.54, against 0.60 for the 40-sign trial and 0.75 for the curated test
set. Of the 60 boxes, 59 coincide with the eBL truth box. First signs are harder than average for two reasons
visible in the overlays. First, they sit at the tablet's left edge, so the padded box often includes background or a broken
margin. Second, the commonest first signs (DIŠ 16, IGI 12 of 60) are a single vertical wedge or a dense cluster, and both are
weakly constrained. Note: `run_test.py` must be run with `MPLBACKEND=Agg` in a headless shell. The Tk backend
aborted the first attempt after 21 targets.

`overlay_results.py` projects each fitted skeleton back onto the unrotated face and onto the gap-bounded line crop of
its line (undoing ProtoSnap's 512-px stretch and y flip, then applying the face deskew). It writes
`output/line_cropping/phase6_OB/overlays/{faces,lines}/` plus `line_contact_sheet.jpg` and `index.csv`.

## Sign classifier from the snippet metadata (20 Sep 2026; negative result for alignment)

`data/ebl/signs_snippets_metadata.json` is a classification set. It holds one cropped sign per entry with its name and script, and no
position on the tablet. It cannot train or fine-tune the detector, but it can train a sign classifier, which was tried
as a way to improve the sign-to-box alignment in `protosnap/protosnap_prep.py`.

- `signs/build_dataset.py` builds NA and OB datasets with hold-out at FACE level: a snippet's eBL line index is mapped
  to its face through the Phase 1 line table, and snippets on Phase 5 test faces never enter training. Tablet-level
  hold-out left OB with 3,767 training snippets, and face-level hold-out gives 8,472. Classes are signs with >= 20 training samples.
- `signs/train_classifier.py` trains a ResNet-34 from cached ImageNet weights on 112 px letterboxed inputs held in RAM,
  with class-balanced sampling, for 15 epochs. Training takes under 3 minutes per script on the RTX 4060. The weights are in `weights/sign_classifier/`.

| script | classes | train / val / test snippets | held-out test top-1 | top-5 |
|---|---|---|---|---|
| OB | 108 | 8,472 / 887 / 7,684 | 0.81 | 0.95 |
| NA | 104 | 14,913 / 2,333 / 3,685 | 0.82 (intact 0.87, damaged 0.50, partially broken 0.29) | 0.93 |

(NA validation top-1 is 0.94, but its validation tablets are mostly detector-derived crops of clean signs. Quote the
held-out figure.)

**Effect on alignment (Phase 5 test faces only, `protosnap_prep --classifier --split test --no-write`):** There is no effect for OB
(the matched box is the eBL truth box of that sign in 66% of cases, with and without the classifier) and a slightly negative effect for NA (from 63% to 59%).
The table below shows the reason, measured on detector boxes matched to truth boxes by IoU alone (no labels involved):

| | detector label correct | classifier correct | classifier correct where the detector is wrong |
|---|---|---|---|
| OB (4,795 boxes) | 92% (97% on signs the classifier knows) | 85% | 50% of 125 boxes |
| NA (1,823 boxes) | 83% (96%) | 83% | 50% of 56 boxes |

The eBL detector was trained on a superset of the same annotations, with context, and its label is already right on
about 97% of correctly localised boxes. The classifier is weaker and knows fewer classes (104-108 against 173), so it
adds noise rather than information. The alignment's remaining errors are not labelling errors: the detector finds only
about 55% of the signs a transliteration lists, so the sequence alignment has gaps to bridge. Effort belongs in
detection recall, not classification. `--classifier` therefore stays off by default. Where a snippet-trained
classifier could still matter is a script the detector handles badly, such as Old Assyrian (9,175 OA snippets exist).

## Phase 7 — learned line model (status: run 6 trained, `run6_heights_s1` is the default; see `docs/line_model/line_model.md` for the full write-up)

`linemodel/` implements section 14 of the plan. Run from the repo root with the ProtoSnap venv.

| step | command | what to inspect |
|---|---|---|
| 1. targets | `python -m line_cropping.linemodel.targets --preview 12` | `tablet_data/ebl_tablets/linemodel/target_previews/`: green = band (<= 0.7 pitch wide), yellow = kernel, red hatch = ignore (gaps > 1.5 pitch, one pitch beyond first/last line). Bands must sit on lines and never merge. |
| 2. index | `python -c "from line_cropping.linemodel.dataset import build_index as b; i=b(); print(len(i['train']), len(i['val']))"` | training faces = tune-split eBL faces with coverage >= 0.9 + labelled Old Assyrian faces (302 train / 38 val at the time of writing; only 16 NA faces pass 0.9, use `--min-cov 0.8` in train.py to admit more) |
| 3. dry run | `MPLBACKEND=Agg .venv/bin/python -m line_cropping.linemodel.train --dry-run --out weights/linemodel/dryrun` | three steps and a 3-face validation; proves the loop, ~10 s |
| 4. train | `MPLBACKEND=Agg .venv/bin/python -m line_cropping.linemodel.train --epochs 40 --out weights/linemodel/run1` | `history.csv` (loss parts, val line recall/precision per epoch) and `plots/` (loss, loss components, validation recall/precision, checkpoint-selection score, `dashboard.png`), drawn over the whole run when training ends, or every N epochs with `--plot-every N`; `model.pt` = best val recall+precision. About 30 min for 40 epochs on the RTX 4060. `--no-oa` trains on eBL only for the transfer test; `--resume` continues a run. Read-out options `--kernel-thr`, `--band-thr`, `--min-area` change only validation scoring and checkpoint choice (saved in `config.json`, reused by `infer.py` unless overridden); target options `--kernel-shrink`, `--band-frac`, `--ignore-gap` change what the network learns. `python -m line_cropping.linemodel.visualize --run <dir>` regenerates the plots. |
| 5. infer | `.venv/bin/python -m line_cropping.linemodel.infer --weights weights/linemodel/run1/model.pt --faces tablet_data/ebl_tablets/faces/faces.csv --out tablet_data/ebl_tablets/predictions --method unet --split test` | writes `predictions/unet/` in the detector route's format and `predictions/unet_segments/` (raw segments) |
| 5b. look | add `--viz 8` to the infer command | `<out>/unet_viz/`: green = predicted band, lime = predicted kernel, red = read-out segments (the target previews of step 1 use the same colours for the truth) |
| 5c. on the tablet | `python -m line_cropping.linemodel.render_predictions --n 10` | `tablet_data/ebl_tablets/linemodel/prediction_overlays/`: predicted segments (red) and eBL truth (green) on each face and on the original composite photograph (`photos/`, face outlined in orange), plus a contact sheet |
| 6. score | `python -m line_cropping.eval.evaluate_lines --faces tablet_data/ebl_tablets/faces/faces.csv --pred tablet_data/ebl_tablets/predictions` | `unet` appears beside `det`/`det_n`; compare on the same test faces. For Old Assyrian: `infer` on `tablet_data/fat-cross_processed/prepared/faces_accepted.csv` with `--out tablet_data/fat-cross_processed/predictions_unet --n-col atf_lines`, then `evaluate_oa.py` (add the method there). |

**First run through the whole pipeline (run1, 40 epochs, test split, 27 Sep 2026).** `infer.py --use-n` adds the
plan's optional refinement: where a transliteration exists, its line count is reconciled with the model's lines
(`reconcile_count`, which uses the same split/fill/merge rules as the detector route), and the result is written as method `unet_n`. Crops use
`make_crops --bounds segments` (boundaries from the predicted bands) or `midpoint`. `protosnap_prep --crops` runs the
ATF-to-box alignment on any crop table.

| test split | recall | precision | first line < 0.5 pitch | count = ATF | crops: truth centre in exactly one crop / two-line crops | crops with ATF label | ATF signs matched to a box / box is the truth box |
|---|---|---|---|---|---|---|---|
| `unet` (no transliteration) | 0.86 | 0.83 | 81% | 23% | 97% / 3% | 14% | (only 902 labelled lines) |
| `unet_n` (count reconciled) | 0.83 | 0.84 | 82% | 85% | 97% / 3% | 72% | 54% / 67% (OB) |
| `det_n` (detector route) | 0.91 | 0.91 | 90% | 85% | 99% / 3% | 73% | 53% / 66% (OB) |

So the line model already carries the downstream stages at parity with the detector route, without any sign detector
at line-finding time, and trails it by a few points on line recall, mostly on Neo-Assyrian faces (16 in training).
Note that `evaluate_lines` "all faces" rows for `unet` include tune faces the model trained on. Quote test-split numbers.

The model is in `model.py`. It is a U-Net with a ResNet-34 encoder from ImageNet weights and 24.4 M weights, and its outputs at stride 4 are a kernel logit, a band logit,
sin 2θ and cos 2θ. The loss is BCE plus Dice on kernel and band with ignore pixels excluded, and L1 on orientation
inside bands. The read-out (`infer.read_lines`) takes kernel components, assigns band pixels to the nearest component, and fits a PCA
segment per component. The knobs to tune on the val split are `--kernel-thr`, `--band-thr` and `min_area`, and in training
`--win`, `--long-side`, `--per-face` and the scale range in `dataset.py`.

## Line Splitter app (`app/`)

The app is a local web page for trying the line model on any photograph:

```
MPLBACKEND=Agg .venv/bin/python -m line_cropping.app.server [--weights weights/linemodel/run6_heights_s1/model.pt] [--port 8770]
```

Then open http://localhost:8770. Upload a tablet photograph (by drag and drop or file picker). The app finds the tablet's
views without any annotations (threshold against the background, split touching regions at empty gaps, keep regions
over 1% of the image) and lists them, largest first. Select one or more views, or treat the whole image as one face
when the upload is already a single face. Choose whether each line is straightened on its own, choose the detection
threshold, and split. The photograph then shows every crop's outline, and the crops are listed beside it. Hovering over a
crop highlights its outline, and clicking it scrolls the photograph to it. The reverse also works. Crops and `result.json` can
be downloaded as a zip. Sessions are kept in `tablet_data/app_sessions/<id>/` and can be reopened with
`?session=<id>`. The app uses the standard library only. Its files are `server.py` (HTTP), `pipeline.py` (view detection, line splitting)
and `static/index.html` (page).

## Old Assyrian set without ground truth (experiment, 15 Sep 2026)

To work on the parked Old Assyrian set without labels, three additions stay inside the existing layout:

- `crops/make_crops.py --bounds midpoint --atf-source cdli` cuts single-line crops halfway between consecutive
  predicted line centres, so no detector boxes are needed. ATF line labels come from the CDLI dump via the face's
  P-number. Run on the profile `dp` predictions, it produced 30,472 crops from 3,289 faces, 88% of them with an ATF line label and with a median size of
  601 x 48 px, in `tablet_data/fat-cross_processed/line_crops/`.
- `eval/confidence_oa.py` computes per-face confidence without truth from (a) agreement between the two independent line
  finders (`dp` vs `det_n`, within 0.35 pitch), (b) the share of detector boxes within 0.3 pitch of a predicted line
  centre, and (c) spacing regularity. Two tempting signals are uninformative here and are reported but not scored: the
  ATF count (enforced, so always matched) and crop containment (midpoint crops tile the face). The result is
  `tablet_data/fat-cross_processed/confidence.csv`, covering 3,172 faces. Median agreement is 0.50, median box proximity is 0.44, median
  confidence is 0.64, and the lowest decile is below 0.45. The two signals correlate at 0.50.
- `eval/render_lines.py --sheet` draws contact sheets of any ranked list of faces. The eight least and eight most
  confident faces are in `tablet_data/fat-cross_processed/review/`.

The ranking is a review queue, not a metric: look at the bottom of it first.

## Later phases
Phase 7, a learned line model that removes the dependence on the transliteration count and on sign-detector
recall, was planned in section 14 of `docs/line_cropping/line_cropping_plan.md` (a U-Net with a ResNet-34 encoder trained on the
Phase 1 line segments, with kernel, band and orientation maps and the same evaluation harness). It has since been built and trained: the
Phase 7 section above describes the code, and `docs/line_model/line_model.md` describes the runs. The current default model is
`run6_heights_s1`.
