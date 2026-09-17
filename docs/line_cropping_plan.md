# Plan: dividing cuneiform tablet photographs into single-line crops

## 1. Goal

The end goal is a repeatable procedure that takes a photograph of a cuneiform tablet face and
produces one image crop per line of writing, in reading order, with each crop labelled by tablet,
face and line number. These line crops are the input the rest of the ProtoSnap pipeline needs in
order to place sign prototypes along a line, and they are the unit at which the tablet's
transliteration can be matched to the image. A transliteration is the sign-by-sign rendering of a
tablet's text in Latin letters, one text line per line, which is how Assyriologists publish tablet
contents.

The immediate target is the 1,532 Old Assyrian tablets in `fat-cross_processed/`, whose faces have
already been cut out of their museum photographs. The procedure counts as consistent when, on a
held-out set of faces, it recovers at least nine out of ten lines, places the first line within half
a line height of the truth, and produces a number of crops equal to the transliteration's line
count for most faces. Section 7 makes these criteria measurable.

## 2. What we already have

The investigation so far and the data in the repository establish the following.

**Face images.** Museum photographs are laid out as a fat cross, the conventional arrangement in
which the obverse (front face) sits at the centre, the reverse (back face) below it and the four
edges around them. `fat-cross_processed/` holds, for 1,532 tablets named by their CDLI P-number,
one file per face (`_obverse`, `_reverse`, `_left`, `_right`, `_top`, `_bottom`) plus a `_layout`
image showing the detected faces drawn on the composite photograph. 1,505 tablets have an obverse
file and 1,350 have both obverse and reverse; the remainder are missing faces or are single-face
fragments. A subfolder `incomplete/` holds about 155 composites that could not be split, and
`Test_ObvRev/` holds 46 hand-checked faces. `fat-cross_processed/inventory.csv` lists all of this per
tablet and is rebuilt by `line_cropping/ingest/inventory.py` whenever images are added.

**Transliterations.** All 1,532 tablets are in the CDLI dump under `artifacts_json/`, and 1,312 of
them have a transliteration in ATF. ATF (ASCII Transliteration Format) is CDLI's plain-text
convention: structural lines beginning with `@` such as `@obverse` or `@left`, state lines beginning
with `$` such as `$ beginning broken`, and numbered text lines such as `3. a-na ...`. A prime after
a line number, as in `1'.`, means the count is relative because the true beginning is lost. For this
set the ATF says that faces are almost always single-column (only 4 tablets use `@column`), that the
obverse and reverse each carry a median of 12 lines with a maximum near 58, and that 619 tablets
also carry text on the left edge, typically 3 lines. So for most of the set we know in advance
exactly how many lines each face holds.

**Ground truth.** The Electronic Babylonian Library (eBL) publishes sign-level bounding boxes for
about 2,000 tablets, but only one of them is in this set, so eBL cannot serve as line-level truth
here. No line-level truth exists for these tablets yet; section 3 describes how to make some.

**Released detectors.** The eBL group released Deformable DETR sign-detector weights trained on
124,504 signs from many periods including Old Assyrian (Zenodo record 17395154), and an earlier
FCENet model packaged as a Docker API (Zenodo record 10693601). A detector is a neural network that
outputs a bounding box, an axis-aligned rectangle, and a class label for every sign it finds.

**Measurements on this set** (scripts and tables in `fat-cross_processed/experiments/`):

- A projection profile is a one-dimensional signal made by summing a per-pixel quantity along each
  row of the image. Summing the vertical image gradient, which is strong along the shadowed upper
  edges of wedge impressions, gives a signal that peaks once per line of writing when the face is
  upright. Rotating the face to make this signal sharpest is called deskewing; the faces here need
  rotations of up to about eight degrees.
- Counting peaks of that profile without any other knowledge matched the ATF line count within one
  line on only 25 percent of 1,881 faces, and systematically undercounted dense faces. Free counting
  is therefore not sufficient on its own.
- When the line count is taken from the ATF and the profile is only used to place that many lines,
  the fit tracks the lines well on faces of ordinary density (see `experiments/oa_comb_fit.png`).
  This is the key observation: for the 1,312 tablets with an ATF, the task is placing a known number
  of lines, which is far easier than counting them.
- Resolution is the main physical limit. Faces are about 614 by 654 pixels at the median, which
  gives a median of 47 pixels per line of writing, but 9 percent of faces have under 25 pixels per
  line and 5 percent under 15. The source composites are only about 1,160 by 2,020 pixels, so no
  more detail can be recovered by re-cropping.
- Some face crops are wrong: of 2,856 obverse and reverse files, 60 are under 150 pixels in one
  dimension, 233 have an implausible aspect ratio and 94 are too short for their ATF line count, for
  example a 445 by 98 pixel sliver labelled as the obverse of P250549; 243 files carry at least one of
  these flags. `fat-cross_processed/experiments/face_qc.csv` lists them. These need to be caught
  before line finding.

## 3. Phase 1: quality-check the faces and build a small ground-truth set

The first phase makes the inputs trustworthy and creates something to measure against.

Face quality is checked automatically with three rules: an obverse or reverse must be at least 150
pixels in both dimensions, its height divided by its width must lie between 0.45 and 4, and when an
ATF exists its height divided by its ATF line count must be at least 12 pixels. Faces that fail any
rule are listed for review, and the composite `_layout` image is used to decide whether the face
should be re-cropped or the tablet excluded. The `incomplete/` composites are treated the same way.

Edge faces need one extra step. In the composite, the left and right edges are shown as tall
strips, so their lines of writing run vertically in the `_left` and `_right` files. These files are
rotated by 90 degrees before any line finding, in the direction that puts the edge's text in the
same reading orientation as the obverse; the ATF's `@left` block, present for 619 tablets, says
whether text is expected there at all.

Ground truth is then made by hand for a stratified sample, meaning a sample chosen so that each
range of line counts, each face type, and both good and poor resolution are represented. A sample of
about 100 faces is enough to measure the criteria of section 7 with useful confidence. Labelling
consists of marking the vertical centre of each line and the left and right ends of the text
block, which takes under a minute per face when the labeller starts from the automatic fit and only
corrects it. The `Test_ObvRev/` faces, already checked by hand, are the first candidates. The sample
is split so that half is used to tune the method and half is held out for the final numbers.

## 4. Phase 2: prepare each face for line finding

Each accepted face is converted to greyscale, the scale bar that sits at the bottom of many
obverse photographs is removed by discarding the bottom eight percent of the image, and the curved
margins where the tablet surface turns away from the camera are trimmed by discarding about twelve
percent on the left and right. These fractions were used in the experiments and should be re-tuned
on the ground-truth sample rather than kept as constants.

The face is then deskewed by searching rotations between minus twelve and plus twelve degrees and
keeping the angle at which the row profile has the largest variance. The chosen angle is recorded,
because it is also the rotation that a font prototype must be given before it is matched against
signs on that face, which answers the second item of the original plan.

Columns are not a concern for this set, since only four tablets have more than one, but the step
that finds them on other collections is kept in reserve: a vertical projection profile, summed
along image columns instead of rows, dips at the blank strip between text columns.

## 5. Phase 3: find the lines

Two routes are used depending on whether the face has a transliteration.

**Faces with an ATF (most of the set).** The number of lines N is read from the ATF block for that
face. Lines are placed by fitting a comb of N positions to the deskewed row profile: the fit searches
over a line pitch, the vertical distance from one line to the next, between 0.6 and 1.15 times the
body height divided by N, and over the offset of the first line, and keeps the combination with the
largest total profile response. Each line is then snapped to the nearest profile peak within three
tenths of a pitch. This rigid comb is a starting point; the planned improvement is to replace it
with dynamic programming, a method that finds the lowest-cost sequence of N line positions when the
cost penalises both low profile response and departures from a uniform pitch, which lets the spacing
vary gradually down the face as it does on real tablets. The state lines in the ATF are used as
priors: `$ beginning broken` means the first line may start below the top edge, `$ rest broken`
means the last lines may be missing, and `blank space` lines allow a gap.

**Faces without an ATF (220 tablets).** The eBL Deformable DETR detector is run on the face, its
sign boxes are clustered into lines with DBSCAN, a clustering algorithm that groups points with
enough close neighbours and uses an anisotropic distance in which vertical separation counts far
more than horizontal separation, and the number of clusters gives N. The comb fit above is then run
with that N so both routes produce lines in the same form. Whether the detector works at 25 to 45
pixels per line has to be measured on the ground-truth sample; if it does not, the free profile
count with a pitch estimated by autocorrelation is the fallback, accepting the lower accuracy.

Where the detector runs, its boxes are also kept, since they give the left and right ends of each
line and, later, starting positions for ProtoSnap.

## 6. Phase 4: produce the crops and check them

Each line becomes one crop taken from the deskewed face: it spans the text block horizontally with a
small margin, has a height of one pitch centred on the line plus a quarter pitch above and below so
that tall signs are not cut, and is saved as `<P-number>_<face>_<line index>.png`. A CSV per
tablet records the crop's rectangle and rotation in the original face image so that any later
result can be mapped back to the photograph and to the composite.

When an ATF exists, the crop index is matched to the ATF line number directly, since N came from
the ATF. When it does not, the detector's per-line sign counts are compared with the ATF of similar
tablets only as a sanity check, and the face is flagged if the count is far outside the range seen
for its size.

## 7. Phase 5: evaluate

Evaluation uses the held-out half of the ground-truth sample and reports three numbers, separately
for faces above and below 25 pixels per line. Line recall and precision count a predicted line as
correct when its centre lies within a third of a pitch of a true line centre. First-line error is the
distance between the predicted and true first line in units of pitch. Count agreement, for the
no-ATF route only, is the fraction of faces where the detected N equals the true N. The targets are
recall of at least 0.9 and first-line error under 0.5 pitch on at least nine faces in ten for the
higher-resolution group, with the lower-resolution group reported but not gated. Every failure is
inspected by overlaying predicted and true lines on the face, as in `experiments/oa_comb_fit.png`,
and the largest failure category decides what to improve next.

## 8. Phase 6: connect to ProtoSnap

Once line crops are reliable, each crop together with its ATF line gives ProtoSnap an ordered list
of sign names and a narrow strip in which to place them. The deskew angle supplies the prototype
rotation, the pitch supplies the expected sign height, and detector boxes where available supply
starting positions, so the existing best-buddies initialisation can be seeded instead of started
from a random guess. Two things must be settled before this pays off. First, the repository has
fonts for Old Babylonian, Neo-Assyrian and Neo-Babylonian script but none for Old Assyrian, so
either an Old Assyrian font is added through the `add_new_font_skeletons` workflow or the Old
Babylonian font is tested as a stand-in. Second, at 45 pixels per line the signs are far smaller
than the 512-pixel crops ProtoSnap was developed on, so upscaling and its effect on the diffusion
features must be checked on a few lines before committing to the whole set. Training a dedicated
model to find the first sign, the last item of the original plan, is deferred until the phases
above show whether it is needed.

## 9. Risks and open questions

Resolution is the largest risk: the poorest tenth of faces may never yield usable crops and should
be reported as out of scope rather than dragging the averages down. Faces that are strongly curved
or broken produce lines that bend or stop early, which the rigid comb handles badly and the dynamic
programming version handles only partly. Failed face crops must be fixed upstream in the
fat-cross splitter, whose code or origin should be recorded so that new tablets are split the same
way. Old Assyrian letters occasionally run text around the edges or in the margins, which the ATF
marks but the face-based crops will not capture. Finally, the ground-truth sample is small by
design, so the reported numbers carry wide intervals and should be quoted with them.

## 10. Order of work

The order is: face quality check and edge rotation, then the hand-labelled sample, then the
ATF-constrained line fit with dynamic programming tuned on half the sample, then crop generation,
then evaluation on the other half, then the detector route for tablets without an ATF, and only
then the ProtoSnap connection. A complete run on a dozen well-photographed letters should come
before any step is polished, so that the whole chain is exercised early and the evaluation numbers
guide where the effort goes.

## 11. Status after Phase 3 (12 September 2026)

Phases 1 to 3 are implemented in `line_cropping/` (its README carries the numbers). Three findings change
the plan above.

First, the two sets need different line finders. On the eBL photographs, where a line is 120 pixels tall,
the eBL sign detector followed by clustering and the transliteration count finds 87 percent of lines and
places the first line within half a pitch on 89 percent of faces, while the projection-profile methods stop
near 62 percent. On the Old Assyrian faces the order reverses: the detector finds too few signs at 45 pixels
per line, and the profile dynamic programme finds 89 percent of the hand-labelled lines against 43 percent
for the detector route. Phase 3 therefore keeps both, chosen by the resolution and origin of the face.

Second, the transliteration line count is an upper bound, not the number of lines on the face. On half of
the 88 hand-labelled Old Assyrian faces the labeller marked fewer lines than the ATF lists for that side,
typically three fewer, because lines run onto edges, are broken away, or are not visible in the photograph.
Forcing the ATF count therefore costs precision (0.76 against 0.88 with the true count), and choosing the
count automatically is the open problem carried into Phase 4.

Third, the ground truth for the Old Assyrian set came from a click-to-mark page rather than from CSV editing;
90 faces were labelled in one sitting, and the same page can label more faces or a second annotator's pass
whenever the evaluation needs it.

## 12. Status after Phases 4 and 5 (12 September 2026)

Crop generation and the formal evaluation are done on the eBL set; the Old Assyrian set is parked by decision.
Each detected line becomes a strip two pitches tall, because eBL sign boxes turned out to be as tall as the
line pitch and the one-and-a-quarter-pitch crop of section 6 cut the wedges of half the lines. With the
clustering thresholds tuned on one half of the faces and scored on the other, the detector route meets the
three criteria of section 7 on faces with at least 120 pixels per line: recall 0.90, first line within half a
pitch on 92 percent of faces, count agreement 97 percent. Eighty-three percent of annotated lines fall fully inside
one crop and one percent in none. What remains for Phase 6 is the ProtoSnap connection itself, and two known
gaps: faces with more than one column are still excluded, and Neo-Assyrian faces agree with the ATF count far
less often than Old Babylonian ones, because their transliterations count lines the photographs do not show.

## 13. Correction after review (12 September 2026)

The two-pitch crop of section 12 was a wrong turn: it captured whole eBL sign boxes only by including half of
each neighbouring line, which is exactly what the project set out to avoid. Crops are now bounded by the gaps
between lines, found from the detected sign boxes, so each crop holds one line: 99 percent of annotated line
centres fall in exactly one crop, none in two, and only 2 percent of crops contain the centre of a second line.
The sign-level ProtoSnap trial that followed is kept as groundwork for aligning transliterated signs to detected
boxes, but the deliverable of Phase 6 is the clean line crop itself.
