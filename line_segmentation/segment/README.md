# Running the line model on a new data set

`segment_images.py` runs the current line model on any folder or list of tablet images and writes the lines it finds, as
polygons and boxes in the pixels of each image as provided. `preprocess.py` prepares the images. Run everything from the
repository root with the project's virtual environment.

## Quick start

For a folder of images, with every image treated as one tablet side:

```
.venv/bin/python -m line_segmentation.segment.segment_images --images <folder> --out <output folder> --overlays 20
```

For a CSV that lists the images, such as the `sides.csv` of the Ur III request:

```
.venv/bin/python -m line_segmentation.segment.segment_images \
    --list tablet_data/nathan_urIII_data/to_eddie_list_only/sides.csv --id-col side_id --file-col file \
    --out tablet_data/nathan_urIII_data/segmentation_run6 --overlays 232 --crops
```

In `--list` mode, paths in `--file-col` are relative to the CSV's folder unless `--root` says otherwise. In `--images`
mode, the folder is searched recursively for PNG, JPEG, TIFF, BMP and WebP files, and each image's side id is its path
without the extension.

## What the pipeline does

1. **Preprocessing** (`preprocess.load_image`). Each image becomes an 8-bit, three-channel image on the same pixel grid as
   the file, so every coordinate written refers to the image as provided. A transparent background, as in MSII and other
   3D renderings, is flattened onto `--background` (black by default, like most museum photographs). 16-bit images are
   scaled to 8 bits, and greyscale images are copied into three channels. `--clahe` equalises the contrast, and `--exif`
   applies the EXIF orientation; both are off by default.
2. **Views** (`preprocess.find_views`). With `--views whole` (the default) each image is one side. With `--views auto` a
   photograph that shows several views of a tablet, such as a CDLI fat cross or an eBL composite, is split into one image
   per view, using the background detection of the Line Splitter app. Each view is saved as `views/<side id>__v<k>.png`
   (largest view first, with a small margin around it) and is segmented as a side of its own. Each view also gets an
   estimated role (`preprocess.label_views`). The two largest views that are neither small (under 35 percent of the largest)
   nor elongated (sides in a ratio above 2.2) are the obverse and the reverse, the upper one being the obverse, as CDLI fat
   crosses and eBL composites lay them out. Every other view is an edge, named `edge_top`, `edge_bottom`, `edge_left` or
   `edge_right` by its place next to the obverse, or plain `edge` when it is not next to it. With `--edges skip` (the
   default), edges are saved but not segmented, because the model finds false lines on most uninscribed edges; on 30 Ur III
   fat crosses, it put a line on 73 percent of edges, while the transliterations show writing on only 5 of 126. Use
   `--edges segment` to run the model on edges as well.
3. **Line model.** The model (`--weights`, by default `paths.LINE_WEIGHTS`, currently `run6_heights_s1`) runs on each side
   resized to 1024 pixels on the long side. The read-out (`linemodel/infer.read_lines`) uses the settings stored in the
   run's `config.json`; every setting can be overridden with options such as `--kernel-thr`.
4. **Line regions.** Each predicted line becomes a four-cornered region. Its top and bottom are the middle of the gaps to
   the neighbouring lines, extended by `--extend` line spacings (0.2) and moved down by `--shift` spacings (0.06), the crop
   rule calibrated on eBL faces. The region runs `--end-margin` spacings (0.5) past the line's ends and follows the line's
   own angle when it is within `--max-line-dev` degrees (20) of the side's usual angle.
5. **Output.** The pipeline writes the files described below. Lines are ordered top to bottom, every side appears even
   when no line was found, and the output is the model's raw read-out: nothing is corrected by hand, and low-scoring lines
   are kept.

## Output

- `lines.json` has three parts.
  - `segmenter` gives the model's name, the git commit and the weights file with its checksum (`version_or_commit`), and
    the tablets the weights were trained or validated on (`trained_on`, with per-face detail in `trained_on_detail`). It
    also says what the read-out and region rule were tuned on (`tuning`).
  - `settings` records every preprocessing and read-out setting used.
  - `sides` has one entry per side, with `side_id`, `width`, `height` and `lines`. With `--views auto`, a side also has
    `image` (its saved view image), `role` (its estimated role), `skipped` (when it is an edge that was not segmented),
    `source_image` and `source_file` (the photograph it came from) and `view_box`
    (`[x0, y0, x1, y1]`, where the view sits in that photograph). Its coordinates are then in the pixels of the view image;
    add `view_box`'s `x0` and `y0` to place them in the photograph. Each line has `order` (1 is the top
    line), `polygon` (four corners, clockwise from the top left of the line), `bbox` (`[x0, y0, x1, y1]`, clipped to the
    image), `score` (the mean kernel probability, between 0 and 1), `baseline` (the predicted centre line's two ends) and
    `angle_deg`. With `--crops`, each line also names its crop image.
- `summary.csv` has one row per side, with the number of lines, the mean score, the time taken and the preprocessing notes.
- `overlays/<side>.jpg` (with `--overlays N`) draws the regions in orange, the centre lines in red, and each line's order
  and score on the first N images.
- `crops/<side>__L<nn>.jpg` (with `--crops`) holds one image per line, cut along the line's own angle.
- `views/<side>__v<k>.png` (with `--views auto`) holds one image per detected view.

## Training data and leakage

`trained_on` comes from `tablet_data/line_model_training_faces.csv`, which `linemodel/list_training_faces.py` writes. If
that file is missing, run the script first; `trained_on` is otherwise marked unknown. Before sharing results on a data set,
compare its tablets with `trained_on` by museum number and by CDLI P-number. The Ur III request was checked this way: none
of its 118 tablets share a museum number or P-number with the 387 training tablets.

## What to expect on new material

The model was trained on photographs of Old Babylonian, Neo-Assyrian, Neo-Babylonian and Old Assyrian tablets. On the
Ur III MSII renderings, it finds the lines of clean, well-preserved sides well. It also reads the inscriptions of seal
impressions as lines, which the transliteration lists separately from the side's own lines, and it finds few lines on
badly worn surfaces. Line counts on the Ur III sides matched the CDLI transliteration exactly on 22 percent of sides and
within one line on 62 percent, mostly because of seal impressions and lines that run onto the edges. On CDLI fat-cross photographs
with `--views auto --edges segment`, the model also finds one or two false lines on most uninscribed edges; segmenting edges
at the obverse's scale instead of enlarging them did not prevent this. Roles are estimates from the layout, and other parts
of a photograph, such as a museum logo, can be detected as views and labelled `edge`. Check the overlays before relying on a new data set.
