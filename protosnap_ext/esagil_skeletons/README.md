# Adding skeletons for a new cuneiform font

For each font, ProtoSnap needs glyph images (`prototypes/<Font>/0x12xxx/<hex>.png`) and annotated
skeletons (`skeletons/<Font>/<hex>_adf.csv` and `_con.csv`). These scripts bootstrap both for a new
font by transferring the existing Assurbanipal skeletons. Run everything from the repo root.

1. **renderFont.py** — This script rasterizes the SVG glyph collection into the repo's prototype layout. It skips
   "Variant Form" and compound-sign files:
   `python protosnap_ext/esagil_skeletons/renderFont.py "data/fonts/CuneiformFonts/Esagil (Neo-Babylonian)" Esagil`
2. **bootstrapSkeletons.py prepare** — This step builds transfer targets and a CSV for every sign that has both a
   new-font glyph and an Assurbanipal skeleton:
   `python protosnap_ext/esagil_skeletons/bootstrapSkeletons.py prepare Esagil`
3. **Run the normal pipeline.** It snaps the Assurbanipal skeleton onto each new glyph and needs a GPU:
   `python run_test.py --samples_df_path test_set/transfer_Esagil.csv --font_dir prototypes/Assurbanipal --con_dir skeletons/Assurbanipal --output_folder protosnap/esagil/transfer_Esagil --ignore_errors`
4. **detectWedges.py triage** — This step ranks signs by how likely it is that their wedge inventory changed between
   the fonts (`data/fonts/fontTargets/Esagil_triage.csv`). Review the overlays (`output/protosnap/esagil/transfer_Esagil/*/itr99_viz.png`)
   for flagged signs first. The detector is a heuristic screen, not ground truth.
5. **bootstrapSkeletons.py convert** — This step turns the fitted points into skeleton CSVs:
   `python protosnap_ext/esagil_skeletons/bootstrapSkeletons.py convert Esagil`
   Hand-fix the signs whose transfers failed. These are typically the signs whose stroke count changed between
   periods, because the method can move strokes but never add or remove them.

Then use the new font like any other: `--font_dir prototypes/Esagil --con_dir skeletons/Esagil`.
