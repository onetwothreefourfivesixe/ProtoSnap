# Adding skeletons for a new cuneiform font

ProtoSnap needs, per font: glyph images (`prototypes/<Font>/0x12xxx/<hex>.png`) and annotated
skeletons (`skeletons/<Font>/<hex>_adf.csv` + `_con.csv`). These scripts bootstrap both for a new
font by transferring the existing Assurbanipal skeletons. Run everything from the repo root.

1. **renderFont.py** — rasterize the SVG glyph collection into the repo's prototype layout:
   `python add_new_font_skeletons/renderFont.py "CuneiformFonts/Esagil (Neo-Babylonian)" Esagil`
   (skips "Variant Form" and compound-sign files).
2. **bootstrapSkeletons.py prepare** — build transfer targets + CSV for every sign that has both a
   new-font glyph and an Assurbanipal skeleton:
   `python add_new_font_skeletons/bootstrapSkeletons.py prepare Esagil`
3. **Run the normal pipeline** (snaps the Assurbanipal skeleton onto each new glyph; needs GPU):
   `python run_test.py --samples_df_path test_set/transfer_Esagil.csv --font_dir prototypes/Assurbanipal --con_dir skeletons/Assurbanipal --output_folder transfer_Esagil --ignore_errors`
4. **detectWedges.py triage** — rank signs by how likely their wedge inventory changed between the
   fonts (`fontTargets/Esagil_triage.csv`); review the overlays (`output/transfer_Esagil/*/itr99_viz.png`)
   for flagged signs first. The detector is a heuristic screen, not ground truth.
5. **bootstrapSkeletons.py convert** — turn the fitted points into skeleton CSVs:
   `python add_new_font_skeletons/bootstrapSkeletons.py convert Esagil`
   Hand-fix the signs whose transfers failed (typically those whose stroke count changed between
   periods — the method can move strokes but never add or remove them).

Then use the new font like any other: `--font_dir prototypes/Esagil --con_dir skeletons/Esagil`.
