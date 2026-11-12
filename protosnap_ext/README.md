# protosnap_ext — additions to ProtoSnap

The original ProtoSnap code is unchanged at the repository root (`main.py`, `run_test.py`, `src/`, `prototypes/`,
`skeletons/`, `test_set/`). This folder holds everything this fork adds around it. Run every command from the
repository root.

| folder | what it does | docs | results |
|---|---|---|---|
| `font_tests/` | Runs ProtoSnap on photographed single signs with the Assurbanipal and Santakku prototypes, and measures how far the fitted skeletons move from the font. See [font_tests/README.md](font_tests/README.md). | `docs/protosnap/font_tests/` | `output/protosnap/font_tests/` |
| `esagil_skeletons/` | Builds Esagil (Neo-Babylonian) skeletons, both transferred from Assurbanipal and generated from the font, and samples the results. The skeletons themselves are in `skeletons/Esagil/` and `prototypes/Esagil/`. See [esagil_skeletons/README.md](esagil_skeletons/README.md). | `docs/protosnap/esagil/` | `output/protosnap/esagil/` |
| `line_inputs/` | Turns the line crops made by `line_segmentation/` into ProtoSnap inputs (`protosnap_prep.py`), draws and summarises the runs (`overlay_results.py`, `phase6_summary.py`), and runs the line-envelope pilot (`line_envelope_prep.py`, `line_envelope_analyze.py`). Run as `python -m protosnap_ext.line_inputs.<module>`. | phase 6 in `line_segmentation/README.md` | `output/line_segmentation/phase6_*` |
| `sign_classifier/` | Trains and runs a ResNet-34 sign classifier on the eBL sign crops (`build_dataset.py`, `train_classifier.py`, `classify.py`). Run as `python -m protosnap_ext.sign_classifier.<module>`. | `line_segmentation/README.md` | `weights/sign_classifier/` |

ProtoSnap writes each run to `output/<output_folder>/`, so pass a nested name such as
`--output_folder protosnap/esagil/transfer_Esagil` to keep results in this layout.

`line_inputs/` and `sign_classifier/` read their data locations from `line_segmentation/paths.py`.
