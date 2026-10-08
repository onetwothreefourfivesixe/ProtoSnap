# Font tests

ProtoSnap on photographed single signs (eBL sign crops) with the Assurbanipal and Santakku font prototypes. Write-up
and results: `docs/font_tests/font_tests.md`. Run everything from the repo root.

- `testSingleSigns.py` samples sign crops that ProtoSnap can run with one font (set `fontDir` at the top) and writes
  `test_set/singleSigns_<font>.csv`, then prints the `run_test.py` command. ProtoSnap writes its results to
  `output/font_tests/singleSigns_<font>/`.
- `compareToFont.py font_tests/singleSigns_<font>` measures how far each fitted skeleton moved from the font glyph
  (keypoint shift, IoU with the glyph's ink) and writes `font_compare.csv` and per-sign `font_compare.png`.
