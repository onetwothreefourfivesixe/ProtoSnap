# Zenodo record 10693601 (eBL "Sign Detection for Cuneiform Tablets", 2024) — raw-data.tar.gz

Extracted to `raw/raw-data/`. No licence is stated on the record; treat as research-use only.

| folder | content |
|---|---|
| `ebl-30-11/detection/` | 315 eBL composite museum photos (`imgs/<museum no>.jpg`, same files and pixel sizes as the eBL API photos, median 2,779 x 5,053 px) with `annotations/gt_<museum no>.txt`: one sign per row, `x,y,w,h,sign` in pixels. No line indices, sides or transliterations. 89 of the 315 tablets are also in `ebl_tablets/` (NA/OB); the rest are other periods. |
| `ebl-30-11/classification/` | 476 sign-crop images with class annotations. |
| `heidelberg/` | 81 Neo-Assyrian tablet images with XML sign boxes (the Dencker et al. 2020 test tablets). |
| `deepscribe/` | 1,239 Persepolis Fortification (Elamite) images with hotspot JSON. |
| `jooch/`, `urschrei-CDP/` | ~15k and ~12k images from other cuneiform datasets (JOOCH, CDP/Urschrei). |

Conclusion for line cropping: these are NOT cropped tablet sides. The eBL portion is the same photographs we
already have from the API, with fewer fields. Its only extra value is 226 annotated tablets outside our
Neo-Assyrian/Old Babylonian selection, usable if other periods are added later, and the Heidelberg set as an
independent Neo-Assyrian check.
