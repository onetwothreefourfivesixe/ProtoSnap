# CDLI transliterations extracted from artifacts_json/

Source: `artifacts_json/artifacts_json/*.json` (421,387 CDLI artifact records, one per P-number;
file `NNNNNN.json` = `PNNNNNN`). The transliteration is embedded in each record under
`inscription.atf` (CDLI ATF format). 150,196 records carry an ATF text.

Kuyunjik tablets (K, Sm, Rm, DT ...) have `museum_no = "BM —"`; their number is in `accession_no`
(e.g. `K 04426`). BM registration numbers appear as `1881-02-04, 0233`; Rm-II as `Rm 2, 0481`.

## Files produced (by `line_cropping/ingest/extract_atf.py` then `line_cropping/ingest/match_museum.py`, run from the repo root)

| file | content |
|---|---|
| `artifacts_index.csv` | one row per artifact: p_number, designation, museum_no, accession_no, period, provenience, genre, language, has_atf, atf_lines |
| `transliterations.jsonl` | one record per artifact that has an ATF text: p_number, designation, museum_no, accession_no, period, atf |
| `protosnap_tablets_to_cdli.csv` | every `tabletNumber` in `signs_snippets_metadata.json` joined to CDLI by museum/accession number, with snippet count, eBL script label, P-number, period, has_atf |
| `protosnap_tablets_atf.jsonl` | the ATF texts for the repo tablets that have one (`tabletNumbers` lists the repo ids that map to that P-number) |

## Join results (repo tablets from signs_snippets_metadata.json)

- 9,276 tablets; 8,929 matched to a CDLI record (71 match more than one, mostly joins/fragments).
- Only 723 of the matched tablets have an ATF in this dump (37,544 of 158,946 snippets).
  The rest (mostly Neo-Assyrian K-tablets) exist in CDLI without a transliteration here;
  eBL / Oracc are the places to look for those.

ATF line conventions: `&P...` header, `@obverse`/`@column` structure, `$` state lines, `#` comments,
`>>Q...` composite links; numbered lines (`1.`, `2'.`) hold the signs.
