# CDLI transliterations extracted from data/cdli/artifacts_json/

The source is `data/cdli/artifacts_json/artifacts_json/*.json`, which holds 421,387 CDLI artifact records, one per P-number.
The file `NNNNNN.json` corresponds to `PNNNNNN`. The transliteration is embedded in each record under
`inscription.atf` (CDLI ATF format). 150,196 records carry an ATF text.

Kuyunjik tablets (K, Sm, Rm, DT ...) have `museum_no = "BM —"`. Their number is in `accession_no`
(e.g. `K 04426`). BM registration numbers appear as `1881-02-04, 0233`, and Rm-II numbers appear as `Rm 2, 0481`.

## Files produced (by `line_cropping/ingest/extract_atf.py` then `line_cropping/ingest/match_museum.py`, run from the repo root)

| file | content |
|---|---|
| `artifacts_index.csv` | one row per artifact: p_number, designation, museum_no, accession_no, period, provenience, genre, language, has_atf, atf_lines |
| `transliterations.jsonl` | one record per artifact that has an ATF text: p_number, designation, museum_no, accession_no, period, atf |
| `protosnap_tablets_to_cdli.csv` | every `tabletNumber` in `data/ebl/signs_snippets_metadata.json` joined to CDLI by museum/accession number, with snippet count, eBL script label, P-number, period, has_atf |
| `protosnap_tablets_atf.jsonl` | the ATF texts for the repo tablets that have one (`tabletNumbers` lists the repo ids that map to that P-number) |

## Join results (repo tablets from data/ebl/signs_snippets_metadata.json)

- There are 9,276 tablets, and 8,929 of them matched to a CDLI record. Of these, 71 match more than one record; these are mostly joins/fragments.
- Only 723 of the matched tablets have an ATF in this dump, covering 37,544 of 158,946 snippets.
  The rest, mostly Neo-Assyrian K-tablets, exist in CDLI without a transliteration here.
  eBL / Oracc are the places to look for those.

ATF uses these line conventions: a `&P...` header, `@obverse`/`@column` structure, `$` state lines, `#` comments
and `>>Q...` composite links. The numbered lines (`1.`, `2'.`) hold the signs.
