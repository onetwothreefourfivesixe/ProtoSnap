"""Fetch Gottstein codes (wedge counts by orientation) for cuneiform sign variants from Wikidata.

The Gottstein system encodes a sign's wedge inventory, e.g. 'a1b2' = 1 horizontal + 2 vertical
wedges (a=horizontal, b=vertical, c=diagonal, w=Winkelhaken). Wikidata stores one code per sign
PER PERIOD (Neo-Assyrian, Neo-Babylonian, Old Babylonian, Hittite, ...), which gives an external,
expert-curated expected wedge count for each sign in each font's period -- far more reliable than
estimating counts from glyph images.

  python add_new_font_skeletons/fetchGottstein.py
      Writes add_new_font_skeletons/gottstein_counts.csv with columns:
      hex, sign_name, period, code, total_wedges
"""
import json
import re
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

QUERY = """SELECT ?label ?code WHERE {
  ?item wdt:P11957 ?code ; rdfs:label ?label .
  FILTER(lang(?label)='en')
}"""


def sign_hex(name):
    try:
        return f"0x{ord(unicodedata.lookup('CUNEIFORM SIGN ' + name.strip().upper())):x}"
    except KeyError:
        try:
            return f"0x{ord(unicodedata.lookup('CUNEIFORM NUMERIC SIGN ' + name.strip().upper())):x}"
        except KeyError:
            return None


def main():
    url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode(
        {"query": QUERY, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": "ProtoSnap-skeletons/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        bindings = json.load(r)["results"]["bindings"]

    pat = re.compile(r"^Cuneiform Sign Variant (?:of )?(?:CUNEIFORM (?:NUMERIC )?SIGN )?(.+?)\s*\((.+?)\)")
    rows = []
    for b in bindings:
        m = pat.match(b["label"]["value"])
        if not m:
            continue
        name, period = m.group(1), m.group(2)
        code = b["code"]["value"]
        counts = re.findall(r"([a-z])(\d+)", code)
        if not counts:
            continue
        rows.append({"hex": sign_hex(name), "sign_name": name, "period": period,
                     "code": code, "total_wedges": sum(int(n) for _, n in counts)})
    df = pd.DataFrame(rows)
    out = Path(__file__).parent / "gottstein_counts.csv"
    df.to_csv(out, index=False)
    resolved = df["hex"].notna().sum()
    print(f"{len(df)} variant codes fetched, {resolved} resolved to a codepoint -> {out}")
    print(df["period"].value_counts().head(8).to_string())


if __name__ == "__main__":
    main()
