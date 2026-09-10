"""Rasterize the CuneiformFonts SVG glyphs into the repo's prototype layout.

Usage: python add_new_font_skeletons/renderFont.py "CuneiformFonts/Esagil (Neo-Babylonian)" Esagil
Writes prototypes/Esagil/0x12xxx/<hex>.png (primary forms; files named '... Variant Form.svg'
are skipped so each sign gets one canonical glyph).
"""
import io
import re
import sys
from pathlib import Path

import cairosvg
from PIL import Image

HEIGHT = 512


def render(svg_path, out_path):
    png_bytes = cairosvg.svg2png(url=str(svg_path), output_height=HEIGHT, background_color="white")
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    img.save(out_path)


def main(src_dir, font_name):
    out_root = Path("prototypes") / font_name
    count, variants, compounds, unparsed = 0, 0, 0, []
    for svg in sorted(Path(src_dir).glob("*.svg")):
        if "Variant Form" in svg.name:
            variants += 1
            continue
        if re.search(r"U\+?1[0-9A-Fa-f]{4}\+", svg.name):
            compounds += 1  # ligature of several signs; not a single codepoint, would overwrite one
            continue
        m = re.search(r"U\+?(1[0-9A-Fa-f]{4})\b", svg.name)
        if not m:
            unparsed.append(svg.name)
            continue
        hex_code = f"0x{m.group(1).lower()}"
        sub = out_root / f"{hex_code[:5]}xx"
        sub.mkdir(parents=True, exist_ok=True)
        render(svg, sub / f"{hex_code}.png")
        count += 1
    print(f"Rendered {count} glyphs to {out_root} (skipped {variants} variant forms, {compounds} compound signs).")
    for name in unparsed:
        print(f"  could not parse codepoint from: {name}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
