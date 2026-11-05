"""Why signs in signs_outside_crops.csv (report/export_model_line_signs.py) fall in no crop.

  python -m line_cropping.report.explain_signs_outside [--export tablet_data/exports/line_signs_run6]

Each sign is placed by its box centre against every crop of its face, in that crop's frame. `line_caught` is how many other signs of its eBL
line are in a crop. Causes, checked in this order:
  off_its_line    the box centre is at least --off pitch from the median row of its eBL line's boxes: a box assigned to the wrong line, or a stray box
  line_missed     no crop's rows hold the sign and no other sign of its line is in a crop: the model found no line there
  between_crops   no crop's rows hold the sign, though other signs of its line are in a crop: the sign sits above or below its crop
  past_line_end   a crop's rows hold the sign and that crop has signs of its line: the crop stops short (past_end_pitch says by how much)
  beside_other_line   a crop's rows hold the sign, but that crop has no sign of its line, usually a short line or fragment beside it, not found separately
Writes signs_outside_crops_causes.csv next to the input and prints the counts.
"""
import argparse
import collections
import csv
import os

import numpy as np

from line_cropping.crops.crop_frame import crop_matrix
from line_cropping.paths import EBL


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--export', default='tablet_data/exports/line_signs_run6'); ap.add_argument('--off', type=float, default=0.75)
    ap.add_argument('--crops', default=os.path.join(EBL, 'line_crops_unet', 'crops.csv'))
    a = ap.parse_args()
    O = list(csv.DictReader(open(os.path.join(a.export, 'signs_outside_crops.csv'), encoding='utf-8')))
    faces = {f['face']: f for f in csv.DictReader(open(os.path.join(EBL, 'faces', 'faces.csv')))}
    crops = collections.defaultdict(list)
    for r in csv.DictReader(open(a.crops)):
        crops[r['face']].append(r)
    inside = collections.defaultdict(list)                      # (face, line index) -> [(crop, centre x, centre y)] of the signs that are in a crop
    crop_lines = collections.defaultdict(set)
    for s in csv.DictReader(open(os.path.join(a.export, 'signs.csv'), encoding='utf-8')):
        cx, cy = (float(s['face_x0']) + float(s['face_x1'])) / 2, (float(s['face_y0']) + float(s['face_y1'])) / 2
        inside[(s['face'], s['ebl_line_index'])].append((cx, cy)); crop_lines[s['line_id']].add(s['ebl_line_index'])
    outs = collections.defaultdict(list)
    for s in O:
        outs[(s['face'], s['ebl_line_index'])].append(((float(s['face_x0']) + float(s['face_x1'])) / 2, (float(s['face_y0']) + float(s['face_y1'])) / 2))
    rows = []
    for s in O:
        f = faces[s['face']]; W, H = int(f['width']), int(f['height']); key = (s['face'], s['ebl_line_index'])
        p = np.array([(float(s['face_x0']) + float(s['face_x1'])) / 2, (float(s['face_y0']) + float(s['face_y1'])) / 2, 1.0])
        pitch = float(f['pitch_px'] or crops[s['face']][0]['pitch_px']); best = None
        for c in crops[s['face']]:
            x, y = crop_matrix(c, W, H) @ p; y0, y1, x0, x1 = float(c['y0']), float(c['y1']), float(c['x0']), float(c['x1'])
            row_gap = max(0.0, y0 - y, y - y1); x_gap = max(0.0, x0 - x, x - x1); name = os.path.splitext(os.path.basename(c['crop']))[0]
            cand = (row_gap > 0, x_gap if row_gap == 0 else row_gap, name, x_gap, row_gap, 'left' if x < x0 else 'right' if x >= x1 else '')
            best = cand if best is None or cand[:2] < best[:2] else best
        own = inside.get(key, []); allp = np.array(own + outs[key]); M = crop_matrix(crops[s['face']][0], W, H)
        rowsy = allp @ M[1, :2] + M[1, 2] if len(allp) else np.array([]); off = abs(float(M[1] @ p) - float(np.median(rowsy))) / pitch if len(allp) > 1 else 0.0
        in_rows = not best[0]; caught = len(own); same = s['ebl_line_index'] in crop_lines.get(best[2], ())
        cause = ('off_its_line' if off >= a.off else 'line_missed' if not in_rows and not caught else 'between_crops' if not in_rows
                 else 'past_line_end' if same else 'beside_other_line')
        rows.append(dict(s, script=f['script'], cause=cause, nearest_crop=best[2], line_signs=len(allp), line_caught=caught, off_line_pitch=round(off, 2),
                         past_end_pitch=round(best[3] / pitch, 2) if in_rows else '', row_gap_pitch=round(best[4] / pitch, 2), side=best[5]))
    out = os.path.join(a.export, 'signs_outside_crops_causes.csv')
    with open(out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(len(rows), collections.Counter(r['cause'] for r in rows).most_common(), '->', out)


if __name__ == '__main__':
    main()
