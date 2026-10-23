"""Line envelopes from ProtoSnap, step 2: join each line's fitted sign skeletons and compare them with the line crop.

  python -m line_cropping.protosnap.line_envelope_analyze [--inputs tablet_data/protosnap_inputs/envelope_pilot]
      [--runs output/line_cropping/envelope_pilot_{script}] [--min-score 0.5] [--render 12] [--out <inputs>/analysis]

For every sign target fitted by ProtoSnap (step 1: line_envelope_prep.py), the fitted skeleton points (dst_points_100.pt,
512 x 512 frame of the stretched target, y flipped) are mapped back into the deskewed face, the frame the crop was cut
in. Fits whose ProtoSnap score (agg_score.pt, 0-1) is below --min-score are left out of the envelopes and reported.

Per sign: the skeleton's extent against the eBL box (how loose the boxes are). Per line, with all kept skeletons:
  clip_top / clip_bottom / clip_left / clip_right   share of skeleton points outside that crop edge
  over_*_pitch                                      how far the wedge envelope reaches past that edge, in line pitches
  slack_*                                           crop space beyond the wedge envelope, in pitches
  The envelope is the 2.5th-97.5th percentile of the line's wedge points (one stretched fit must not define it);
  envelope_h_max_pitch keeps the full extent.
  slant_deg, bend_pitch                             angle of the line through the sign centres in the crop frame, and
                                                   the largest sign-centre deviation from that straight line
  neighbour_box_in_crop                             share of the eBL boxes of other lines' signs that falls inside the crop
Writes <out>/signs.csv, lines.csv, summary.json and renders/<crop>.jpg (crop red, eBL boxes grey, kept fits green, fits
below --min-score orange, convex hull of the kept wedges cyan).
"""
import argparse
import collections
import csv
import json
import os

import cv2
import numpy as np
import torch

IMG = 512


def load_fit(run_dir):
    pts = torch.load(os.path.join(run_dir, 'dst_points_100.pt')).numpy().astype(float)
    pts[:, 1] = IMG - pts[:, 1]
    con = torch.load(os.path.join(run_dir, 'connectivity_100.pt'))
    score = float(torch.load(os.path.join(run_dir, 'agg_score.pt')))
    return pts, con, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default='tablet_data/protosnap_inputs/envelope_pilot')
    ap.add_argument('--runs', default='output/line_cropping/envelope_pilot_{script}')
    ap.add_argument('--export', default='tablet_data/exports/line_crops_60')
    ap.add_argument('--min-score', type=float, default=0.5)
    ap.add_argument('--render', type=int, default=12)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.inputs, 'analysis')
    os.makedirs(os.path.join(out, 'renders'), exist_ok=True)
    faces = {r['face']: r for r in csv.DictReader(open(os.path.join(a.export, 'faces.csv')))}
    all_signs = collections.defaultdict(list)            # every eBL box of a face, for neighbour intrusion
    for s in csv.DictReader(open(os.path.join(a.export, 'signs.csv'))):
        all_signs[s['face']].append(s)
    T = []
    for sc in ('OB', 'NA'):
        p = os.path.join(a.inputs, sc, 'targets.csv')
        if os.path.exists(p):
            T += list(csv.DictReader(open(p)))
    sign_rows, by_line = [], collections.defaultdict(list)
    for t in T:
        rd = os.path.join(a.runs.format(script=t['script']), t['target'] + '_results')
        if not os.path.exists(os.path.join(rd, 'dst_points_100.pt')):
            continue
        pts, con, score = load_fit(rd)
        tx0, ty0, tx1, ty1 = (float(t[k]) for k in ('target_x0', 'target_y0', 'target_x1', 'target_y1'))
        P = np.c_[tx0 + pts[:, 0] / IMG * (tx1 - tx0), ty0 + pts[:, 1] / IMG * (ty1 - ty0)]
        bx0, by0, bx1, by1 = (float(t[k]) for k in ('box_x0', 'box_y0', 'box_x1', 'box_y1'))
        pitch = float(faces[t['face']]['line_pitch_px'])
        kept = score >= a.min_score
        sign_rows.append(dict(target=t['target'], crop_id=t['crop_id'], script=t['script'], sign=t['sign'], score=round(score, 3), kept=kept,
                              skel_w_over_box=round((P[:, 0].max() - P[:, 0].min()) / max(1, bx1 - bx0), 3),
                              skel_h_over_box=round((P[:, 1].max() - P[:, 1].min()) / max(1, by1 - by0), 3),
                              skel_outside_box=round(float(np.mean((P[:, 0] < bx0) | (P[:, 0] > bx1) | (P[:, 1] < by0) | (P[:, 1] > by1))), 3),
                              centre_dy_pitch=round(((P[:, 1].mean()) - (by0 + by1) / 2) / pitch, 3)))
        by_line[t['crop_id']].append(dict(t=t, P=P, con=con, score=score, kept=kept))
    line_rows = []
    for cid, items in by_line.items():
        t0 = items[0]['t']; f = faces[t0['face']]; pitch = float(f['line_pitch_px'])
        cx0, cy0, cx1, cy1 = (float(t0[k]) for k in ('crop_x0', 'crop_y0', 'crop_x1', 'crop_y1'))
        K = [it for it in items if it['kept']]
        if not K:
            continue
        P = np.vstack([it['P'] for it in K])
        # robust envelope: 2.5th-97.5th percentile of the wedge points, so one stretched fit does not define the line
        ylo, yhi = np.percentile(P[:, 1], [2.5, 97.5]); xlo, xhi = np.percentile(P[:, 0], [2.5, 97.5])
        out_top, out_bot = P[:, 1] < cy0, P[:, 1] > cy1
        out_l, out_r = P[:, 0] < cx0, P[:, 0] > cx1
        cen = np.array([it['P'].mean(0) for it in K])
        if len(cen) >= 2:
            k, b = np.polyfit(cen[:, 0], cen[:, 1], 1)
            slant = float(np.degrees(np.arctan(k))); bend = float(np.max(np.abs(cen[:, 1] - (k * cen[:, 0] + b)))) / pitch
        else:
            slant = bend = float('nan')
        # neighbour intrusion: eBL boxes of signs from other lines on the same face (deskewed frame) inside this crop
        nb_area = nb_in = 0.0
        for s in all_signs[t0['face']]:
            if s['crop_id'] == cid:
                continue
            x0, y0, x1, y1 = (float(s[k]) for k in ('deskew_x0', 'deskew_y0', 'deskew_x1', 'deskew_y1'))
            ix = max(0.0, min(x1, cx1) - max(x0, cx0)); iy = max(0.0, min(y1, cy1) - max(y0, cy0))
            if ix * iy > 0:
                nb_in += ix * iy; nb_area += (x1 - x0) * (y1 - y0)
        line_rows.append(dict(crop_id=cid, script=t0['script'], face=t0['face'], line_no=t0['line_no'], signs=len(items), signs_kept=len(K), pitch_px=round(pitch, 1),
                              crop_h_pitch=round((cy1 - cy0) / pitch, 3),
                              clip_top=round(float(out_top.mean()), 4), clip_bottom=round(float(out_bot.mean()), 4),
                              clip_left=round(float(out_l.mean()), 4), clip_right=round(float(out_r.mean()), 4),
                              over_top_pitch=round(max(0.0, cy0 - ylo) / pitch, 3), over_bottom_pitch=round(max(0.0, yhi - cy1) / pitch, 3),
                              over_left_pitch=round(max(0.0, cx0 - xlo) / pitch, 3), over_right_pitch=round(max(0.0, xhi - cx1) / pitch, 3),
                              slack_top=round(max(0.0, ylo - cy0) / pitch, 3), slack_bottom=round(max(0.0, cy1 - yhi) / pitch, 3),
                              slack_left=round(max(0.0, xlo - cx0) / pitch, 3), slack_right=round(max(0.0, cx1 - xhi) / pitch, 3),
                              envelope_h_pitch=round((yhi - ylo) / pitch, 3), envelope_h_max_pitch=round((P[:, 1].max() - P[:, 1].min()) / pitch, 3),
                              slant_deg=round(slant, 2), bend_pitch=round(bend, 3), neighbour_box_in_crop=round(nb_in / nb_area, 4) if nb_area else 0.0))
    for name, rows in (('signs.csv', sign_rows), ('lines.csv', line_rows)):
        with open(os.path.join(out, name), 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    S, L = sign_rows, line_rows
    med = lambda rows, k: float(np.median([r[k] for r in rows])) if rows else float('nan')
    share = lambda rows, f: float(np.mean([f(r) for r in rows])) if rows else float('nan')
    summary = dict(targets=len(T), fitted=len(S), kept=sum(r['kept'] for r in S), score_median=med(S, 'score'), lines=len(L),
                   sign_box=dict(skel_w_over_box=med([r for r in S if r['kept']], 'skel_w_over_box'), skel_h_over_box=med([r for r in S if r['kept']], 'skel_h_over_box'),
                                 skel_points_outside_box=med([r for r in S if r['kept']], 'skel_outside_box')),
                   lines_with_wedges_cut={k: share(L, lambda r, k=k: r[f'clip_{k}'] > 0) for k in ('top', 'bottom', 'left', 'right')},
                   median_overshoot_when_cut_pitch={k: med([r for r in L if r[f'clip_{k}'] > 0], f'over_{k}_pitch') for k in ('top', 'bottom', 'left', 'right')},
                   median_slack_pitch={k: med(L, f'slack_{k}') for k in ('top', 'bottom', 'left', 'right')},
                   crop_h_pitch=med(L, 'crop_h_pitch'), envelope_h_pitch=med(L, 'envelope_h_pitch'),
                   slant_deg_abs_median=float(np.median([abs(r['slant_deg']) for r in L if np.isfinite(r['slant_deg'])])),
                   bend_pitch_median=float(np.nanmedian([r['bend_pitch'] for r in L])), neighbour_box_in_crop_median=med(L, 'neighbour_box_in_crop'),
                   by_script={sc: dict(lines=len([r for r in L if r['script'] == sc]),
                                       wedges_cut_top=share([r for r in L if r['script'] == sc], lambda r: r['clip_top'] > 0),
                                       wedges_cut_bottom=share([r for r in L if r['script'] == sc], lambda r: r['clip_bottom'] > 0),
                                       slack_top=med([r for r in L if r['script'] == sc], 'slack_top'), slack_bottom=med([r for r in L if r['script'] == sc], 'slack_bottom'))
                              for sc in ('OB', 'NA')})
    json.dump(summary, open(os.path.join(out, 'summary.json'), 'w'), indent=1)
    print(json.dumps(summary, indent=1))
    # renders: the lines with the most wedges cut first, then the rest
    order = sorted(L, key=lambda r: -(r['clip_top'] + r['clip_bottom'] + r['clip_left'] + r['clip_right']))[:a.render]
    rot_cache = {}
    for r in order:
        items = by_line[r['crop_id']]; t0 = items[0]['t']; f = faces[t0['face']]
        if t0['face'] not in rot_cache:
            im = cv2.imread(os.path.join(a.export, f['image'])); H, W = im.shape[:2]
            M = cv2.getRotationMatrix2D((W / 2, H / 2), float(f['skew_deg']), 1.0); rot_cache = {t0['face']: cv2.warpAffine(im, M, (W, H))}
        rot = rot_cache[t0['face']].copy(); pitch = float(f['line_pitch_px'])
        cx0, cy0, cx1, cy1 = (int(float(t0[k])) for k in ('crop_x0', 'crop_y0', 'crop_x1', 'crop_y1'))
        for it in items:
            t = it['t']; cv2.rectangle(rot, (int(t['box_x0']), int(t['box_y0'])), (int(t['box_x1']), int(t['box_y1'])), (170, 170, 170), 1)
            col = (40, 200, 40) if it['kept'] else (0, 140, 255)
            for i, nb in enumerate(it['con']):
                for j in nb:
                    cv2.line(rot, tuple(int(v) for v in it['P'][i]), tuple(int(v) for v in it['P'][j]), col, 2, cv2.LINE_AA)
        K = np.vstack([it['P'] for it in items if it['kept']]) if any(it['kept'] for it in items) else None
        if K is not None:
            hull = cv2.convexHull(K.astype(np.int32)); cv2.polylines(rot, [hull], True, (220, 200, 0), 2)
        cv2.rectangle(rot, (cx0, cy0), (cx1, cy1), (30, 30, 230), 2)
        pad = int(1.2 * pitch); y0, y1 = max(0, cy0 - pad), min(rot.shape[0], cy1 + pad); x0, x1 = max(0, cx0 - pad), min(rot.shape[1], cx1 + pad)
        view = rot[y0:y1, x0:x1]; s = min(1.0, 1400 / view.shape[1]); view = cv2.resize(view, None, fx=s, fy=s)
        bar = np.full((28, view.shape[1], 3), 245, np.uint8)
        cv2.putText(bar, f"{r['crop_id']}: cut top {r['clip_top']:.0%} bottom {r['clip_bottom']:.0%} left {r['clip_left']:.0%} right {r['clip_right']:.0%}; slant {r['slant_deg']:.1f} deg",
                    (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out, 'renders', r['crop_id'] + '.jpg'), np.vstack([bar, view]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f'-> {out}: signs.csv, lines.csv, summary.json, renders/ ({len(order)})')


if __name__ == '__main__':
    main()
