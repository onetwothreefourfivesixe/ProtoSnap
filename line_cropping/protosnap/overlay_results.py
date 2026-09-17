"""Overlay ProtoSnap results back onto the tablet: on the original (unrotated) face and on the gap-bounded line crop.

ProtoSnap saves the fitted skeleton as dst_points_<itr>.pt in the 512 x 512 frame of the (stretched) target image,
with y flipped for plotting. This tool undoes the flip, maps the points through the padded sign box that the target
was cut from (alignment.csv, same padding as protosnap_prep.py) into face pixels, rotates them into the deskewed
frame of the face's line crops, and draws them.

  python -m line_cropping.protosnap.overlay_results --run output/phase6_OB --samples protosnap_inputs/OB/samples_run40.csv \
      --alignment protosnap_inputs/OB/alignment.csv --out output/phase6_OB/overlays
Writes <out>/faces/<face>.jpg, <out>/lines/<face>__L<nn>.jpg, <out>/line_contact_sheet.jpg, <out>/index.csv.
"""
import argparse, csv, os, collections
import numpy as np, cv2, torch
from line_cropping.lines.profile_lines import rotate

IMG = 512
COLORS = [(60, 76, 231), (219, 152, 52), (34, 189, 34), (182, 89, 155), (0, 165, 255), (128, 128, 0), (200, 50, 200), (30, 200, 200)]


def load_fit(run_dir, itr=100):
    pts = torch.load(os.path.join(run_dir, f'dst_points_{itr}.pt')).numpy().astype(float); pts[:, 1] = IMG - pts[:, 1]
    con = torch.load(os.path.join(run_dir, f'connectivity_{itr}.pt'))
    try: labels = torch.load(os.path.join(run_dir, f'labels_{itr}.pt'))
    except Exception: labels = None
    return pts, con, (list(labels) if labels is not None else None)


def draw_skeleton(img, pts, con, labels, thickness=2):
    for i, nbrs in enumerate(con):
        c = COLORS[(labels[i] if labels else 0) % len(COLORS)]
        for j in nbrs: cv2.line(img, (int(pts[i][0]), int(pts[i][1])), (int(pts[j][0]), int(pts[j][1])), c, thickness, cv2.LINE_AA)
    for i, p in enumerate(pts): cv2.circle(img, (int(p[0]), int(p[1])), thickness + 1, COLORS[(labels[i] if labels else 0) % len(COLORS)], -1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); ap.add_argument('--samples', required=True); ap.add_argument('--alignment', required=True)
    ap.add_argument('--faces', default='ebl_tablets/faces/faces.csv'); ap.add_argument('--crops', default='ebl_tablets/line_crops/crops.csv'); ap.add_argument('--out', required=True)
    ap.add_argument('--pad', type=float, default=0.15); ap.add_argument('--itr', type=int, default=100)
    a = ap.parse_args(); os.makedirs(os.path.join(a.out, 'faces'), exist_ok=True); os.makedirs(os.path.join(a.out, 'lines'), exist_ok=True)
    faces = {f['face']: f for f in csv.DictReader(open(a.faces))}
    crops = {(r['face'], int(r['line_no'])): r for r in csv.DictReader(open(a.crops))}
    align = {r['target']: r for r in csv.DictReader(open(a.alignment, encoding='utf-8')) if r.get('target')}
    samples = {r['fn']: r for r in csv.DictReader(open(a.samples, encoding='utf-8'))}
    per_face = collections.defaultdict(list)
    for fn, s in samples.items():
        run_dir = os.path.join(a.run, fn[:-4] + '_results')
        if not os.path.exists(os.path.join(run_dir, f'dst_points_{a.itr}.pt')) or fn not in align: continue
        al = align[fn]; f = faces[al['face']]; W, H = int(f['width']), int(f['height'])
        x0, y0, x1, y1 = map(float, al['box'].split()); pw, ph = (x1 - x0) * a.pad, (y1 - y0) * a.pad
        X0, Y0, X1, Y1 = int(max(0, x0 - pw)), int(max(0, y0 - ph)), int(min(W, x1 + pw)), int(min(H, y1 + ph))
        pts, con, labels = load_fit(run_dir, a.itr)
        face_pts = np.stack([X0 + pts[:, 0] * (X1 - X0) / IMG, Y0 + pts[:, 1] * (Y1 - Y0) / IMG], axis=1)
        per_face[al['face']].append(dict(fn=fn, line_no=int(al['line_no']), sign=s['sign'], name=s['name'], box=(X0, Y0, X1, Y1), pts=face_pts, con=con, labels=labels))
    index = []; line_tiles = []
    for face, items in per_face.items():
        f = faces[face]; img = cv2.imread(f['file']); H, W = img.shape[:2]
        # --- face overlay (unrotated) ---
        ov = img.copy()
        for it in items:
            X0, Y0, X1, Y1 = it['box']; cv2.rectangle(ov, (X0, Y0), (X1, Y1), (0, 255, 0), 3); draw_skeleton(ov, it['pts'], it['con'], it['labels'], 3)
            cv2.putText(ov, it['name'], (X0, max(20, Y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        sc = min(1.0, 1600 / max(H, W)); cv2.imwrite(os.path.join(a.out, 'faces', face + '.jpg'), cv2.resize(ov, None, fx=sc, fy=sc), [cv2.IMWRITE_JPEG_QUALITY, 85])
        # --- line overlays (deskewed gap crops) ---
        by_line = collections.defaultdict(list)
        for it in items: by_line[it['line_no']].append(it)
        for ln, its in by_line.items():
            c = crops.get((face, ln))
            if c is None: continue
            skew = float(c['skew_deg']); M = cv2.getRotationMatrix2D((W / 2, H / 2), skew, 1.0)
            cx0, cy0 = int(c['x0']), int(c['y0']); crop = cv2.imread(c['crop'])
            if crop is None: continue
            up = 2 if crop.shape[0] < 300 else 1
            if up > 1: crop = cv2.resize(crop, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
            for it in its:
                P = (M @ np.hstack([it['pts'], np.ones((len(it['pts']), 1))]).T).T; P = (P - [cx0, cy0]) * up
                X0, Y0, X1, Y1 = it['box']; corners = np.array([[X0, Y0, 1], [X1, Y0, 1], [X1, Y1, 1], [X0, Y1, 1]], float)
                Q = ((M @ corners.T).T - [cx0, cy0]) * up; cv2.polylines(crop, [Q.astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)
                draw_skeleton(crop, P, it['con'], it['labels'], 2)
                cv2.putText(crop, it['name'], (int(Q[0][0]), max(16, int(Q[0][1]) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                index.append(dict(face=face, line_no=ln, target=it['fn'], sign=it['sign'], prototype=it['name'], face_overlay=os.path.join(a.out, 'faces', face + '.jpg'), line_overlay=os.path.join(a.out, 'lines', f'{face}__L{ln:02d}.jpg')))
            cv2.putText(crop, f'{face}  line {ln}' + (f"  (ATF {c['atf_line']})" if c['atf_line'] else ''), (6, crop.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(a.out, 'lines', f'{face}__L{ln:02d}.jpg'), crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
            s = 1200 / crop.shape[1]; line_tiles.append(cv2.resize(crop, (1200, max(1, int(crop.shape[0] * s)))))
    if line_tiles:
        sheet = np.vstack([t if k % 2 else t for k, t in enumerate(sum([[t, np.zeros((8, 1200, 3), np.uint8)] for t in line_tiles], []))])
        cv2.imwrite(os.path.join(a.out, 'line_contact_sheet.jpg'), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    with open(os.path.join(a.out, 'index.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(index[0].keys())); w.writeheader(); w.writerows(index)
    print(f'{len(index)} results drawn on {len(per_face)} faces and {len(line_tiles)} line crops -> {a.out}')


if __name__ == '__main__':
    main()
