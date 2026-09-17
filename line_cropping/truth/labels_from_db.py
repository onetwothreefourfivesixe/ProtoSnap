"""Turn label documents pulled from the labelling page's database into Old Assyrian line ground truth.

Pull first (from the Claude session that owns the artifact):
  Artifact read_db  url=<labeler url>  db_op=list  collection=labels  out_dir=fat-cross_processed/ground_truth/db
Then:  python -m line_cropping.truth.labels_from_db
Writes fat-cross_processed/ground_truth/labels.csv: face_id, split, status, line_no, y_face_deskewed, skew_deg, src
(y is a row in the full-size face after rotation by skew_deg with cv2.getRotationMatrix2D(centre, skew_deg)).
"""
import csv, glob, json, os

def main():
    SAMPLE = {r['face_id']: r for r in csv.DictReader(open('fat-cross_processed/ground_truth/sample_faces.csv'))}
    rows = []; n_done = 0
    for p in glob.glob('fat-cross_processed/ground_truth/db/labels/*.json'):
        d = json.load(open(p, encoding='utf-8')); d = d.get('data', d)
        fid = d.get('face'); s = SAMPLE.get(fid)
        if not fid or not s: continue
        if d.get('status') == 'done': n_done += 1
        for i, l in enumerate(sorted(d.get('lines', []), key=lambda l: l['y']), 1):
            rows.append(dict(face_id=fid, split=s['split'], status=d.get('status', ''), line_no=i, y_face_deskewed=round(float(l['y']) / float(d['scale']), 1), skew_deg=d.get('skew_deg'), src=l.get('src', '')))
    os.makedirs('fat-cross_processed/ground_truth', exist_ok=True)
    with open('fat-cross_processed/ground_truth/labels.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['face_id', 'split', 'status', 'line_no', 'y_face_deskewed', 'skew_deg', 'src']); w.writeheader(); w.writerows(rows)
    print(f'{len({r["face_id"] for r in rows})} faces labelled ({n_done} marked done), {len(rows)} lines -> fat-cross_processed/ground_truth/labels.csv')



if __name__ == '__main__':
    main()
