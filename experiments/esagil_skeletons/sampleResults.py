"""Write a markdown page with a random sample of ProtoSnap results for a font run (default: the Esagil run).

  python experiments/esagil_skeletons/sampleResults.py [--run output/esagil/singleSigns_Esagil_New] [--samples test_set/singleSigns_Esagil.csv]
      [--n 10] [--seed 7] [--include <result id> ...] [--out docs/esagil/esagil_results.md]

Only finished results are eligible (dst_points_100.pt and itr99_viz.png present). --include forces specific results
into the sample (ids as in the run folder without the _results suffix, e.g. BM.38166_1); the rest are drawn at random
with the given seed. Each sample's itr99_viz.png panel is copied to img/<font>/ next to the output page (docs/esagil/img/esagil/) as a JPEG.
"""
import argparse, csv, glob, os, random, statistics
import cv2, torch


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', default='output/esagil/singleSigns_Esagil_New'); ap.add_argument('--samples', default='test_set/singleSigns_Esagil.csv')
    ap.add_argument('--font', default='Esagil'); ap.add_argument('--n', type=int, default=10); ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--include', nargs='*', default=[]); ap.add_argument('--out', default='docs/esagil/esagil_results.md')
    a = ap.parse_args()
    meta = {r['fn'].rsplit('.', 1)[0]: r for r in csv.DictReader(open(a.samples, encoding='utf-8'))}
    done = {}
    for d in sorted(glob.glob(os.path.join(a.run, '*_results'))):
        rid = os.path.basename(d)[:-len('_results')]
        if os.path.exists(os.path.join(d, 'dst_points_100.pt')) and os.path.exists(os.path.join(d, 'itr99_viz.png')):
            try: score = float(torch.load(os.path.join(d, 'agg_score.pt')))
            except Exception: score = float('nan')
            done[rid] = dict(dir=d, score=score)
    missing = [i for i in a.include if i not in done]
    if missing: raise SystemExit(f'not finished or not found in {a.run}: {missing}')
    random.seed(a.seed); pool = [i for i in sorted(done) if i not in a.include]; pick = list(a.include) + random.sample(pool, max(0, a.n - len(a.include)))
    img_dir = os.path.join(os.path.dirname(a.out), 'img', a.font.lower()); os.makedirs(img_dir, exist_ok=True)
    scores = [v['score'] for v in done.values() if v['score'] == v['score']]
    n_skel = len([f for f in os.listdir(f'skeletons/{a.font}') if f.endswith('_adf.csv')])
    L = [f'# ProtoSnap on {a.font} (Neo-Babylonian): a random sample of results', '',
         f'This page shows {len(pick)} results drawn at random (seed {a.seed}' + (f', with {", ".join(a.include)} included by request' if a.include else '') + f') from the {a.font} run in `{a.run}/`. They are not selected for quality.', '',
         '## What this is', '',
         f'ProtoSnap aligns a font-derived **prototype** of a cuneiform sign, with a **skeleton** recording each wedge as a small stroke graph, to a photograph of that sign. Until recently the repository had skeletons only for the Santakku (Old Babylonian) and Assurbanipal (Neo-Assyrian) fonts, so Neo-Babylonian signs could not be processed. The {n_skel} skeletons now in `skeletons/{a.font}/` were generated with **ProtoSnapVectorMapper** [1], which predicts ProtoSnap-format skeletons (`*_adf.csv` points, `*_con.csv` connectivity) directly from rendered font glyphs and provides a browser editor for reviewing them, instead of annotating every sign by hand.', '',
         f'The run uses the Neo-Babylonian sign photographs of the eBL sign-crop dataset (`data/ebl/singleSignImages/`, listed in `{a.samples}`) whose sign has an {a.font} prototype and skeleton:', '',
         '```', f'python run_test.py --samples_df_path {a.samples} --font_dir prototypes/{a.font} --con_dir skeletons/{a.font} --output_folder {os.path.relpath(a.run, "output")} --ignore_errors', '```', '',
         f'At the time of writing {len(done)} of {len(meta)} listed signs had finished. The **initialisation score** quoted below is ProtoSnap\'s own agreement measure for the initial placement (0 to 1, higher is better); over the finished results its median is {statistics.median(scores):.2f} (range {min(scores):.2f} to {max(scores):.2f}). It is not an accuracy: there is no ground truth for these alignments, so the panels have to be judged by eye.', '',
         '## How to read a panel', '',
         'Each panel is ProtoSnap\'s `itr99_viz.png`. Top row: the photographed sign, the font prototype, and the prototype with its skeleton (one colour per wedge). Middle row: the skeleton drawn on the photograph at initialisation, after the global transform, and after the per-stroke refinement. Bottom row: the same three stages over the diffusion-feature similarity map that drives the alignment. A good result has every coloured stroke lying along a wedge in the right-hand column.', '',
         '## The sample', '', '| # | result | sign | Unicode | tablet | init score |', '|---|---|---|---|---|---|']
    for k, rid in enumerate(pick, 1):
        m = meta.get(rid, {}); tablet = rid.rsplit('_', 1)[0]
        L.append(f"| {k} | `{rid}` | {m.get('name', '?')} | {m.get('sign', '')} `{m.get('hex', '')}` | {tablet} | {done[rid]['score']:.2f} |")
    L.append('')
    for k, rid in enumerate(pick, 1):
        m = meta.get(rid, {}); im = cv2.imread(os.path.join(done[rid]['dir'], 'itr99_viz.png')); dst = os.path.join(img_dir, rid.replace(',', '_') + '.jpg'); cv2.imwrite(dst, im, [cv2.IMWRITE_JPEG_QUALITY, 88])
        L += [f"### {k}. {m.get('name', '?')} {m.get('sign', '')} on {rid.rsplit('_', 1)[0]}", '', f"![ProtoSnap result {rid}](img/{a.font.lower()}/{os.path.basename(dst)})", '',
              f"*`{rid}`: sign {m.get('name', '?')} (`{m.get('hex', '')}`), initialisation score {done[rid]['score']:.2f}.*", '']
    L += ['## Sources', '',
          '1. onetwothreefourfivesixe, *ProtoSnapVectorMapper*: automatic generation of ProtoSnap prototype skeletons from cuneiform font glyphs, with a review editor. https://github.com/onetwothreefourfivesixe/ProtoSnapVectorMapper',
          '2. Rachel Mikulinsky, Morris Alper, Shai Gordin, Enrique Jiménez, Yoram Cohen, Hadar Averbuch-Elor, *ProtoSnap: Prototype Alignment for Cuneiform Signs*, ICLR 2025. https://arxiv.org/abs/2502.00129',
          '3. Or Lewenstein et al., *A Large-Scale Dataset of Annotated Cuneiform Sign Images for Digital Palaeography*, Journal of Open Humanities Data, 2026 (the sign photographs). https://doi.org/10.5334/johd.503', '',
          f'Regenerate this page with `python experiments/esagil_skeletons/sampleResults.py --seed {a.seed}' + (f' --include {" ".join(a.include)}' if a.include else '') + '`.', '']
    open(a.out, 'w', encoding='utf-8').write('\n'.join(L)); print(f'{len(done)} finished results; wrote {a.out} with {len(pick)} samples:', pick)


if __name__ == '__main__':
    main()
