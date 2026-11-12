"""Build per-script sign-classification datasets from data/ebl/signs_snippets_metadata.json (the eBL sign-crop release).

  python -m line_cropping.signs.build_dataset [--scripts NA OB] [--min-train 20]

The metadata is a classification set: one cropped sign per entry with its sign name and script, no position on the
tablet. It therefore cannot train the detector; it trains a sign classifier used to score detector boxes against
the transliteration's expected sign (see train_classifier.py and protosnap/protosnap_prep.py --classifier).

Split keeps everything evaluated downstream out of training:
  test  = snippets on faces in the Phase 5 test split (tablet_data/ebl_tablets/split.csv); a snippet's face comes from its eBL
          line index via the Phase 1 line table; snippets with no line index on a tablet that has a test face
  val   = the remaining snippets of 10% of tablets (seeded, by tablet)
  train = the rest
Classes are the signs with at least --min-train TRAINING samples; rarer signs are dropped from all splits.
Writes tablet_data/ebl_tablets/sign_classifier/<script>/{train,val,test}.csv (id,path,sign,label,tablet,type), classes.json, stats.json.
"""
from line_cropping.paths import EBL
import argparse, csv, json, os, random, collections

KEEP_TYPES = {'MANUALLY_ANNOTATED_SIGN', 'OCR', 'MANUALLY_ANNOTATED_SIGN_DAMAGED', 'PARTIALLY_BROKEN'}
OUT = f'{EBL}/sign_classifier'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--scripts', nargs='+', default=['NA', 'OB']); ap.add_argument('--min-train', type=int, default=20)
    ap.add_argument('--val-frac', type=float, default=0.1); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    meta = json.load(open('data/ebl/signs_snippets_metadata.json'))
    have = set(os.listdir('data/ebl/singleSignImages'))
    # Phase 5 split is per face. A snippet's label is its eBL path, whose first element is the text-line index, and
    # Phase 1 knows which face each line index belongs to, so hold-out can be decided per face: a snippet is held
    # out if its face is a test face, or if its face cannot be determined and its tablet has any test face.
    face_split = {r['face']: r['split'] for r in csv.DictReader(open(f'{EBL}/split.csv'))} if os.path.exists(f'{EBL}/split.csv') else {}
    held = {f.split('__')[0] for f, sp in face_split.items() if sp == 'test'}
    line_face = {}
    for tab in {f.split('__')[0] for f in face_split}:
        lp = f'{EBL}/ground_truth/lines/{tab}.csv'
        if not os.path.exists(lp): continue
        for l in csv.DictReader(open(lp)):
            face = f"{tab}__{l['side']}" + (f"__{l['object']}" if l['object'] else '')
            line_face[(tab, int(l['line_index']))] = face
    def snippet_held(x):
        tab = x['tabletNumber']
        if tab not in held: return False
        try: li = int(json.loads(x['label'])[0])
        except Exception: return True
        face = line_face.get((tab, li))
        return face_split.get(face, 'test') == 'test'
    for sc in a.scripts:
        rows = [x for x in meta if x['script'] == sc and x['type'] in KEEP_TYPES and x['signName'] and x['signName'] != 'UnclearSign' and (x['_id'] + '.jpeg') in have]
        tablets = sorted({x['tabletNumber'] for x in rows}); random.seed(a.seed)
        shuffled = tablets[:]; random.shuffle(shuffled); nval = max(1, int(a.val_frac * len(shuffled))); val_tabs = set(shuffled[:nval])
        for x in rows: x['_part'] = 'test' if snippet_held(x) else ('val' if x['tabletNumber'] in val_tabs else 'train')
        part = None
        cnt = collections.Counter(x['signName'] for x in rows if x['_part'] == 'train')
        classes = sorted(s for s, n in cnt.items() if n >= a.min_train); idx = {s: i for i, s in enumerate(classes)}
        d = os.path.join(OUT, sc); os.makedirs(d, exist_ok=True); stats = {}
        for split in ('train', 'val', 'test'):
            R = [x for x in rows if x['_part'] == split and x['signName'] in idx]
            with open(os.path.join(d, split + '.csv'), 'w', newline='', encoding='utf-8') as fh:
                w = csv.writer(fh); w.writerow(['id', 'path', 'sign', 'label', 'tablet', 'type'])
                for x in R: w.writerow([x['_id'], f"data/ebl/singleSignImages/{x['_id']}.jpeg", x['signName'], idx[x['signName']], x['tabletNumber'], x['type']])
            stats[split] = dict(samples=len(R), tablets=len({x['tabletNumber'] for x in R}), dropped_rare=sum(1 for x in rows if x['_part'] == split and x['signName'] not in idx))
        json.dump(classes, open(os.path.join(d, 'classes.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
        stats.update(script=sc, n_classes=len(classes), min_train=a.min_train, tablets_with_a_test_face=len([t for t in tablets if t in held]), all_tablets=len(tablets),
                     train_per_class_median=int(sorted(cnt[s] for s in classes)[len(classes) // 2]) if classes else 0)
        json.dump(stats, open(os.path.join(d, 'stats.json'), 'w'), indent=1)
        print(f"{sc}: {len(classes)} classes (>= {a.min_train} train samples) | train {stats['train']['samples']} ({stats['train']['tablets']} tablets) | val {stats['val']['samples']} | test {stats['test']['samples']} ({stats['test']['tablets']} held-out tablets) | rare signs dropped from train: {stats['train']['dropped_rare']}")


if __name__ == '__main__':
    main()
