"""Does a cached lane reach the ground truth nothing in our pool ever reached?

Generalises `m2f_miss_profile.py`, which asked this of the query lane by running
the model inline. Every lane we consider from here already has to be cached in
the oof_cache schema before the fusion harness can consume it, so the reach
question is better asked of the cache than of the weights: it costs no GPU, and
the same script then serves every future source.

The decision this gates. Absolute PQ is the wrong bar for a lane that is being
proposed as a SOURCE rather than as a replacement -- `eval_tiled.py` made that
mistake, requiring the tiled lane to beat our oracle before the pivot counted.
A source earns its place by reaching ground truth the existing sources miss,
because that is the only property that has ever added at the pipeline level
here. Fold-0's never-proposed rows are the population that settles it: ground
truth for which no existing candidate reaches IoU > 0.5 at any confidence, so
the pool cannot recover them no matter how the selector is tuned.

Reference points from the same measurement, for reading the number: seeded
multi-threshold extraction reached 3 of 440 (0.7%) and went on to lose 0.0161 at
the pipeline level; pair-unions reached 57; the M2F query lane reached 121
(27.5%) and went on to add +0.0025. So reach is necessary and clearly not
sufficient, and a lane that reaches fewer than about 50 is not worth the harness
run.

Usage:
  python src/eda/lane_miss_profile.py --preds outputs/oof_cache/fold0_preds_tiled.parquet
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament.rle import decode_rle

OUT = ROOT / "outputs" / "eda"


def log(*a):
    print(*a, flush=True)


def boxed(m):
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    return (y0, y1, x0, x1), m[y0:y1, x0:x1], int(m.sum())


def iou(a, b):
    (ay0, ay1, ax0, ax1), asub, asum = a
    (by0, by1, bx0, bx1), bsub, bsum = b
    y0, y1 = max(ay0, by0), min(ay1, by1)
    x0, x1 = max(ax0, bx0), min(ax1, bx1)
    if y0 >= y1 or x0 >= x1:
        return 0.0
    i = int(np.logical_and(asub[y0 - ay0:y1 - ay0, x0 - ax0:x1 - ax0],
                           bsub[y0 - by0:y1 - by0, x0 - bx0:x1 - bx0]).sum())
    u = asum + bsum - i
    return i / u if u else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True,
                    help="cached lane predictions in the oof_cache schema")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.0,
                    help="loose gate: reach is a ceiling question, not a policy")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    tag = a.tag or Path(a.preds).stem

    pred = pd.read_parquet(ROOT / a.preds if not Path(a.preds).is_absolute()
                           else a.preds)
    pred = pred[pred["conf"] >= a.conf]
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    f0 = set(folds.loc[folds["fold"] == a.fold, "file_name"]
             .str.replace(".jpeg", "", regex=False))

    # The gate population is recomputed against the pool we actually ship from.
    # It used to be read straight out of error_profile_fn_gt.parquet as
    # ~matched & ~in_raw_pool, and that population is STALE: the profile was
    # built against an earlier, smaller pool. Measured on fold 0, the stale gate
    # and the correct one overlap on only 335 rows -- 105 of the stale 440 ARE
    # reached by today's pool (up to IoU 0.858) and 79 genuinely unreachable rows
    # were invisible to it. A lane could therefore score on the gate by
    # re-finding candidates the pool already has, which is worth nothing.
    tu = OUT / "true_unreachable.parquet"
    if tu.exists():
        t = pd.read_parquet(tu)
        keep = set(t.loc[t["best"] <= 0.5, "filament_id"])
        d = pd.read_parquet(OUT / "error_profile_fn_gt.parquet")
        never = d[d["stem"].isin(f0) & d["filament_id"].isin(keep)]
        log("gate population: recomputed against the shipped pool "
            "(src/eda/true_unreachable.py)")
    else:
        raise SystemExit(
            "run src/eda/true_unreachable.py first -- the old "
            "~matched & ~in_raw_pool population is stale and overstates every "
            "lane's reach")
    log(f"{tag}: {len(pred)} candidates over {pred['stem'].nunique()} images")
    log(f"fold-{a.fold} never-proposed GT rows: {len(never)}")

    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt = gt.set_index("filament_id")["segmentation_rle"]
    by_stem = {s: g for s, g in pred.groupby("stem")}

    rows, t0 = [], time.time()
    for k, (stem, g) in enumerate(never.groupby("stem")):
        p = by_stem.get(stem)
        if p is None or not len(p):
            for r in g.itertuples():
                rows.append(dict(filament_id=r.filament_id, stem=stem, best=0.0))
            continue
        pm = [boxed(decode_rle(x).astype(bool)) for x in p["rle"]]
        pm = [x for x in pm if x is not None]
        for r in g.itertuples():
            gm = boxed(decode_rle(gt.loc[r.filament_id]).astype(bool))
            best = max((iou(gm, x) for x in pm), default=0.0) if gm else 0.0
            rows.append(dict(filament_id=r.filament_id, stem=stem, best=best))
        if (k + 1) % 30 == 0:
            log(f"  {k+1} stems ({time.time()-t0:.0f}s)")

    rf = pd.DataFrame(rows)
    rf.to_parquet(OUT / f"missprofile_{tag}.parquet", index=False)
    reach = int((rf["best"] > 0.5).sum())
    log(f"\nof {len(rf)} never-proposed rows, {tag} (conf>={a.conf}) reaches:")
    for t in (0.3, 0.4, 0.5, 0.6):
        n = int((rf["best"] > t).sum())
        log(f"  IoU > {t}: {n:4d} ({100*n/max(len(rf),1):.1f}%)")
    log(f"  zero overlap at all: {int((rf['best'] == 0).sum())}")
    log("\ncontext: seeded multithresh reached 3 of 440 (0.7%) and lost 0.0161 at")
    log("the pipeline level; pair-unions 57; the M2F query lane 121 (27.5%) and")
    log("added +0.0025. Below about 50 the harness run is not worth its cost.")

    json.dump(dict(lane=tag, n_never=len(rf), reach_050=reach,
                   reach_frac=float(reach / max(len(rf), 1)),
                   n_pred=int(len(pred))),
              open(OUT / f"missprofile_{tag}.json", "w"), indent=1)
    log(f"\nwrote {OUT/f'missprofile_{tag}.json'}")


if __name__ == "__main__":
    main()
