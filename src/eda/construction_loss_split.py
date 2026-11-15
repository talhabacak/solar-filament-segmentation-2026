"""Which construction step destroys the recoverable rows -- fusion or de-overlap?

The finding this follows up, on the CORRECTED population. Fold-0's true
pool-unreachable set is 414 rows (recomputed against the shipped pool by
src/eda/true_unreachable.py; the 440 figure inherited from
error_profile_fn_gt.parquet was stale and wrong on about 42% of its members).
Of those 414, the raw e013 detector -- our own primary source, before fusion --
reaches 147 at IoU > 0.5. They are lost between its output and the constructed
pool. That is more than three times the best new lane's real increment (M2F 44,
tiled 22 beyond our own sources) and it is not a reach problem: the candidates
exist and something in `build_fused_pool` destroys them.

Two steps can do it, and they call for completely different remedies:

  clustering + fusion   union-find joins candidates at IoU > 0.5 and averages
                        their masks by confidence. Averaging two masks that
                        disagree yields one that matches NEITHER, and no
                        downstream stage can undo it -- the damage is baked into
                        the mask before any selector sees it.
  greedy de-overlap     pixels are carved from the lower-ranked candidate so the
                        submission stays pixel-disjoint. `build_fused_pool.py`
                        already records this at 74 fold-0 rows (conf-vote fusion
                        reaches 1170/1510 without it against 1096 with it).

The split decides what to attack, and it is free to measure because the
de-overlap-free pool is already on disk. A row reachable by a raw source that
SURVIVES in `fused_pool_nodeov` but is missing from `fused_pool_all` was killed
by de-overlap; a row missing from both was killed by clustering and fusion.

Note this is a different question from the one Experiment A asked. That moved
de-overlap from build time to selection time and returned +0.0022 in-sample
(-0.0011 cross-fitted). It changed WHEN pixels are carved; it never touched
fusion averaging a mask into something that matches nothing.

Usage: python src/eda/construction_loss_split.py
"""

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
SEL = ROOT / "outputs" / "selector"


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


def best_by_stem(pool, gt_rows, gt_rle, label):
    """Best IoU each ground-truth row achieves against any candidate of a pool."""
    by = {s: list(g["rle"]) for s, g in pool.groupby("stem")}
    out, t0 = {}, time.time()
    for stem, g in gt_rows.groupby("stem"):
        cands = [boxed(decode_rle(r).astype(bool)) for r in by.get(stem, [])]
        cands = [c for c in cands if c is not None]
        for r in g.itertuples():
            gm = boxed(decode_rle(gt_rle.loc[r.filament_id]).astype(bool))
            out[r.filament_id] = (max((iou(gm, c) for c in cands), default=0.0)
                                  if gm else 0.0)
    log(f"  {label}: {len(out)} rows scored ({time.time()-t0:.0f}s)")
    return pd.Series(out)


def main():
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    f0 = set(folds.loc[folds["fold"] == 0, "file_name"]
             .str.replace(".jpeg", "", regex=False))
    # population recomputed against the shipped pool: the profile's
    # ~matched & ~in_raw_pool set is stale (see src/eda/true_unreachable.py --
    # 105 of its 440 rows ARE reached by today's pool, and it misses 79 that are
    # genuinely unreachable)
    t = pd.read_parquet(OUT / "true_unreachable.parquet")
    keep = set(t.loc[t["best"] <= 0.5, "filament_id"])
    d = pd.read_parquet(OUT / "error_profile_fn_gt.parquet")
    never = d[d["stem"].isin(f0) & d["filament_id"].isin(keep)]
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt_rle = gt.set_index("filament_id")["segmentation_rle"]

    # the population: rows a raw source of ours already reaches
    raw = pd.read_parquet(OUT / "missprofile_true_raw_e013.parquet").set_index(
        "filament_id")["best"]
    reach = pd.DataFrame({"raw_det": raw}).reindex(
        never["filament_id"]).fillna(0.0)
    recoverable = reach[reach["raw_det"] > 0.5]
    log(f"fold-0 never-proposed rows: {len(never)}")
    log(f"  of those, reachable by a raw source we already have: "
        f"{len(recoverable)}\n")

    sub = never[never["filament_id"].isin(recoverable.index)]
    nodeov = pd.read_parquet(SEL / "fused_pool_nodeov.parquet")
    nodeov = nodeov[nodeov["stem"].isin(f0)]
    full = pd.read_parquet(SEL / "fused_pool_all.parquet")
    full = full[full["stem"].isin(f0)]
    log(f"pools on fold 0: fused_pool_all {len(full)} candidates, "
        f"fused_pool_nodeov {len(nodeov)}")

    b_nodeov = best_by_stem(nodeov, sub, gt_rle, "no-deoverlap pool")
    b_full = best_by_stem(full, sub, gt_rle, "production pool")

    j = pd.DataFrame({"nodeov": b_nodeov, "full": b_full}).fillna(0.0)
    survives_nodeov = j["nodeov"] > 0.5
    survives_full = j["full"] > 0.5

    killed_by_deov = int((survives_nodeov & ~survives_full).sum())
    killed_by_fusion = int((~survives_nodeov).sum())
    alive_in_full = int(survives_full.sum())

    log(f"\nof the {len(j)} rows a raw source reaches:")
    log(f"  still reachable in the PRODUCTION pool       : {alive_in_full:4d}")
    log(f"  survive de-overlap-free build, die WITH it   : {killed_by_deov:4d}"
        f"   <- greedy de-overlap")
    log(f"  already gone before de-overlap               : {killed_by_fusion:4d}"
        f"   <- clustering + confidence-weighted fusion")
    log(f"\n  build_fused_pool.py records de-overlap at 74 fold-0 rows on the")
    log(f"  whole GT set; this counts only the recoverable subset, so the two")
    log(f"  numbers are not the same quantity.")

    if killed_by_fusion:
        f = j[~survives_nodeov]
        log(f"\n  rows killed by fusion -- how close does the fused pool get?")
        for q in (0.5, 0.75, 0.9):
            log(f"    best IoU in the no-deov pool, p{int(q*100)}: "
                f"{f['nodeov'].quantile(q):.3f}")
        log(f"    fraction with NO overlapping candidate at all: "
            f"{100*(f['nodeov'] == 0).mean():.1f}%")
        log("    a high near-0.5 mass means fusion is degrading a good mask;")
        log("    a mass near zero means clustering never grouped it at all.")

    json.dump(dict(n_never=len(never), n_recoverable=len(j),
                   alive_in_production=alive_in_full,
                   killed_by_deoverlap=killed_by_deov,
                   killed_by_fusion=killed_by_fusion),
              open(OUT / "construction_loss_split.json", "w"), indent=1)
    j.to_parquet(OUT / "construction_loss_split.parquet")
    log(f"\nwrote {OUT/'construction_loss_split.json'}")


if __name__ == "__main__":
    main()
