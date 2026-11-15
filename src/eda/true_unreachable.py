"""The precondition gate has been measured against a pool that no longer exists.

Since `m2f_miss_profile.py`, every candidate lane has been judged on fold-0's 440
"never-proposed" ground-truth rows, taken from `error_profile_fn_gt.parquet` as
`~matched & ~in_raw_pool`. Two checks show that population is stale:

  * the profile's own `pool_iou` column is NaN for all 440, i.e. it recorded no
    pool candidate reaching any of them at IoU > 0.5;
  * scanning today's `fused_pool_all.parquet` finds a candidate above 0.5 for
    104 of the 174 that a raw source reaches, with IoUs up to 0.858.

Both cannot be true of the same pool. The profile was built against an earlier,
smaller pool, and the gate has been inherited unchanged through every lane
decision since.

This recomputes the population from scratch against the pool we actually ship
from, and reports what the gate should have said. It does not re-litigate any
lane verdict on its own -- the realised cross-fitted numbers stand regardless --
but the REASONS recorded for those verdicts depend on this population, and so
does every future lane decision.

Usage: python src/eda/true_unreachable.py
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
    return ((int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1),
            m[ys.min():ys.max() + 1, xs.min():xs.max() + 1], int(m.sum()))


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
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    f0 = set(folds.loc[folds["fold"] == 0, "file_name"]
             .str.replace(".jpeg", "", regex=False))
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt = gt[gt["stem"].isin(f0)]
    pool = pd.read_parquet(SEL / "fused_pool_all.parquet")
    pool = pool[pool["stem"].isin(f0)]
    log(f"fold 0: {len(gt)} ground-truth rows, {len(pool)} pool candidates")

    by = {s: list(g["rle"]) for s, g in pool.groupby("stem")}
    rows, t0 = [], time.time()
    for k, (stem, g) in enumerate(gt.groupby("stem")):
        cands = [boxed(decode_rle(r).astype(bool)) for r in by.get(stem, [])]
        cands = [c for c in cands if c is not None]
        for r in g.itertuples():
            gm = boxed(decode_rle(r.segmentation_rle).astype(bool))
            rows.append(dict(filament_id=r.filament_id, stem=stem,
                             best=(max((iou(gm, c) for c in cands), default=0.0)
                                   if gm else 0.0)))
        if (k + 1) % 40 == 0:
            log(f"  {k+1} stems ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "true_unreachable.parquet", index=False)
    reach = (df["best"] > 0.5).sum()
    log(f"\nTRUE fold-0 pool reachability: {reach}/{len(df)} "
        f"({100*reach/len(df):.1f}%)")
    log(f"  genuinely unreachable rows: {len(df) - reach}")

    old = pd.read_parquet(OUT / "error_profile_fn_gt.parquet")
    old["matched"] = old["matched"].astype(bool)
    old["in_raw_pool"] = old["in_raw_pool"].astype(bool)
    old = old[old["stem"].isin(f0)]
    stale = set(old.loc[~old["matched"] & ~old["in_raw_pool"], "filament_id"])
    true_un = set(df.loc[df["best"] <= 0.5, "filament_id"])
    log(f"\nthe gate used since m2f_miss_profile: {len(stale)} rows")
    log(f"the population it should have been  : {len(true_un)} rows")
    log(f"  in both                           : {len(stale & true_un)}")
    log(f"  stale-only (pool DOES reach them) : {len(stale - true_un)}")
    log(f"  missed by the stale gate          : {len(true_un - stale)}")

    log("\nwhat the lane reach numbers become on the correct population:")
    for name, path, col in (
            ("raw e013 detector", "missprofile_raw_e013.parquet", "best"),
            ("tiled lane", "missprofile_fold0_preds_tiled.parquet", "best"),
            ("M2F lane", "m2f_miss_profile.parquet", "iou")):
        p = OUT / path
        if not p.exists():
            continue
        s = pd.read_parquet(p).set_index("filament_id")[col]
        on_stale = int((s.reindex(sorted(stale)).fillna(0) > 0.5).sum())
        on_true = int((s.reindex(sorted(true_un)).fillna(0) > 0.5).sum())
        cov = s.reindex(sorted(true_un)).notna().mean()
        log(f"  {name:20s} stale gate {on_stale:4d}   correct population "
            f"{on_true:4d}   (measured for {100*cov:.0f}% of it)")
    log("  note: the lane profiles were only ever COMPUTED on the stale 440, so")
    log("  the right-hand column is a lower bound, not a re-measurement.")

    json.dump(dict(n_gt=len(df), reachable=int(reach),
                   unreachable=int(len(df) - reach),
                   stale_gate=len(stale), correct_population=len(true_un),
                   overlap=len(stale & true_un),
                   stale_only=len(stale - true_un),
                   missed_by_stale=len(true_un - stale)),
              open(OUT / "true_unreachable.json", "w"), indent=1)
    log(f"\nwrote {OUT/'true_unreachable.json'}")


if __name__ == "__main__":
    main()
