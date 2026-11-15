"""E142 -- what is left in the pool, and is the selector or the candidate set the cap?

E140 closed the ensembling family with a law: averaging k semantic members pays
A*(1 - 1/k) with A = +0.0092 and R^2 = 0.995. The deployed field already has
twelve members, so 92% of that is banked and at most +0.0008 remains. That
removes the whole "add members" direction and forces the question of where the
remaining headroom actually is.

Two ceilings bracket it:

  selector ceiling   keep exactly the pool candidates that match a ground-truth
                     row at IoU > 0.5, de-overlapped greedily by IoU. This is
                     what a PERFECT selector would score on TODAY's candidates.
                     The gap to the deployed number is what better selection can
                     still buy.

  pool ceiling       the same, but crediting every ground-truth row the pool can
                     reach at all. The gap from here to 1.0 is what only better
                     candidates (detector, semantic, resolution) can buy.

Reported per fold and pooled, disc and limb separately, so the next GPU hour is
aimed by measurement rather than by intuition.

Usage: python src/eda/pool_ceiling.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/eda", "src/models"):
    sys.path.insert(0, str(ROOT / p))
from filament.pq import score  # noqa: E402

OUT = ROOT / "outputs" / "eda"
GATE = (0.40, 0.60)


def log(*a):
    print(*a, flush=True)


def main():
    a = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    a = a.merge(pd.read_parquet(OUT / "t020pox_oof_scores.parquet"), on="cand_id", how="left")
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    a["fold"] = a["stem"].map(fold_of).astype(int)
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    g = gt[gt["stem"].isin(set(a["stem"]))]

    def sc(df):
        return score(g[["filament_id", "segmentation_rle"]],
                     df[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"}))

    deployed = a[(a["conf"] >= GATE[0]) & (a["p"] >= GATE[1])]
    # a perfect selector keeps every true candidate; where several cover the same
    # ground-truth row, the greedy de-overlap the pipeline already uses picks the best
    oracle = (a[a["y"] == 1].sort_values("iou_sum", ascending=False)
              .drop_duplicates(subset=["stem", "gt_key"]) if "gt_key" in a.columns
              else a[a["y"] == 1].sort_values("iou_sum", ascending=False))
    rows = []
    for name, df in (("deployed", deployed), ("perfect selector, same pool", oracle)):
        r = sc(df)
        rows.append((name, len(df), r["pq"], r["sq"], r["rq"], r["tp"], r["fp"], r["fn"]))
        log(f"{name:28s} n {len(df):5d}  PQ {r['pq']:.4f}  SQ {r['sq']:.4f}  RQ {r['rq']:.4f}  "
            f"TP {r['tp']:5d} FP {r['fp']:5d} FN {r['fn']:5d}")
    gap = rows[1][2] - rows[0][2]
    log(f"\nheadroom inside today's candidate set: {gap:+.4f} PQ")
    log(f"  the selector currently converts {rows[0][2] / rows[1][2]:.1%} of what the pool offers")

    # where is that headroom, by radius and by size
    log("\nheadroom by radial band (perfect selector minus deployed, same pool):")
    for lo, hi, nm in [(0.0, 0.8, "disc"), (0.8, 0.9, "limb"), (0.9, 1.01, "far limb")]:
        m = (a["r_norm"] >= lo) & (a["r_norm"] < hi)
        d_n = int(((a["conf"] >= GATE[0]) & (a["p"] >= GATE[1]) & m).sum())
        o_n = int((m & (a["y"] == 1)).sum())
        miss = int((m & (a["y"] == 1) & ~((a["conf"] >= GATE[0]) & (a["p"] >= GATE[1]))).sum())
        log(f"  {nm:9s} deployed {d_n:5d}  true in pool {o_n:5d}  true but dropped {miss:5d} "
            f"({miss / max(o_n, 1):5.1%})")
    log("\nheadroom by candidate size (log10 area quintile):")
    q = pd.qcut(a["log10_area"], 5, labels=False, duplicates="drop")
    for i in range(5):
        m = q == i
        o_n = int((m & (a["y"] == 1)).sum())
        miss = int((m & (a["y"] == 1) & ~((a["conf"] >= GATE[0]) & (a["p"] >= GATE[1]))).sum())
        log(f"  quintile {i} (area {10 ** a.loc[m, 'log10_area'].min():7.0f}-{10 ** a.loc[m, 'log10_area'].max():8.0f} px)"
            f"  true in pool {o_n:5d}  dropped {miss:5d} ({miss / max(o_n, 1):5.1%})")
    json.dump(dict(deployed=rows[0][2], perfect_selector=rows[1][2], headroom=float(gap),
                   converted=float(rows[0][2] / rows[1][2])),
              open(OUT / "pool_ceiling.json", "w"), indent=1, default=float)
    log(f"\nwrote {OUT / 'pool_ceiling.json'}")


if __name__ == "__main__":
    main()
