"""Do near-miss GT rows become reachable when two overlapping candidates merge?

The organizer has said, verbatim, that the judging committee will look at the
distribution of many-to-many relations, and our own FN decomposition puts 36% of
the never-proposed consensus rows at best-candidate IoU 0.3-0.5 -- the near-miss
band. One mechanism that produces exactly that band is FRAGMENTATION: the
pipeline emits a long filament as two candidates, each matching the GT at 0.3-
0.4, while their union would clear 0.5.

This measures the precondition before any merge machinery is built (the
MISTAKES.md rule): for every fold-0 GT row, the best single-candidate IoU versus
the best 2-candidate-union IoU over candidates that each overlap the GT. If the
union rescues only a handful, fragmentation is not the mechanism behind the
near-miss band and no merge stage is worth building; if it rescues many, merged
pairs become a new candidate source for the selector.

Usage: python src/eda/pair_union_recovery.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament.rle import decode_rle

OUT = ROOT / "outputs" / "eda"


def log(*a):
    print(*a, flush=True)


def main():
    cand = pd.read_parquet(ROOT / "outputs" / "selector" / "fused_pool_all.parquet")
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    f0 = {f.rsplit(".", 1)[0] for f in folds.loc[folds["fold"] == 0, "file_name"]}
    cand = cand[cand["stem"].isin(f0)]
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.rsplit(".", n=1).str[0]
    gt = gt[gt["stem"].isin(f0)]
    log(f"fold 0: {len(cand)} candidates, {len(gt)} GT rows")

    rows = []
    for stem, g in gt.groupby("stem"):
        cs = cand[cand["stem"] == stem]
        if not len(cs):
            continue
        cmasks = [decode_rle(r).astype(bool) for r in cs["rle"]]
        careas = np.array([m.sum() for m in cmasks])
        for _, r in g.iterrows():
            gm = decode_rle(r["segmentation_rle"]).astype(bool)
            ga = gm.sum()
            inters = np.array([(m & gm).sum() for m in cmasks])
            ious = inters / (careas + ga - inters)
            best1 = float(ious.max()) if len(ious) else 0.0
            best2 = best1
            # only pairs where both members genuinely overlap the GT
            idx = np.argsort(ious)[::-1][:6]
            idx = [i for i in idx if inters[i] > 0]
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    u = cmasks[idx[a]] | cmasks[idx[b]]
                    iou = (u & gm).sum() / (u.sum() + ga - (u & gm).sum())
                    best2 = max(best2, float(iou))
            rows.append(dict(filament_id=r["filament_id"], stem=stem,
                             best1=best1, best2=best2))
    df = pd.DataFrame(rows)

    near = df[(df["best1"] > 0.3) & (df["best1"] <= 0.5)]
    rescued = near[near["best2"] > 0.5]
    below = df[df["best1"] <= 0.3]
    rescued_lo = below[below["best2"] > 0.5]
    log(f"\nGT rows: {len(df)}")
    log(f"  already reached (best1 > 0.5):      {int((df['best1'] > 0.5).sum())}")
    log(f"  near-miss (0.3 < best1 <= 0.5):     {len(near)}")
    log(f"    -> RESCUED by a 2-union:          {len(rescued)} "
        f"({100*len(rescued)/max(len(near),1):.0f}% of near-misses)")
    log(f"  far (best1 <= 0.3):                 {len(below)}")
    log(f"    -> rescued by a 2-union:          {len(rescued_lo)}")
    tot = len(rescued) + len(rescued_lo)
    log(f"\ntotal newly reachable via pair-merge on fold 0: {tot} "
        f"(x5 folds ~ {5*tot}; each worth its IoU into a ~7000 denominator)")
    log("reading: a large rescue count makes merged pairs a candidate source "
        "worth adding; a small one closes the fragmentation hypothesis.")
    df.to_parquet(OUT / "pair_union_recovery.parquet", index=False)
    log(f"\nsaved -> {OUT/'pair_union_recovery.parquet'}")


if __name__ == "__main__":
    main()
