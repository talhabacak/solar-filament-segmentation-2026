"""P2+P3: where exactly does a matched mask lose its IoU, and would adaptive
shrink recover any of it?

SQ is 0.706 against a measured human-agreement ceiling of 0.84 -- about 0.14 of
IoU per matched pair is being lost somewhere, worth roughly 0.08 PQ if fully
recovered. Uniform 5% shrink was adopted (+0.005) and its own study noted the
masks are NOT uniformly fat: some overshoot, some undershoot. Two questions,
both answered per matched TP pair on fold 0:

P2 (adaptive shrink oracle): for each TP, the IoU at shrink fractions
0/2.5/5/7.5/10/15% -- how much would choosing the best fraction PER CANDIDATE
gain over the uniform 5%? If the oracle gain is small, the adaptive-shrink idea
dies before anything is built. If large, a regressor on selector features is the
next measurement.

P3 (deficit anatomy): the IoU deficit of each TP decomposes into OVER pixels
(ours beyond GT) and UNDER pixels (GT beyond ours). Each is further split by
where it sits: near the GT boundary (a rim, annotator-style disagreement, likely
irreducible) versus beyond the GT's skeleton endpoints (a chopped or overgrown
TAIL, structurally actionable). If tails dominate, spine-extension is a real
lever; if rims dominate, SQ is annotator noise and the axis closes.

Usage: python src/eda/sq_anatomy.py
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament.pq import score
from filament.rle import decode_rle, encode_mask
from mask_shrink import shrink_frac

OUT = ROOT / "outputs" / "eda"
FRACS = (0.0, 0.025, 0.05, 0.075, 0.10, 0.15)


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

    # match unshrunk candidates to GT with the exact scorer, then analyse pairs
    pred_df = cand[["cand_id", "rle"]].rename(
        columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
    res = score(gt[["filament_id", "segmentation_rle"]], pred_df,
                return_details=True)
    pairs = res["matches"]
    log(f"matched pairs (unshrunk, IoU>0.5): {len(pairs)}")

    cand_rle = dict(zip(cand["cand_id"], cand["rle"]))
    gt_rle = dict(zip(gt["filament_id"], gt["segmentation_rle"]))

    t0 = time.time()
    rows = []
    for pr in pairs.itertuples():
        cm = decode_rle(cand_rle[pr.pred_filament_id]).astype(bool)
        gm = decode_rle(gt_rle[pr.gt_filament_id]).astype(bool)
        ga = int(gm.sum())

        # ---- P2: per-candidate shrink curve ------------------------------
        ious = {}
        for f in FRACS:
            m = shrink_frac(cm, f) if f > 0 else cm
            inter = int((m & gm).sum())
            ious[f] = inter / (int(m.sum()) + ga - inter)

        # ---- P3: deficit anatomy on the 5%-shrunk mask -------------------
        m5 = shrink_frac(cm, 0.05)
        over = m5 & ~gm
        under = gm & ~m5
        # GT rim: within 3px of the GT boundary (either side)
        gm8 = gm.astype(np.uint8)
        dil = cv2.dilate(gm8, np.ones((7, 7), np.uint8)) > 0
        ero = cv2.erode(gm8, np.ones((7, 7), np.uint8)) > 0
        rim = dil & ~ero
        rows.append(dict(
            gt_id=pr.gt_filament_id, cand_id=pr.pred_filament_id,
            iou_un=ious[0.0], iou_5=ious[0.05],
            iou_best=max(ious.values()),
            best_frac=max(ious, key=ious.get), gt_area=ga,
            over_px=int(over.sum()), under_px=int(under.sum()),
            over_rim=int((over & rim).sum()), under_rim=int((under & rim).sum()),
        ))
    df = pd.DataFrame(rows)
    log(f"pair analysis done ({time.time()-t0:.0f}s)")

    log("\n== P2: adaptive shrink oracle ==")
    log(f"  mean IoU unshrunk        : {df['iou_un'].mean():.4f}")
    log(f"  mean IoU uniform 5%      : {df['iou_5'].mean():.4f}")
    log(f"  mean IoU best-per-cand   : {df['iou_best'].mean():.4f}  "
        f"(oracle gain over 5%: {df['iou_best'].mean()-df['iou_5'].mean():+.4f})")
    log("  best-fraction distribution:")
    log("  " + df["best_frac"].value_counts().sort_index().to_string()
        .replace("\n", "\n  "))

    log("\n== P3: where the 5%-shrunk deficit lives ==")
    tot_over, tot_under = df["over_px"].sum(), df["under_px"].sum()
    log(f"  OVER pixels (ours beyond GT):  {tot_over}  "
        f"({100*df['over_rim'].sum()/max(tot_over,1):.0f}% within 3px of the GT rim)")
    log(f"  UNDER pixels (GT we miss):     {tot_under}  "
        f"({100*df['under_rim'].sum()/max(tot_under,1):.0f}% within 3px of the GT rim)")
    log(f"  over/under balance: {tot_over/max(tot_under,1):.2f}")
    log("  rim-dominated deficits are annotator-style noise (irreducible); the "
        "non-rim remainder is the structurally actionable part (tails, offsets).")

    df.to_parquet(OUT / "sq_anatomy.parquet", index=False)
    log(f"\nsaved -> {OUT/'sq_anatomy.parquet'}")


if __name__ == "__main__":
    main()
