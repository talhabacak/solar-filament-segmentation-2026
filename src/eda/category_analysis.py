"""Is 'Unidentifiable' garbage to abandon, or the biggest untapped block?

The hypothesis under test (Talha's): U instances are mostly noise, so the loss
should care less about them. Down-weighting U to 0.35 was trained and measured
neutral-to-negative (pixel-Dice 0.6956 vs 0.7019, standalone instance PQ 0.4045
vs 0.4066), which says the idea does not pay — but not WHY.

The metric answer is arithmetic: U is 3074 of 8199 GT rows (37.5%). Every U we
fail to match is half an FN in the denominator, so abandoning them cannot be
cheap no matter how noisy they are. This script measures the other half — how
well we currently do on each class — to tell apart the two readings:

  * U recall is already comparable to L/R  -> nothing special about U, drop it
  * U recall is much worse                 -> 37.5% of the GT is under-served,
                                              and that is an opportunity, not
                                              garbage to discard

Reports per category: GT count, matched fraction, IoU distribution of matches,
and the counterfactual PQ with U rows excluded from scoring, which bounds what
perfect U handling would be worth.

Usage: python src/eda/category_analysis.py --fold 0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament.pq import score

OUT = ROOT / "outputs" / "eda"
CAT = {1: "Left", 2: "Right", 3: "Unidentifiable", 4: "Ambiguous"}


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=-1, help="-1 = all pooled")
    ap.add_argument("--pool", default="outputs/selector/fused_pool_all_scored.parquet")
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--p-min", type=float, default=0.0,
                    help="selector threshold; 0 = conf/area baseline only")
    a = ap.parse_args()

    cand = pd.read_parquet(ROOT / a.pool)
    if a.fold >= 0 and "fold" in cand.columns:
        cand = cand[cand["fold"] == a.fold]
    keep = cand[cand["conf"] >= a.conf]
    if a.p_min > 0 and "p_BASE+SRC+CROSS" in keep.columns:
        keep = keep[keep["p_BASE+SRC+CROSS"] >= a.p_min]
    else:
        keep = keep[keep["area"] >= 128]

    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    files = set(keep["stem"] + ".jpeg")
    gt_sub = gt[gt["file_name"].isin(files)].copy()
    log(f"{len(keep)} predictions over {keep['stem'].nunique()} images; "
        f"{len(gt_sub)} GT rows")

    pred = keep[["cand_id", "rle"]].rename(
        columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
    res = score(gt_sub[["filament_id", "segmentation_rle"]], pred, return_details=True)
    log(f"overall: PQ {res['pq']:.4f}  TP {res['tp']}  FP {res['fp']}  FN {res['fn']}")

    m = res["matches"]
    cat_of = dict(zip(gt_sub["filament_id"], gt_sub["category_id"]))
    m["category_id"] = m["gt_filament_id"].map(cat_of)
    area_of = dict(zip(gt_sub["filament_id"], gt_sub["area"]))
    m["gt_area"] = m["gt_filament_id"].map(area_of)

    log("\nper category (a GT row is 'matched' if some prediction cleared IoU>0.5):")
    rows = []
    for c, g in gt_sub.groupby("category_id"):
        hit = m[m["category_id"] == c]
        n_gt, n_hit = len(g), hit["gt_filament_id"].nunique()
        ious = hit["iou"].to_numpy()
        rows.append(dict(category=CAT.get(c, c), n_gt=n_gt, matched=n_hit,
                         recall=n_hit / n_gt, mean_iou=float(ious.mean()) if len(ious) else 0.0,
                         median_gt_area=float(g["area"].median()),
                         median_gt_area_matched=float(
                             g.loc[g["filament_id"].isin(hit["gt_filament_id"]), "area"].median()
                             if n_hit else np.nan),
                         median_gt_area_missed=float(
                             g.loc[~g["filament_id"].isin(hit["gt_filament_id"]), "area"].median()
                             if n_hit < n_gt else np.nan)))
    tab = pd.DataFrame(rows)
    log(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # counterfactual: score with U rows removed entirely
    no_u = gt_sub[gt_sub["category_id"] != 3]
    res_no_u = score(no_u[["filament_id", "segmentation_rle"]], pred)
    log(f"\ncounterfactual PQ with Unidentifiable GT removed: {res_no_u['pq']:.4f} "
        f"(vs {res['pq']:.4f} with them)  TP {res_no_u['tp']} FP {res_no_u['fp']} "
        f"FN {res_no_u['fn']}")

    # what perfect U handling would be worth, holding L/R as-is
    u_gt = int((gt_sub["category_id"] == 3).sum())
    u_hit = int(m[m["category_id"] == 3]["gt_filament_id"].nunique())
    ideal = (sum(res["tp_ious"]) + (u_gt - u_hit)) / (
        res["tp"] + (u_gt - u_hit) + 0.5 * res["fp"] + 0.5 * (res["fn"] - (u_gt - u_hit))
    ) if "tp_ious" in res else None
    if ideal is not None:
        log(f"ceiling if every missed U were matched at IoU 1.0: {ideal:.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "category_analysis.json", "w") as f:
        json.dump(dict(overall={k: v for k, v in res.items()
                                if k not in ("matches", "tp_ious")},
                       per_category=rows, pq_without_u=res_no_u["pq"]),
                  f, indent=1, default=float)
    log(f"saved -> {OUT/'category_analysis.json'}")


if __name__ == "__main__":
    main()
