"""Is the limb actually under-selected, and would a radial keep-threshold help?

Two things get conflated in the proposal and only one of them is ours:

  IoU threshold      the METRIC's, fixed at 0.5 by the official scorer. Making it
                     radius-dependent locally would only flatter our own numbers.
  confidence/keep    ours. A candidate at r > 0.7 could be kept on weaker
                     evidence than one at disk centre.

The physical argument for it is sound: at the limb a filament is foreshortened
by cos(theta) and limb darkening cuts its contrast, so the same object presents
as a thinner, fainter thing. But our selector already reads r_norm as one of its
32 features (5th by permutation importance, +0.0048), so it can in principle
already apply a different effective threshold by radius, and a hand-written
radial rule would be re-doing what the model can learn.

So this measures, in order:
  1. how PQ / recall / SQ actually vary with r_norm — is there a limb deficit at all
  2. whether the learned selector's kept-rate already falls or rises with radius
  3. whether a hand-set radial relaxation beats the flat threshold, cross-fitted

Usage: python src/eda/radial_threshold.py
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

from selector import fast_pq
from selector_v5 import BASE, SRC, CROSS
from selector_v6 import make_regressor

OUT = ROOT / "outputs" / "eda"
FEATS = BASE + SRC + CROSS
BINS = [0.0, 0.3, 0.5, 0.65, 0.75, 0.85, 0.92, 1.0]


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="outputs/selector/fused_pool_all.parquet")
    a = ap.parse_args()

    cand = pd.read_parquet(ROOT / a.pool)
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt_sub = gt[gt["file_name"].isin(set(cand["stem"] + ".jpeg"))]
    total_gt = len(gt_sub)
    cand["value"] = cand["iou_sum"] / cand["n_copies"].clip(lower=1)
    cand["rbin"] = pd.cut(cand["r_norm"], BINS, include_lowest=True)

    # 1. where does the value actually sit, by radius?
    log("candidate pool by radius (value = iou_sum / n_copies, the metric's own "
        "worth of a candidate):")
    g = cand.groupby("rbin", observed=True).agg(
        n=("cand_id", "size"),
        matched_frac=("value", lambda s: float((s > 0).mean())),
        mean_value=("value", "mean"),
        mean_conf=("conf", "mean"),
        median_area=("area", "median"))
    log(g.to_string(float_format=lambda v: f"{v:.4f}"))

    # 2. what does the learned selector already do by radius?
    from sklearn.model_selection import GroupKFold
    X = cand[FEATS].to_numpy(np.float64)
    y = cand["value"].to_numpy(np.float64)
    grp = cand["stem"].to_numpy()
    p = np.full(len(cand), np.nan)
    for tr, va in GroupKFold(n_splits=5).split(X, y, grp):
        p[va] = make_regressor().fit(X[tr], y[tr]).predict(X[va])
    cand["v_oof"] = p
    thr = 0.4255 / 2          # the fixed-point rule's threshold
    cand["kept"] = cand["v_oof"] > thr
    log(f"\nlearned selector at the derived threshold {thr:.4f}, by radius:")
    g2 = cand.groupby("rbin", observed=True).agg(
        n=("cand_id", "size"),
        kept_frac=("kept", "mean"),
        kept_precision=("value", lambda s: np.nan),
        mean_pred=("v_oof", "mean"),
        mean_true=("value", "mean"))
    k = cand[cand["kept"]]
    g2["kept_precision"] = k.groupby("rbin", observed=True)["value"].apply(
        lambda s: float((s > 0).mean()))
    log(g2.to_string(float_format=lambda v: f"{v:.4f}"))
    log("  (mean_pred vs mean_true shows whether the model is already "
        "calibrated per radius — a gap is what a manual rule would exploit)")

    # 3. does a radial relaxation beat the flat threshold?
    log("\nradial relaxation: keep if v_oof > thr * (1 - slope * max(0, r-r0))")
    res = {}
    base = fast_pq(cand[cand["kept"]], total_gt)["pq"]
    log(f"  flat threshold                      PQ {base:.4f} "
        f"(kept {int(cand['kept'].sum())})")
    res["flat"] = base
    for r0 in (0.6, 0.7):
        for slope in (0.5, 1.0, 1.5):
            adj = thr * (1 - slope * np.clip(cand["r_norm"] - r0, 0, None))
            keep = cand[cand["v_oof"] > adj]
            q = fast_pq(keep, total_gt)["pq"]
            log(f"  r0={r0} slope={slope:<4} PQ {q:.4f} ({q-base:+.4f}, "
                f"kept {len(keep)})")
            res[f"r0={r0},slope={slope}"] = q
    # and the opposite direction, in case the limb should be STRICTER
    for r0 in (0.6, 0.7):
        for slope in (-0.5, -1.0):
            adj = thr * (1 - slope * np.clip(cand["r_norm"] - r0, 0, None))
            keep = cand[cand["v_oof"] > adj]
            q = fast_pq(keep, total_gt)["pq"]
            log(f"  r0={r0} slope={slope:<4} PQ {q:.4f} ({q-base:+.4f}, "
                f"kept {len(keep)})")
            res[f"r0={r0},slope={slope}"] = q

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "radial_threshold.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    log(f"\nsaved -> {OUT/'radial_threshold.json'}")


if __name__ == "__main__":
    main()
