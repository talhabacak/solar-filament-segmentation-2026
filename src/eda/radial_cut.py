"""E132 -- drop every prediction whose body sits outside r 0.9 (and 0.8).

Talha's rule: a candidate whose centroid is inside the radius keeps its whole
mask (its continuation beyond the circle is kept); a candidate whose centroid is
outside is dropped entirely. The production pipeline already suppresses detector
candidates beyond r/R 0.97, because no ground-truth centroid lies past 0.94.

Why it might pay: at the limb the pool is impure. Measured in ledger 16, the
false-positive share is 0.46 beyond r 0.8 against 0.13 on the disc, and the miss
rate is 0.50 in 0.8-0.9 and 0.93 beyond 0.9 -- so those rows are mostly lost
anyway. PQ charges 0.5 for a false positive and 0.5 for a miss, so dropping a
band pays exactly when its kept candidates are less than about 45% true (the
empirical break-even of E128).

Measured on the champion configuration (semantic threshold 0.2 pool, 42-feature
selector, out-of-fold scores, keep gate swept in E129d), with the exact scorer
over all 703 images. Also reported per band so the trade is visible: how many
true positives and false positives each cut removes.

Usage: python src/eda/radial_cut.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "eda"))
sys.path.insert(0, str(ROOT / "src" / "models"))
from filament.pq import score  # noqa: E402
from e115c_threshold_ci import oof_scores  # noqa: E402
from extra_features import EXTRA  # noqa: E402

OUT = ROOT / "outputs" / "eda"
GATES = [(0.40, 0.60), (0.45, 0.575)]
CUTS = [None, 0.95, 0.9, 0.85, 0.8, 0.75]


def log(*a):
    print(*a, flush=True)


def main():
    feats = list(joblib.load(ROOT / "outputs/selector/selector_v5_shrunkblob5prodts.joblib")["features"]) + list(EXTRA)
    a = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    a = a.merge(pd.read_parquet(OUT / "extra_features_t020.parquet")[["cand_id"] + EXTRA], on="cand_id", how="left")
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    a["fold"] = a["stem"].map(fold_of).astype(int)
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    g = gt[gt["stem"].isin(set(a["stem"]))][["filament_id", "segmentation_rle"]]
    cache = OUT / "t020pox_oof_scores.parquet"
    if cache.exists():
        a = a.merge(pd.read_parquet(cache), on="cand_id", how="left")
        log("out-of-fold scores from cache")
    else:
        a["p"] = oof_scores(a, feats, a["fold"].to_numpy())
        a[["cand_id", "p"]].to_parquet(cache, index=False)
    log(f"pool {len(a)} rows; GT rows beyond r 0.9: "
        f"{int((a['y'] == 1).sum())} positives in pool, candidates beyond 0.9: {int((a['r_norm'] > 0.9).sum())}")

    def sc(sub):
        return score(g, sub[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"}))

    res = {}
    for conf, pc in GATES:
        keep = a[(a["conf"] >= conf) & (a["p"] >= pc)]
        base = sc(keep)
        log(f"\ngate ({conf}, {pc}): {len(keep)} kept, PQ {base['pq']:.4f} "
            f"TP {base['tp']} FP {base['fp']} FN {base['fn']}")
        log(f"  {'cut':>6s} {'kept':>6s} {'dropped':>8s} {'true dropped':>13s} {'purity':>7s} {'PQ':>8s} {'dPQ':>9s}")
        for cut in CUTS:
            sub = keep if cut is None else keep[keep["r_norm"] <= cut]
            dropped = keep if cut is None else keep[keep["r_norm"] > cut]
            r = sc(sub)
            d = r["pq"] - base["pq"]
            pur = float(dropped["y"].mean()) if len(dropped) and cut is not None else float("nan")
            res[f"{conf}_{pc}_{cut}"] = dict(kept=len(sub), pq=r["pq"], tp=r["tp"], fp=r["fp"], fn=r["fn"],
                                             dropped=0 if cut is None else len(dropped), d_pq=float(d))
            log(f"  {str(cut):>6s} {len(sub):6d} {0 if cut is None else len(dropped):8d} "
                f"{'' if cut is None else int(dropped['y'].sum()):>13} {pur:7.3f} {r['pq']:8.4f} {d:+9.4f}")
    # what the pool holds per band, for the report
    log("\ncandidates and purity by radial band (whole pool):")
    a["band"] = pd.cut(a["r_norm"], [0, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01])
    t = a.groupby("band", observed=True).agg(n=("cand_id", "size"), purity=("y", "mean"))
    log(t.round(3).to_string())
    json.dump(res, open(OUT / "radial_cut.json", "w"), indent=1, default=float)
    log(f"\nwrote {OUT / 'radial_cut.json'}")


if __name__ == "__main__":
    main()
