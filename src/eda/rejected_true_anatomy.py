"""E137 -- the 1029 rows the pool already reaches and the selector throws away.

Reachability is not the binding constraint people assume. The champion pool
reaches 79% of ground-truth rows; the deployed set reaches 60.6% on the disc.
The stage decomposition (ledger 22b) puts the difference in one column:

    disc      817 rows (11.5%)   a pool candidate matches at IoU > 0.5, not kept
    limb      174 rows (18.0%)
    far limb   38 rows (36.2%)

That is ~1029 rows, and recovering them all would be worth roughly +0.05 PQ --
more than every open axis combined. Adding new candidates (a third lane, a
multi-threshold extraction, a query model) fights for the same rows while adding
false positives; converting these costs nothing new.

So: why are they rejected? This describes them against the accepted true
positives on the same features the selector sees, and answers three questions
that decide whether the population is recoverable at all:

  1. how far below the gate do they sit? (just under -> a threshold or
     calibration problem; far below -> the model genuinely disbelieves them)
  2. what do they look like? (per-feature medians against kept true positives
     and against kept false positives -- if they resemble the FPs, the selector
     is right and the ceiling is real)
  3. is there a rule that separates them from the false positives at their own
     score level? (a small honest classifier trained ONLY inside the band, with
     the same out-of-fold discipline, reports its AUC; below ~0.6 there is no
     separating signal and the population is closed)

Usage: python src/eda/rejected_true_anatomy.py
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
from extra_features import EXTRA  # noqa: E402

OUT = ROOT / "outputs" / "eda"
GATE = (0.40, 0.60)


def log(*a):
    print(*a, flush=True)


def main():
    feats = list(joblib.load(ROOT / "outputs/selector/selector_v5_shrunkblob5prodts.joblib")["features"]) + list(EXTRA)
    a = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    a = a.merge(pd.read_parquet(OUT / "extra_features_t020.parquet")[["cand_id"] + EXTRA], on="cand_id", how="left")
    a = a.merge(pd.read_parquet(OUT / "t020pox_oof_scores.parquet"), on="cand_id", how="left")
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    a["fold"] = a["stem"].map(fold_of).astype(int)
    kept = (a["conf"] >= GATE[0]) & (a["p"] >= GATE[1])
    a["grp"] = np.where(kept & (a["y"] == 1), "kept true",
                np.where(kept & (a["y"] == 0), "kept false",
                np.where(~kept & (a["y"] == 1), "REJECTED TRUE", "rejected false")))
    log(a["grp"].value_counts().to_string())

    # 1. how far below the gate
    rt = a[a["grp"] == "REJECTED TRUE"]
    log(f"\nrejected-true candidates: {len(rt)}")
    log("  selector score distribution (gate is 0.60):")
    for lo, hi in [(0.55, 0.60), (0.50, 0.55), (0.40, 0.50), (0.20, 0.40), (0.0, 0.20)]:
        n = int(((rt["p"] >= lo) & (rt["p"] < hi)).sum())
        log(f"    p in [{lo:.2f}, {hi:.2f}): {n:5d} ({n/len(rt):5.1%})")
    below_conf = int((rt["conf"] < GATE[0]).sum())
    log(f"  rejected by the confidence gate rather than the score: {below_conf} ({below_conf/len(rt):.1%})")
    log(f"  by radial band: " + ", ".join(
        f"{b} {int(v)}" for b, v in pd.cut(rt["r_norm"], [0, 0.8, 0.9, 1.01]).value_counts().sort_index().items()))

    # 2. what they look like
    cols = ["conf", "det_conf", "sem_conf", "log10_area", "n_sources", "n_det_models", "det_conf_std",
            "sem_prob_mean", "sem_frac_over_50", "sem_contrast", "member_iou_mean", "img_contrast",
            "dark_core", "coh_in", "boundary_complexity", "width_proxy", "nn_dist", "n_within_200px"]
    med = a.groupby("grp")[cols].median().T
    log("\nmedians by group:")
    log(med.round(3).to_string())

    # 3. is there signal inside the band?
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    band = a[(a["p"] >= 0.40) & (a["p"] < 0.60) & (a["conf"] >= GATE[0])].copy()
    log(f"\nband p in [0.40, 0.60): {len(band)} candidates, {int(band['y'].sum())} true ({band['y'].mean():.1%})")
    X, y, f = band[feats].to_numpy(np.float64), band["y"].to_numpy(), band["fold"].to_numpy()
    oof = np.zeros(len(band))
    for k in range(5):
        tr, te = np.where(f != k)[0], np.where(f == k)[0]
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                           min_samples_leaf=25, l2_regularization=1.0, random_state=42)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y, oof)
    log(f"  a fresh classifier trained ONLY inside this band: out-of-fold AUC {auc:.4f}")
    log("  (the production selector's own AUC on the whole pool is 0.907; an in-band AUC near 0.5")
    log("   means the band is genuinely unseparable and the population is closed)")
    top = band.assign(oof=oof).nlargest(int(len(band) * 0.2), "oof")
    log(f"  purity of the top 20% of the band by that classifier: {top['y'].mean():.1%} "
        f"(band base rate {band['y'].mean():.1%})")
    json.dump(dict(counts=a["grp"].value_counts().to_dict(), n_rejected_true=len(rt),
                   in_band_auc=float(auc), band_n=len(band), band_purity=float(band["y"].mean()),
                   top20_purity=float(top["y"].mean())),
              open(OUT / "rejected_true_anatomy.json", "w"), indent=1, default=float)
    log(f"\nwrote {OUT / 'rejected_true_anatomy.json'}")


if __name__ == "__main__":
    main()
