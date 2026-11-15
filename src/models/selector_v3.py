"""Selector v3 target ablation on the pooled 5-fold candidates (CPU only).

Variants (same features, same HGB, same GroupKFold-by-file honesty):
  A binary          -- y = matched >= 1 copy (v2 reference)
  B copy-weighted   -- classifier trained with sample_weight favoring
                       multi-copy-matched positives and multi-copy FPs
  C expected-gain   -- regression on the candidate's exact marginal PQ
                       numerator/denominator contribution: keep if
                       iou_sum - lambda*(0.5*fp_copies - 0.5*tp_copies)... we use
                       g = iou_sum - 0.5*(n_copies - n_entries_hit) + 0.5*n_entries_hit
                       trained as regression, thresholded on predicted g
Policy evaluation: cross-fitted (policy chosen on half the files, PQ on the
other half, 10 reps x 2 directions) vs the conf.40/area128 baseline.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from selector import FEATURES, fast_pq

OUT = ROOT / "outputs" / "selector"
cand = pd.read_parquet(OUT / "pooled_candidates.parquet")
gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
total_gt = len(gt)
print(f"{len(cand)} candidates, {cand['y'].mean():.3f} positive", flush=True)

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

X = cand[FEATURES].to_numpy(np.float64)
groups = cand["stem"].to_numpy()

# exact per-candidate marginal contribution terms
cand["tp_copies"] = cand["n_entries_hit"]
cand["fp_copies"] = cand["n_copies"] - cand["n_entries_hit"]
cand["gain"] = cand["iou_sum"] - 0.0  # numerator part; denominator handled in policy


def oof_scores(model_factory, y, sample_weight=None, regress=False):
    p = np.full(len(cand), np.nan)
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
        m = model_factory()
        sw = None if sample_weight is None else sample_weight[tr]
        m.fit(X[tr], y[tr], sample_weight=sw)
        p[va] = m.predict(X[va]) if regress else m.predict_proba(X[va])[:, 1]
    return p


def crossfit_gain(score_col, grid_thresholds, conf_grid=(0.20, 0.25, 0.30, 0.35)):
    files = np.array(sorted(cand["stem"].unique()))
    rng = np.random.default_rng(7)
    stems_gt = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gains = []
    for rep in range(10):
        perm = rng.permutation(len(files))
        half = set(files[perm[:len(files) // 2]])
        for tr_half, ev_half in [(half, set(files) - half), (set(files) - half, half)]:
            ctr = cand[cand["stem"].isin(tr_half)]
            cev = cand[cand["stem"].isin(ev_half)]
            gt_tr = int(stems_gt.isin(tr_half).sum())
            gt_ev = total_gt - gt_tr
            best, best_pq = None, -1
            for c in conf_grid:
                for t in grid_thresholds:
                    k = ctr[(ctr["conf"] >= c) & (ctr[score_col] >= t)]
                    pq = fast_pq(k, gt_tr)["pq"]
                    if pq > best_pq:
                        best, best_pq = (c, t), pq
            k_ev = cev[(cev["conf"] >= best[0]) & (cev[score_col] >= best[1])]
            base = cev[(cev["conf"] >= 0.40) & (cev["area"] >= 128)]
            gains.append(fast_pq(k_ev, gt_ev)["pq"] - fast_pq(base, gt_ev)["pq"])
    return float(np.mean(gains)), float(np.std(gains))


mk_clf = lambda: HistGradientBoostingClassifier(
    max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
    min_samples_leaf=25, l2_regularization=1.0, early_stopping=False, random_state=42)
mk_reg = lambda: HistGradientBoostingRegressor(
    max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
    min_samples_leaf=25, l2_regularization=1.0, early_stopping=False, random_state=42)

# A: binary (v2 reference)
cand["pA"] = oof_scores(mk_clf, cand["y"].to_numpy())
gA = crossfit_gain("pA", np.arange(0.05, 0.60, 0.05))
print(f"A binary        : cross-fitted gain {gA[0]:+.4f} +/- {gA[1]:.4f}", flush=True)

# B: copy-weighted classifier (positives weighted by tp_copies, negatives by n_copies)
w = np.where(cand["y"] == 1, cand["tp_copies"].clip(lower=1), cand["n_copies"])
cand["pB"] = oof_scores(mk_clf, cand["y"].to_numpy(), sample_weight=w.astype(float))
gB = crossfit_gain("pB", np.arange(0.05, 0.60, 0.05))
print(f"B copy-weighted : cross-fitted gain {gB[0]:+.4f} +/- {gB[1]:.4f}", flush=True)

# C: expected marginal-contribution regression
# marginal numerator iou_sum; marginal denominator tp + 0.5*fp; omit constant global part.
# target: iou_sum - PQ0*(tp_copies + 0.5*fp_copies) with PQ0 ~ 0.41 (local linearization)
PQ0 = 0.41
g_target = (cand["iou_sum"] - PQ0 * (cand["tp_copies"] + 0.5 * cand["fp_copies"])
            + PQ0 * 0.5 * cand["tp_copies"])   # removing an FN when matched
cand["pC"] = oof_scores(mk_reg, g_target.to_numpy(), regress=True)
gC = crossfit_gain("pC", np.arange(-0.2, 0.6, 0.05))
print(f"C expected-gain : cross-fitted gain {gC[0]:+.4f} +/- {gC[1]:.4f}", flush=True)

pd.DataFrame({"variant": ["A", "B", "C"],
              "gain_mean": [gA[0], gB[0], gC[0]],
              "gain_std": [gA[1], gB[1], gC[1]]}).to_csv(OUT / "selector_v3_ablation.csv", index=False)
print("saved -> selector_v3_ablation.csv", flush=True)
