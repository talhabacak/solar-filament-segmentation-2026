"""Selector v6: predict expected PQ value, not "does it match anything".

v5 asked a classifier `y = weight >= 1` — does this candidate match some GT? —
and then swept a (conf, p) grid for the keep policy. Both halves are wrong, and
the metric says why.

Take a GT instance that k of an image's n annotator copies drew. Adding a
prediction for it changes the pooled PQ = sum(IoU_TP) / (TP + FP/2 + FN/2) by:

    with the prediction   numerator += k*IoU     denominator += k + (n-k)/2
    without it            numerator += 0         denominator += k/2   (k FNs)

so the denominator cost of predicting is (k + (n-k)/2) - k/2 = n/2 — the SAME
for every candidate on that image, whatever its support. The benefit is k*IoU,
i.e. `iou_sum`. Predicting is worth it exactly when

    iou_sum > (n/2) * PQ            <=>    iou_sum/n > PQ/2

PQ sits on both sides, so the operating point is a fixed point, not a free
parameter — the grid search disappears. And the rule depends on iou_sum/n, a
per-candidate quantity (support fraction times mean IoU), so nothing needs to
know n at test time.

Measured on the fold-pooled fused pool, this is a strictly better TARGET than
v5's: the value rule with true labels scores 0.5078 against 0.5024 for "any
match", keeping 3862 candidates instead of 4161 — it drops 299 that do match
something but whose IoU is too low to pay for the n/2 they cost.

So v6 regresses value = iou_sum/n_copies and keeps by the closed-form rule.
Reported honestly: regressor fit out-of-fold by image, and the whole procedure
(fit + fixed point) run on one half of the images and scored on the other.

Usage: python src/models/selector_v6.py [--pool ...] [--label v6]
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from selector import fast_pq
from selector_v5 import BASE, SRC, CROSS, log

OUT = ROOT / "outputs" / "selector"


def make_regressor():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=25,
        l2_regularization=1.0, early_stopping=False, random_state=42)


def oof_values(cand, feats):
    from sklearn.model_selection import GroupKFold
    X = cand[feats].to_numpy(np.float64)
    y = cand["value"].to_numpy(np.float64)
    g = cand["stem"].to_numpy()
    p = np.full(len(cand), np.nan)
    for tr, va in GroupKFold(n_splits=5).split(X, y, g):
        p[va] = make_regressor().fit(X[tr], y[tr]).predict(X[va])
    return p


def fixed_point(cand, col, total_gt, q0=0.40, iters=40):
    """Keep iff predicted value > q/2, with q the PQ that rule itself produces."""
    q = q0
    keep = cand.iloc[:0]
    for _ in range(iters):
        keep = cand[cand[col] > q / 2.0]
        q2 = fast_pq(keep, total_gt)["pq"] if len(keep) else 0.0
        if abs(q2 - q) < 1e-6:
            break
        q = q2
    return q, keep


def crossfit(cand, feats, total_gt, gt_stems, base_pol, repeats=10):
    """Fit and choose the operating point on half the images, score on the other."""
    files = np.array(sorted(cand["stem"].unique()))
    rng = np.random.default_rng(13)
    gains, abs_pq = [], []
    for _ in range(repeats):
        perm = rng.permutation(len(files))
        half = set(files[perm[:len(files) // 2]])
        for tr_h, ev_h in ((half, set(files) - half), (set(files) - half, half)):
            ctr = cand[cand["stem"].isin(tr_h)]
            cev = cand[cand["stem"].isin(ev_h)].copy()
            gt_ev = total_gt - int(gt_stems.isin(tr_h).sum())
            m = make_regressor().fit(ctr[feats].to_numpy(np.float64),
                                     ctr["value"].to_numpy(np.float64))
            cev["v"] = m.predict(cev[feats].to_numpy(np.float64))
            # operating point comes from the TRAINING half, then is applied blind
            ctr2 = ctr.copy()
            ctr2["v"] = m.predict(ctr[feats].to_numpy(np.float64))
            q_tr, _ = fixed_point(ctr2, "v", int(gt_stems.isin(tr_h).sum()))
            keep = cev[cev["v"] > q_tr / 2.0]
            base_ev = cev[(cev["conf"] >= base_pol[0]) & (cev["area"] >= base_pol[1])]
            pq_ev = fast_pq(keep, gt_ev)["pq"]
            abs_pq.append(pq_ev)
            gains.append(pq_ev - fast_pq(base_ev, gt_ev)["pq"])
    return float(np.mean(gains)), float(np.std(gains)), float(np.mean(abs_pq))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="outputs/selector/fused_pool_all.parquet")
    ap.add_argument("--label", default="v6")
    a = ap.parse_args()

    cand = pd.read_parquet(ROOT / a.pool)
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    files = set(cand["stem"] + ".jpeg")
    gt_sub = gt[gt["file_name"].isin(files)]
    total_gt = len(gt_sub)
    gt_stems = gt_sub["file_name"].str.replace(".jpeg", "", regex=False)

    cand["value"] = cand["iou_sum"] / cand["n_copies"].clip(lower=1)
    base_pol = (0.40, 128)
    base = fast_pq(cand[(cand["conf"] >= base_pol[0]) & (cand["area"] >= base_pol[1])],
                   total_gt)["pq"]
    orc_any = fast_pq(cand[cand["weight"] >= 1], total_gt)["pq"]
    q_orc, keep_orc = fixed_point(cand, "value", total_gt)
    log(f"pool {len(cand)} candidates, GT {total_gt}")
    log(f"conf/area baseline            {base:.4f}")
    log(f"oracle 'any match'            {orc_any:.4f}  ({int((cand['weight']>=1).sum())} kept)")
    log(f"oracle value rule (true)      {q_orc:.4f}  ({len(keep_orc)} kept)")

    feats = BASE + SRC + CROSS
    cand["v_oof"] = oof_values(cand, feats)
    from scipy.stats import spearmanr
    rho = spearmanr(cand["v_oof"], cand["value"]).statistic
    q_hat, keep_hat = fixed_point(cand, "v_oof", total_gt)
    log(f"\nOOF value regression: Spearman rho {rho:.4f}")
    log(f"in-sample fixed point: q={q_hat:.4f} keep>{q_hat/2:.4f} -> {len(keep_hat)} kept")

    # is the closed-form threshold actually where the optimum sits?
    best_t, best_q = None, -1.0
    for t in np.arange(0.02, 0.60, 0.01):
        k = cand[cand["v_oof"] > t]
        if k.empty:
            continue
        q = fast_pq(k, total_gt)["pq"]
        if q > best_q:
            best_t, best_q = float(t), q
    log(f"swept best threshold {best_t:.2f} -> {best_q:.4f}  "
        f"(closed form {q_hat/2:.4f} -> {q_hat:.4f}, gap {best_q-q_hat:+.4f})")

    gm, gs, apq = crossfit(cand, feats, total_gt, gt_stems, base_pol)
    log(f"cross-fitted: {gm:+.4f} +/- {gs:.4f} (abs PQ {apq:.4f})")

    X = cand[feats].to_numpy(np.float64)
    joblib.dump({"model": make_regressor().fit(X, cand["value"].to_numpy(np.float64)),
                 "features": feats, "kind": "value_regressor",
                 "threshold": q_hat / 2.0, "q": q_hat},
                OUT / f"selector_{a.label}.joblib")
    with open(OUT / f"selector_{a.label}_results.json", "w") as f:
        json.dump(dict(baseline=base, oracle_any=orc_any, oracle_value=q_orc,
                       spearman=float(rho), insample_pq=q_hat,
                       threshold=q_hat / 2.0, swept_threshold=best_t,
                       swept_pq=best_q, crossfit_gain=gm, crossfit_std=gs,
                       crossfit_abs_pq=apq), f, indent=1, default=float)
    log(f"saved -> {OUT / f'selector_{a.label}.joblib'}")


if __name__ == "__main__":
    main()
