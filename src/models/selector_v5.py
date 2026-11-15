"""Selector v5: does cross-lane agreement close part of the +0.088 selection gap?

Measured motivation: on the fold-0 fused pool, perfect selection scores PQ 0.5262
against our realised 0.4383. The previous selectors ranked candidates on shape,
position and the producing model's own confidence — never on what the OTHER lane
thought of the same pixels. A detector box the semantic map lights up is very
different evidence from one it ignores.

Feature blocks, evaluated as a ladder so each block's contribution is separable:
  BASE   shape/position/conf                      (selector v2)
  +SRC   which lane(s) produced it                (fused selector)
  +CROSS semantic-probability statistics under and around the mask, detector-fold
         consensus, cluster tightness, neighbourhood and image activity

Honesty: probabilities are GroupKFold-by-image out-of-fold, and the keep policy
is chosen on one half of the images and scored on the other (10 repeats x 2
directions), so the reported gain is cross-fitted, not in-sample.

Usage: python src/models/selector_v5.py
"""

import json
import os
import sys
from pathlib import Path

# leave cores for the training dataloader running alongside
os.environ.setdefault("OMP_NUM_THREADS", "8")

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from selector import fast_pq

OUT = ROOT / "outputs" / "selector"

BASE = ["conf", "log10_area", "r_norm", "abs_lat", "bbox_aspect", "eccentricity",
        "boundary_complexity", "skel_len", "width_proxy", "n_cands_img", "rank_conf"]
SRC = ["det_conf", "sem_conf", "from_det", "from_sem", "n_sources"]
CROSS = ["sem_prob_mean", "sem_prob_max", "sem_prob_p90", "sem_frac_over_50",
         "sem_ring_mean", "sem_contrast", "n_det_models", "det_conf_mean",
         "det_conf_std", "det_conf_max", "member_iou_mean", "member_iou_min",
         "nn_dist", "n_within_200px", "img_cand_density", "img_sem_area_frac"]


def log(*a):
    print(*a, flush=True)


def make_model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=25,
        l2_regularization=1.0, early_stopping=False, random_state=42)


def oof_probs(cand, feats):
    from sklearn.model_selection import GroupKFold
    X = cand[feats].to_numpy(np.float64)
    y = cand["y"].to_numpy()
    g = cand["stem"].to_numpy()
    p = np.full(len(cand), np.nan)
    for tr, va in GroupKFold(n_splits=5).split(X, y, g):
        p[va] = make_model().fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    return p


def crossfit_gain(cand, col, total_gt, gt_stems, baseline):
    """Policy chosen on half the images, PQ measured on the other half."""
    files = np.array(sorted(cand["stem"].unique()))
    rng = np.random.default_rng(13)
    grid = [(c, p) for c in (0.0, 0.2, 0.25, 0.3, 0.35, 0.4)
            for p in np.arange(0.05, 0.70, 0.05)]
    gains, abs_pq = [], []
    for _ in range(10):
        perm = rng.permutation(len(files))
        half = set(files[perm[:len(files) // 2]])
        for tr_h, ev_h in ((half, set(files) - half), (set(files) - half, half)):
            ctr, cev = cand[cand["stem"].isin(tr_h)], cand[cand["stem"].isin(ev_h)]
            gt_tr = int(gt_stems.isin(tr_h).sum())
            gt_ev = total_gt - gt_tr
            best, best_pq = None, -1.0
            for c, p in grid:
                k = ctr[(ctr["conf"] >= c) & (ctr[col] >= p)]
                if k.empty:
                    continue
                q = fast_pq(k, gt_tr)["pq"]
                if q > best_pq:
                    best, best_pq = (c, p), q
            k_ev = cev[(cev["conf"] >= best[0]) & (cev[col] >= best[1])]
            base_ev = cev[(cev["conf"] >= baseline[0]) & (cev["area"] >= baseline[1])]
            pq_ev = fast_pq(k_ev, gt_ev)["pq"]
            abs_pq.append(pq_ev)
            gains.append(pq_ev - fast_pq(base_ev, gt_ev)["pq"])
    return float(np.mean(gains)), float(np.std(gains)), float(np.mean(abs_pq))


def main():
    cand = pd.read_parquet(OUT / "fused_pool_all.parquet")
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    files = set(cand["stem"] + ".jpeg")
    gt_sub = gt[gt["file_name"].isin(files)]
    total_gt = len(gt_sub)
    gt_stems = gt_sub["file_name"].str.replace(".jpeg", "", regex=False)
    log(f"pool {len(cand)} candidates over {cand['stem'].nunique()} images, "
        f"{int(cand['y'].sum())} positive; GT rows {total_gt}")

    # reference points
    base_pol = (0.40, 128)
    ref = fast_pq(cand[(cand["conf"] >= base_pol[0]) & (cand["area"] >= base_pol[1])], total_gt)
    orc = fast_pq(cand[cand["y"] == 1], total_gt)
    log(f"conf/area baseline : PQ {ref['pq']:.4f}  TP {ref['tp']} FP {ref['fp']} FN {ref['fn']}")
    log(f"oracle (perfect sel): PQ {orc['pq']:.4f}  -> selection gap {orc['pq']-ref['pq']:+.4f}")

    from sklearn.metrics import roc_auc_score
    results = {"baseline": ref, "oracle": orc}
    ladder = [("BASE", BASE), ("BASE+SRC", BASE + SRC), ("BASE+SRC+CROSS", BASE + SRC + CROSS)]
    for name, feats in ladder:
        col = f"p_{name}"
        cand[col] = oof_probs(cand, feats)
        auc = roc_auc_score(cand["y"], cand[col])
        gm, gs, apq = crossfit_gain(cand, col, total_gt, gt_stems, base_pol)
        # in-sample best policy, for reference only
        best, best_pq = None, -1.0
        for c in (0.0, 0.2, 0.25, 0.3, 0.35, 0.4):
            for p in np.arange(0.05, 0.70, 0.05):
                k = cand[(cand["conf"] >= c) & (cand[col] >= p)]
                if k.empty:
                    continue
                q = fast_pq(k, total_gt)["pq"]
                if q > best_pq:
                    best, best_pq = (c, round(float(p), 2)), q
        results[name] = dict(n_features=len(feats), auc=float(auc),
                             crossfit_gain=gm, crossfit_std=gs, crossfit_abs_pq=apq,
                             insample_policy={"conf": best[0], "p": best[1]},
                             insample_pq=best_pq)
        log(f"{name:16s} ({len(feats):2d} feats): AUC {auc:.4f} | cross-fitted gain "
            f"{gm:+.4f} +/- {gs:.4f} (abs PQ {apq:.4f}) | in-sample best "
            f"conf>={best[0]} p>={best[1]} -> {best_pq:.4f}")

    # permutation importance of the winning set, on held-out folds
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import GroupKFold
    feats = BASE + SRC + CROSS
    X = cand[feats].to_numpy(np.float64)
    y = cand["y"].to_numpy()
    g = cand["stem"].to_numpy()
    imps = []
    for tr, va in GroupKFold(n_splits=5).split(X, y, g):
        m = make_model().fit(X[tr], y[tr])
        pi = permutation_importance(m, X[va], y[va], scoring="roc_auc",
                                    n_repeats=5, random_state=0, n_jobs=1)
        imps.append(pi.importances_mean)
    imp = pd.Series(np.mean(imps, axis=0), index=feats).sort_values(ascending=False)
    log("\ntop features (permutation AUC drop):")
    for k, v in imp.head(12).items():
        log(f"  {k:22s} {v:+.4f}")
    results["importance"] = {k: float(v) for k, v in imp.items()}

    joblib.dump({"model": make_model().fit(X, y), "features": feats},
                OUT / "selector_v5.joblib")
    cand.to_parquet(OUT / "fused_pool_all_scored.parquet", index=False)
    with open(OUT / "selector_v5_results.json", "w") as f:
        json.dump(results, f, indent=1, default=float)
    log(f"\nsaved -> {OUT/'selector_v5.joblib'}")


if __name__ == "__main__":
    main()
