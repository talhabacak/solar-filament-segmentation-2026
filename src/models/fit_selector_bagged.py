"""Fit the deployed two-stage selector with image-bootstrap bagging, as a deployable bundle.

SI-S2 (src/eda/selector_image_bagging.py) measured five image-bootstrap copies of the selector,
scores averaged, at +0.0010 CI [-0.0009, +0.0030] over five honest folds on the champion pool.
This builds that selector for submission, changing exactly one thing against the sub033 bundle
(outputs/selector/selector_v5_shrunk_t020pox.joblib, fitted by fused_shrink_eval.py --two-stage):

  same pool      the shrunk, labelled t0.2 pool with the ten extra features (10042 candidates)
  same learners  selector_v5.make_model() classifier, HGB regressor lr 0.06 on positives only
  same quantile  the two-stage score is mapped through the CDF of its out-of-fold values over
                 GroupKFold(5) by image, exactly as fused_shrink_eval builds `two_stage_cdf`
  one change     every (classifier, regressor) pair is replaced by NBAG pairs, each fitted on a
                 bootstrap sample of IMAGES, and the product score is their mean -- in the
                 out-of-fold CDF and in the final model alike, so the gate keeps its meaning

Usage: python src/models/fit_selector_bagged.py [--nbag 5]
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "16")
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
from selector_v5 import make_model  # noqa: E402
from extra_features import EXTRA  # noqa: E402

REF_BUNDLE = ROOT / "outputs/selector/selector_v5_shrunk_t020pox.joblib"


def make_cls(reg_arm, slow=False):
    """The deployed classifier; E122's stronger regularisation (15 leaves, min leaf 60, l2 5);
    or the slower schedule (lr 0.03, 500 iterations) measured at +0.0006 over bag5 on 2026-09-16
    (outputs/eda/selector_learner_arms.json)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    if slow:
        return HistGradientBoostingClassifier(max_iter=500, learning_rate=0.03, max_leaf_nodes=31,
                                              min_samples_leaf=25, l2_regularization=1.0,
                                              early_stopping=False, random_state=42)
    if not reg_arm:
        return make_model()
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=60, l2_regularization=5.0,
                                          early_stopping=False, random_state=42)


def fit_bag(df, feats, nbag, rng, reg_arm=False, slow=False):
    stems = df["stem"].unique()
    by_stem = {s: g for s, g in df.groupby("stem")}
    pairs = []
    for _ in range(nbag):
        boot = (df if nbag == 1 and not rng else
                pd.concat([by_stem[s] for s in rng.choice(stems, len(stems), replace=True)],
                          ignore_index=True))
        X, y = boot[feats].to_numpy(np.float64), boot["y"].to_numpy()
        pos = y == 1
        c = make_cls(reg_arm, slow).fit(X, y)
        r = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06).fit(
            X[pos], boot.loc[pos, "iou_sum"].to_numpy(np.float64))
        pairs.append((c, r))
    return pairs


def bag_score(pairs, X):
    return np.mean([c.predict_proba(X)[:, 1] * np.clip(r.predict(X), 0.0, None)
                    for c, r in pairs], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbag", type=int, default=5)
    ap.add_argument("--out", default="outputs/selector/selector_t020pox_bag5.joblib")
    ap.add_argument("--reg", action="store_true",
                    help="E122 'reg' arm: 15 leaves, min leaf 60, l2 5.0 in the classifier, "
                         "everything else the deployed recipe. Re-measured on this pool at "
                         "+0.0014, CI [-0.0004, +0.0032], no fold negative (outputs/eda/reg_t020pox.json)")
    ap.add_argument("--slow", action="store_true", help="classifier lr 0.03, 500 iterations")
    ap.add_argument("--relational", action="store_true",
                    help="append the thirteen neighbourhood-geometry features of E195 "
                         "(src/eda/relational_features.py). On the bagged machinery they read "
                         "+0.0022, CI [+0.0003, +0.0043], 4/5 folds (relational_selector_bag.json)")
    ap.add_argument("--regime", action="store_true",
                    help="append the twenty frame-relative features of E207 "
                         "(src/eda/regime_relative_features.py): the percentile and z-score, "
                         "within the candidate's own frame, of the ten quantities the score leans "
                         "on. Motivated by E205b -- across high-frequency quartiles SQ and "
                         "precision are flat and the whole loss is recall, half of it the gate "
                         "discarding real candidates -- i.e. the gate judges on an absolute "
                         "evidence scale while the scale moves with the frame. Bagged five-fold "
                         "OOF: +0.0009, CI [-0.0009, +0.0027], 4/5 folds, and the gain does "
                         "concentrate in the target slice (hf Q4 +0.0026, recall +0.0023) "
                         "(quality_features_regime.json)")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed (0 = sub036)")
    ap.add_argument("--pool", default="fused_pool_all_blob5_t020",
                    help="labelled training pool, without the _labelled suffix. A pool built at "
                         "another extraction threshold needs its own feature block; the guard "
                         "below only applies to the sub036 pool (ledger 40).")
    a = ap.parse_args()
    ref = joblib.load(REF_BUNDLE)
    feats = list(ref["features"])
    pool = pd.read_parquet(ROOT / f"outputs/eda/{a.pool}_labelled.parquet")
    tag = a.pool.split("_")[-1]
    pool = pool.merge(pd.read_parquet(ROOT / f"outputs/eda/extra_features_{tag}.parquet")[["cand_id"] + EXTRA],
                      on="cand_id", how="left")
    if a.relational:
        sys.path.insert(0, str(ROOT / "src" / "eda"))
        from relational_features import REL
        pool = pool.merge(pd.read_parquet(ROOT / "outputs/eda/relational_features_t020.parquet"),
                          on="cand_id", how="left", validate="one_to_one")
        feats = feats + list(REL)
    if a.regime:
        sys.path.insert(0, str(ROOT / "src" / "eda"))
        from regime_relative_features import REG
        pool = pool.merge(pd.read_parquet(ROOT / "outputs/eda/regime_features_t020.parquet"),
                          on="cand_id", how="left", validate="one_to_one")
        feats = feats + list(REG)
    assert not pool[feats].isna().any().any(), f"{a.pool}: missing features"
    if a.pool == "fused_pool_all_blob5_t020":
        assert len(pool) == 10042, "pool is not the sub033 training pool"
    rng = np.random.default_rng(a.seed)
    X = pool[feats].to_numpy(np.float64)
    raw = np.zeros(len(pool))
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, pool["y"], pool["stem"])):
        raw[te] = bag_score(fit_bag(pool.iloc[tr], feats, a.nbag, rng if a.nbag > 1 else None, a.reg, a.slow), X[te])
        print(f"  oof split {k}: done", flush=True)
    cdf = np.sort(raw)
    pairs = fit_bag(pool, feats, a.nbag, rng if a.nbag > 1 else None, a.reg, a.slow)
    joblib.dump({"pairs": pairs, "features": feats, "two_stage_cdf": cdf, "bagged": True,
                 "nbag": a.nbag, "reg_arm": a.reg, "slow": a.slow, "seed": a.seed, "shrink": ref["shrink"]}, ROOT / a.out)
    print(f"saved {a.out}: {a.nbag} pairs, cdf over {len(cdf)} out-of-fold scores")


if __name__ == "__main__":
    main()
