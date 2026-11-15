"""E176 -- are the 63 near-gate selector drops a FEATURE problem or an INFORMATION limit?

E173 found 63 unambiguous ground-truth copies (every annotator drew them, they agree) whose
best pool candidate reaches IoU 0.637 and is still dropped: selector score 0.532 against a 0.60
gate. Two very different diagnoses fit that:

  feature problem      candidates like these are separable from the false positives that share
                       their score, and we are missing the feature that separates them
  information limit    they sit in a region where the true positive rate really is about the
                       gate, and no feature computable from the image can do better --
                       the same verdict the limb axis got ("signal strength, not a feature")

The test is the honest one: take the candidates in the score band around the gate, restrict to
the profile of these drops (small, faint, crowded), and measure the actual positive rate there.
If the rate is near 0.5, the selector is calibrated and right, and the 63 are the price of a
correct decision rule -- the fix has to come from the model making a stronger mask, not from the
selector. If the rate is high, the gate is leaving value on the table in a describable region,
and a region-specific feature or threshold is worth building.

Also fitted: a selector RESTRICTED to that region, with the strongest extra information the
atlas found (the candidate's own crowding, contrast and size), to see whether the separation
exists at all inside the region.

Usage: python src/eda/selector_information_limit.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/eda", "src/models"):
    sys.path.insert(0, str(ROOT / p))
from e115c_threshold_ci import make_cls, make_reg  # noqa: E402
from extra_features import EXTRA  # noqa: E402

OUT = ROOT / "outputs" / "eda"


def log(*a):
    print(*a, flush=True)


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    feats = list(joblib.load(ROOT / "outputs/selector/selector_v5_shrunkblob5prodts.joblib")["features"]) + list(EXTRA)
    folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    pool = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    pool = pool.merge(pd.read_parquet(OUT / "extra_features_t020.parquet")[["cand_id"] + list(EXTRA)],
                      on="cand_id", how="left")
    pool["fold"] = pool["stem"].map(fold_of).astype(int)
    p = np.zeros(len(pool))
    for F in range(5):
        trn = pool[pool["fold"] != F]
        te = (pool["fold"] == F).to_numpy()
        X, y, v = trn[feats].to_numpy(np.float64), trn["y"].to_numpy(), trn["iou_sum"].to_numpy(np.float64)
        inner = np.zeros(len(trn))
        for g in [k for k in range(5) if k != F]:
            itr, msk = (trn["fold"] != g).to_numpy(), (trn["fold"] == g).to_numpy()
            c = make_cls().fit(X[itr], y[itr])
            r = make_reg().fit(X[itr][y[itr] == 1], v[itr][y[itr] == 1])
            inner[msk] = c.predict_proba(X[msk])[:, 1] * np.clip(r.predict(X[msk]), 0, None)
        ref = np.sort(inner)
        cls = make_cls().fit(X, y)
        reg = make_reg().fit(X[y == 1], v[y == 1])
        Z = pool.loc[te, feats].to_numpy(np.float64)
        p[te] = np.searchsorted(ref, cls.predict_proba(Z)[:, 1] * np.clip(reg.predict(Z), 0, None),
                                side="right") / len(ref)
    pool["p"] = p
    pool["gated"] = pool["conf"] >= 0.40

    log(f"{len(pool)} candidates; positive rate overall {pool['y'].mean():.3f}")
    log("\npositive rate and realised value by selector score band (conf >= 0.40 only):")
    g = pool[pool["gated"]].assign(b=pd.cut(pool.loc[pool["gated"], "p"], [0, .3, .4, .5, .55, .6, .65, .7, .8, 1.0]))
    log(g.groupby("b", observed=True).apply(
        lambda x: pd.Series(dict(n=len(x), pos_rate=x["y"].mean(), mean_iou_sum=x["iou_sum"].mean())),
        include_groups=False).round(3).to_string())

    # the profile of the 63: small, faint, crowded. Express it in CANDIDATE features.
    prof = ((pool["log10_area"] < 3.0) & (pool["n_within_200px"] >= 2))
    band = pool["gated"] & (pool["p"] >= 0.45) & (pool["p"] < 0.60)
    log(f"\nthe near-gate band (0.45 <= p < 0.60, conf >= 0.40): {int(band.sum())} candidates, "
        f"positive rate {pool.loc[band, 'y'].mean():.3f}, mean iou_sum {pool.loc[band, 'iou_sum'].mean():.3f}")
    sub = band & prof
    log(f"  of those, small (<1000 px) and crowded (>=2 within 200 px): {int(sub.sum())} candidates, "
        f"positive rate {pool.loc[sub, 'y'].mean():.3f}")
    keep_value = pool.loc[sub, "iou_sum"].mean()
    # the metric's break-even: keeping a candidate pays when iou_sum > 0.5 * copies * PQ
    ncopy = 1.635
    pq = 0.4421
    be = 0.5 * ncopy * pq
    log(f"  mean iou_sum there {keep_value:.3f} against the break-even {be:.3f} "
        f"-> keeping them is {'PROFITABLE' if keep_value > be else 'a LOSS'}")

    # is there separation INSIDE the region? fit only there, out of fold
    reg_cols = feats + ["p"]
    d = pool[sub].copy()
    if len(d) > 200 and 0.02 < d["y"].mean() < 0.98:
        oof = np.zeros(len(d))
        for F in range(5):
            tr, te = (d["fold"] != F).to_numpy(), (d["fold"] == F).to_numpy()
            if te.sum() == 0 or tr.sum() < 50:
                continue
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=15,
                                               min_samples_leaf=20, l2_regularization=1.0, random_state=0)
            m.fit(d.loc[tr, reg_cols], d.loc[tr, "y"])
            oof[te] = m.predict_proba(d.loc[te, reg_cols])[:, 1]
        auc_in = roc_auc_score(d["y"], oof) if 0 < d["y"].mean() < 1 else np.nan
        auc_p = roc_auc_score(d["y"], d["p"]) if 0 < d["y"].mean() < 1 else np.nan
        log(f"\ninside the region: the deployed score separates at AUC {auc_p:.4f}; "
            f"a selector refitted ONLY there reaches AUC {auc_in:.4f}")
        log("  -> " + ("a region-specific model finds separation the global one misses"
                       if auc_in - auc_p > 0.03 else
                       "no separation left to find: the region is at the information limit"))
    else:
        auc_in = auc_p = float("nan")
        log("\nregion too small or degenerate to refit")
    json.dump(dict(n_band=int(band.sum()), pos_rate_band=float(pool.loc[band, "y"].mean()),
                   n_region=int(sub.sum()), pos_rate_region=float(pool.loc[sub, "y"].mean()),
                   mean_iou_sum_region=float(keep_value), break_even=float(be),
                   auc_deployed_in_region=float(auc_p), auc_refit_in_region=float(auc_in)),
              open(OUT / "selector_information_limit.json", "w"), indent=1)
    log("\nINFO_LIMIT_DONE")


if __name__ == "__main__":
    main()
