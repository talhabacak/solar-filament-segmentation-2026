"""E151b -- does the image-fed selector actually rank better?

E151 trains a ResNet-18 on a three-channel patch per candidate and emits two
out-of-fold columns, patch_p and patch_v. This is the single-variable test of
whether they help: the same pool, the same masks, the same gate (conf 0.40,
p_sel 0.60), the same two-stage GBDT recipe, fitted once on 42 features and once
on 44.

The protocol is E115b's, unchanged. For each fold the classifier and regressor are
fitted on the other four folds, the quantile reference comes from an inner
leave-one-fold-out pass over those same four, and the held-out fold is scored by
the exact scorer. Nothing about the deployment is altered except the feature
count, so a difference here is the ranking and only the ranking.

Two numbers are reported next to the result, because they say whether a gain is
worth anything:

  AUC              the ranking quality the selector achieves, 42 against 44
  E150's ceiling   a perfect ranking at this budget is worth +0.0321, so the
                   fraction of that captured is the honest way to read the size

Bands are the pre-registered ones: pays at >= +0.003, hurts at <= -0.003.

Usage: python src/eda/patch_selector_eval.py [--patch patch128_oof.parquet]
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/eda", "src/models"):
    sys.path.insert(0, str(ROOT / p))
from filament.pq import score  # noqa: E402
from e115c_threshold_ci import make_cls, make_reg, per_image, pq  # noqa: E402
from extra_features import EXTRA  # noqa: E402

OUT = ROOT / "outputs" / "eda"
GATE = (0.40, 0.60)
REPS, FLOOR, EQUIV = 4000, 0.003, 0.005
CEILING = 0.0321


def log(*a):
    print(*a, flush=True)


def run_selector(a, feats, gt, fold_of, stems_by_fold, ncopy_by_fold):
    """The deployed recipe, out of fold, returning per-image statistics and totals."""
    parts, kept_n = {}, 0
    for F in range(5):
        trn = a[a["fold"] != F]
        X = trn[feats].to_numpy(np.float64)
        y = trn["y"].to_numpy()
        v = trn["iou_sum"].to_numpy(np.float64)
        inner = np.zeros(len(trn))
        for g in [k for k in range(5) if k != F]:
            itr = (trn["fold"] != g).to_numpy()
            msk = (trn["fold"] == g).to_numpy()
            c = make_cls().fit(X[itr], y[itr])
            r = make_reg().fit(X[itr][y[itr] == 1], v[itr][y[itr] == 1])
            inner[msk] = c.predict_proba(X[msk])[:, 1] * np.clip(r.predict(X[msk]), 0, None)
        ref = np.sort(inner)
        cls, reg = make_cls().fit(X, y), make_reg().fit(X[y == 1], v[y == 1])
        te = a[a["fold"] == F].copy()
        Z = te[feats].to_numpy(np.float64)
        raw = cls.predict_proba(Z)[:, 1] * np.clip(reg.predict(Z), 0, None)
        te["p"] = np.searchsorted(ref, raw, side="right") / len(ref)
        sub = te[(te["conf"] >= GATE[0]) & (te["p"] >= GATE[1])]
        kept_n += len(sub)
        gF = gt[gt["stem"].map(fold_of) == F]
        parts[F] = per_image(sub[["cand_id", "rle", "stem"]], gF, stems_by_fold[F], ncopy_by_fold[F])[0]
    return parts, kept_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", default="patch128_oof.parquet")
    a_ = ap.parse_args()
    from sklearn.metrics import roc_auc_score

    base_feats = list(joblib.load(ROOT / "outputs/selector/selector_v5_shrunkblob5prodts.joblib")["features"]) + list(EXTRA)
    a = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    a = a.merge(pd.read_parquet(OUT / "extra_features_t020.parquet")[["cand_id"] + EXTRA], on="cand_id", how="left")
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    a["fold"] = a["stem"].map(fold_of).astype(int)
    patch = pd.read_parquet(OUT / a_.patch)
    a = a.merge(patch, on="cand_id", how="left")
    miss = a["patch_p"].isna().sum()
    log(f"{len(a)} pool rows, {len(patch)} patch rows, {miss} without a patch prediction "
        f"(they sit below the confidence gate and are never kept anyway)")
    a["patch_p"] = a["patch_p"].fillna(0.0)
    a["patch_v"] = a["patch_v"].fillna(0.0)

    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt = gt[gt["stem"].isin(set(a["stem"]))]
    stems_by_fold, ncopy_by_fold = {}, {}
    for F in range(5):
        s = np.array(sorted(set(a.loc[a["fold"] == F, "stem"])))
        stems_by_fold[F] = s
        gF = gt[gt["stem"].map(fold_of) == F]
        ncopy_by_fold[F] = gF.groupby("stem")["entry_id"].nunique().reindex(s).fillna(1).to_numpy()

    gated = a[a["conf"] >= GATE[0]]
    # the production selector's headline AUC (0.907) is measured on the FULL pool,
    # where most negatives are easy. The patch head only ever sees candidates past
    # the confidence gate, so the two are only comparable on that subset.
    prod_p = pd.read_parquet(OUT / "t020pox_oof_scores.parquet")
    gp = gated.merge(prod_p, on="cand_id", how="left")
    log(f"on the gated pool ({len(gated)} candidates, {gated['y'].mean():.1%} true):")
    log(f"  production selector AUC {roc_auc_score(gp['y'], gp['p']):.4f}")
    log(f"  patch head AUC          {roc_auc_score(gated['y'], gated['patch_p']):.4f}")

    res = {}
    parts = {}
    for name, feats in (("42 features", base_feats),
                        ("44 features (+patch)", base_feats + ["patch_p", "patch_v"])):
        p, kept = run_selector(a, feats, gt, fold_of, stems_by_fold, ncopy_by_fold)
        parts[name] = p
        cat = lambda k: np.concatenate([p[F][k] for F in range(5)])
        val = pq(cat("tp").sum(), cat("fp").sum(), cat("fn").sum(), cat("iou").sum())
        res[name] = dict(pq=float(val), kept=kept)
        log(f"{name:22s} kept {kept:5d}  PQ {val:.4f}  "
            f"TP {int(cat('tp').sum())} FP {int(cat('fp').sum())} FN {int(cat('fn').sum())}")

    A, B = parts["42 features"], parts["44 features (+patch)"]
    cat = lambda P, k: np.concatenate([P[F][k] for F in range(5)])
    t0, f0, n0, i0 = cat(A, "tp"), cat(A, "fp"), cat(A, "fn"), cat(A, "iou")
    t1, f1, n1, i1 = cat(B, "tp"), cat(B, "fp"), cat(B, "fn"), cat(B, "iou")
    d = pq(t1.sum(), f1.sum(), n1.sum(), i1.sum()) - pq(t0.sum(), f0.sum(), n0.sum(), i0.sum())
    rng = np.random.default_rng(0)
    n = len(t0)
    boot = np.empty(REPS)
    for i in range(REPS):
        idx = rng.integers(0, n, n)
        boot[i] = (pq(t1[idx].sum(), f1[idx].sum(), n1[idx].sum(), i1[idx].sum())
                   - pq(t0[idx].sum(), f0[idx].sum(), n0[idx].sum(), i0[idx].sum()))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    verdict = ("PAYS" if d >= FLOOR else "HURTS" if d <= -FLOOR
               else ("TIED" if lo > -EQUIV and hi < EQUIV else "UNDERPOWERED"))
    log(f"\ndelta {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  P(delta>0) {(boot > 0).mean():.3f} "
        f"-> {verdict}")
    log(f"E150's perfect-ranking ceiling is +{CEILING:.4f}, so this captures "
        f"{100 * d / CEILING:.0f}% of it")
    res["delta"] = float(d)
    res["ci"] = [float(lo), float(hi)]
    res["verdict"] = verdict
    res["fraction_of_ceiling"] = float(d / CEILING)
    json.dump(res, open(OUT / "patch_selector_eval.json", "w"), indent=1, default=float)
    log(f"wrote {OUT / 'patch_selector_eval.json'}")


if __name__ == "__main__":
    main()
