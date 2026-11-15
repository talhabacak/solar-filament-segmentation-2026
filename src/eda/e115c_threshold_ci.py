"""E115c -- a paired interval on the honest p 0.55 vs 0.60 contrast.

E115b gives a point estimate (-0.0021 mean over folds) and the board gives a
sign on one comparison. Neither bounds the effect, and the whole disagreement
is about whether the truth is "small positive", "small negative" or "nothing".
This resamples IMAGES with replacement and recomputes pooled PQ for both
policies on the same resample, so the two arms share every resample and the
image-to-image variance that dominates each arm cancels.

Pooled PQ is not an average of per-image scores, so it cannot be bootstrapped
by resampling per-image PQs. It decomposes instead through its four sufficient
statistics, which ARE additive over images:

    PQ = sum(IoU of matches) / (TP + FP/2 + FN/2)

with, for image i carrying c_i annotator copies and k_i emitted predictions,
    tp_i  = match rows for that image
    fp_i  = k_i * c_i - tp_i          (a prediction is charged once per copy it misses)
    fn_i  = (GT rows of image i) - tp_i
The script verifies this decomposition against the exact scorer's own totals
before bootstrapping, and refuses to continue if they disagree.

PRE-REGISTERED READING, fixed before the run. An interval covering zero proves
nothing: a wide interval covers zero while still admitting effects worth
chasing. The claim being tested is EQUIVALENCE, so the band of indifference is
declared here, BAND = +/-0.005, taken from the project's own record -- the
transfer floor is +0.003 and the operating-point work measured every threshold
from 0.26 to 0.60 as within 0.005 of the peak.
    interval entirely inside +/-0.005   -> the contrast is measurably flat
    interval reaching past +/-0.005     -> the instrument is underpowered and
                                           §11 must say that instead
Only images are exchangeable here, so images are resampled whole; candidates
and the annotator copies inside one image are correlated. PQ is a pooled ratio,
so each resample recomputes it from summed numerator and denominator rather
than averaging per-image PQ, which would be a different estimand.

Usage: python src/eda/e115c_threshold_ci.py [--reps 4000]
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
sys.path.insert(0, str(ROOT / "src" / "eda"))

from filament.pq import score

OUT = ROOT / "outputs" / "eda"
CACHE = OUT / "e115b_labelled_pool.parquet"
BAND = 0.005  # pre-registered band of indifference; see the module docstring
BUNDLE = ROOT / "outputs" / "selector" / "selector_v5_shrunkblob5prodts.joblib"


def log(*a):
    print(*a, flush=True)


def make_cls():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=25, l2_regularization=1.0, random_state=42)


def make_reg():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=42)


def oof_scores(cand, feats, fold):
    """Deployment-faithful per fold: model fit on the other four folds, scores
    mapped through a reference built from INNER leave-one-fold-out raws of
    those same four folds (never in-sample, never the held-out fold's own)."""
    X = cand[feats].to_numpy(np.float64)
    y = cand["y"].to_numpy()
    v = cand["iou_sum"].to_numpy(np.float64)
    p = np.zeros(len(cand))
    for f in range(5):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        inner = np.zeros(len(tr))
        for g in [k for k in range(5) if k != f]:
            itr = np.where((fold != f) & (fold != g))[0]
            msk = fold[tr] == g
            ci = make_cls().fit(X[itr], y[itr])
            ri = make_reg().fit(X[itr][y[itr] == 1], v[itr][y[itr] == 1])
            Xg = X[tr][msk]
            inner[msk] = ci.predict_proba(Xg)[:, 1] * np.clip(ri.predict(Xg), 0, None)
        ref = np.sort(inner)
        c = make_cls().fit(X[tr], y[tr])
        r = make_reg().fit(X[tr][y[tr] == 1], v[tr][y[tr] == 1])
        raw = c.predict_proba(X[te])[:, 1] * np.clip(r.predict(X[te]), 0, None)
        p[te] = np.searchsorted(ref, raw, side="right") / len(ref)
        log(f"    fold {f} scored")
    return p


def per_image(cand_sub, gt, stems, ncopy):
    """The four additive statistics per image, plus the exact scorer's totals
    for the verification."""
    pred = cand_sub[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
    res = score(gt[["filament_id", "segmentation_rle"]], pred, return_details=True)
    m = res["matches"].copy()
    cid_stem = dict(zip(cand_sub["cand_id"], cand_sub["stem"]))
    m["stem"] = m["pred_filament_id"].map(cid_stem)
    tp = m.groupby("stem").size().reindex(stems).fillna(0).to_numpy()
    iou = m.groupby("stem")["iou"].sum().reindex(stems).fillna(0.0).to_numpy()
    k = cand_sub.groupby("stem").size().reindex(stems).fillna(0).to_numpy()
    gtn = gt.groupby("stem").size().reindex(stems).fillna(0).to_numpy()
    fp = k * ncopy - tp
    fn = gtn - tp
    return dict(tp=tp, fp=fp, fn=fn, iou=iou), res


def pq(tp, fp, fn, iou):
    d = tp + 0.5 * fp + 0.5 * fn
    return iou / d if d > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4000)
    a = ap.parse_args()
    import joblib
    feats = joblib.load(BUNDLE)["features"]
    if not CACHE.exists():
        sys.exit(f"{CACHE} not there yet -- run src/eda/e115b_oof_threshold.py first")
    cand = pd.read_parquet(CACHE)
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt = gt[gt["stem"].isin(set(cand["stem"]))]
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    cand["fold"] = cand["stem"].map(fold_of).astype(int)
    log(f"pool {len(cand)}, stems {cand['stem'].nunique()}, GT rows {len(gt)}")
    pcache = OUT / "e115b_p_oof.parquet"
    if pcache.exists():
        pc = pd.read_parquet(pcache).set_index("cand_id")["p_oof"]
        cand["p_oof"] = cand["cand_id"].map(pc).to_numpy()
        log("  out-of-fold selector scores from cache")
    else:
        log("  fitting out-of-fold selector scores:")
        cand["p_oof"] = oof_scores(cand, feats, cand["fold"].to_numpy())
        cand[["cand_id", "p_oof"]].to_parquet(pcache, index=False)

    stems = np.array(sorted(cand["stem"].unique()))
    ncopy = gt.groupby("stem")["entry_id"].nunique().reindex(stems).fillna(1).to_numpy()
    arms = {}
    for pcut in (0.55, 0.60):
        sub = cand[(cand["conf"] >= 0.40) & (cand["p_oof"] >= pcut)]
        st, res = per_image(sub, gt, stems, ncopy)
        tot = pq(st["tp"].sum(), st["fp"].sum(), st["fn"].sum(), st["iou"].sum())
        log(f"  p>={pcut:.2f}: decomposed PQ {tot:.4f} vs exact scorer {res['pq']:.4f}  "
            f"(TP {int(st['tp'].sum())}/{res['tp']}, FP {int(st['fp'].sum())}/{res['fp']}, "
            f"FN {int(st['fn'].sum())}/{res['fn']})")
        if abs(tot - res["pq"]) > 5e-4:
            sys.exit("per-image decomposition does not reproduce the exact scorer; "
                     "the bootstrap would be measuring the wrong thing")
        arms[pcut] = st

    rng = np.random.default_rng(0)
    n = len(stems)
    d = np.empty(a.reps)
    for b in range(a.reps):
        idx = rng.integers(0, n, n)
        vals = [pq(s["tp"][idx].sum(), s["fp"][idx].sum(), s["fn"][idx].sum(), s["iou"][idx].sum())
                for s in (arms[0.55], arms[0.60])]
        d[b] = vals[1] - vals[0]
    obs = (pq(arms[0.60]["tp"].sum(), arms[0.60]["fp"].sum(), arms[0.60]["fn"].sum(), arms[0.60]["iou"].sum())
           - pq(arms[0.55]["tp"].sum(), arms[0.55]["fp"].sum(), arms[0.55]["fn"].sum(), arms[0.55]["iou"].sum()))
    lo, hi = np.percentile(d, [2.5, 97.5])
    inside = (lo > -BAND) and (hi < BAND)
    log(f"\n  observed Δ (p0.60 - p0.55), honest scores: {obs:+.4f}")
    log(f"  paired image bootstrap, {a.reps} reps over {n} images: "
        f"95% CI [{lo:+.4f}, {hi:+.4f}], sd {d.std():.4f}, P(Δ>0) = {(d > 0).mean():.3f}")
    log(f"  pre-registered band ±{BAND:.3f}: interval is "
        f"{'INSIDE -> the contrast is measurably flat' if inside else 'NOT inside -> underpowered, no equivalence claim'}")
    json.dump(dict(observed=float(obs), ci=[float(lo), float(hi)], sd=float(d.std()),
                   p_gt0=float((d > 0).mean()), reps=a.reps, band=BAND, equivalent=bool(inside)),
              open(OUT / "e115c_threshold_ci.json", "w"), indent=1)
    log(f"\nwrote {OUT/'e115c_threshold_ci.json'}")


if __name__ == "__main__":
    main()
