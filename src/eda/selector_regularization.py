"""E122 -- is the selector overfit, and does a physics-shaped prior fix it?

Evidence: E115b measured the two-stage selector at AUC 0.9751 in-sample against
0.9065 out-of-fold -- a 0.07 gap on 9917 candidates -- and every board read of
a selector-side change landed inside the display step. The production recipe
(HGB 300 iterations, 31 leaves, min leaf 25, l2 1.0, one seed) was tuned for
in-sample cross-fit PQ, not for transfer. This measures, under the honest E115b
protocol (fit on four folds, inner leave-one-fold-out quantile reference,
score the held-out fold; exact scorer at (0.40, 0.60)), five alternatives that
only change how the SAME features are learned:

  base      production recipe
  reg       stronger regularisation: 15 leaves, min leaf 60, l2 5.0
  bag5      production recipe averaged over five seeds (raw scores averaged
            before the quantile mapping)
  mono      monotone constraints on features whose direction is physics or
            construction, not data: candidate/detector/semantic confidences,
            member agreement, source count, semantic contrast (all increasing)
  mono+reg  both
  mono+bag  monotone + five seeds

Reported per arm: honest OOF AUC of p(match), exact PQ per fold and pooled, the
paired image bootstrap of the pooled delta against base, and the E120 bands
(pays >= +0.003, hurts <= -0.003, TIED if the 95% interval is inside +-0.005).

Usage: python src/eda/selector_regularization.py
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
from e115c_threshold_ci import per_image, pq  # noqa: E402

OUT = ROOT / "outputs" / "eda"
BUNDLE = ROOT / "outputs" / "selector" / "selector_v5_shrunkblob5prodts.joblib"
A_POOL = OUT / "e098d_fused_pool_all_blob5_rebuild_labelled.parquet"
EXTRA_CACHES = ["shape_photometry_features.parquet", "orientation_context_features.parquet"]
POLICY, REPS, FLOOR, EQUIV = (0.40, 0.60), 4000, 0.003, 0.005
MONO_UP = ["conf", "det_conf", "sem_conf", "n_sources", "sem_prob_mean", "sem_prob_max", "sem_prob_p90",
           "sem_frac_over_50", "sem_contrast", "det_conf_mean", "det_conf_max", "member_iou_mean", "member_iou_min"]


def log(*a):
    print(*a, flush=True)


def factories(name, feats):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    mono = [1 if f in MONO_UP else 0 for f in feats] if "mono" in name else None
    if "reg" in name:
        kw = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=60, l2_regularization=5.0)
    else:
        kw = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=25, l2_regularization=1.0)
    seeds = [42, 7, 13, 29, 61] if "bag" in name else [42]

    def cls(seed):
        return HistGradientBoostingClassifier(random_state=seed, monotonic_cst=mono, **kw)

    def reg(seed):
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=seed, monotonic_cst=mono)
    return cls, reg, seeds


def raw_score(cls, reg, seeds, Xtr, ytr, vtr, Xte):
    out, pm = np.zeros(len(Xte)), np.zeros(len(Xte))
    for s in seeds:
        c = cls(s).fit(Xtr, ytr)
        r = reg(s).fit(Xtr[ytr == 1], vtr[ytr == 1])
        p = c.predict_proba(Xte)[:, 1]
        pm += p / len(seeds)
        out += p * np.clip(r.predict(Xte), 0, None) / len(seeds)
    return out, pm


def oof(name, cand, feats, fold):
    cls, reg, seeds = factories(name, feats)
    X, y, v = cand[feats].to_numpy(np.float64), cand["y"].to_numpy(), cand["iou_sum"].to_numpy(np.float64)
    p, pmatch = np.zeros(len(cand)), np.zeros(len(cand))
    for f in range(5):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        inner = np.zeros(len(tr))
        for g in [k for k in range(5) if k != f]:
            itr = np.where((fold != f) & (fold != g))[0]
            msk = fold[tr] == g
            inner[msk], _ = raw_score(cls, reg, seeds, X[itr], y[itr], v[itr], X[tr][msk])
        ref = np.sort(inner)
        raw, pm = raw_score(cls, reg, seeds, X[tr], y[tr], v[tr], X[te])
        p[te] = np.searchsorted(ref, raw, side="right") / len(ref)
        pmatch[te] = pm
    return p, pmatch


def exact(cand, gt, col, fold=None):
    sub = cand[(cand["conf"] >= POLICY[0]) & (cand[col] >= POLICY[1])]
    g = gt
    if fold is not None:
        sub, g = sub[sub["fold"] == fold], gt[gt["fold"] == fold]
    pred = sub[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
    return score(g[["filament_id", "segmentation_rle"]], pred)


def main():
    from sklearn.metrics import roc_auc_score
    feats = list(joblib.load(BUNDLE)["features"])
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    gt["fold"] = gt["stem"].map(fold_of)
    gt = gt[gt["fold"].notna()].copy()
    gt["fold"] = gt["fold"].astype(int)
    a = pd.read_parquet(A_POOL)
    a["fold"] = a["stem"].map(fold_of).astype(int)
    stems = np.array(sorted(set(a["stem"])))
    gts = gt[gt["stem"].isin(stems)]
    ncopy = gts.groupby("stem")["entry_id"].nunique().reindex(stems).fillna(1).to_numpy()
    fold = a["fold"].to_numpy()
    log(f"pool {len(a)} rows, {len(feats)} features; monotone(+) on {sum(f in MONO_UP for f in feats)} of them")

    if "--extra" in sys.argv:
        # the 42-feature selector that sub032 shipped: re-ask the regularisation
        # question on the model that is actually deployed
        from extra_features import EXTRA
        for c in EXTRA_CACHES:
            ex = pd.read_parquet(OUT / c)
            cols = [k for k in EXTRA if k in ex.columns]
            a = a.merge(ex[["cand_id"] + cols], on="cand_id", how="left")
        feats = feats + list(EXTRA)
        log(f"extra features joined -> {len(feats)} total")
    arms = ["base", "reg", "mono+reg"] if "--extra" in sys.argv else ["base", "reg", "bag5", "mono", "mono+reg", "mono+bag"]
    res, parts = {}, {}
    for name in arms:
        a[f"p_{name}"], pm = oof(name, a, feats, fold)
        auc = roc_auc_score(a["y"], pm)
        per = [exact(a, gts, f"p_{name}", fold=f)["pq"] for f in range(5)]
        r = exact(a, gts, f"p_{name}")
        sub = a[(a["conf"] >= POLICY[0]) & (a[f"p_{name}"] >= POLICY[1])]
        st, rr = per_image(sub, gts, stems, ncopy)
        parts[name] = st
        res[name] = dict(auc=float(auc), pq=r["pq"], tp=r["tp"], fp=r["fp"], fn=r["fn"], kept=int(len(sub)), per_fold=per)
        log(f"{name:9s} OOF AUC {auc:.4f} | kept {len(sub)} PQ {r['pq']:.4f} TP {r['tp']} FP {r['fp']} FN {r['fn']} | per fold "
            + " ".join(f"{x:.4f}" for x in per))
    rng = np.random.default_rng(0)
    n = len(stems)
    log("\ndeltas against base (paired image bootstrap, 4000 reps):")
    for name in arms[1:]:
        boot = np.empty(REPS)
        for i in range(REPS):
            idx = rng.integers(0, n, n)
            v = [pq(s["tp"][idx].sum(), s["fp"][idx].sum(), s["fn"][idx].sum(), s["iou"][idx].sum()) for s in (parts["base"], parts[name])]
            boot[i] = v[1] - v[0]
        d = res[name]["pq"] - res["base"]["pq"]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        dpf = [res[name]["per_fold"][f] - res["base"]["per_fold"][f] for f in range(5)]
        verdict = "PAYS" if d >= FLOOR else "HURTS" if d <= -FLOOR else ("TIED" if lo > -EQUIV and hi < EQUIV else "UNDERPOWERED")
        res[name].update(delta=float(d), ci=[float(lo), float(hi)], p_gt0=float((boot > 0).mean()), verdict=verdict, delta_folds=dpf)
        log(f"  {name:9s} {d:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  P(>0) {(boot > 0).mean():.3f}  folds+ {sum(x > 0 for x in dpf)}/5  -> {verdict}")
    json.dump(res, open(OUT / ("selector_regularization_extra.json" if "--extra" in sys.argv else "selector_regularization.json"), "w"), indent=1, default=float)
    log(f"\nwrote {OUT / 'selector_regularization.json'}")


if __name__ == "__main__":
    main()
