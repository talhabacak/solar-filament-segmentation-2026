"""E115b -- the selector-threshold sweep of ledger §11 (p 0.55 -> 0.60,
"+0.0053 on 5/5 folds") redone with OUT-OF-FOLD selector scores.

What §11 got right: the threshold was chosen on four folds of folds_v1 and
scored on the fifth. What it got wrong: the p_sel it thresholded came from the
single deployed bundle (selector_v5_shrunkblob5prodts.joblib) applied to the
whole pool, i.e. every held-out candidate carried a probability from a model
trained on its own label. In-sample probabilities are over-separated, so
raising a cut harvests an unrealistically clean slice, and holding out images
cannot detect a leak in the score being tuned.

Here the deployed recipe (HistGradientBoosting classifier 300/0.05/31/25/1.0
seed 42 on the bundle's 32 features; positives-only IoU regressor 300/0.06;
score = p x E[iou_sum|match]; quantile map through the TRAINING scores, as
deploy does) is refit per fold of data/splits/folds_v1.parquet, so p_sel on a
fold comes from a model that never saw it. Labels: shrink 0.05 (bundle's
shrink), the same label_pool as two_stage_selector.py.

Then, per fold: the baseline policy (conf >= 0.40, p_sel >= 0.55) and the
fixed alternative (p_sel >= 0.60) are scored with the EXACT scorer; the
fold-honest grid search (argmax on the other four folds, cached-match fast
formula as in §11) is repeated and its choice is also scored exactly. Both a
paired per-fold delta and the pooled 5-fold delta are reported.

Usage: python src/eda/e115b_oof_threshold.py
"""

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
from two_stage_selector import label_pool

OUT = ROOT / "outputs" / "eda"
POOL = "fused_pool_all_blob5.parquet"
BUNDLE = ROOT / "outputs" / "selector" / "selector_v5_shrunkblob5prodts.joblib"
BASE = (0.40, 0.55)
ALT = (0.40, 0.60)


def log(*a):
    print(*a, flush=True)


def make_cls():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=25, l2_regularization=1.0, random_state=42)


def make_reg():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=42)


def exact(cand_sub, gt_df):
    pred = cand_sub[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
    return score(gt_df, pred)


def main():
    import joblib
    feats = joblib.load(BUNDLE)["features"]
    # labelling shrinks 9917 masks and runs the exact scorer once (~8 min);
    # cache it so the policy arms can be re-run in seconds
    cache = OUT / "e115b_labelled_pool.parquet"
    if cache.exists():
        cand = pd.read_parquet(cache)
        log(f"labelled pool from cache: {cache.name}")
    else:
        cand, _, _ = label_pool(POOL, 0.05, False)
        cand.to_parquet(cache, index=False)
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt = gt[gt["stem"].isin(set(cand["stem"]))]
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    cand["fold"] = cand["stem"].map(fold_of).astype(int)
    gt["fold"] = gt["stem"].map(fold_of).astype(int)
    log(f"pool {len(cand)} candidates, {cand['stem'].nunique()} stems, {len(gt)} GT rows; "
        f"positives {int(cand['y'].sum())} ({100*cand['y'].mean():.1f}%)")

    # --- out-of-fold two-stage score, deployment-faithful per fold
    X = cand[feats].to_numpy(np.float64)
    y = cand["y"].to_numpy()
    v = cand["iou_sum"].to_numpy(np.float64)
    fold = cand["fold"].to_numpy()
    p_oof = np.zeros(len(cand))
    p_ins = np.zeros(len(cand))   # the leaked version, for a side-by-side
    from sklearn.metrics import roc_auc_score
    for f in range(5):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        # The quantile reference must be built from scores of data the model did
        # NOT see, because that is what deploy does: submit_v2 maps genuinely
        # unseen test raws through the training pool's OOF raws. Building it
        # from the training folds' in-sample raws (inflated) or from the
        # held-out fold's own raws (its own distribution) both change what the
        # threshold means. So: inner leave-one-fold-out inside the four
        # training folds gives the reference.
        inner = np.zeros(len(tr))
        for g in [k for k in range(5) if k != f]:
            itr = np.where((fold != f) & (fold != g))[0]
            ite_mask = fold[tr] == g
            ci = make_cls().fit(X[itr], y[itr])
            ri = make_reg().fit(X[itr][y[itr] == 1], v[itr][y[itr] == 1])
            Xg = X[tr][ite_mask]
            inner[ite_mask] = ci.predict_proba(Xg)[:, 1] * np.clip(ri.predict(Xg), 0, None)
        ref = np.sort(inner)
        c = make_cls().fit(X[tr], y[tr])
        pos = tr[y[tr] == 1]
        r = make_reg().fit(X[pos], v[pos])
        raw_te = c.predict_proba(X[te])[:, 1] * np.clip(r.predict(X[te]), 0, None)
        p_oof[te] = np.searchsorted(ref, raw_te, side="right") / len(ref)
        log(f"  fold {f}: train {len(tr)} test {len(te)}  ref from inner OOF (n={len(ref)})  "
            f"held-out AUC {roc_auc_score(y[te], p_oof[te]):.4f}")
    c = make_cls().fit(X, y)
    r = make_reg().fit(X[y == 1], v[y == 1])
    raw = c.predict_proba(X)[:, 1] * np.clip(r.predict(X), 0, None)
    ref = np.sort(raw)
    p_ins[:] = np.searchsorted(ref, raw, side="right") / len(ref)
    cand["p_oof"], cand["p_ins"] = p_oof, p_ins
    log(f"  AUC over all candidates: OOF {roc_auc_score(y, p_oof):.4f} vs in-sample {roc_auc_score(y, p_ins):.4f}")
    for col in ("p_oof", "p_ins"):
        k55 = (cand["conf"] >= 0.40) & (cand[col] >= 0.55)
        k60 = (cand["conf"] >= 0.40) & (cand[col] >= 0.60)
        drop = k55 & ~k60
        log(f"  {col}: kept@0.55 {int(k55.sum())}, kept@0.60 {int(k60.sum())}, dropped {int(drop.sum())} "
            f"of which positives {int(cand.loc[drop, 'y'].sum())} "
            f"(FP:TP among dropped = {int((cand.loc[drop, 'y'] == 0).sum())}:{int(cand.loc[drop, 'y'].sum())})")

    # --- per-fold exact scores at the two fixed policies, OOF and in-sample
    res = {"per_fold": []}
    log(f"\n  {'fold':<5}{'score':<8}{'base PQ':>9}{'p60 PQ':>9}{'Δ':>9}{'FP':>10}{'TP':>10}")
    for col in ("p_oof", "p_ins"):
        for f in range(5):
            gf = gt[gt["fold"] == f][["filament_id", "segmentation_rle"]]
            cf = cand[cand["fold"] == f]
            b = exact(cf[(cf["conf"] >= BASE[0]) & (cf[col] >= BASE[1])], gf)
            a = exact(cf[(cf["conf"] >= ALT[0]) & (cf[col] >= ALT[1])], gf)
            res["per_fold"].append(dict(score=col, fold=f, base=b["pq"], p60=a["pq"], delta=a["pq"] - b["pq"],
                                        fp=(b["fp"], a["fp"]), tp=(b["tp"], a["tp"])))
            log(f"  {f:<5}{col:<8}{b['pq']:>9.4f}{a['pq']:>9.4f}{a['pq']-b['pq']:>+9.4f}"
                f"{b['fp']:>5}>{a['fp']:<4}{b['tp']:>5}>{a['tp']:<4}")
        d = [x["delta"] for x in res["per_fold"] if x["score"] == col]
        log(f"  {col}: mean Δ {np.mean(d):+.4f}, folds positive {sum(x > 0 for x in d)}/5, "
            f"min {min(d):+.4f} max {max(d):+.4f}")

    # --- fold-honest grid search on OOF scores (fast formula), exact on the held-out fold
    log("\n  fold-honest grid on OOF scores (argmax on 4 folds by cached-match formula, exact on the 5th):")
    # The match cache must span every candidate the grid can select, so it is
    # built at conf >= 0.40 with NO p filter. Building it on the p >= 0.55 pool
    # (the first version of this script) let the grid pick p = 0.30 and then
    # scored a pool the cache had never seen, which made that arm meaningless.
    kept = cand[cand["conf"] >= BASE[0]].copy()
    r_all = score(gt[["filament_id", "segmentation_rle"]],
                  kept[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"}),
                  return_details=True)
    m = r_all["matches"].groupby("pred_filament_id").agg(tp=("iou", "count"), iou_sum=("iou", "sum"))
    kept["tp_m"] = kept["cand_id"].map(m["tp"]).fillna(0).astype(int)
    kept["iou_m"] = kept["cand_id"].map(m["iou_sum"]).fillna(0.0)
    ncopy = gt.groupby("stem")["entry_id"].nunique()
    kept["fp_m"] = kept["stem"].map(ncopy).fillna(1).astype(int) - kept["tp_m"]
    gt_per_fold = gt.groupby("fold").size().to_dict()
    # §11's grid started at the incumbent (0.55), so it could not pick DOWN and a
    # null result was indistinguishable from a floor effect. Search both ways.
    grid = [(round(cc, 2), round(pp, 3)) for cc in np.arange(0.40, 0.85, 0.05) for pp in np.arange(0.30, 0.90, 0.025)]

    def fast(sub, n_gt):
        tp, fp, iou = sub["tp_m"].sum(), sub["fp_m"].sum(), sub["iou_m"].sum()
        fn = n_gt - tp
        return iou / (tp + 0.5 * fp + 0.5 * fn)

    res["grid"] = []
    for f in range(5):
        tr = kept[kept["fold"] != f]
        n_tr = sum(gt_per_fold[k] for k in range(5) if k != f)
        best = max(grid, key=lambda cp: fast(tr[(tr["conf"] >= cp[0]) & (tr["p_oof"] >= cp[1])], n_tr))
        gf = gt[gt["fold"] == f][["filament_id", "segmentation_rle"]]
        cf = cand[cand["fold"] == f]
        b = exact(cf[(cf["conf"] >= BASE[0]) & (cf["p_oof"] >= BASE[1])], gf)["pq"]
        a = exact(cf[(cf["conf"] >= best[0]) & (cf["p_oof"] >= best[1])], gf)["pq"]
        res["grid"].append(dict(fold=f, policy=best, base=b, chosen=a, delta=a - b))
        log(f"    fold {f}: chosen {best}  base {b:.4f} -> {a:.4f}  Δ {a-b:+.4f}")
    d = [x["delta"] for x in res["grid"]]
    log(f"    mean Δ {np.mean(d):+.4f}, folds positive {sum(x > 0 for x in d)}/5")

    # --- where is the optimum on the honest axis? exact per fold, both directions.
    # The board has already settled 0.60 > 0.55 (sub025 #1 over sub020 #3); the
    # open question is whether it keeps paying above 0.60, which is the next
    # submission decision.
    log("\n  exact PQ by p_oof cut at conf >= 0.40 (mean over folds, and per fold):")
    sweep = {}
    for pcut in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        per = []
        for f in range(5):
            gf = gt[gt["fold"] == f][["filament_id", "segmentation_rle"]]
            cf = cand[cand["fold"] == f]
            per.append(exact(cf[(cf["conf"] >= 0.40) & (cf["p_oof"] >= pcut)], gf)["pq"])
        sweep[pcut] = per
        base_per = sweep.get(0.55)
        rel = f"  Δ vs 0.55 {np.mean(per) - np.mean(base_per):+.4f}" if base_per else ""
        log(f"    p>={pcut:.2f}  mean {np.mean(per):.4f}  per fold " +
            " ".join(f"{x:.4f}" for x in per) + rel)
    res["sweep"] = {str(k): v for k, v in sweep.items()}

    # --- pooled 5-fold exact (single number the board would see)
    gt_df = gt[["filament_id", "segmentation_rle"]]
    for col in ("p_oof", "p_ins"):
        b = exact(cand[(cand["conf"] >= BASE[0]) & (cand[col] >= BASE[1])], gt_df)
        a = exact(cand[(cand["conf"] >= ALT[0]) & (cand[col] >= ALT[1])], gt_df)
        res[f"pooled_{col}"] = dict(base=b["pq"], p60=a["pq"], delta=a["pq"] - b["pq"])
        log(f"\n  pooled 5-fold exact, {col}: base {b['pq']:.4f} (TP {b['tp']} FP {b['fp']} FN {b['fn']}) -> "
            f"p60 {a['pq']:.4f} (TP {a['tp']} FP {a['fp']} FN {a['fn']})  Δ {a['pq']-b['pq']:+.4f}")
    json.dump(res, open(OUT / "e115b_oof_threshold.json", "w"), indent=1, default=float)
    log(f"\nwrote {OUT/'e115b_oof_threshold.json'}")


if __name__ == "__main__":
    main()
