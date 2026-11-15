"""E189 -- the selector's LOSS, which has never been changed.

Talha, 2026-09-14: "how is the selector's loss set up?" As of today:

    score = P(y = 1 | x)  x  E[iou_sum | x, y = 1]
            HGB classifier         HGB regressor
            log-loss, UNWEIGHTED   squared error, fitted on POSITIVES ONLY

Both are sklearn defaults with no sample weights, which means a candidate worth 1.6 copies of
IoU counts exactly as much in the loss as one worth 0.55 -- while the metric does not treat them
equally at all. PQ = sum IoU / (TP + FP/2 + FN/2) makes the net value of keeping a candidate

    value = iou_sum - 0.5 x copies x PQ

(the break-even derived in ledger 23d and confirmed calibrated in E176). Three formulations are
compared under the identical deployment-faithful protocol:

  A  baseline            unweighted classifier x positives-only regressor  (production)
  B  value-weighted      the same two models, each row weighted by |value| -- the loss now
                         counts PQ at stake rather than rows, so a marginal candidate whose
                         decision changes almost nothing stops competing for capacity with one
                         that carries 1.6 copies
  C  direct regression   one HGB regressor on ALL rows with target iou_sum (0 for negatives),
                         i.e. estimate E[iou_sum] in one step instead of factorising it

C is the formulation the metric literally asks for; A factorises it (which helps when positives
are rare, and they are 41%); B keeps A's structure but fixes the weighting. Whichever wins has
to win by more than the +/-0.003 floor to be adopted.

Usage: python src/eda/selector_loss_variants.py
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
from filament.pq import score  # noqa: E402
from e115c_threshold_ci import per_image, pq  # noqa: E402
from extra_features import EXTRA  # noqa: E402

OUT = ROOT / "outputs" / "eda"
PQ0, NCOPY = 0.4421, 1.635          # the deployed operating point's own PQ and mean copies


def log(*a):
    print(*a, flush=True)


def cls_(w=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=25, l2_regularization=1.0, random_state=42)


def reg_():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=42)


def fit_score(trn, tst, feats, variant):
    X = trn[feats].to_numpy(np.float64)
    y = trn["y"].to_numpy()
    v = trn["iou_sum"].to_numpy(np.float64)
    Z = tst[feats].to_numpy(np.float64)
    if variant == "A":
        c = cls_().fit(X, y)
        r = reg_().fit(X[y == 1], v[y == 1])
        return c.predict_proba(Z)[:, 1] * np.clip(r.predict(Z), 0, None)
    if variant == "B":
        w = np.abs(v - 0.5 * NCOPY * PQ0)
        w = np.clip(w / max(w.mean(), 1e-9), 0.05, 20.0)
        c = cls_().fit(X, y, sample_weight=w)
        pos = y == 1
        r = reg_().fit(X[pos], v[pos], sample_weight=w[pos])
        return c.predict_proba(Z)[:, 1] * np.clip(r.predict(Z), 0, None)
    if variant == "C":
        r = reg_().fit(X, v)          # every row, target iou_sum (0 for negatives)
        return np.clip(r.predict(Z), 0, None)
    raise ValueError(variant)


def main():
    feats = list(joblib.load(ROOT / "outputs/selector/selector_v5_shrunkblob5prodts.joblib")["features"]) + list(EXTRA)
    folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    pool = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    pool = pool.merge(pd.read_parquet(OUT / "extra_features_t020.parquet")[["cand_id"] + list(EXTRA)],
                      on="cand_id", how="left")
    pool["fold"] = pool["stem"].map(fold_of).astype(int)
    stems = np.array(sorted(fold_of))
    ncopy = gt.groupby("stem")["entry_id"].nunique().reindex(stems).fillna(1).to_numpy()

    res, per = {}, {}
    for variant in ("A", "B", "C"):
        sel = []
        for F in range(5):
            trn, tst = pool[pool["fold"] != F], pool[pool["fold"] == F].copy()
            inner = np.zeros(len(trn))
            for g in [k for k in range(5) if k != F]:
                itr, msk = trn["fold"] != g, trn["fold"] == g
                inner[msk.to_numpy()] = fit_score(trn[itr], trn[msk], feats, variant)
            ref = np.sort(inner)
            raw = fit_score(trn, tst, feats, variant)
            tst["p"] = np.searchsorted(ref, raw, side="right") / len(ref)
            sel.append(tst[(tst["conf"] >= 0.40) & (tst["p"] >= 0.60)])
        sub = pd.concat(sel)
        r = score(gt[["filament_id", "segmentation_rle"]],
                  sub[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"}))
        res[variant], per[variant] = r, per_image(sub, gt, stems, ncopy)[0]
        log(f"{variant}: kept {len(sub):5d}  PQ {r['pq']:.4f}  SQ {r['sq']:.4f}  RQ {r['rq']:.4f}  "
            f"TP {r['tp']} FP {r['fp']} FN {r['fn']}")
    rng = np.random.default_rng(0)
    n = len(stems)
    out = {}
    for v in ("B", "C"):
        pa, pb = per["A"], per[v]
        boot = np.empty(4000)
        for i in range(4000):
            idx = rng.integers(0, n, n)
            boot[i] = (pq(pb["tp"][idx].sum(), pb["fp"][idx].sum(), pb["fn"][idx].sum(), pb["iou"][idx].sum())
                       - pq(pa["tp"][idx].sum(), pa["fp"][idx].sum(), pa["fn"][idx].sum(), pa["iou"][idx].sum()))
        d = res[v]["pq"] - res["A"]["pq"]
        log(f"  A -> {v}: delta {d:+.4f}  95% CI [{np.quantile(boot, .025):+.4f}, "
            f"{np.quantile(boot, .975):+.4f}]  P(delta>0) {(boot > 0).mean():.3f}")
        out[v] = dict(delta=float(d), lo=float(np.quantile(boot, .025)), hi=float(np.quantile(boot, .975)))
    json.dump(dict(pq={k: float(v["pq"]) for k, v in res.items()}, paired=out),
              open(OUT / "selector_loss_variants.json", "w"), indent=1)
    log("LOSS_VARIANTS_DONE")


if __name__ == "__main__":
    main()
