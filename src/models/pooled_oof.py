"""Pooled 5-fold OOF evaluation + selector v2 (all-folds training data).

Each fold model predicts its own val fold; the union covers all 707 train files.
One exact-scorer pass against the FULL gt (1154 annotator entries) gives the
honest pooled OOF PQ — the gold-standard number for model selection.

Outputs: outputs/selector/pooled_candidates.parquet, selector_v2.joblib,
pooled_results.json. Policy grid evaluated with the verified fast-PQ
decomposition (exact for pixel-disjoint predictions).
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament import IMG_H, IMG_W
from filament.pq import score
from filament.rle import decode_rle, encode_mask
from selector import FEATURES, mask_features, fast_pq

R_MAX = 0.97
CACHE_DIR = ROOT / "outputs" / "oof_cache"
OUT = ROOT / "outputs" / "selector"


def log(*a):
    print(*a, flush=True)


def build_all_candidates():
    rows = []
    for fold in range(5):
        preds = pd.read_parquet(CACHE_DIR / f"fold{fold}_preds_e013.parquet")
        preds = preds[(preds["conf"] >= 0.05) & (preds["r_norm"] <= R_MAX)]
        for stem, g in preds.groupby("stem"):
            g = g.sort_values("conf", ascending=False)
            taken = np.zeros((IMG_H, IMG_W), dtype=bool)
            k = 0
            start = len(rows)
            for r in g.itertuples():
                mm = decode_rle(r.rle).astype(bool) & ~taken
                if not mm.any():
                    continue
                taken |= mm
                k += 1
                f = mask_features(mm)
                rows.append(dict(stem=stem, fold=fold, cand_id=f"{stem}_{k}",
                                 conf=float(r.conf),
                                 rle_deov=encode_mask(mm.astype(np.uint8)), **f))
            for i in range(start, len(rows)):
                rows[i]["n_cands_img"] = k
        log(f"fold {fold}: cumulative candidates {len(rows)}")
    df = pd.DataFrame(rows)
    df["rank_conf"] = df.groupby("stem")["conf"].rank(ascending=False, method="first")
    df["log10_area"] = np.log10(df["area"])
    return df


def main():
    t0 = time.time()
    cand = build_all_candidates()
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt_all = gt[["filament_id", "segmentation_rle", "entry_id", "file_name"]]

    # one exact-scorer pass labels everything
    pred_df = cand[["cand_id", "rle_deov"]].rename(
        columns={"cand_id": "filament_id", "rle_deov": "segmentation_rle"})
    res = score(gt_all[["filament_id", "segmentation_rle"]], pred_df, return_details=True)
    log(f"ALL candidates pooled: PQ={res['pq']:.4f} TP={res['tp']} FP={res['fp']} "
        f"FN={res['fn']} ({time.time()-t0:.0f}s)")
    m = res["matches"]
    agg = m.groupby("pred_filament_id").agg(
        weight=("gt_filament_id", "size"),
        n_entries_hit=("annotator_image", "nunique"),
        iou_sum=("iou", "sum"))
    cand = cand.merge(agg, left_on="cand_id", right_index=True, how="left")
    for c in ["weight", "n_entries_hit", "iou_sum"]:
        cand[c] = cand[c].fillna(0)
    cand["weight"] = cand["weight"].astype(int)
    cand["n_entries_hit"] = cand["n_entries_hit"].astype(int)
    cand["y"] = (cand["weight"] >= 1).astype(int)
    n_copies = gt.groupby("file_name")["entry_id"].nunique()
    stem2copies = {fn.rsplit(".", 1)[0]: int(v) for fn, v in n_copies.items()}
    cand["n_copies"] = cand["stem"].map(stem2copies)
    total_gt = len(gt_all)

    # baselines (fold-agnostic thresholds)
    results = {"n_candidates": int(len(cand)), "total_gt": int(total_gt),
               "baselines": {}, "selector": {}}
    for conf_t in [0.30, 0.35, 0.40]:
        kept = cand[(cand["conf"] >= conf_t) & (cand["area"] >= 128)]
        r = fast_pq(kept, total_gt)
        results["baselines"][f"conf{conf_t}_area128"] = r
        log(f"baseline conf>={conf_t}/area128: PQ={r['pq']:.4f} "
            f"TP={r['tp']} FP={r['fp']} FN={r['fn']}")

    # selector v2: honest OOF probs via GroupKFold by file
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    X = cand[FEATURES].to_numpy(np.float64)
    y = cand["y"].to_numpy()
    groups = cand["stem"].to_numpy()

    def make_model():
        return HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=25, l2_regularization=1.0,
            early_stopping=False, random_state=42)

    p_oof = np.full(len(cand), np.nan)
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
        p_oof[va] = make_model().fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    auc = roc_auc_score(y, p_oof)
    cand["p_sel"] = p_oof
    log(f"selector v2 OOF AUC = {auc:.4f}  ({len(cand)} candidates)")
    results["selector"]["oof_auc"] = float(auc)

    # policy grid with nested honesty: choose policy on 4/5 files, eval on 1/5
    grid = [(c, p) for c in [0.20, 0.25, 0.30, 0.35] for p in np.arange(0.05, 0.55, 0.05)]
    files = np.array(sorted(cand["stem"].unique()))
    rng = np.random.default_rng(2026)
    gains = []
    for rep in range(10):
        perm = rng.permutation(len(files))
        half_a = set(files[perm[:len(files) // 2]])
        for train_half, eval_half in [(half_a, set(files) - half_a),
                                      (set(files) - half_a, half_a)]:
            ctr = cand[cand["stem"].isin(train_half)]
            cev = cand[cand["stem"].isin(eval_half)]
            gt_tr = int(gt_all["file_name"].str.replace(".jpeg", "", regex=False)
                        .isin(train_half).sum())
            gt_ev = total_gt - gt_tr
            best, best_pq = None, -1
            for c, p in grid:
                k = ctr[(ctr["conf"] >= c) & (ctr["p_sel"] >= p)]
                pq = fast_pq(k, gt_tr)["pq"]
                if pq > best_pq:
                    best, best_pq = (c, p), pq
            k_ev = cev[(cev["conf"] >= best[0]) & (cev["p_sel"] >= best[1])]
            base_ev = cev[(cev["conf"] >= 0.40) & (cev["area"] >= 128)]
            gains.append(fast_pq(k_ev, gt_ev)["pq"] - fast_pq(base_ev, gt_ev)["pq"])
    results["selector"]["cv_gain_mean"] = float(np.mean(gains))
    results["selector"]["cv_gain_std"] = float(np.std(gains))
    log(f"cross-fitted selector gain vs conf.40/area128: "
        f"{np.mean(gains):+.4f} +/- {np.std(gains):.4f} (n={len(gains)})")

    # in-sample best policy (for reporting + test-time default)
    best, best_pq = None, -1
    for c, p in grid:
        k = cand[(cand["conf"] >= c) & (cand["p_sel"] >= p)]
        pq = fast_pq(k, total_gt)["pq"]
        if pq > best_pq:
            best, best_pq = (c, p), pq
    k = cand[(cand["conf"] >= best[0]) & (cand["p_sel"] >= best[1])]
    results["selector"]["best_policy"] = {"conf_min": best[0], "p_min": round(float(best[1]), 2)}
    results["selector"]["best_policy_pq"] = fast_pq(k, total_gt)
    log(f"pooled best policy conf>={best[0]} & p>={best[1]:.2f}: "
        f"PQ={best_pq:.4f}")

    # oracle
    results["oracle"] = fast_pq(cand[cand["y"] == 1], total_gt)
    log(f"oracle: PQ={results['oracle']['pq']:.4f}")

    final_model = make_model().fit(X, y)
    joblib.dump({"model": final_model, "features": FEATURES}, OUT / "selector_v2.joblib")
    cand.drop(columns=["rle_deov"]).to_parquet(OUT / "pooled_candidates.parquet", index=False)
    with open(OUT / "pooled_results.json", "w") as f:
        json.dump(results, f, indent=1, default=str)
    log(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
