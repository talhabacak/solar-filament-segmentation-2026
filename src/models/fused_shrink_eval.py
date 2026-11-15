"""Do mask shrinking and the selector stack, or do they eat each other?

Both were measured on different pools: shrinking on the detector pool
(+0.0128 pooled) and selector v5 on the un-shrunk fused pool (+0.0297). Shrinking
changes which candidates clear IoU>0.5, so it moves the selector's own labels —
the combined effect has to be measured, not added.

This shrinks every fused candidate, re-labels the pool with the exact scorer,
retrains the selector out-of-fold and cross-fits the keep policy, so the final
number is directly comparable to selector_v5's 0.4251.

Shrinking after de-overlap can only remove pixels, so pixel-disjointness (a
submission requirement) is preserved.

Usage: python src/models/fused_shrink_eval.py --shrink 0.05
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament.pq import score
from filament.rle import decode_rle, encode_mask
from selector import fast_pq
from selector_v5 import BASE, SRC, CROSS, make_model, oof_probs, crossfit_gain, log

OUT = ROOT / "outputs" / "selector"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shrink", type=float, default=0.05)
    ap.add_argument("--pool", default="fused_pool_all.parquet",
                    help="pool parquet under outputs/selector/ to evaluate")
    ap.add_argument("--tag", default="",
                    help="suffix for the result json, so variants do not clobber")
    ap.add_argument("--two-stage", action="store_true",
                    help="rank by p(match) x E[iou_sum|match] rather than "
                         "p(match); see the note at the fit site")
    ap.add_argument("--true-frac", action="store_true",
                    help="use the rank-based shrink whose parameter is the area "
                         "fraction actually removed; see mask_shrink.shrink_true_frac")
    ap.add_argument("--extra-feats", default="",
                    help="comma-separated extra feature columns the pool carries "
                         "(E125 photometry+orientation: img_contrast,dark_core,fg_p05,"
                         "ring_med,coh_in,coh_ring,tex_align,ring_align,grad_norm,aniso_ratio). "
                         "They must be computed by src/models/extra_features.py on the SHRUNK "
                         "mask, which is what submit_v2 does at deployment.")
    ap.add_argument("--drop", default="",
                    help="comma-separated selector features to withhold; see the "
                         "note at the drop site for why any would be withheld")
    a = ap.parse_args()
    from mask_shrink import shrink_frac, shrink_true_frac
    # shrink_frac's parameter is inert below 0.19 (it removes a whole
    # tied boundary ring regardless); --true-frac uses the rank-based
    # version whose fraction is the fraction of area actually removed
    shrink = shrink_true_frac if a.true_frac else shrink_frac

    cand = pd.read_parquet(OUT / a.pool)
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    files = set(cand["stem"] + ".jpeg")
    gt_sub = gt[gt["file_name"].isin(files)]
    total_gt = len(gt_sub)
    gt_stems = gt_sub["file_name"].str.replace(".jpeg", "", regex=False)

    t0 = time.time()
    new_rle, areas = [], []
    for r in cand.itertuples():
        m = decode_rle(r.rle).astype(bool)
        m2 = shrink(m, a.shrink)
        new_rle.append(encode_mask(m2.astype(np.uint8)))
        areas.append(int(m2.sum()))
    cand["rle"] = new_rle
    cand["area"] = areas
    cand["log10_area"] = np.log10(np.maximum(cand["area"], 1))
    log(f"shrunk {len(cand)} masks by {a.shrink:.0%} ({time.time()-t0:.0f}s)")

    pred_df = cand[["cand_id", "rle"]].rename(
        columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
    res = score(gt_sub[["filament_id", "segmentation_rle"]], pred_df, return_details=True)
    log(f"ALL shrunk fused: PQ={res['pq']:.4f} TP={res['tp']} FP={res['fp']} FN={res['fn']}")
    m = res["matches"]
    agg = m.groupby("pred_filament_id").agg(weight=("gt_filament_id", "size"),
                                            n_entries_hit=("annotator_image", "nunique"),
                                            iou_sum=("iou", "sum"))
    for c in ["weight", "n_entries_hit", "iou_sum"]:
        cand[c] = cand["cand_id"].map(agg[c]).fillna(0)
    cand["y"] = (cand["weight"] >= 1).astype(int)
    log(f"labels: {int(cand['y'].sum())}/{len(cand)} positive")

    base_pol = (0.40, 128)
    ref = fast_pq(cand[(cand["conf"] >= base_pol[0]) & (cand["area"] >= base_pol[1])], total_gt)
    orc = fast_pq(cand[cand["y"] == 1], total_gt)
    log(f"conf/area baseline (shrunk): PQ {ref['pq']:.4f}")
    log(f"oracle (shrunk)            : PQ {orc['pq']:.4f}")

    feats = BASE + SRC + CROSS
    if a.extra_feats:
        extra = [x.strip() for x in a.extra_feats.split(",") if x.strip()]
        missing = [f for f in extra if f not in cand.columns]
        if missing:
            raise SystemExit(f"pool lacks extra feature(s): {missing}")
        feats = feats + extra
        log(f"extra features: +{len(extra)} -> {len(feats)} total")
    if a.drop:
        # Train/deploy definitional mismatch, not a distribution shift. The
        # selector trains on a pool built from ONE detector cache per fold
        # (build_fused_pool.py, fold{f}_preds_e013) because the other four models
        # saw fold 0, but the submission concatenates FIVE (submit_v2.py). So
        # n_det_models counts one model's duplicate boxes here and five-model
        # cross-model consensus there; det_conf_std is exactly 0.0 on 85% of
        # training rows and member_iou_min exactly 1.0 on 46%. The same
        # redefinition reaches the per-image activity features through the
        # candidate count. It cannot be fixed at source without leakage, so the
        # only remedy is to stop feeding the selector columns that mean
        # different things in the two places.
        drop = set(x.strip() for x in a.drop.split(",") if x.strip())
        unknown = drop - set(feats)
        if unknown:
            raise SystemExit(f"unknown feature(s) to drop: {sorted(unknown)}")
        feats = [f for f in feats if f not in drop]
        log(f"dropping {len(drop)} features -> {len(feats)} remain: "
            f"{sorted(drop)}")
    cand["p_sel"] = oof_probs(cand, feats)
    if a.two_stage:
        # Rank by p(match) x E[iou_sum | match] instead of p(match). The metric
        # does not pay in matches: keeping a candidate adds its summed IoU and
        # charges 0.5 per annotator entry it misses, so two candidates with equal
        # P(match) are worth different amounts and a classifier cannot say so.
        # The expectation is fit on POSITIVES ONLY -- iou_sum is 58.2% exact
        # zeros and fitting it unconditionally with a squared-error loss is the
        # form already measured to lose (-0.005 to -0.057).
        # Verified leak-free (thresholds from the training half only) over six
        # seeds x 20 paired splits: +0.0010, seed-to-seed se 0.0002, 71/120
        # splits won. See src/eda/two_stage_selector.py.
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.model_selection import GroupKFold
        X = cand[feats].to_numpy(np.float64)
        v = cand["iou_sum"].to_numpy(np.float64)
        yv = cand["y"].to_numpy()
        g = cand["stem"].to_numpy()
        e = np.zeros(len(cand))
        for tr, te in GroupKFold(n_splits=5).split(X, yv, g):
            pos = tr[yv[tr] == 1]
            r = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06)
            r.fit(X[pos], v[pos])
            e[te] = np.clip(r.predict(X[te]), 0.0, None)
        raw = cand["p_sel"].to_numpy() * e
        # The keep-threshold grid downstream is a fixed arange(0.05, 0.70, 0.05)
        # built for a probability, while this product runs to about 2 --
        # E[iou_sum|match] exceeds 1 whenever a candidate matches more than one
        # annotator entry. A plain rescale is not enough either: the product's
        # DISTRIBUTION is nothing like a probability's, so a 0.05-step grid still
        # lands in the wrong places. Measured: the coarse grid put this arm at
        # 0.4328 against the classifier's 0.4331, while the same arm with
        # training-half quantile thresholds scores +0.0011 over it, paired, on
        # four seeds of twenty splits.
        #
        # Mapping the score through its own empirical CDF makes it uniform on
        # [0,1], so the fixed grid becomes exactly a quantile grid and the
        # downstream code needs no change. The transform is monotone, so nothing
        # about the ranking moves. The sorted training scores are stored in the
        # bundle because deploy must use the TRAINING quantiles -- the test pool
        # has its own distribution, and re-deriving the mapping there would give
        # the threshold a different meaning than the one it was tuned with.
        cdf_ref = np.sort(raw)
        cand["p_sel"] = np.searchsorted(cdf_ref, raw, side="right") / len(cdf_ref)
        log(f"two-stage score in use: p(match) x E[iou_sum|match], "
            f"E ranges {e.min():.3f}-{e.max():.3f}, mapped through its own CDF "
            f"so the fixed grid acts as a quantile grid")
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(cand["y"], cand["p_sel"])
    gm, gs, apq = crossfit_gain(cand, "p_sel", total_gt, gt_stems, base_pol)
    best, best_pq = None, -1.0
    for c in (0.0, 0.2, 0.25, 0.3, 0.35, 0.4):
        for p in np.arange(0.05, 0.70, 0.05):
            k = cand[(cand["conf"] >= c) & (cand["p_sel"] >= p)]
            if k.empty:
                continue
            q = fast_pq(k, total_gt)["pq"]
            if q > best_pq:
                best, best_pq = (c, round(float(p), 2)), q
    log(f"selector v5 on shrunk pool: AUC {auc:.4f} | cross-fitted {gm:+.4f} +/- {gs:.4f} "
        f"(abs PQ {apq:.4f}) | in-sample best conf>={best[0]} p>={best[1]} -> {best_pq:.4f}")

    X = cand[feats].to_numpy(np.float64)
    bundle = {"model": make_model().fit(X, cand["y"].to_numpy()), "features": feats,
              "shrink": a.shrink, "policy": {"conf": best[0], "p": best[1]},
              "two_stage": bool(a.two_stage),
              "two_stage_cdf": (cdf_ref if a.two_stage else None)}
    if a.two_stage:
        from sklearn.ensemble import HistGradientBoostingRegressor
        pos = cand["y"].to_numpy() == 1
        bundle["iou_model"] = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06).fit(
                X[pos], cand.loc[pos, "iou_sum"].to_numpy(np.float64))
    joblib.dump(bundle, OUT / f"selector_v5_shrunk{a.tag}.joblib")
    with open(OUT / f"fused_shrink_results{a.tag}.json", "w") as f:
        json.dump(dict(shrink=a.shrink, baseline=ref, oracle=orc, auc=float(auc),
                       crossfit_gain=gm, crossfit_abs_pq=apq,
                       insample_pq=best_pq, policy={"conf": best[0], "p": best[1]}),
                  f, indent=1, default=float)
    log(f"saved -> {OUT / f'selector_v5_shrunk{a.tag}.joblib'}")


if __name__ == "__main__":
    main()
