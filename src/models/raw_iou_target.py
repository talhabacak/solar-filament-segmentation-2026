"""Regress a smooth IoU, not the discontinuous value the metric pays.

Every selector so far has capped near Spearman 0.68, and the target may be why.
`value = iou_sum / n_copies` is what PQ actually pays, but it is a step function
of the underlying overlap: a candidate at IoU 0.49 is worth 0 and one at 0.51 is
worth 0.51. Two nearly identical masks get labels an order of magnitude apart,
which is close to unlearnable at the boundary — and the boundary is exactly
where the selection decisions are.

The overlap itself is smooth. This records, for every candidate and without any
threshold:

    iou_mean   mean over the image's annotator entries of the best IoU with any
               GT in that entry
    iou_max    the best of those
    frac_over  fraction of entries where that best exceeds 0.5

and then compares two ways of ranking:

    direct     regress `value` (what v6 does)
    smooth     regress `iou_mean`, then rank by the implied value
               E[value] ~= iou_mean * P(over threshold), with P estimated by a
               second, equally smooth regression of frac_over

Both are fit out-of-fold by image and scored with the exact scorer after
de-overlap, so the comparison is like for like.

Usage: python src/models/raw_iou_target.py --pool outputs/selector/fused_pool_all.parquet
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament.pq import _entry_iou_matrix
from filament.rle import rles_to_layers
from selector_v5 import BASE, SRC, CROSS, log
from selector_v6 import make_regressor
from eval_nodeov import select_and_deoverlap, solve

OUT = ROOT / "outputs" / "selector"
FEATS = BASE + SRC + CROSS


def raw_targets(cand, gt_sub):
    """Best IoU per candidate per annotator entry, with no threshold applied."""
    gt_entry = gt_sub["filament_id"].str.split("_", n=1).str[0]
    best = {c: [] for c in cand["cand_id"]}
    t0 = time.time()
    for n, entry in enumerate(gt_entry.unique()):
        image_id = entry.split("-", maxsplit=1)[1]
        gt_sel = gt_sub.loc[gt_entry == entry]
        pred_sel = cand.loc[cand["stem"] == image_id]
        if gt_sel.empty or pred_sel.empty:
            continue
        gl = rles_to_layers(gt_sel["segmentation_rle"].tolist())
        pl = rles_to_layers(pred_sel["rle"].tolist())
        iou = _entry_iou_matrix(gl, pl)            # (n_gt, n_pred)
        per_pred = iou.max(axis=0)
        for cid, v in zip(pred_sel["cand_id"], per_pred):
            best[cid].append(float(v))
        if (n + 1) % 200 == 0:
            log(f"  {n+1} entries ({time.time()-t0:.0f}s)")
    rows = []
    for cid, vals in best.items():
        a = np.array(vals) if vals else np.zeros(1)
        rows.append(dict(cand_id=cid, iou_mean=float(a.mean()),
                         iou_max=float(a.max()),
                         frac_over=float((a > 0.5).mean())))
    return pd.DataFrame(rows)


def oof_col(cand, target, feats=FEATS):
    from sklearn.model_selection import GroupKFold
    X = cand[feats].to_numpy(np.float64)
    y = cand[target].to_numpy(np.float64)
    g = cand["stem"].to_numpy()
    p = np.full(len(cand), np.nan)
    for tr, va in GroupKFold(n_splits=5).split(X, y, g):
        p[va] = make_regressor().fit(X[tr], y[tr]).predict(X[va])
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="outputs/selector/fused_pool_all.parquet")
    a = ap.parse_args()

    cand = pd.read_parquet(ROOT / a.pool)
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt_sub = gt[gt["file_name"].isin(set(cand["stem"] + ".jpeg"))]
    cand["value"] = cand["iou_sum"] / cand["n_copies"].clip(lower=1)

    cache = OUT / (Path(a.pool).stem + "_rawiou.parquet")
    if cache.exists():
        raw = pd.read_parquet(cache)
        log(f"reusing raw IoU targets from {cache.name}")
    else:
        raw = raw_targets(cand, gt_sub)
        raw.to_parquet(cache, index=False)
    cand = cand.merge(raw, on="cand_id", how="left")
    log(f"pool {len(cand)}; iou_mean>0 for {int((cand['iou_mean']>0).sum())}, "
        f"value>0 for {int((cand['value']>0).sum())}")

    from scipy.stats import spearmanr
    cand["p_direct"] = oof_col(cand, "value")
    cand["p_iou"] = oof_col(cand, "iou_mean")
    cand["p_frac"] = oof_col(cand, "frac_over")
    cand["p_smooth"] = cand["p_iou"] * cand["p_frac"]

    for col in ("p_direct", "p_iou", "p_smooth"):
        log(f"{col:9s} Spearman vs value {spearmanr(cand[col], cand['value']).statistic:.4f}"
            f" | vs iou_mean {spearmanr(cand[col], cand['iou_mean']).statistic:.4f}")

    gt_cache, res = {}, {}
    for col in ("p_direct", "p_smooth"):
        thr, r, df = solve(cand, col, gt_sub, gt_cache)
        log(f"{col:9s} PQ {r['pq']:.4f} SQ {r['sq']:.4f} TP {r['tp']:5d} "
            f"FP {r['fp']:5d} FN {r['fn']:5d} (kept {len(df)}, thr {thr:.4f})")
        res[col] = dict(pq=r["pq"], sq=r["sq"], tp=r["tp"], fp=r["fp"], fn=r["fn"],
                        kept=len(df), thr=thr)

    with open(OUT / "raw_iou_target.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    log(f"saved -> {OUT/'raw_iou_target.json'}")


if __name__ == "__main__":
    main()
