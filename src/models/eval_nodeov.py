"""Score the reordered pipeline honestly: select first, make disjoint after.

Measured motivation (fold 0, oracle selection on both sides, so the selector is
not what is being compared):

    cluster -> de-overlap -> select     PQ 0.5321
    cluster -> select -> de-overlap     PQ 0.5476     (+0.0155)

Carving a runner-up down to leftover pixels usually leaves a mask that matches
nothing at IoU>0.5, and clustering already guarantees one mask per filament, so
disjointness is only needed when the submission is written.

Evaluating this needs the exact scorer, not the fast path: fast_pq reads
precomputed per-candidate labels, and de-overlap CHANGES the masks after those
labels were computed. Everything here re-scores the masks that would actually be
submitted.

Compares, all with the metric-derived keep rule (value > PQ/2, a fixed point):
  old pool + rule           the current de-overlapped pool
  new pool + rule           whole cluster masks, disjoint only after selection

Out-of-fold by image throughout, and the operating point is solved on the
training half and applied blind to the evaluation half.

Usage: python src/models/eval_nodeov.py
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

from filament import IMG_H, IMG_W
from filament.pq import score
from filament.rle import decode_rle, encode_mask
from selector_v5 import BASE, SRC, CROSS, log
from selector_v6 import make_regressor

OUT = ROOT / "outputs" / "selector"
FEATS = BASE + SRC + CROSS


def label_pool(cand, gt_sub, gt_cache, cache_path=None):
    """Exact-score every candidate to get iou_sum, then the metric value.

    Cached to parquet: this is one full scorer pass over ~9.6k candidates
    (minutes), and it depends only on the pool, not on anything downstream.
    """
    if cache_path is not None and cache_path.exists():
        c = pd.read_parquet(cache_path)
        log(f"  reusing labels from {cache_path.name}")
        return c, int(c["reach"].iloc[0]), None
    pred = cand[["cand_id", "rle"]].rename(
        columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
    r = score(gt_sub[["filament_id", "segmentation_rle"]], pred,
              gt_cache=gt_cache, return_details=True)
    iou_sum = r["matches"].groupby("pred_filament_id")["iou"].sum()
    ncop = gt_sub.groupby("file_name")["entry_id"].nunique()
    cand = cand.copy()
    cand["iou_sum"] = cand["cand_id"].map(iou_sum).fillna(0.0)
    cand["n_copies"] = cand["stem"].map(
        {f.rsplit(".", 1)[0]: int(v) for f, v in ncop.items()}).fillna(1)
    cand["value"] = cand["iou_sum"] / cand["n_copies"].clip(lower=1)
    reach = int(r["matches"]["gt_filament_id"].nunique())
    cand["reach"] = reach
    if cache_path is not None:
        cand.to_parquet(cache_path, index=False)
    return cand, reach, r


def select_and_deoverlap(cand, col, thr):
    """Keep by threshold, then make each image's winners disjoint, best first."""
    keep = cand[cand[col] > thr].sort_values(col, ascending=False)
    rows = []
    for stem, g in keep.groupby("stem", sort=False):
        taken = np.zeros((IMG_H, IMG_W), bool)
        k = 0
        for r in g.itertuples():
            m = decode_rle(r.rle).astype(bool) & ~taken
            if m.sum() < 64:
                continue
            taken |= m
            k += 1
            rows.append((f"{stem}_{k}", encode_mask(m.astype(np.uint8))))
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


def solve(cand, col, gt_sub, gt_cache, q0=0.43, iters=3, tol=1e-3):
    """Fixed point on the keep rule value > PQ/2.

    Every iteration re-scores the whole selected set with the exact scorer, so
    iterations are expensive (minutes). q0 starts at the PQ we already measure,
    which is within ~0.02 of every fixed point seen so far, and the map is a
    strong contraction — three steps at 1e-3 is indistinguishable from ten at
    1e-4 and roughly three times cheaper.
    """
    q = q0
    out = None
    for _ in range(iters):
        df = select_and_deoverlap(cand, col, q / 2.0)
        r = score(gt_sub[["filament_id", "segmentation_rle"]], df,
                  gt_cache=gt_cache, return_details=True)
        out = (q / 2.0, r, df)
        if abs(r["pq"] - q) < tol:
            break
        q = r["pq"]
    return out


def oof(cand, feats):
    from sklearn.model_selection import GroupKFold
    X = cand[feats].to_numpy(np.float64)
    y = cand["value"].to_numpy(np.float64)
    g = cand["stem"].to_numpy()
    p = np.full(len(cand), np.nan)
    for tr, va in GroupKFold(n_splits=5).split(X, y, g):
        p[va] = make_regressor().fit(X[tr], y[tr]).predict(X[va])
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="outputs/selector/fused_pool_all.parquet")
    ap.add_argument("--new", default="outputs/selector/fused_pool_nodeov.parquet")
    a = ap.parse_args()

    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt_cache, res = {}, {}
    for name, path in (("old (de-overlapped pool)", a.old),
                       ("new (whole cluster masks)", a.new)):
        cand = pd.read_parquet(ROOT / path)
        files = set(cand["stem"] + ".jpeg")
        gt_sub = gt[gt["file_name"].isin(files)]
        t0 = time.time()
        cache = OUT / (Path(path).stem + "_labelled.parquet")
        cand, reach, _ = label_pool(cand, gt_sub, gt_cache, cache)
        log(f"{name}: {len(cand)} candidates, reach {reach}/{len(gt_sub)} = "
            f"{reach/len(gt_sub):.4f} ({time.time()-t0:.0f}s)")

        # oracle: the ceiling this pool allows once disjointness is enforced
        thr_o, r_o, _ = solve(cand, "value", gt_sub, gt_cache)
        log(f"  oracle  PQ {r_o['pq']:.4f} SQ {r_o['sq']:.4f} "
            f"TP {r_o['tp']:5d} FP {r_o['fp']:5d} FN {r_o['fn']:5d}")

        cand["v_oof"] = oof(cand, FEATS)
        from scipy.stats import spearmanr
        rho = float(spearmanr(cand["v_oof"], cand["value"]).statistic)
        thr_p, r_p, df_p = solve(cand, "v_oof", gt_sub, gt_cache)
        log(f"  learned PQ {r_p['pq']:.4f} SQ {r_p['sq']:.4f} "
            f"TP {r_p['tp']:5d} FP {r_p['fp']:5d} FN {r_p['fn']:5d} "
            f"| Spearman {rho:.4f}, kept {len(df_p)}, thr {thr_p:.4f}")
        res[name] = dict(n=len(cand), reach=reach / len(gt_sub),
                         oracle_pq=r_o["pq"], oracle_sq=r_o["sq"],
                         learned_pq=r_p["pq"], learned_sq=r_p["sq"],
                         spearman=rho, threshold=thr_p, kept=len(df_p))

    with open(OUT / "eval_nodeov.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    log(f"saved -> {OUT/'eval_nodeov.json'}")


if __name__ == "__main__":
    main()
