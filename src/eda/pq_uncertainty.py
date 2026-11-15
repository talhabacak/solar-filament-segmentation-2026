"""How uncertain is the pooled out-of-fold PQ of the submitted system?

The report template asks for error bars. This reads the per-image out-of-fold statistics of the
first final submission (tp, fp, fn and the IoU sum of the matches, one row per training image,
written by src/eda/final_pair_selection.py) and reports

  * the pooled PQ, SQ, RQ, precision and recall over all 707 images,
  * a 95 % interval from a bootstrap over images (images are the independent unit: all annotator
    copies and all candidates of an image are resampled together),
  * the same quantities per fold of data/splits/folds_v1.parquet.

Input : outputs/eda/sub036_oof_per_image.parquet, data/splits/folds_v1.parquet
Output: outputs/eda/pq_uncertainty.json

Usage: python src/eda/pq_uncertainty.py [--reps 4000] [--seed 0]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "eda"


def scores(tp, fp, fn, iou):
    """PQ and its factors from summed counts."""
    sq = iou / tp
    rq = tp / (tp + 0.5 * fp + 0.5 * fn)
    return {"pq": sq * rq, "sq": sq, "rq": rq, "precision": tp / (tp + fp), "recall": tp / (tp + fn)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    d = pd.read_parquet(OUT / "sub036_oof_per_image.parquet")
    folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    d["fold"] = d["stem"].map(fold_of).astype(int)
    x = d[["tp", "fp", "fn", "iou"]].to_numpy(float)

    pooled = scores(*x.sum(0))
    rng = np.random.default_rng(a.seed)
    idx = rng.integers(0, len(x), size=(a.reps, len(x)))
    boot = pd.DataFrame([scores(*x[i].sum(0)) for i in idx])
    ci = {k: [float(boot[k].quantile(0.025)), float(boot[k].quantile(0.975))] for k in boot.columns}
    per_fold = {int(k): scores(*g[["tp", "fp", "fn", "iou"]].to_numpy(float).sum(0)) | {"images": int(len(g))}
                for k, g in d.groupby("fold")}
    fold_pq = [per_fold[k]["pq"] for k in sorted(per_fold)]

    res = {"images": int(len(d)), "reps": a.reps, "pooled": pooled, "ci95": ci, "bootstrap_sd": float(boot["pq"].std()),
           "per_fold": per_fold, "fold_pq_mean": float(np.mean(fold_pq)), "fold_pq_sd": float(np.std(fold_pq, ddof=1))}
    (OUT / "pq_uncertainty.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for k in ("pq", "sq", "rq", "precision", "recall"):
        print(f"{k:9s} {pooled[k]:.4f}   95% CI [{ci[k][0]:.4f}, {ci[k][1]:.4f}]")
    print("per-fold PQ:", " ".join(f"{v:.4f}" for v in fold_pq),
          f"  mean {np.mean(fold_pq):.4f}  sd {np.std(fold_pq, ddof=1):.4f}")


if __name__ == "__main__":
    main()
