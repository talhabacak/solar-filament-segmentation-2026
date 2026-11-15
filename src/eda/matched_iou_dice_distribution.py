"""What do the IoU and Dice of our matched masks look like as distributions, not as a mean?

The evaluation rubric asks for the distribution of Dice and of IoU next to PQ. SQ is the mean IoU
of the matched copies and hides the spread, so this reads the per-copy table that
src/eda/pq_decomposition.py writes (one row per annotator copy, `iou_acc` = IoU with the accepted
prediction it is matched to, 0 when unmatched) and reports quantiles of both.

A copy is matched when IoU > 0.5, the metric's own rule. Dice is computed per matched copy as
2*IoU / (1 + IoU); for one mask pair this identity is exact, so no second pass over the masks is
needed. The table is the out-of-fold output at the unbagged operating point (pooled PQ 0.4421),
which differs from the submitted system only by selector bagging (0.4431).

Input : outputs/eda/pq_decomposition.parquet
Output: outputs/eda/matched_iou_dice_distribution.json

Usage: python src/eda/matched_iou_dice_distribution.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "eda"
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]


def describe(x):
    """Mean and the fixed quantiles of a one-dimensional sample."""
    out = {"mean": float(np.mean(x))}
    out.update({f"q{int(q * 100):02d}": float(v) for q, v in zip(QUANTILES, np.quantile(x, QUANTILES))})
    return out


def main():
    d = pd.read_parquet(OUT / "pq_decomposition.parquet")
    iou = d.loc[d["iou_acc"] > 0.5, "iou_acc"].to_numpy()
    dice = 2 * iou / (1 + iou)
    res = {"copies": int(len(d)), "matched": int(len(iou)), "recall": float(len(iou) / len(d)),
           "iou": describe(iou), "dice": describe(dice),
           "share_iou_ge_0.7": float((iou >= 0.7).mean()), "share_iou_ge_0.8": float((iou >= 0.8).mean())}
    (OUT / "matched_iou_dice_distribution.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"{res['matched']} of {res['copies']} copies matched (IoU > 0.5)")
    for name in ("iou", "dice"):
        r = res[name]
        print(f"{name:4s} mean {r['mean']:.4f}   5/25/50/75/95 %: "
              f"{r['q05']:.3f} {r['q25']:.3f} {r['q50']:.3f} {r['q75']:.3f} {r['q95']:.3f}")


if __name__ == "__main__":
    main()
