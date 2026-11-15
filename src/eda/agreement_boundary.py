"""E126 -- is our inclusion policy at the annotator-agreement optimum?

Talha's question: how are the annotator copies handled, and is our pixel-level
loss the right one for them?

The metric scores one prediction set against every copy, so for a pixel drawn by
k of N annotators the change in mean IoU from including it is

    d(mean IoU) = [k - (N - k)*IoU] / |union|,

which is positive exactly when  k/N > IoU/(1+IoU).  At our matched-instance IoU
of 0.679 that threshold is **0.404**: a pixel should be included if at least 40%
of the annotators drew it, and excluded otherwise. This is a property of the
metric, not a hyper-parameter.

The production pipeline never states that number anywhere. It extracts semantic
components at p >= 0.3 (BCE on all copies converges in expectation to the
per-pixel agreement fraction, so p is an estimate of k/N), fuses, and then
erodes the outermost ring (~19% of the pixels of a thin mask). Whether the
composition of those three steps lands on 0.404 has never been measured.

This measures it directly on the 296 multi-copy training images: build the
agreement map k/N from the copies, take the deployed masks (honest out-of-fold
pool at the production policy), and report

    - the mean k/N of the pixels we INCLUDE and of the pixels we EXCLUDE,
      restricted to the disputed band (0 < k/N < 1) around each matched mask;
    - the agreement profile in 1-px rings from 3 px inside to 3 px outside the
      boundary, which shows where the decision boundary actually sits;
    - the counterfactual: mean IoU against all copies if we included every pixel
      with k/N > t, for t on a grid, i.e. the achievable optimum for this field.

Usage: python src/eda/agreement_boundary.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from filament.rle import decode_rle  # noqa: E402

OUT = ROOT / "outputs" / "eda"
CONF, PCUT = 0.40, 0.60
K3 = np.ones((3, 3), np.uint8)


def log(*a):
    print(*a, flush=True)


def main():
    cand = pd.read_parquet(OUT / "e115b_labelled_pool.parquet").merge(
        pd.read_parquet(OUT / "e115b_p_oof.parquet"), on="cand_id")
    keep = cand[(cand["conf"] >= CONF) & (cand["p_oof"] >= PCUT)]
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    ncop = gt.groupby("stem")["entry_id"].nunique()
    multi = set(ncop[ncop > 1].index) & set(keep["stem"])
    log(f"multi-copy images with kept masks: {len(multi)} (of {len(ncop[ncop > 1])} multi-copy images)")

    rings = {d: [] for d in (-3, -2, -1, 1, 2, 3)}
    inc, exc = [], []
    grid = np.round(np.arange(0.1, 0.95, 0.05), 2)
    iou_at_t = {t: [0.0, 0.0] for t in grid}   # [sum of per-copy intersections, sum of unions]
    for n, stem in enumerate(sorted(multi)):
        g = gt[gt["stem"] == stem]
        entries = sorted(g["entry_id"].unique())
        N = len(entries)
        acc = np.zeros((2048, 2048), np.float32)
        per_copy = []
        for e in entries:
            m = np.zeros((2048, 2048), bool)
            for r in g.loc[g["entry_id"] == e, "segmentation_rle"]:
                m |= decode_rle(r).astype(bool)
            per_copy.append(m)
            acc += m
        agree = acc / N                                   # k/N per pixel
        ours = np.zeros((2048, 2048), bool)
        for r in keep.loc[keep["stem"] == stem, "rle"]:
            ours |= decode_rle(r).astype(bool)
        band = (agree > 0) & (agree < 1)                  # disputed pixels only
        if band.any():
            inc.append(float(agree[band & ours].mean()) if (band & ours).any() else np.nan)
            exc.append(float(agree[band & ~ours].mean()) if (band & ~ours).any() else np.nan)
        # agreement in 1-px rings around our boundary
        cur = ours.astype(np.uint8)
        for d in (1, 2, 3):
            outer = cv2.dilate(cur, K3, iterations=d).astype(bool) & ~cv2.dilate(cur, K3, iterations=d - 1).astype(bool)
            inner = cv2.erode(cur, K3, iterations=d - 1).astype(bool) & ~cv2.erode(cur, K3, iterations=d).astype(bool)
            if outer.any():
                rings[d].append(float(agree[outer].mean()))
            if inner.any():
                rings[-d].append(float(agree[inner].mean()))
        # counterfactual inclusion policies on this image's own agreement map
        for t in grid:
            pred = agree > t
            for m in per_copy:
                iou_at_t[t][0] += float((pred & m).sum())
                iou_at_t[t][1] += float((pred | m).sum())
        if n % 50 == 0:
            log(f"  {n}/{len(multi)} images")

    log("\nagreement k/N of DISPUTED pixels, per image:")
    log(f"  pixels we include: mean {np.nanmean(inc):.3f}   pixels we exclude: mean {np.nanmean(exc):.3f}")
    log("\nagreement profile across our mask boundary (mean k/N):")
    for d in (-3, -2, -1, 1, 2, 3):
        side = "inside" if d < 0 else "outside"
        log(f"  {abs(d)} px {side:7s}: {np.mean(rings[d]):.3f}")
    log("\ncounterfactual: pixel-level IoU against all copies if we included every pixel with k/N > t")
    best = max(grid, key=lambda t: iou_at_t[t][0] / max(iou_at_t[t][1], 1))
    for t in grid:
        i, u = iou_at_t[t]
        mark = "  <- best" if t == best else ""
        log(f"  t = {t:.2f}: IoU {i / max(u, 1):.4f}{mark}")
    log(f"\ntheory says the optimum is IoU/(1+IoU) = {0.679 / 1.679:.3f}; the counterfactual grid says {best:.2f}")
    json.dump(dict(include_mean=float(np.nanmean(inc)), exclude_mean=float(np.nanmean(exc)),
                   rings={str(k): float(np.mean(v)) for k, v in rings.items()},
                   counterfactual={str(t): iou_at_t[t][0] / max(iou_at_t[t][1], 1) for t in grid},
                   best_t=float(best), theory_t=0.679 / 1.679),
              open(OUT / "agreement_boundary.json", "w"), indent=1)
    log(f"wrote {OUT / 'agreement_boundary.json'}")


if __name__ == "__main__":
    main()
