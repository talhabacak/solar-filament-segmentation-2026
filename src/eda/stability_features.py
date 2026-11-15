"""E188 -- threshold-stability features: new INFORMATION for the selector, not a new fitter.

Talha, 2026-09-14: "can we give extra features to the selector?" Everything tried so far was
either a different fitter (bagging, monotone constraints, regularisation, a set-level encoding)
or a different source (the magnetic PIL summaries). The one thing never given to the selector is
how the candidate behaves when the probability field is thresholded at a DIFFERENT level.

The pool extracts candidates at p >= 0.2 and describes them with the field's mean, max, p90 and
ring contrast inside the final mask. None of those says whether the object is crisp or diffuse,
and none can see the merge: a candidate that covers two filaments is one component at 0.2 and
TWO at 0.7. That is a GT-free detector of the mechanism E173 measured (34 of the 166 addressable
misses are 'merged_into', plus 17 'split').

Six features per candidate, all from the cached out-of-fold probability field:

  stab_area_53     area(p>=0.5) / area(p>=0.3) inside the mask -- how fast the object shrinks
  stab_area_75     area(p>=0.7) / area(p>=0.5)
  stab_iou_37      IoU between (p>=0.3) and (p>=0.7) inside the mask -- crispness
  ncomp_07         connected components of (p>=0.7) inside the mask -- 2+ means a merge
  ncomp_05         the same at 0.5
  edge_grad        mean |grad p| on the mask's own boundary -- a sharp edge is a confident one

Written to outputs/eda/stability_features_t020.parquet, keyed by cand_id, to be read with
  python outputs/eda/_feature_add_test.py outputs/eda/stability_features_t020.parquet \\
      stab_area_53 stab_area_75 stab_iou_37 ncomp_07 ncomp_05 edge_grad

Usage: python src/eda/stability_features.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/eda", "src/models"):
    sys.path.insert(0, str(ROOT / p))
from filament.rle import decode_rle  # noqa: E402

OUT = ROOT / "outputs" / "eda"
SEM = ROOT / "outputs" / "semantic"
NEW = ["stab_area_53", "stab_area_75", "stab_iou_37", "ncomp_07", "ncomp_05", "edge_grad"]


def log(*a):
    print(*a, flush=True)


def feats_for(m, prob):
    """m: boolean mask at 2048; prob: the fold's OOF probability field at 2048, 0..1."""
    ys, xs = np.nonzero(m)
    y0, y1 = max(0, ys.min() - 4), min(2048, ys.max() + 5)
    x0, x1 = max(0, xs.min() - 4), min(2048, xs.max() + 5)
    sm, sp = m[y0:y1, x0:x1], prob[y0:y1, x0:x1]
    a3 = (sp >= 0.3) & sm
    a5 = (sp >= 0.5) & sm
    a7 = (sp >= 0.7) & sm
    n3, n5, n7 = int(a3.sum()), int(a5.sum()), int(a7.sum())
    inter = int((a3 & a7).sum())
    union = int((a3 | a7).sum())
    gy, gx = np.gradient(sp.astype(np.float32))
    grad = np.hypot(gx, gy)
    edge = (cv2.dilate(sm.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) ^ \
           (cv2.erode(sm.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    return dict(
        stab_area_53=float(n5 / max(n3, 1)),
        stab_area_75=float(n7 / max(n5, 1)),
        stab_iou_37=float(inter / max(union, 1)),
        ncomp_07=int(max(cv2.connectedComponents(a7.astype(np.uint8), connectivity=8)[0] - 1, 0)),
        ncomp_05=int(max(cv2.connectedComponents(a5.astype(np.uint8), connectivity=8)[0] - 1, 0)),
        edge_grad=float(grad[edge].mean()) if edge.any() else 0.0)


def main():
    folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    pool = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    pool["fold"] = pool["stem"].map(fold_of).astype(int)
    probs = {k: np.load(SEM / f"fold{k}_probs_blob5ens.npz") for k in range(5)}
    rows = []
    for n, (stem, g) in enumerate(pool.groupby("stem")):
        f = fold_of[stem]
        if stem not in probs[f].files:
            continue
        pr = cv2.resize(probs[f][stem], (2048, 2048), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        for r in g.itertuples():
            m = decode_rle(r.rle).astype(bool)
            if m.sum() < 16:
                continue
            d = feats_for(m, pr)
            d["cand_id"] = r.cand_id
            rows.append(d)
        if (n + 1) % 100 == 0:
            log(f"  {n + 1} images, {len(rows)} candidates")
    d = pd.DataFrame(rows)
    d.to_parquet(OUT / "stability_features_t020.parquet", index=False)
    log(f"wrote {len(d)} rows -> stability_features_t020.parquet")
    m = pool[["cand_id", "y", "iou_sum"]].merge(d, on="cand_id")
    log("\nmean by label:")
    log(m.groupby("y")[NEW].mean().round(3).to_string())
    log("\nthe merge signal: candidates whose high-threshold core splits in two")
    for k in (1, 2, 3):
        sel = m["ncomp_07"] == k if k < 3 else m["ncomp_07"] >= 3
        if sel.sum():
            log(f"  ncomp_07 {'>=3' if k == 3 else k}: {int(sel.sum()):5d} candidates, "
                f"positive rate {m.loc[sel, 'y'].mean():.3f}, mean iou_sum {m.loc[sel, 'iou_sum'].mean():.3f}")
    log("STABILITY_FEATURES_DONE")


if __name__ == "__main__":
    main()
