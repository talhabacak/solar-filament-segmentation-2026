"""E124 -- mask-shape and photometric features the selector does not have.

Talha's question: does the selector see the bounding box and the segmentation
mask? Today it sees a SUMMARY of the mask -- log-area, bbox aspect ratio,
eccentricity, compactness (perimeter^2/4piA), skeleton length, width proxy,
centroid position -- plus semantic-probability statistics under the mask and in
a 25 px ring. It never sees the mask itself (a CNN over the crop was measured:
Spearman 0.604 against the tabular 0.678, blend gain ~0), and three quantities
that the honest false-positive taxonomy (ledger 18b) showed DIFFER between true
positives and hallucinations are absent:

    solidity   area / convex-hull area        TP 0.623  vs  hallucination 0.736
    extent     area / bbox area               (not measured before)
    dark_core  fg 5th percentile / ring median  TP 0.736  vs  0.758
    contrast   (ring median - fg mean) / ring   TP 0.137  vs  0.127

The bbox enters only as an aspect ratio, so absolute size and fill are missing
too. This adds the family and measures it under the E120 protocol on the full
production pool (not just the kept set):

    solidity, extent, bbox_w, bbox_h, hull_gap (hull area / area - 1),
    img_contrast, dark_core, fg_p05, ring_med

Arms: base (32 features) vs +shape (solidity, extent, bbox_w, bbox_h, hull_gap)
vs +photometry (img_contrast, dark_core, fg_p05, ring_med) vs both. Honest
out-of-fold selector, exact scorer at (0.40, 0.60), paired image bootstrap,
bands pays >= +0.003 / hurts <= -0.003 / TIED inside +-0.005.

Usage: python src/eda/shape_photometry_features.py
"""

import json
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "eda"))
from filament.pq import score  # noqa: E402
from filament.rle import decode_rle  # noqa: E402
from e115c_threshold_ci import per_image, pq, oof_scores  # noqa: E402

OUT = ROOT / "outputs" / "eda"
IMG = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
BUNDLE = ROOT / "outputs" / "selector" / "selector_v5_shrunkblob5prodts.joblib"
A_POOL = OUT / "e098d_fused_pool_all_blob5_rebuild_labelled.parquet"
POLICY, REPS, FLOOR, EQUIV = (0.40, 0.60), 4000, 0.003, 0.005
SHAPE = ["solidity", "extent", "bbox_w", "bbox_h", "hull_gap"]
PHOTO = ["img_contrast", "dark_core", "fg_p05", "ring_med"]
RING = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))


def log(*a):
    print(*a, flush=True)


def feats_for(mask, img):
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    h, w = float(y1 - y0), float(x1 - x0)
    area = float(mask.sum())
    m8 = mask[y0:y1, x0:x1].astype(np.uint8)
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        hull_a = max(cv2.contourArea(cv2.convexHull(c)), 1.0)
    else:
        hull_a = max(area, 1.0)
    fg = img[mask]
    ring = cv2.dilate(mask.astype(np.uint8), RING).astype(bool) & ~mask
    rg = img[ring]
    rm = float(np.median(rg)) if rg.size else float(fg.mean())
    return dict(solidity=area / hull_a, extent=area / max(h * w, 1.0),
                bbox_w=w, bbox_h=h, hull_gap=hull_a / max(area, 1.0) - 1.0,
                img_contrast=(rm - float(fg.mean())) / max(rm, 1.0),
                dark_core=float(np.percentile(fg, 5)) / max(rm, 1.0),
                fg_p05=float(np.percentile(fg, 5)), ring_med=rm)


def main():
    feats = list(joblib.load(BUNDLE)["features"])
    a = pd.read_parquet(A_POOL)
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    a["fold"] = a["stem"].map(fold_of).astype(int)
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt["fold"] = gt["stem"].map(fold_of)
    gt = gt[gt["fold"].notna()].copy()
    gt["fold"] = gt["fold"].astype(int)

    cache = OUT / "shape_photometry_features.parquet"
    if cache.exists():
        ext = pd.read_parquet(cache)
        log(f"features from cache: {len(ext)} rows")
    else:
        rows, t0 = [], time.time()
        for n, (stem, g) in enumerate(a.groupby("stem")):
            img = cv2.imread(str(IMG / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            for cid, rle in zip(g["cand_id"], g["rle"]):
                m = decode_rle(rle).astype(bool)
                d = feats_for(m, img)
                d["cand_id"] = cid
                rows.append(d)
            if n % 100 == 0:
                log(f"  {n}/{a['stem'].nunique()} images ({time.time()-t0:.0f}s)")
        ext = pd.DataFrame(rows)
        ext.to_parquet(cache, index=False)
        log(f"features for {len(ext)} candidates ({time.time()-t0:.0f}s)")
    a = a.merge(ext, on="cand_id", how="left")

    # what the new features look like per class, for the report
    lab = np.where(a["y"] == 1, "positive", "negative")
    log("\nmedians by pool label:")
    log(a.groupby(lab)[SHAPE + PHOTO].median().round(3).T.to_string())

    stems = np.array(sorted(set(a["stem"])))
    gts = gt[gt["stem"].isin(stems)]
    ncopy = gts.groupby("stem")["entry_id"].nunique().reindex(stems).fillna(1).to_numpy()

    def exact(col, fold=None):
        sub = a[(a["conf"] >= POLICY[0]) & (a[col] >= POLICY[1])]
        g = gts
        if fold is not None:
            sub, g = sub[sub["fold"] == fold], gts[gts["fold"] == fold]
        pred = sub[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
        return score(g[["filament_id", "segmentation_rle"]], pred)

    arms = {"base": feats, "shape": feats + SHAPE, "photometry": feats + PHOTO, "both": feats + SHAPE + PHOTO}
    res, parts = {}, {}
    from sklearn.metrics import roc_auc_score
    for name, fl in arms.items():
        col = f"p_{name}"
        a[col] = oof_scores(a, fl, a["fold"].to_numpy())
        per = [exact(col, f)["pq"] for f in range(5)]
        r = exact(col)
        sub = a[(a["conf"] >= POLICY[0]) & (a[col] >= POLICY[1])]
        st, _ = per_image(sub, gts, stems, ncopy)
        parts[name] = st
        res[name] = dict(pq=r["pq"], tp=r["tp"], fp=r["fp"], fn=r["fn"], kept=int(len(sub)),
                         per_fold=per, auc=float(roc_auc_score(a["y"], a[col])))
        log(f"{name:11s} PQ {r['pq']:.4f} kept {len(sub)} TP {r['tp']} FP {r['fp']} FN {r['fn']} | per fold "
            + " ".join(f"{x:.4f}" for x in per))
    rng = np.random.default_rng(0)
    n = len(stems)
    for name in ("shape", "photometry", "both"):
        boot = np.empty(REPS)
        for i in range(REPS):
            idx = rng.integers(0, n, n)
            v = [pq(s["tp"][idx].sum(), s["fp"][idx].sum(), s["fn"][idx].sum(), s["iou"][idx].sum())
                 for s in (parts["base"], parts[name])]
            boot[i] = v[1] - v[0]
        d = res[name]["pq"] - res["base"]["pq"]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        dpf = [res[name]["per_fold"][f] - res["base"]["per_fold"][f] for f in range(5)]
        verdict = "PAYS" if d >= FLOOR else "HURTS" if d <= -FLOOR else ("TIED" if lo > -EQUIV and hi < EQUIV else "UNDERPOWERED")
        res[name].update(delta=float(d), ci=[float(lo), float(hi)], p_gt0=float((boot > 0).mean()),
                         delta_folds=dpf, verdict=verdict)
        log(f"  {name:11s} {d:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  P(>0) {(boot > 0).mean():.3f}  "
            f"folds+ {sum(x > 0 for x in dpf)}/5  -> {verdict}")
    json.dump(res, open(OUT / "shape_photometry_features.json", "w"), indent=1, default=float)
    log(f"\nwrote {OUT / 'shape_photometry_features.json'}")


if __name__ == "__main__":
    main()
