"""Intensity-guided mask snapping: push near-miss predictions over the IoU 0.5 line.

Measured target (outputs/eda/error_profile_fp.json): 617 of 3431 FN have best IoU
> 0.40 with a kept prediction; converting them is worth +0.0404 PQ and removes
617 FP at the same time. SQ is exactly linear in mask quality
(dPQ = 0.6181 x mean IoU gain), so boundary work pays predictably.

Method (adapted from a public kernel's idea, reimplemented and measured here):
for each predicted instance, dilate its mask to form a local search region,
compute an Otsu threshold on the ORIGINAL intensities inside that region only
(so the cut adapts to local limb darkening and per-site JPEG stretch), keep dark
pixels within the region, and retain only the connected component that touches
the original prediction. Guarded by an area-ratio band so a snap can never
explode into a neighbouring structure.

Usage:
  python src/models/otsu_snap.py --fold 0 --tag _e021       # measure on OOF
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament import IMG_H, IMG_W
from filament.pq import score
from filament.rle import decode_rle, encode_mask

IMG_DIR = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
CACHE = ROOT / "outputs" / "oof_cache"
R_MAX = 0.97
DISK_C, DISK_R = 1023.5, 900.0


def log(*a):
    print(*a, flush=True)


def snap_mask(img, mask, margin=10, lo=0.7, hi=2.0):
    """Return the intensity-snapped mask, or the original if the snap is unsafe."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask
    y0, y1 = max(0, ys.min() - margin * 2), min(IMG_H, ys.max() + margin * 2 + 1)
    x0, x1 = max(0, xs.min() - margin * 2), min(IMG_W, xs.max() + margin * 2 + 1)
    win = img[y0:y1, x0:x1]
    m = mask[y0:y1, x0:x1].astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin * 2 + 1,) * 2)
    region = cv2.dilate(m, k) > 0
    vals = win[region]
    if vals.size < 50:
        return mask
    thr, _ = cv2.threshold(vals.astype(np.uint8), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = (win <= thr) & region
    if not dark.any():
        return mask
    n, lab = cv2.connectedComponents(dark.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(dark)
    touched = set(lab[m > 0]) - {0}
    for c in touched:
        keep |= lab == c
    if not keep.any():
        return mask
    ratio = keep.sum() / max(m.sum(), 1)
    if ratio < lo or ratio > hi:                 # unsafe growth/shrink -> keep original
        return mask
    out = mask.copy()
    out[y0:y1, x0:x1] = keep
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--tag", default="_e021")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--min-area", type=int, default=128)
    ap.add_argument("--margins", default="6,10,16")
    a = ap.parse_args()

    preds = pd.read_parquet(CACHE / f"fold{a.fold}_preds{a.tag}.parquet")
    preds = preds[(preds["conf"] >= a.conf) & (preds["r_norm"] <= R_MAX)]
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    val_files = sorted(folds.loc[folds["fold"] == a.fold, "file_name"])
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt_val = gt[gt["file_name"].isin(val_files)][["filament_id", "segmentation_rle"]]

    gt_cache: dict = {}
    results = {}
    for margin in [0] + [int(x) for x in a.margins.split(",")]:
        rows = []
        t0 = time.time()
        for stem, g in preds.groupby("stem"):
            img = cv2.imread(str(IMG_DIR / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
            taken = np.zeros((IMG_H, IMG_W), dtype=bool)
            k = 0
            for r in g.sort_values("conf", ascending=False).itertuples():
                m = decode_rle(r.rle).astype(bool) & ~taken
                if m.sum() < a.min_area:
                    continue
                if margin > 0:
                    m = snap_mask(img, m, margin=margin) & ~taken
                    if m.sum() < a.min_area:
                        continue
                taken |= m
                k += 1
                rows.append((f"{stem}_{k}", encode_mask(m.astype(np.uint8))))
        pred_df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
        r = score(gt_val, pred_df, gt_cache=gt_cache)
        results[f"margin{margin}"] = {k2: v for k2, v in r.items()
                                      if k2 in ("pq", "sq", "rq", "tp", "fp", "fn")}
        log(f"margin={margin:2d}: PQ={r['pq']:.4f} SQ={r['sq']:.4f} RQ={r['rq']:.4f} "
            f"TP={r['tp']} FP={r['fp']} FN={r['fn']}  ({time.time()-t0:.0f}s)")

    with open(ROOT / "outputs" / "eda" / f"otsu_snap_fold{a.fold}{a.tag}.json", "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
