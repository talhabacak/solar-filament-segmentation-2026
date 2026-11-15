"""Model-free floor: how far does plain darkness thresholding get on PQ?

Filaments are H-alpha absorption features, so the naive classical pipeline is
"threshold the dark pixels, take connected components, call each one a filament".
This measures that floor with the exact competition scorer, on the same fold-0
split every model result is quoted on (142 images, 1510 GT rows), so the numbers
are directly comparable.

Variants:
  otsu      Otsu threshold over the disk interior
  otsu_hp   Otsu after removing large-scale structure (I - blur), which is the
            fairer classical baseline since chromospheric background varies
  pct{p}    keep the darkest p% of disk pixels

Each is followed by an opening to drop speckle and a minimum-area filter.

Usage: python src/eda/classical_baseline.py --fold 0
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

from filament import IMG_H, IMG_W
from filament.pq import score
from filament.rle import encode_mask

IMG_DIR = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
DISK_C, DISK_R = 1023.5, 900.0
R_MAX = 0.97


def log(*a):
    print(*a, flush=True)


def disk_mask(r_max=R_MAX):
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W].astype(np.float32)
    return np.hypot(xx - DISK_C, yy - DISK_C) / DISK_R <= r_max


DISK = disk_mask()


def dark_mask(img, variant):
    inside = DISK
    if variant.startswith("otsu_hp"):
        blur = cv2.GaussianBlur(img, (101, 101), 0).astype(np.int16)
        work = np.clip(img.astype(np.int16) - blur + 128, 0, 255).astype(np.uint8)
    else:
        work = img
    vals = work[inside]
    if variant.startswith("pct"):
        p = float(variant[3:])
        thr = np.percentile(vals, p)
    else:
        thr, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (work <= thr) & inside


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--variants", default="otsu,otsu_hp,pct1.0,pct2.0,pct5.0")
    ap.add_argument("--min-areas", default="128,400")
    a = ap.parse_args()

    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    val_files = sorted(folds.loc[folds["fold"] == a.fold, "file_name"])
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt_val = gt[gt["file_name"].isin(val_files)][["filament_id", "segmentation_rle"]]
    log(f"fold {a.fold}: {len(val_files)} images, {len(gt_val)} GT rows")

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gt_cache: dict = {}
    results = {}
    for variant in a.variants.split(","):
        per_image = {}
        t0 = time.time()
        for fn in val_files:
            img = cv2.imread(str(IMG_DIR / fn), cv2.IMREAD_GRAYSCALE)
            m = dark_mask(img, variant).astype(np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)
            n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            per_image[fn.rsplit(".", 1)[0]] = (lab, stats)
        for min_area in [int(x) for x in a.min_areas.split(",")]:
            rows = []
            for stem, (lab, stats) in per_image.items():
                k = 0
                for c in range(1, stats.shape[0]):
                    if stats[c, cv2.CC_STAT_AREA] < min_area:
                        continue
                    k += 1
                    rows.append((f"{stem}_{k}",
                                 encode_mask((lab == c).astype(np.uint8))))
            pred = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
            if pred.empty:
                continue
            r = score(gt_val, pred, gt_cache=gt_cache)
            key = f"{variant}_area{min_area}"
            results[key] = dict(n_pred=len(pred),
                                **{k2: v for k2, v in r.items()
                                   if k2 in ("pq", "sq", "rq", "tp", "fp", "fn")})
            log(f"{key:20s} preds={len(pred):6d}  PQ={r['pq']:.4f} SQ={r['sq']:.4f} "
                f"TP={r['tp']:5d} FP={r['fp']:6d} FN={r['fn']:5d}  ({time.time()-t0:.0f}s)")

    dest = ROOT / "outputs" / "eda" / f"classical_baseline_fold{a.fold}.json"
    with open(dest, "w") as f:
        json.dump(results, f, indent=1)
    log(f"saved -> {dest}")


if __name__ == "__main__":
    main()
