"""Build the full GT RLE cache and smoke-test the scorer (PQ=1 self-score)."""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from filament.gt import build_gt_df
from filament.pq import score

COCO = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
OUT = ROOT / "outputs" / "gt_cache"
OUT.mkdir(parents=True, exist_ok=True)

t0 = time.time()
gt = build_gt_df(str(COCO))
gt.to_parquet(OUT / "gt_full.parquet", index=False)
print(f"gt_full.parquet: {len(gt)} rows, {time.time()-t0:.1f}s")

# Smoke: on single-copy files, predictions = the annotator's own masks (bare-stem ids)
copies = gt.groupby("file_name")["entry_id"].nunique()
single_files = set(copies[copies == 1].index)
gt_single = gt[gt["file_name"].isin(single_files)].copy()
pred = gt_single.copy()
pred["stem"] = pred["file_name"].str.replace(".jpeg", "", regex=False)
pred["filament_id"] = pred["stem"] + "_" + (pred.groupby("stem").cumcount() + 1).astype(str)
pred = pred[["filament_id", "segmentation_rle"]]

t0 = time.time()
r = score(gt_single[["filament_id", "segmentation_rle"]], pred)
print(f"self-score on {len(single_files)} single-copy files: "
      f"PQ={r['pq']:.6f} SQ={r['sq']:.6f} RQ={r['rq']:.6f} "
      f"TP={r['tp']} FP={r['fp']} FN={r['fn']}  ({time.time()-t0:.1f}s)")
assert r["pq"] == 1.0, "self-score must be exactly 1.0"
print("SMOKE OK")
