"""Representational ceiling imposed by YOLO's mask_ratio (segmentation target grid).

Ultralytics stores segmentation targets at imgsz/mask_ratio with NEAREST
downsampling (data/dataset.py) and upsamples the proto to that grid in the loss
(utils/loss.py). So mask_ratio sets the fidelity of what the mask head is taught.

This measures, per GT instance, the IoU between the true 2048 mask and the same
mask after a round trip through the target grid — i.e. the best IoU any model
could score if it reproduced its training target perfectly. Anything below 0.5
can never become a PQ true positive.

Run: python src/eda/mask_ratio_ceiling.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament.rle import rles_to_layers

N_SAMPLE = 1500
GRIDS = {448: "mask_ratio=4 @1792 (current)", 896: "mask_ratio=2 @1792",
         1792: "mask_ratio=1 @1792", 256: "mask_ratio=4 @1024 (E010 setting)"}

gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
morph = pd.read_parquet(ROOT / "outputs" / "eda" / "morphology_instances.parquet").set_index("ann_id")
sample = gt.sample(N_SAMPLE, random_state=1)
w = morph.reindex(sample["ann_id"])["width_proxy"].to_numpy()
area = morph.reindex(sample["ann_id"])["poly_area"].to_numpy()

results = {}
for grid, label in GRIDS.items():
    ious = np.empty(len(sample))
    for i, r in enumerate(sample.itertuples()):
        m = rles_to_layers([r.segmentation_rle])[0].astype(np.uint8)
        small = cv2.resize(m, (grid, grid), interpolation=cv2.INTER_NEAREST)
        back = cv2.resize(small.astype(np.float32), (2048, 2048),
                          interpolation=cv2.INTER_LINEAR) >= 0.5
        inter = np.logical_and(back, m > 0).sum()
        union = np.logical_or(back, m > 0).sum()
        ious[i] = inter / max(union, 1)
    thin, thick = w < 8, w >= 12
    results[label] = dict(
        grid=grid, median=float(np.median(ious)), p10=float(np.percentile(ious, 10)),
        frac_below_050=float((ious < 0.5).mean()), frac_below_070=float((ious < 0.7).mean()),
        median_thin=float(np.median(ious[thin])), median_thick=float(np.median(ious[thick])),
        n_thin=int(thin.sum()), n=len(ious))
    print(f"{label:32s} median {results[label]['median']:.3f}  p10 {results[label]['p10']:.3f}  "
          f"<0.5 {results[label]['frac_below_050']*100:5.1f}%  "
          f"thin {results[label]['median_thin']:.3f}  thick {results[label]['median_thick']:.3f}",
          flush=True)

with open(ROOT / "outputs" / "eda" / "mask_ratio_ceiling.json", "w") as f:
    json.dump(results, f, indent=1)
print("saved -> outputs/eda/mask_ratio_ceiling.json")
