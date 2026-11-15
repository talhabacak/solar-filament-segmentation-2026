"""NMS-threshold ablation (inference-only axis, never tested before).

Filaments are long, thin and curved: two distinct filaments can have heavily
overlapping bounding boxes, so box-NMS at the default iou=0.5 may suppress true
detections (recall loss). Conversely, looser NMS may create duplicate proposals
that become FPs. This sweeps the NMS iou for a fixed model and scores each
setting with the exact PQ scorer on the fold's val split.

Usage: python src/models/nms_sweep.py --weights W --fold 0 --imgsz 1792 \
         --ious "0.3,0.5,0.7,0.9"
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament import IMG_H, IMG_W
from filament.pq import score
from filament.rle import encode_mask

DISK_C, DISK_R = 1023.5, 900.0
R_MAX = 0.97


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=1792)
    ap.add_argument("--ious", default="0.3,0.5,0.7,0.9")
    ap.add_argument("--confs", default="0.3,0.4")
    ap.add_argument("--min-area", type=int, default=128)
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.weights)
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    val_files = sorted(folds.loc[folds["fold"] == a.fold, "file_name"])
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt_val = gt[gt["file_name"].isin(val_files)][["filament_id", "segmentation_rle"]]
    img_dir = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"

    gt_cache: dict = {}
    rows = []
    for nms_iou in [float(x) for x in a.ious.split(",")]:
        # collect raw candidates once per NMS setting
        per_stem = {}
        for fn in val_files:
            stem = fn.rsplit(".", 1)[0]
            res = model.predict(str(img_dir / fn), imgsz=a.imgsz, conf=0.05,
                                iou=nms_iou, max_det=300, verbose=False)[0]
            items = []
            if res.masks is not None and len(res.masks.data):
                raw = res.masks.data.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                for i in np.argsort(-confs):
                    m = cv2.resize(raw[i], (IMG_W, IMG_H),
                                   interpolation=cv2.INTER_LINEAR) >= 0.5
                    if not m.any():
                        continue
                    items.append((float(confs[i]), m))
            per_stem[stem] = items

        for conf_t in [float(x) for x in a.confs.split(",")]:
            pred_rows = []
            for stem, items in per_stem.items():
                taken = np.zeros((IMG_H, IMG_W), dtype=bool)
                k = 0
                for c, m in items:
                    if c < conf_t:
                        continue
                    mm = m & ~taken
                    if mm.sum() < a.min_area:
                        continue
                    ys, xs = np.nonzero(mm)
                    if np.hypot(xs.mean() - DISK_C, ys.mean() - DISK_C) / DISK_R > R_MAX:
                        continue
                    taken |= mm
                    k += 1
                    pred_rows.append((f"{stem}_{k}", encode_mask(mm.astype(np.uint8))))
            pred_df = pd.DataFrame(pred_rows, columns=["filament_id", "segmentation_rle"])
            r = score(gt_val, pred_df, gt_cache=gt_cache)
            rows.append(dict(nms_iou=nms_iou, conf=conf_t, **{k: v for k, v in r.items()
                                                              if k in ("pq", "sq", "rq", "tp", "fp", "fn")}))
            print(f"nms_iou={nms_iou} conf={conf_t}: PQ={r['pq']:.4f} SQ={r['sq']:.4f} "
                  f"RQ={r['rq']:.4f} TP={r['tp']} FP={r['fp']} FN={r['fn']}", flush=True)

    out = pd.DataFrame(rows).sort_values("pq", ascending=False)
    dest = ROOT / "outputs" / "oof_cache" / f"fold{a.fold}_nms_sweep.csv"
    out.to_csv(dest, index=False)
    print("\nbest:\n", out.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
