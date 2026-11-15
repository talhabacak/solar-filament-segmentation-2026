"""Fold-0 PQ for a single YOLO checkpoint, organiser-parity scorer, conf sweep.

The decision this feeds: e018 (yolov8x initialised from crack segmentation --
Talha's find) beats its matched COCO-init control by +0.0275 mask mAP50 and
+0.049 recall on the same fold-0 val. mAP is the training-time proxy; the
competition scores PQ against every annotator copy, so nothing is adopted until
it wins THERE. Same de-overlap and NMS iou 0.0 as the verified public recipe,
conf swept because the optimum may shift between models.

Usage:
  python src/eda/pq_single_yolo.py --weights outputs/runs/<name>/weights/best.pt
"""

import argparse
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

IMDIR = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"


def log(*a):
    print(*a, flush=True)


def deoverlap(masks, conf):
    acc = np.zeros(masks.shape[1:], np.uint8)
    out = []
    for i in np.argsort(conf)[::-1]:
        out.append(masks[i] & ~acc)
        acc |= masks[i]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=1792)
    ap.add_argument("--confs", default="0.2,0.3,0.4")
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.weights)
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    stems = sorted(f.rsplit(".", 1)[0]
                   for f in folds.loc[folds["fold"] == a.fold, "file_name"].unique())
    paths = {p.stem: p for p in IMDIR.rglob("*.jpeg")}
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt = gt[gt["file_name"].isin({s + ".jpeg" for s in stems})]
    log(f"{Path(a.weights).parts[-3]}: fold {a.fold}, {len(stems)} images, "
        f"{len(gt)} GT rows")

    confs = [float(c) for c in a.confs.split(",")]
    preds = {c: [] for c in confs}
    t0 = time.time()
    for stem in stems:
        for c in confs:
            r = model.predict(str(paths[stem]), imgsz=a.imgsz, conf=c, iou=0.0,
                              max_det=100, verbose=False)[0]
            if r.masks is None:
                continue
            md = (r.masks.data.cpu().numpy() > 0.5).astype(np.uint8)
            cf = r.boxes.conf.cpu().numpy()
            for i, m in enumerate(deoverlap(md, cf)):
                mr = cv2.resize(m, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
                if mr.sum() >= 5:
                    preds[c].append((f"{stem}_{i+1}", encode_mask(mr)))
    log(f"inference done ({time.time()-t0:.0f}s)")

    for c in confs:
        pdf = pd.DataFrame(preds[c], columns=["filament_id", "segmentation_rle"])
        s = score(gt[["filament_id", "segmentation_rle"]], pdf)
        log(f"  conf {c:.2f}: PQ {s['pq']:.4f} SQ {s['sq']:.4f} RQ {s['rq']:.4f}  "
            f"TP {s['tp']} FP {s['fp']} FN {s['fn']}  n={len(pdf)}")


if __name__ == "__main__":
    main()
