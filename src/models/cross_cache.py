"""Cross-prediction cache: every fold model predicts ALL train + test images.

Enables fold-ensemble construction and its (mildly optimistic — off-diagonal
models saw the images in training; documented) policy calibration.
Output: outputs/oof_cache/cross_m{fold}_{split}.parquet with stem, conf, area,
r_norm, rle per raw candidate (conf>=0.05, pre-de-overlap).
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament import IMG_H, IMG_W
from filament.rle import encode_mask

DISK_C, DISK_R = 1023.5, 900.0
DATA = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"
OUTD = ROOT / "outputs" / "oof_cache"


def run_model(fold):
    from ultralytics import YOLO
    w = ROOT / "outputs" / "runs" / f"e013_yolo11m_f{fold}_1792" / "weights" / "best.pt"
    if not w.exists():
        w = ROOT / "outputs" / "runs" / "e013_yolo11m_f0_1792" / "weights" / "best.pt" \
            if fold == 0 else w
    model = YOLO(str(w))
    for split, d in [("train", DATA / "train" / "train_images"),
                     ("test", DATA / "test" / "test_images")]:
        out = OUTD / f"cross_m{fold}_{split}.parquet"
        if out.exists():
            print(f"m{fold} {split}: exists, skip", flush=True)
            continue
        rows = []
        t0 = time.time()
        for p in sorted(d.glob("*.jpeg")):
            res = model.predict(str(p), imgsz=1792, conf=0.05, iou=0.5,
                                max_det=300, verbose=False)[0]
            if res.masks is None:
                continue
            raw = res.masks.data.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for m, c in zip(raw, confs):
                mm = cv2.resize(m, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR) >= 0.5
                if not mm.any():
                    continue
                ys, xs = np.nonzero(mm)
                rows.append(dict(stem=p.stem, conf=float(c), area=int(mm.sum()),
                                 r_norm=float(np.hypot(xs.mean() - DISK_C,
                                                       ys.mean() - DISK_C) / DISK_R),
                                 rle=encode_mask(mm.astype(np.uint8))))
        pd.DataFrame(rows).to_parquet(out, index=False)
        print(f"m{fold} {split}: {len(rows)} candidates ({time.time()-t0:.0f}s) -> {out.name}",
              flush=True)


def run_weights(model_id, weights, imgsz=1792, splits=("train", "test"),
                nms_iou=0.5, flip=""):
    """Cache an arbitrary weights file under cross_m{model_id}_{split}.parquet.

    flip: '' (original view) or 'h'/'v'/'hv' — the view is flipped before
    inference and predicted masks are un-flipped, so the cache is always in
    original image coordinates.
    """
    from ultralytics import YOLO
    model = YOLO(str(weights))
    for split in splits:
        d = DATA / split / f"{split}_images"
        out = OUTD / f"cross_m{model_id}_{split}.parquet"
        if out.exists():
            print(f"m{model_id} {split}: exists, skip", flush=True)
            continue
        rows = []
        t0 = time.time()
        for p in sorted(d.glob("*.jpeg")):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if "h" in flip:
                img = img[:, ::-1]
            if "v" in flip:
                img = img[::-1, :]
            src = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_GRAY2BGR)
            res = model.predict(src, imgsz=imgsz, conf=0.05, iou=nms_iou,
                                max_det=300, verbose=False)[0]
            if res.masks is None:
                continue
            raw = res.masks.data.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for m, c in zip(raw, confs):
                if "h" in flip:
                    m = m[:, ::-1]
                if "v" in flip:
                    m = m[::-1, :]
                mm = cv2.resize(np.ascontiguousarray(m), (IMG_W, IMG_H),
                                interpolation=cv2.INTER_LINEAR) >= 0.5
                if not mm.any():
                    continue
                ys, xs = np.nonzero(mm)
                rows.append(dict(stem=p.stem, conf=float(c), area=int(mm.sum()),
                                 r_norm=float(np.hypot(xs.mean() - DISK_C,
                                                       ys.mean() - DISK_C) / DISK_R),
                                 rle=encode_mask(mm.astype(np.uint8))))
        pd.DataFrame(rows).to_parquet(out, index=False)
        print(f"m{model_id} {split}: {len(rows)} candidates ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--spec", default=None,
                    help="comma list id:weights_path to cache arbitrary models")
    ap.add_argument("--nms", type=float, default=0.5)
    ap.add_argument("--flip", default="", help="'' or h/v/hv view for TTA members")
    ap.add_argument("--splits", default="train,test")
    a = ap.parse_args()
    OUTD.mkdir(parents=True, exist_ok=True)
    if a.spec:
        for item in a.spec.split(","):
            mid, path = item.split(":", 1)
            run_weights(mid, ROOT / path, nms_iou=a.nms, flip=a.flip,
                        splits=tuple(a.splits.split(",")))
    else:
        for f in [int(x) for x in a.folds.split(",")]:
            run_model(f)
