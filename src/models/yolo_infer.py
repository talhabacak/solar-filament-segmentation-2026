"""YOLO-seg inference -> pixel-disjoint instance masks -> submission CSV / OOF scoring.

Post-process chain (each step OOF-tunable):
  1. model(imgsz, conf, iou NMS) -> per-instance soft masks at model resolution
  2. upsample masks to 2048x2048, threshold
  3. greedy de-overlap in confidence order (competition requires pixel-disjoint)
  4. min-area filter, radial suppression (centroid r_norm > r_max)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament import IMG_H, IMG_W
from filament.rle import encode_mask, write_submission
from filament.pq import score

DISK_C = 1023.5
DISK_R = 900.0


CROP0, CROP1 = 112, 1936          # disk-ROI crop box (see export_yolo_crop.py)


def predict_image(model, img_path, imgsz=1024, conf=0.25, iou_nms=0.5,
                  mask_thr=0.5, min_area=64, r_max=0.97, max_det=100, crop=False):
    """Return (masks, confs) with masks pixel-disjoint at 2048x2048.

    crop=True: predict on the constant 1824x1824 disk ROI and paste masks back
    at the (112,112) offset — for models trained on *_crop datasets.
    """
    if crop:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)[CROP0:CROP1, CROP0:CROP1]
        src = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        src = str(img_path)
    res = model.predict(src, imgsz=imgsz, conf=conf, iou=iou_nms,
                        max_det=max_det, verbose=False, retina_masks=False)[0]
    if res.masks is None or len(res.masks.data) == 0:
        return [], []
    raw = res.masks.data.cpu().numpy()          # (n, h, w) at model proto res
    confs = res.boxes.conf.cpu().numpy()
    order = np.argsort(-confs)

    crop_size = CROP1 - CROP0
    taken = np.zeros((IMG_H, IMG_W), dtype=bool)
    masks, kept_confs = [], []
    for i in order:
        if crop:
            mc = cv2.resize(raw[i], (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
            m = np.zeros((IMG_H, IMG_W), dtype=np.float32)
            m[CROP0:CROP1, CROP0:CROP1] = mc
        else:
            m = cv2.resize(raw[i], (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
        m = (m >= mask_thr)
        m &= ~taken                              # greedy de-overlap
        area = int(m.sum())
        if area < min_area:
            continue
        ys, xs = np.nonzero(m)
        r_norm = np.hypot(xs.mean() - DISK_C, ys.mean() - DISK_C) / DISK_R
        if r_norm > r_max:
            continue
        taken |= m
        masks.append(m.astype(np.uint8))
        kept_confs.append(float(confs[i]))
    return masks, kept_confs


def run_test_submission(weights, out_csv, imgsz=1024, **kw):
    from ultralytics import YOLO
    model = YOLO(weights)
    test_dir = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "test" / "test_images"
    preds = {}
    for p in sorted(test_dir.glob("*.jpeg")):
        masks, _ = predict_image(model, p, imgsz=imgsz, **kw)
        preds[p.stem] = masks
    n_inst = sum(len(v) for v in preds.values())
    n_empty = sum(1 for v in preds.values() if not v)
    print(f"test inference: {n_inst} instances over 180 images ({n_empty} empty)")
    df = write_submission({k: v for k, v in preds.items() if v}, out_csv)
    print(f"wrote {len(df)} rows -> {out_csv}")
    return preds


def score_fold(weights, fold, imgsz=1024, **kw):
    """OOF PQ of a fold model on its validation fold, official semantics."""
    from ultralytics import YOLO
    model = YOLO(weights)
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    val_files = folds.loc[folds["fold"] == fold, "file_name"].tolist()
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt_val = gt[gt["file_name"].isin(val_files)]

    img_dir = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
    rows = []
    for fn in sorted(val_files):
        stem = fn.rsplit(".", 1)[0]
        masks, _ = predict_image(model, img_dir / fn, imgsz=imgsz, **kw)
        for k, m in enumerate(masks, start=1):
            rows.append((f"{stem}_{k}", encode_mask(m)))
    pred_df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    if pred_df.empty:
        pred_df = pd.DataFrame([("NONE_1", encode_mask(np.zeros((IMG_H, IMG_W), np.uint8)))],
                               columns=["filament_id", "segmentation_rle"])
    r = score(gt_val[["filament_id", "segmentation_rle"]], pred_df, return_details=True)
    print(f"fold {fold}: PQ={r['pq']:.4f} SQ={r['sq']:.4f} RQ={r['rq']:.4f} "
          f"TP={r['tp']} FP={r['fp']} FN={r['fn']}  (pred rows={len(pred_df)})")
    return r


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["oof", "submit"])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--mask-thr", type=float, default=0.5)
    ap.add_argument("--min-area", type=int, default=64)
    ap.add_argument("--out", default="outputs/submissions/sub.csv")
    a = ap.parse_args()
    kw = dict(conf=a.conf, mask_thr=a.mask_thr, min_area=a.min_area)
    if a.mode == "oof":
        score_fold(a.weights, a.fold, imgsz=a.imgsz, **kw)
    else:
        out = ROOT / a.out
        out.parent.mkdir(parents=True, exist_ok=True)
        run_test_submission(a.weights, out, imgsz=a.imgsz, **kw)
