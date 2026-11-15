"""OOF postprocess calibration: cache low-conf predictions once, sweep thresholds on PQ.

Stage 1 (cache): run the model at very low conf on a fold's val images; store every
instance mask (RLE at 2048) + confidence, pre-de-overlap.
Stage 2 (sweep): for each (conf, min_area) combo: filter, greedy de-overlap in conf
order, radial suppression, score with the exact official-equivalent scorer.

Usage:
  python src/models/sweep_oof.py cache --weights W --fold 0 --imgsz 1024
  python src/models/sweep_oof.py sweep --fold 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament import IMG_H, IMG_W
from filament.rle import encode_mask, decode_rle
from filament.pq import score

DISK_C = 1023.5
DISK_R = 900.0
CACHE_DIR = ROOT / "outputs" / "oof_cache"


CROP0, CROP1 = 112, 1936


def cache_fold(weights, fold, imgsz=1024, conf=0.05, max_det=300, retina=False,
               tag="", crop=False, tta="", nms_iou=0.5):
    """tta: comma list of extra views ('h','v','hv'); their proposals are
    un-flipped and pooled with the original view's (greedy de-overlap in the
    sweep resolves duplicates, so pooling raises the proposal recall pool)."""
    from ultralytics import YOLO
    model = YOLO(weights)
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    val_files = sorted(folds.loc[folds["fold"] == fold, "file_name"])
    img_dir = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
    views = [""] + [v for v in tta.split(",") if v]

    rows = []
    for fn in val_files:
        stem = fn.rsplit(".", 1)[0]
        base = cv2.imread(str(img_dir / fn), cv2.IMREAD_GRAYSCALE)
        if crop:
            base = base[CROP0:CROP1, CROP0:CROP1]
        for view in views:
            img = base
            if "h" in view:
                img = img[:, ::-1]
            if "v" in view:
                img = img[::-1, :]
            src = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_GRAY2BGR)
            res = model.predict(src, imgsz=imgsz, conf=conf, iou=nms_iou,
                                max_det=max_det, retina_masks=retina, verbose=False)[0]
            if res.masks is None:
                continue
            raw = res.masks.data.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for m, c in zip(raw, confs):
                if "h" in view:
                    m = m[:, ::-1]
                if "v" in view:
                    m = m[::-1, :]
                m = np.ascontiguousarray(m)
                if crop:
                    mc = cv2.resize(m, (CROP1 - CROP0, CROP1 - CROP0),
                                    interpolation=cv2.INTER_LINEAR)
                    up = np.zeros((IMG_H, IMG_W), dtype=np.float32)
                    up[CROP0:CROP1, CROP0:CROP1] = mc
                else:
                    up = cv2.resize(m, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
                m2 = up >= 0.5
                if m2.sum() == 0:
                    continue
                ys, xs = np.nonzero(m2)
                rows.append(dict(stem=stem, view=view or "o", conf=float(c),
                                 area=int(m2.sum()),
                                 r_norm=float(np.hypot(xs.mean() - DISK_C,
                                                       ys.mean() - DISK_C) / DISK_R),
                                 rle=encode_mask(m2.astype(np.uint8))))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(CACHE_DIR / f"fold{fold}_preds{tag}.parquet", index=False)
    print(f"cached {len(df)} instances over {len(val_files)} images -> fold{fold}_preds{tag}.parquet")


def sweep_fold(fold, conf_grid=None, area_grid=None, mask_grid=None, r_max=0.97,
               tag="", fill_holes=False, largest_cc=False):
    from scipy.ndimage import binary_fill_holes, label as cc_label
    conf_grid = conf_grid or [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    area_grid = area_grid or [32, 64, 128]
    preds = pd.read_parquet(CACHE_DIR / f"fold{fold}_preds{tag}.parquet")
    # mask-threshold variants only if the cache carries them
    mask_grid = [m for m in (mask_grid or ["lo", "mid", "hi"])
                 if m == "mid" or f"rle_{m}" in preds.columns]
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    val_files = sorted(folds.loc[folds["fold"] == fold, "file_name"])
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt_val = gt[gt["file_name"].isin(val_files)][["filament_id", "segmentation_rle"]]

    results = []
    gt_layer_cache: dict = {}
    for mvar in mask_grid:
        col = {"lo": "rle_lo", "mid": "rle", "hi": "rle_hi"}[mvar]
        by_stem = {}
        for stem, g in preds.groupby("stem"):
            g = g.sort_values("conf", ascending=False)
            by_stem[stem] = [(r.conf, r.r_norm, decode_rle(getattr(r, col)).astype(bool))
                             for r in g.itertuples()]
        for conf_t in conf_grid:
            for area_t in area_grid:
                rows = []
                for stem, items in by_stem.items():
                    taken = np.zeros((IMG_H, IMG_W), dtype=bool)
                    k = 0
                    for c, rn, m in items:
                        if c < conf_t or rn > r_max:
                            continue
                        mm = m & ~taken
                        if largest_cc and mm.any():
                            lab, n = cc_label(mm)
                            if n > 1:
                                sizes = np.bincount(lab.ravel())[1:]
                                mm = lab == (int(np.argmax(sizes)) + 1)
                        if fill_holes and mm.any():
                            mm = binary_fill_holes(mm) & ~taken
                        if mm.sum() < area_t:
                            continue
                        taken |= mm
                        k += 1
                        rows.append((f"{stem}_{k}", encode_mask(mm.astype(np.uint8))))
                pred_df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
                if pred_df.empty:
                    continue
                r = score(gt_val, pred_df, gt_cache=gt_layer_cache)
                results.append(dict(mask=mvar, conf=conf_t, min_area=area_t,
                                    fill=fill_holes, lcc=largest_cc,
                                    **{k: v for k, v in r.items()
                                       if k in ("pq", "sq", "rq", "tp", "fp", "fn")}))
                print(f"mask={mvar} conf={conf_t:.2f} area>={area_t}: PQ={r['pq']:.4f} "
                      f"SQ={r['sq']:.4f} RQ={r['rq']:.4f} TP={r['tp']} FP={r['fp']} FN={r['fn']}",
                      flush=True)
    out = pd.DataFrame(results).sort_values("pq", ascending=False)
    suffix = tag + ("_fill" if fill_holes else "") + ("_lcc" if largest_cc else "")
    out.to_csv(CACHE_DIR / f"fold{fold}_sweep{suffix}.csv", index=False)
    print("\nbest:\n", out.head(5).to_string(index=False))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cache", "sweep"])
    ap.add_argument("--weights")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--tag", default="")
    ap.add_argument("--crop", action="store_true")
    ap.add_argument("--tta", default="", help="extra views, e.g. 'h' or 'h,v,hv'")
    ap.add_argument("--nms", type=float, default=0.5, help="NMS iou threshold")
    ap.add_argument("--retina", action="store_true")
    ap.add_argument("--fill-holes", action="store_true")
    ap.add_argument("--largest-cc", action="store_true")
    ap.add_argument("--masks", default=None, help="comma list of mask variants, e.g. mid")
    ap.add_argument("--confs", default=None, help="comma list of conf thresholds")
    a = ap.parse_args()
    if a.mode == "cache":
        cache_fold(a.weights, a.fold, imgsz=a.imgsz, retina=a.retina, tag=a.tag,
                   crop=a.crop, tta=a.tta, nms_iou=a.nms)
    else:
        kw = dict(tag=a.tag, fill_holes=a.fill_holes, largest_cc=a.largest_cc)
        if a.masks:
            kw["mask_grid"] = a.masks.split(",")
        if a.confs:
            kw["conf_grid"] = [float(x) for x in a.confs.split(",")]
        sweep_fold(a.fold, **kw)
