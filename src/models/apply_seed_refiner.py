"""Run the seed refiner on detector boxes and score the lane it produces.

Training validates with GT boxes, which is the easy case; what the pipeline
actually gets is a detector box. This takes each detector candidate, throws away
its mask, keeps only its box, and re-segments the crop — so the comparison
against the detector's own masks on the SAME boxes isolates mask quality from
detection.

The detector's masks come from YOLO prototypes at input/4 and are stretched to
2048; the refiner works in a 1.5x-box square resized to 512, which for a typical
73 px box is an upscale rather than a downscale.

Reports, for a few detector confidence floors:
  detector masks   what that set of candidates scores today
  refined masks    the same boxes, re-segmented
optionally with box-jitter TTA (the notebook shifts the seed +-8 px and averages).

Usage:
  python src/models/apply_seed_refiner.py --fold 0 --weights outputs/runs_seed/seed_b3_512/best.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament import IMG_H, IMG_W
from filament.pq import score
from filament.rle import decode_rle, encode_mask
from build_seed_crops import square_bounds
from train_seed_refiner import seed_from_box, PAD_RATIO
from semantic_candidates import R_MAX

DATA = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
OUT = ROOT / "outputs" / "eda"


def log(*a):
    print(*a, flush=True)


def load(weights):
    import segmentation_models_pytorch as smp
    ck = torch.load(weights, map_location="cuda", weights_only=False)
    m = smp.Unet(encoder_name=ck["encoder"], encoder_weights=None, in_channels=3,
                 classes=1, decoder_attention_type="scse").cuda().eval()
    m.load_state_dict(ck["model"])
    return (m, int(ck["crop"]), float(ck.get("pad_ratio", PAD_RATIO)),
            float(ck.get("context", 1.5)))


def bbox_of(mask):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), \
        float(ys.max() - ys.min() + 1)


@torch.no_grad()
def refine_image(model, S, pad, img, boxes, jitter=0, thr=0.5, context=1.5):
    """Re-segment every box of one image; returns full-frame masks."""
    clahe = cv2.createCLAHE(2.0, (8, 8))
    out = []
    for bbox in boxes:
        x0, y0, x1, y1 = square_bounds(bbox, context=context)
        if x1 - x0 < 8 or y1 - y0 < 8:
            out.append(None)
            continue
        sub = cv2.resize(img[y0:y1, x0:x1], (S, S), interpolation=cv2.INTER_AREA)
        cl = clahe.apply(sub)
        sx, sy = S / (x1 - x0), S / (y1 - y0)
        bx, by = (bbox[0] - x0) * sx, (bbox[1] - y0) * sy
        bw, bh = bbox[2] * sx, bbox[3] * sy
        shifts = [(0, 0)]
        if jitter:
            j = jitter
            shifts += [(j, 0), (-j, 0), (0, j), (0, -j)]
        xs = []
        for dx, dy in shifts:
            seed = seed_from_box(bx + dx * sx, by + dy * sy, bw, bh, S, pad)
            xs.append(np.stack([sub.astype(np.float32) / 255.0,
                                cl.astype(np.float32) / 255.0, seed]))
        x = torch.from_numpy(np.stack(xs)).cuda()
        with torch.amp.autocast("cuda"):
            p = torch.sigmoid(model(x)).float().mean(dim=0)[0].cpu().numpy()
        small = (p >= thr).astype(np.uint8)
        full = np.zeros((IMG_H, IMG_W), np.uint8)
        full[y0:y1, x0:x1] = cv2.resize(small, (x1 - x0, y1 - y0),
                                        interpolation=cv2.INTER_NEAREST)
        out.append(full.astype(bool) if full.any() else None)
    return out


def deoverlap(items):
    """items: list of (conf, mask). Greedy by confidence, as the pipeline does."""
    taken = np.zeros((IMG_H, IMG_W), bool)
    kept = []
    for c, m in sorted(items, key=lambda t: -t[0]):
        mm = m & ~taken
        if mm.sum() < 64:
            continue
        taken |= mm
        kept.append(mm)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--weights", default="outputs/runs_seed/seed_b3_512/best.pt")
    ap.add_argument("--confs", default="0.05,0.20,0.28,0.40")
    ap.add_argument("--jitter", type=int, default=0)
    a = ap.parse_args()

    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    stems = [f.rsplit(".", 1)[0]
             for f in sorted(folds.loc[folds["fold"] == a.fold, "file_name"])]
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt_f = gt[gt["file_name"].isin({s + ".jpeg" for s in stems})]
    N = len(gt_f)

    det = pd.read_parquet(ROOT / "outputs" / "oof_cache" / f"fold{a.fold}_preds_e013.parquet")
    det = det[det["r_norm"] <= R_MAX]
    if "view" in det.columns:
        det = det[det["view"] == "o"]

    model, S, pad, ctx = load(ROOT / a.weights)
    log(f"refiner crop {S}, pad {pad}, context {ctx:g}, jitter {a.jitter}px")

    gt_cache, res = {}, {}
    for conf in [float(c) for c in a.confs.split(",")]:
        sel = det[det["conf"] >= conf]
        det_rows, ref_rows = [], []
        t0 = time.time()
        for stem, g in sel.groupby("stem", sort=False):
            img = cv2.imread(str(DATA / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            masks = [decode_rle(r).astype(bool) for r in g["rle"]]
            boxes = [bbox_of(m) for m in masks]
            ok = [i for i, b in enumerate(boxes) if b is not None]
            refined = refine_image(model, S, pad, img, [boxes[i] for i in ok],
                                   jitter=a.jitter, context=ctx)
            confs = g["conf"].to_numpy()
            for k, m in enumerate(deoverlap([(confs[i], masks[i]) for i in ok])):
                det_rows.append((f"{stem}_{k}", encode_mask(m.astype(np.uint8))))
            pairs = [(confs[i], r) for i, r in zip(ok, refined) if r is not None]
            for k, m in enumerate(deoverlap(pairs)):
                ref_rows.append((f"{stem}_{k}", encode_mask(m.astype(np.uint8))))
        for name, rows in (("detector masks", det_rows), ("refined masks", ref_rows)):
            df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
            r = score(gt_f[["filament_id", "segmentation_rle"]], df,
                      gt_cache=gt_cache, return_details=True)
            hit = r["matches"]["gt_filament_id"].nunique()
            log(f"conf>={conf:.2f}  {name:15s} preds {len(df):5d}  "
                f"reach {hit:4d}/{N}={hit/N:.4f}  PQ {r['pq']:.4f} SQ {r['sq']:.4f} "
                f"TP {r['tp']:4d} FP {r['fp']:5d} FN {r['fn']:4d}")
            res[f"{conf}|{name}"] = dict(n=len(df), reach=hit / N, pq=r["pq"],
                                         sq=r["sq"], tp=r["tp"], fp=r["fp"],
                                         fn=r["fn"])
        log(f"  ({time.time()-t0:.0f}s)")

    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"_j{a.jitter}" if a.jitter else ""
    with open(OUT / f"seed_refiner_fold{a.fold}{tag}.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    log(f"saved -> {OUT / f'seed_refiner_fold{a.fold}{tag}.json'}")


if __name__ == "__main__":
    main()
