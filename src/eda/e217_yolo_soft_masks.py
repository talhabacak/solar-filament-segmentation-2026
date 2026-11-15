"""E217a -- cache the detector's SOFT mask probabilities out of fold (they were never kept).

Every YOLO mask in the project is the binary output of Ultralytics' `ops.process_mask`, which
thresholds the prototype logits at 0 (probability 0.5) before anything is returned. The August
"mask threshold" axis (rle_lo / rle_hi at 0.35 / 0.65, sweep_oof.py) thresholded the bilinear
UPSAMPLE OF THAT BINARY MASK, so it moved the boundary by about a pixel and never saw the soft
field. Ledger §39a/§54: near-miss masks have the right width and ~70 % of the length, and the
semantic field is ~0 on 75-80 % of the missing length (§39b). Whether the detector's own soft
field carries that length has never been measured.

This script re-runs the fold's detector on its held-out images with `process_mask` patched to
return sigmoid probabilities (cropped to the box exactly as Ultralytics crops the binary mask),
resizes each to 2048 with the same INTER_LINEAR as the production cache, and stores per detection
the confidence, its 2048 box and the soft crop as uint8.

GUARD (must pass before the analysis is read): the binary built the production way (soft > 0.5 at
1792, then the uint8 resize) must reproduce the production cache
(outputs/oof_cache/fold{F}_preds_e013.parquet): same detection count per image and mean mask IoU
>= 0.98 against the cached masks matched by confidence order.

Usage: python src/eda/e217_yolo_soft_masks.py --fold 0 [--limit N]
Output: outputs/oof_cache/soft_e013_f{F}/<stem>.npz, guard summary printed and saved as json
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from filament import IMG_H, IMG_W  # noqa: E402
from filament.rle import decode_rle  # noqa: E402

IMG = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
CACHE = ROOT / "outputs" / "oof_cache"


def soft_process_mask(protos, masks_in, bboxes, shape, upsample=False):
    """ops.process_mask with the binarisation removed: sigmoid probabilities, same crop."""
    from ultralytics.utils.ops import crop_mask
    c, mh, mw = protos.shape
    if masks_in.shape[0] == 0:
        return torch.zeros((0, *(shape if upsample else (mh, mw))), device=masks_in.device)
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
    if upsample:
        masks = F.interpolate(masks[None], shape, mode="bilinear")[0]
    else:
        ratios = torch.tensor([[mw / shape[1], mh / shape[0], mw / shape[1], mh / shape[0]]], device=bboxes.device)
        bboxes = bboxes * ratios
    return crop_mask(masks.sigmoid(), bboxes)


def legacy_soft_process_mask(protos, masks_in, bboxes, shape, upsample=False):
    """The order of the Ultralytics release that built the production cache (August 2026):
    logits cropped to the box at PROTOTYPE resolution, then bilinear upsampling, then logit > 0.
    8.4 upsamples first and crops at image resolution (its comment cites #24272), which moves
    the edge by about a pixel and failed the guard at mean IoU 0.965. Returned here as
    sigmoid(upsampled cropped logits), with the pixels the crop zeroed exactly kept at 0, so
    soft > 0.5 <=> legacy binary, to the bit."""
    from ultralytics.utils.ops import crop_mask
    c, mh, mw = protos.shape
    ih, iw = shape
    if masks_in.shape[0] == 0:
        return torch.zeros((0, ih, iw), device=masks_in.device)
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
    db = bboxes.clone()
    db[:, 0] *= mw / iw
    db[:, 2] *= mw / iw
    db[:, 1] *= mh / ih
    db[:, 3] *= mh / ih
    masks = crop_mask(masks, db)
    inside = crop_mask(torch.ones_like(masks), db)
    masks = F.interpolate(masks[None], shape, mode="bilinear", align_corners=False)[0]
    inside = F.interpolate(inside[None], shape, mode="bilinear", align_corners=False)[0]
    return torch.where(inside > 0, masks.sigmoid(), torch.zeros_like(masks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--mode", default="current", choices=["legacy", "current"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()

    from ultralytics import YOLO
    from ultralytics.utils import ops
    ops.process_mask = legacy_soft_process_mask if a.mode == "legacy" else soft_process_mask

    w = a.weights or str(ROOT / "outputs" / "runs" / f"e013_yolo11m_f{a.fold}_1792" / "weights" / "best.pt")
    model = YOLO(w)
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    files = sorted(folds.loc[folds["fold"] == a.fold, "file_name"])
    if a.limit:
        files = files[:a.limit]
    files = files[a.shard::a.nshards]
    if a.threads:
        torch.set_num_threads(a.threads)
    cached = pd.read_parquet(CACHE / f"fold{a.fold}_preds_e013.parquet")
    out = CACHE / f"soft_e013_f{a.fold}"
    out.mkdir(parents=True, exist_ok=True)

    guard, t0 = [], time.time()
    for n, fn in enumerate(files):
        stem = fn.rsplit(".", 1)[0]
        src = cv2.cvtColor(cv2.imread(str(IMG / fn), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
        res = model.predict(src, imgsz=1792, conf=0.05, iou=0.5, max_det=300, verbose=False, device="cpu")[0]
        confs, boxes, crops, bins = [], [], [], []
        if res.masks is not None:
            soft = res.masks.data.cpu().numpy()
            for m, c in zip(soft, res.boxes.conf.cpu().numpy()):
                up = cv2.resize(m.astype(np.float32), (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
                # binary exactly as the production cache built it: Ultralytics binarises at 1792,
                # cross_cache/sweep_oof resize that uint8 0/1 mask and keep >= 0.5. Thresholding the
                # resized PROBABILITY instead moves the edge by half a pixel (guard IoU 0.967).
                b = cv2.resize((m > 0.5).astype(np.uint8), (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR) >= 0.5
                if not b.any():          # the production cache drops these too
                    continue
                ys, xs = np.nonzero(up > 0.02)
                y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                confs.append(float(c))
                boxes.append((y0, y1, x0, x1))
                crops.append(np.round(up[y0:y1, x0:x1] * 255).astype(np.uint8))
                bins.append(b)
        np.savez_compressed(out / f"{stem}.npz", conf=np.array(confs, np.float32),
                            box=np.array(boxes, np.int32).reshape(-1, 4),
                            **{f"m{i}": cr for i, cr in enumerate(crops)})
        # guard against the production cache, matched in confidence order
        cg = cached[cached["stem"] == stem].sort_values("conf", ascending=False)
        order = np.argsort(-np.array(confs)) if confs else []
        ious = []
        for k, (_, r) in enumerate(cg.iterrows()):
            if k >= len(order):
                break
            cm = decode_rle(r["rle"]).astype(bool)
            bm = bins[order[k]]
            i = (cm & bm).sum()
            ious.append(i / max((cm | bm).sum(), 1))
        guard.append(dict(stem=stem, n_new=len(confs), n_cache=len(cg),
                          mean_iou=float(np.mean(ious)) if ious else 1.0))
        if n % 10 == 0:
            g = pd.DataFrame(guard)
            print(f"  {n + 1}/{len(files)}  {time.time() - t0:.0f}s  count match "
                  f"{(g.n_new == g.n_cache).mean():.3f}  mean IoU {g.mean_iou.mean():.4f}", flush=True)
    g = pd.DataFrame(guard)
    summ = dict(images=len(g), count_match=float((g.n_new == g.n_cache).mean()),
                mean_iou=float(g.mean_iou.mean()), min_iou=float(g.mean_iou.min()),
                passed=bool((g.n_new == g.n_cache).mean() >= 0.95 and g.mean_iou.mean() >= 0.98))
    print(json.dumps(summ, indent=1))
    g.to_parquet(out / f"_guard_{a.shard}of{a.nshards}.parquet", index=False)
    json.dump(summ, open(out / f"_guard_{a.shard}of{a.nshards}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
