"""Crops for the mask refiner, at native 2048 resolution.

Why this stage exists (measured, not assumed). Under oracle selection the pool
scores PQ 0.5078 at SQ 0.6666. Perfect masks at the same matching would be
0.7618 and zero FN only 0.6262, so mask quality is worth about twice recall. The
reachable target is the majority-consensus mask, which scores SQ 0.7903 against
individual annotators — a single human only manages 0.6337, so we are already
past one person and chasing the consensus.

Decomposing that 0.12 gap (sq_headroom.py, 1500 matched pairs):

    best GLOBAL offset          +0.0000   nothing left, fusion already sized right
    per-instance oracle offset  +0.0305   a width regressor could reach this
    shape residual              +0.0906   no uniform offset reaches it

Three quarters of the gap is shape, which is what a refiner is for. It sees a
320x320 window at FULL resolution — the semantic lane predicts at 1536 and
upsamples, so a filament ~11 px wide loses its boundary to interpolation, and
93.4% of candidates fit in that window (median bbox side 73 px).

Two crop sets:
  gt    one per GT instance (8199), target only. The input mask is synthesised
        by perturbing the target at train time, which is how CascadePSP-style
        refiners get enough data — 4161 matched candidates alone is too few.
  cand  one per pooled candidate, carrying the real mask the pipeline produced,
        for honest evaluation and for fine-tuning on the true error distribution.

Usage:
  python src/models/build_refine_data.py --which gt
  python src/models/build_refine_data.py --which cand
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
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament import IMG_H, IMG_W
from filament.rle import decode_rle, rles_to_layers

DATA = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
OUT = ROOT / "data" / "refine_crops"
CROP = 320


def log(*a):
    print(*a, flush=True)


def crop_box(mask):
    """320x320 window centred on the mask, clamped to the frame."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    cy, cx = int(ys.mean()), int(xs.mean())
    y0 = int(np.clip(cy - CROP // 2, 0, IMG_H - CROP))
    x0 = int(np.clip(cx - CROP // 2, 0, IMG_W - CROP))
    return y0, x0


def build_gt(gt):
    """One crop per GT instance: image + that annotator's mask as the target."""
    n = len(gt)
    imgs = np.lib.format.open_memmap(OUT / "gt_img.npy", mode="w+",
                                     dtype=np.uint8, shape=(n, CROP, CROP))
    msks = np.lib.format.open_memmap(OUT / "gt_msk.npy", mode="w+",
                                     dtype=np.uint8, shape=(n, CROP, CROP))
    meta, t0, k = [], time.time(), 0
    for fn, g in gt.groupby("file_name", sort=False):
        img = cv2.imread(str(DATA / fn), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        layers = rles_to_layers(g["segmentation_rle"].tolist())
        for row, m in zip(g.itertuples(), layers):
            box = crop_box(m)
            if box is None:
                continue
            y0, x0 = box
            imgs[k] = img[y0:y0 + CROP, x0:x0 + CROP]
            msks[k] = m[y0:y0 + CROP, x0:x0 + CROP]
            meta.append(dict(idx=k, filament_id=row.filament_id, file_name=fn,
                             entry_id=row.entry_id, category_id=int(row.category_id),
                             y0=y0, x0=x0))
            k += 1
        if len(meta) % 1000 < len(g):
            log(f"  {k}/{n} crops ({time.time()-t0:.0f}s)")
    imgs.flush()
    msks.flush()
    log(f"gt crops: {k} written ({time.time()-t0:.0f}s)")
    return meta, k


def build_cand(cand, gt):
    """One crop per pooled candidate: image, the produced mask, and the
    majority mask of whatever GT rows it matched (all zero if it matched none)."""
    from filament.pq import score
    files = set(cand["stem"] + ".jpeg")
    gt_sub = gt[gt["file_name"].isin(files)]
    pred = cand[["cand_id", "rle"]].rename(
        columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
    res = score(gt_sub[["filament_id", "segmentation_rle"]], pred, return_details=True)
    pairs = res["matches"]
    log(f"matched pairs {len(pairs)} over {pairs['pred_filament_id'].nunique()} candidates")
    gt_for = pairs.groupby("pred_filament_id")["gt_filament_id"].apply(list).to_dict()
    rle_g = dict(zip(gt_sub["filament_id"], gt_sub["segmentation_rle"]))

    n = len(cand)
    imgs = np.lib.format.open_memmap(OUT / "cand_img.npy", mode="w+",
                                     dtype=np.uint8, shape=(n, CROP, CROP))
    inm = np.lib.format.open_memmap(OUT / "cand_in.npy", mode="w+",
                                    dtype=np.uint8, shape=(n, CROP, CROP))
    tgt = np.lib.format.open_memmap(OUT / "cand_tgt.npy", mode="w+",
                                    dtype=np.uint8, shape=(n, CROP, CROP))
    meta, t0, k = [], time.time(), 0
    for stem, g in cand.groupby("stem", sort=False):
        img = cv2.imread(str(DATA / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        for row in g.itertuples():
            m = decode_rle(row.rle)
            box = crop_box(m)
            if box is None:
                continue
            y0, x0 = box
            gids = gt_for.get(row.cand_id, [])
            if gids:
                stack = rles_to_layers([rle_g[i] for i in gids])
                # majority over the annotators who drew this filament: the
                # object that scores SQ 0.7903 against individuals
                maj = (stack.mean(axis=0) >= 0.5).astype(np.uint8)
                tgt[k] = maj[y0:y0 + CROP, x0:x0 + CROP]
            imgs[k] = img[y0:y0 + CROP, x0:x0 + CROP]
            inm[k] = m[y0:y0 + CROP, x0:x0 + CROP]
            meta.append(dict(idx=k, cand_id=row.cand_id, stem=stem,
                             fold=int(row.fold), n_gt=len(gids),
                             y0=y0, x0=x0))
            k += 1
        if k % 1000 < len(g):
            log(f"  {k}/{n} crops ({time.time()-t0:.0f}s)")
    for arr in (imgs, inm, tgt):
        arr.flush()
    log(f"cand crops: {k} written ({time.time()-t0:.0f}s)")
    return meta, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["gt", "cand"], required=True)
    ap.add_argument("--pool", default="outputs/selector/fused_pool_all.parquet")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    if a.which == "gt":
        meta, k = build_gt(gt)
    else:
        meta, k = build_cand(pd.read_parquet(ROOT / a.pool), gt)
    pd.DataFrame(meta).to_parquet(OUT / f"{a.which}_meta.parquet", index=False)
    with open(OUT / f"{a.which}_index.json", "w") as f:
        json.dump(dict(n=k, crop=CROP), f, indent=1)
    log(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
