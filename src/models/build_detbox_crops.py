"""Seed crops built from REAL detector boxes, not jittered GT boxes.

The seed refiner beats the detector's masks when it is given GT boxes (mean IoU
0.708 against SQ 0.670) and loses when it is given detector boxes (PQ 0.4049
against 0.4111). Measured on fold 0's matched pairs, the detector's box error is
larger than the jitter it trained on:

    centre |dx| p90   0.106   trained sigma 0.05 -> p90 ~0.082
    log scale p90     0.255   trained sigma 0.08 -> p90 ~0.13

so its box SIZE error is about twice what the model ever saw. Rather than widen
a synthetic jitter and hope it matches, this takes the actual out-of-fold
detector boxes as seeds, with the GT mask they match as the target — train and
apply then draw from the same distribution by construction.

Candidates that match no GT are kept with an all-zero target: at inference the
refiner sees those too, and a lane that hallucinates a filament inside every box
would add FPs.

Usage: python src/models/build_detbox_crops.py --conf 0.20
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
from filament.pq import score
from filament.rle import decode_rle, rles_to_layers
from build_seed_crops import square_bounds
from semantic_candidates import R_MAX

DATA = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
OUT = ROOT / "data" / "detbox_crops"
CROP = 512


def log(*a):
    print(*a, flush=True)


def bbox_of(mask):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), \
        float(ys.max() - ys.min() + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--crop", type=int, default=CROP)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    S = a.crop

    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    parts = []
    for fold in range(5):
        d = pd.read_parquet(ROOT / "outputs" / "oof_cache" / f"fold{fold}_preds_e013.parquet")
        d = d[(d["conf"] >= a.conf) & (d["r_norm"] <= R_MAX)]
        if "view" in d.columns:
            d = d[d["view"] == "o"]
        d = d.reset_index(drop=True)
        d["fold"] = fold
        d["cand_id"] = [f"{s}_{fold}x{i}" for i, s in enumerate(d["stem"])]
        parts.append(d)
    det = pd.concat(parts, ignore_index=True)
    log(f"{len(det)} detector candidates at conf>={a.conf} over 5 folds")

    gt_sub = gt[gt["file_name"].isin(set(det["stem"] + ".jpeg"))]
    pred = det[["cand_id", "rle"]].rename(
        columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
    t0 = time.time()
    r = score(gt_sub[["filament_id", "segmentation_rle"]], pred, return_details=True)
    log(f"scored in {time.time()-t0:.0f}s; {len(r['matches'])} matched pairs")
    # a candidate can match one GT per annotator entry; the target is the union
    # of those, which is the object the consensus vote would produce
    gt_for = r["matches"].groupby("pred_filament_id")["gt_filament_id"].apply(list).to_dict()
    rle_g = dict(zip(gt_sub["filament_id"], gt_sub["segmentation_rle"]))

    n = len(det)
    imgs = np.lib.format.open_memmap(OUT / f"img_{S}.npy", mode="w+",
                                     dtype=np.uint8, shape=(n, S, S))
    tgts = np.lib.format.open_memmap(OUT / f"tgt_{S}.npy", mode="w+",
                                     dtype=np.uint8, shape=(n, S, S))
    meta, k, t0 = [], 0, time.time()
    for stem, g in det.groupby("stem", sort=False):
        img = cv2.imread(str(DATA / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        for row in g.itertuples():
            m = decode_rle(row.rle).astype(bool)
            bb = bbox_of(m)
            if bb is None:
                continue
            x0, y0, x1, y1 = square_bounds(bb)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            gids = gt_for.get(row.cand_id, [])
            if gids:
                stack = rles_to_layers([rle_g[i] for i in gids])
                tgt_full = (stack.mean(axis=0) >= 0.5)
            else:
                tgt_full = np.zeros((IMG_H, IMG_W), bool)
            imgs[k] = cv2.resize(img[y0:y1, x0:x1], (S, S), interpolation=cv2.INTER_AREA)
            tgts[k] = cv2.resize(tgt_full[y0:y1, x0:x1].astype(np.uint8), (S, S),
                                 interpolation=cv2.INTER_NEAREST)
            sx, sy = S / (x1 - x0), S / (y1 - y0)
            meta.append(dict(idx=k, cand_id=row.cand_id, stem=stem,
                             fold=int(row.fold), conf=float(row.conf),
                             n_gt=len(gids),
                             bx=(bb[0] - x0) * sx, by=(bb[1] - y0) * sy,
                             bw=bb[2] * sx, bh=bb[3] * sy))
            k += 1
        if k % 2000 < len(g):
            log(f"  {k}/{n} crops ({time.time()-t0:.0f}s)")
    imgs.flush()
    tgts.flush()
    md = pd.DataFrame(meta)
    md.to_parquet(OUT / f"meta_{S}.parquet", index=False)
    with open(OUT / f"index_{S}.json", "w") as f:
        json.dump(dict(n=k, crop=S, conf=a.conf), f, indent=1)
    log(f"wrote {k} crops ({int((md['n_gt']>0).sum())} with a target, "
        f"{int((md['n_gt']==0).sum())} empty) -> {OUT} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
