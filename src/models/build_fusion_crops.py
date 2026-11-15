"""E191 -- crops for a refiner that sees BOTH lanes, not just their fused result.

Talha, 2026-09-14: "how right would it be to guide the selector with the GT mask -- can it learn
to produce the true mask from the YOLO and U-Net outputs?"

Training on ground truth is ordinary supervised learning; what would be illegitimate is using
ground truth at inference, which is the trap the contour-level oracle fell into (ledger 25). A
refiner sees only the image and the models' own outputs, so it is legal and cross-fittable.

This project already built a refiner once (`train_refiner.py`, +0.0043 then overfitting). Its
input was the image, the CURRENT MASK and that mask's signed distance field -- i.e. the fused
result, after the confidence-weighted vote had already thrown away the disagreement between the
lanes. The measured structure says that is exactly the wrong place to cut: the boundary band is
where members disagree, the vote pulls the fused mask toward the intersection, and the band's
candidates are systematically TOO SMALL (ground-truth-only pixels 608 k against candidate-only
248 k, ledger 29).

So the crops here carry five channels, and two of them are new:

  0  image                 the raw frame at full resolution
  1  U-Net probability     the out-of-fold ensemble field, 0..1          <- new
  2  detector evidence     the union of the YOLO masks in this cluster    <- new
  3  candidate mask        what the pipeline currently proposes
  4  signed distance       of the candidate mask, clipped, so the boundary is explicit

Target: the ground-truth copy the candidate matches best (IoU > 0.05); candidates that match
nothing get an empty target, which teaches the refiner to erase a hallucination rather than
grow it.

Crop is 384 px at full resolution, centred on the candidate; the median candidate bbox side is
73 px, so this holds the object plus its neighbourhood -- and the neighbourhood is the point,
since the addressable misses are crowded (3 neighbours within 200 px against 1 for matched).

Usage: python src/models/build_fusion_crops.py [--crop 384]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/eda", "src/models"):
    sys.path.insert(0, str(ROOT / p))
from filament.rle import decode_rle  # noqa: E402
from filament.pq import _entry_iou_matrix  # noqa: E402

OUT = ROOT / "outputs" / "eda"
SEM = ROOT / "outputs" / "semantic"
OOF = ROOT / "outputs" / "oof_cache"
IMG = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
DEST = ROOT / "data" / "fusion_crops"


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", type=int, default=384)
    a = ap.parse_args()
    C = a.crop
    DEST.mkdir(parents=True, exist_ok=True)
    folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    pool = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    pool["fold"] = pool["stem"].map(fold_of).astype(int)
    probs = {k: np.load(SEM / f"fold{k}_probs_blob5ens.npz") for k in range(5)}
    det = {k: pd.read_parquet(OOF / f"fold{k}_preds_e013.parquet") for k in range(5)}

    X = np.lib.format.open_memmap(DEST / f"x_{C}.npy", mode="w+", dtype=np.uint8,
                                  shape=(len(pool), 5, C, C))
    Y = np.lib.format.open_memmap(DEST / f"y_{C}.npy", mode="w+", dtype=np.uint8,
                                  shape=(len(pool), C, C))
    meta = []
    n = 0
    for stem, g in pool.groupby("stem"):
        f = fold_of[stem]
        if stem not in probs[f].files:
            continue
        img = cv2.imread(str(IMG / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        pr = cv2.resize(probs[f][stem], (2048, 2048), interpolation=cv2.INTER_LINEAR)
        dg = det[f][det[f]["stem"] == stem]
        dmask = np.zeros((2048, 2048), np.uint8)
        for r in dg["rle"]:
            dmask |= decode_rle(r).astype(np.uint8)
        gg = gt[gt["stem"] == stem]
        G = np.stack([decode_rle(r).astype(np.uint8) for r in gg["segmentation_rle"]]) if len(gg) else None
        for r in g.itertuples():
            m = decode_rle(r.rle).astype(np.uint8)
            ys, xs = np.nonzero(m)
            if len(ys) < 16:
                continue
            cy, cx = int(ys.mean()), int(xs.mean())
            y0 = int(np.clip(cy - C // 2, 0, 2048 - C))
            x0 = int(np.clip(cx - C // 2, 0, 2048 - C))
            sl = (slice(y0, y0 + C), slice(x0, x0 + C))
            cm = m[sl]
            # signed distance of the candidate boundary, clipped to +/-32 px and rescaled
            din = cv2.distanceTransform(cm, cv2.DIST_L2, 3)
            dout = cv2.distanceTransform(1 - cm, cv2.DIST_L2, 3)
            sdf = np.clip(din - dout, -32, 32)
            X[n, 0] = img[sl]
            X[n, 1] = pr[sl]
            X[n, 2] = dmask[sl] * 255
            X[n, 3] = cm * 255
            X[n, 4] = ((sdf + 32) * (255.0 / 64.0)).astype(np.uint8)
            tgt = np.zeros((C, C), np.uint8)
            best = 0.0
            if G is not None:
                iou = _entry_iou_matrix(G, m[None])[:, 0]
                j = int(np.argmax(iou))
                best = float(iou[j])
                if best > 0.05:
                    tgt = G[j][sl] * 255
            Y[n] = tgt
            meta.append(dict(cand_id=r.cand_id, stem=stem, fold=f, y0=y0, x0=x0,
                             best_iou=best, y=int(r.y), iou_sum=float(r.iou_sum)))
            n += 1
        if len(meta) and len(meta) % 2000 < 20:
            log(f"  {len(meta)} crops")
    X.flush(); Y.flush()
    md = pd.DataFrame(meta)
    md.to_parquet(DEST / f"meta_{C}.parquet", index=False)
    log(f"\n{n} crops of {C}px written to {DEST}")
    log(f"  with a target (best IoU > 0.05): {int((md['best_iou'] > 0.05).sum())} "
        f"({(md['best_iou'] > 0.05).mean():.1%})")
    log(f"  in the boundary band (0.05 < IoU <= 0.5): {int(((md['best_iou'] > 0.05) & (md['best_iou'] <= 0.5)).sum())}")
    log("CROPS_DONE")


if __name__ == "__main__":
    main()
