"""How much of the SQ gap is width calibration, and how much is shape?

Measured context. Under oracle selection our pool scores PQ 0.5078 with
SQ 0.6666. A single human annotator scored against the others gets SQ 0.6337 —
our masks are already better than one person's — but the majority-consensus mask
gets 0.7903. Closing that 0.12 at the same TP/FP/FN would put PQ at ~0.60.

Two very different projects hide behind that number:

  width   our contours are systematically too fat or too thin, and a per-instance
          radius would fix them. Cheap: predict one scalar per candidate.
  shape   the boundary is in the wrong place in a way no uniform offset fixes.
          Expensive: a crop-level refinement model at native resolution.

The global 5% shrink moved SQ 0.6632 -> 0.6710, so *some* of it is width. This
measures the ceiling of the width story honestly: for every matched
(candidate, GT) pair, sweep a signed morphological offset and record the IoU at
the best global offset and at the best per-instance offset. Whatever the
per-instance oracle cannot reach is shape, and only a refiner can take it.

Work happens in a padded bbox crop rather than at 2048^2.

Usage: python src/eda/sq_headroom.py --sample 1500
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament.rle import decode_rle

OUT = ROOT / "outputs" / "eda"
OFFSETS = list(range(-6, 7))          # negative = erode, positive = dilate
PAD = 24


def log(*a):
    print(*a, flush=True)


def offset_masks(m):
    """Signed distance-transform offsets of a binary mask, in a tight crop."""
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return None, None
    y0, y1 = max(0, ys.min() - PAD), min(m.shape[0], ys.max() + PAD + 1)
    x0, x1 = max(0, xs.min() - PAD), min(m.shape[1], xs.max() + PAD + 1)
    sub = m[y0:y1, x0:x1].astype(np.uint8)
    din = cv2.distanceTransform(sub, cv2.DIST_L2, 5)          # inside depth
    dout = cv2.distanceTransform(1 - sub, cv2.DIST_L2, 5)     # outside distance
    sdf = dout - din                                          # <0 inside
    return {k: (sdf <= k).astype(np.uint8) for k in OFFSETS}, (y0, y1, x0, x1)


def iou_crop(a, b):
    i = int(np.logical_and(a, b).sum())
    u = int(np.logical_or(a, b).sum())
    return i / u if u else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=1500)
    ap.add_argument("--pool", default="outputs/selector/fused_pool_all.parquet")
    a = ap.parse_args()

    cand = pd.read_parquet(ROOT / a.pool)
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    matched = cand[cand["weight"] >= 1]
    log(f"pool {len(cand)}, matched {len(matched)}")

    # rebuild the (candidate -> GT rows it matched) pairing with the exact scorer
    from filament.pq import score
    files = set(cand["stem"] + ".jpeg")
    gt_sub = gt[gt["file_name"].isin(files)]
    pred = matched[["cand_id", "rle"]].rename(
        columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
    res = score(gt_sub[["filament_id", "segmentation_rle"]], pred, return_details=True)
    pairs = res["matches"][["pred_filament_id", "gt_filament_id", "iou"]]
    log(f"matched pairs: {len(pairs)} (scorer SQ {res['sq']:.4f})")

    rng = np.random.default_rng(0)
    take = pairs.iloc[rng.choice(len(pairs), size=min(a.sample, len(pairs)), replace=False)]
    rle_c = dict(zip(cand["cand_id"], cand["rle"]))
    rle_g = dict(zip(gt_sub["filament_id"], gt_sub["segmentation_rle"]))

    rows = []
    for n, r in enumerate(take.itertuples()):
        m = decode_rle(rle_c[r.pred_filament_id]).astype(bool)
        offs, box = offset_masks(m)
        if offs is None:
            continue
        y0, y1, x0, x1 = box
        g = decode_rle(rle_g[r.gt_filament_id]).astype(bool)[y0:y1, x0:x1]
        ious = {k: iou_crop(v, g) for k, v in offs.items()}
        best_k = max(ious, key=ious.get)
        rows.append(dict(base=ious[0], best_k=best_k, best=ious[best_k],
                         **{f"o{k}": v for k, v in ious.items()}))
        if (n + 1) % 300 == 0:
            log(f"  {n+1}/{len(take)}")
    df = pd.DataFrame(rows)
    log(f"\nmeasured {len(df)} matched pairs")

    means = {k: df[f"o{k}"].mean() for k in OFFSETS}
    g_k = max(means, key=means.get)
    log(f"as-is mean IoU                {df['base'].mean():.4f}")
    log(f"best GLOBAL offset {g_k:+d} px      {means[g_k]:.4f}  "
        f"({means[g_k]-df['base'].mean():+.4f})")
    log(f"per-instance ORACLE offset    {df['best'].mean():.4f}  "
        f"({df['best'].mean()-df['base'].mean():+.4f})")
    log("  offset sweep: " + "  ".join(f"{k:+d}:{means[k]:.3f}" for k in OFFSETS))
    log(f"  oracle offset spread: p10 {df['best_k'].quantile(.1):+.0f} "
        f"p50 {df['best_k'].median():+.0f} p90 {df['best_k'].quantile(.9):+.0f}")

    consensus_sq = 0.7903
    reach = df["best"].mean()
    log(f"\nshape residual: consensus SQ {consensus_sq:.4f} - per-instance oracle "
        f"{reach:.4f} = {consensus_sq-reach:+.4f} that NO uniform offset can reach")

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "sq_headroom.parquet", index=False)
    with open(OUT / "sq_headroom.json", "w") as f:
        json.dump(dict(n=len(df), as_is=float(df["base"].mean()),
                       global_offset=int(g_k), global_iou=float(means[g_k]),
                       oracle_iou=float(df["best"].mean()),
                       consensus_sq=consensus_sq,
                       sweep={str(k): float(v) for k, v in means.items()}),
                  f, indent=1)
    log(f"saved -> {OUT/'sq_headroom.json'}")


if __name__ == "__main__":
    main()
