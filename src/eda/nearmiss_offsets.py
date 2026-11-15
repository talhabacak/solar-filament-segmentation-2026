"""The population the scorer never shows us: candidates that just miss IoU 0.5.

Why this is worth a separate measurement. `sq_headroom.py` swept a signed
morphological offset over matched pairs and found no global offset beats
identity (mean IoU 0.6692 at k=0 against 0.6589 at k=-2 and 0.6336 at k=+1),
with a per-instance oracle worth +0.0305. But it drew its pairs from
`score(...)["matches"]`, and the scorer only returns pairs above the threshold.
Every pair sitting at IoU 0.45 was invisible to it.

That band is where the money is, because of how PQ moves. Improving an
already-matched pair from 0.60 to 0.65 adds 0.05 to the numerator. Pushing a
pair from 0.49 to 0.51 converts one FN and one FP into one TP: the denominator
changes by +1 - 0.5 - 0.5 = 0 and the numerator gains the whole ~0.51. A
crossing is worth roughly ten times a typical SQ nudge, and the ledger's largest
open block -- "same detections, correct mask: FP 860 -> 338, RQ 0.5922 ->
0.7810", about 0.19 PQ -- is made of exactly these.

Two numbers come out, and they answer different questions:

  global offset   what a single deployed constant would do, scored NET. Crossings
                  are not free: the same offset drags some matched pairs back
                  below 0.5, and each of those costs a TP the same way. Only the
                  net matters, and `sq_headroom` could not see the crossing half
                  of that trade at all.
  per-instance    the ceiling for any method that predicts one scalar per
                  candidate. Whatever this cannot reach is shape, not width, and
                  only a refiner takes it.

Pairing note: above IoU 0.5 the matching is provably one-to-one (two predictions
cannot each cover more than half of the same ground truth), so counting
crossings needs no assignment step. Below the threshold a candidate may overlap
several ground-truth rows in an entry; we take its best, which is the pair that
would become the match if it crossed.

Usage: python src/eda/nearmiss_offsets.py --shrink 0.05
"""

import argparse
import json
import sys
import time
from collections import defaultdict
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
BAND = (0.25, 0.50)                   # the near-miss window, inclusive of neither end


def log(*a):
    print(*a, flush=True)


def crop_of(m):
    """Tight bounding crop of a binary mask plus its origin, or None if empty."""
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return None
    return (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)


def iou_boxed(a_box, a_sub, b_box, b_sub):
    """IoU of two masks held as (bbox, cropped array), without going back to 2048^2."""
    ay0, ay1, ax0, ax1 = a_box
    by0, by1, bx0, bx1 = b_box
    y0, y1 = max(ay0, by0), min(ay1, by1)
    x0, x1 = max(ax0, bx0), min(ax1, bx1)
    asum = int(a_sub.sum())
    bsum = int(b_sub.sum())
    if y0 >= y1 or x0 >= x1:
        return 0.0, asum, bsum
    a = a_sub[y0 - ay0:y1 - ay0, x0 - ax0:x1 - ax0]
    b = b_sub[y0 - by0:y1 - by0, x0 - bx0:x1 - bx0]
    inter = int(np.logical_and(a, b).sum())
    union = asum + bsum - inter
    return (inter / union if union else 0.0), asum, bsum


def offset_variants(m_full):
    """Signed distance-transform offsets, returned as (bbox, cropped array) pairs.

    The crop is padded by PAD so a dilation of up to PAD pixels stays inside it.
    """
    ys, xs = np.nonzero(m_full)
    if len(ys) == 0:
        return None
    y0 = max(0, int(ys.min()) - PAD)
    y1 = min(m_full.shape[0], int(ys.max()) + PAD + 1)
    x0 = max(0, int(xs.min()) - PAD)
    x1 = min(m_full.shape[1], int(xs.max()) + PAD + 1)
    sub = m_full[y0:y1, x0:x1].astype(np.uint8)
    din = cv2.distanceTransform(sub, cv2.DIST_L2, 5)
    dout = cv2.distanceTransform(1 - sub, cv2.DIST_L2, 5)
    sdf = dout - din
    box = (y0, y1, x0, x1)
    return {k: (box, (sdf <= k)) for k in OFFSETS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shrink", type=float, default=0.05,
                    help="production mask shrink applied before measuring")
    ap.add_argument("--pool", default="fused_pool_all.parquet")
    a = ap.parse_args()
    from mask_shrink import shrink_frac

    cand = pd.read_parquet(ROOT / "outputs" / "selector" / a.pool)
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt = gt[gt["stem"].isin(set(cand["stem"]))]
    log(f"pool {len(cand)} candidates over {cand['stem'].nunique()} stems; "
        f"{len(gt)} ground-truth rows")

    gt_by_stem = defaultdict(list)
    for r in gt.itertuples():
        gt_by_stem[r.stem].append((r.filament_id, r.entry_id, r.segmentation_rle))
    cand_by_stem = defaultdict(list)
    for r in cand.itertuples():
        cand_by_stem[r.stem].append((r.cand_id, r.rle))

    rows = []
    t0 = time.time()
    for n, (stem, cands) in enumerate(cand_by_stem.items()):
        gts = gt_by_stem.get(stem, [])
        if not gts:
            continue
        gt_masks = []
        for fid, eid, rle in gts:
            g = decode_rle(rle).astype(bool)
            box = crop_of(g)
            if box is None:
                continue
            gt_masks.append((fid, eid, box, g[box[0]:box[1], box[2]:box[3]]))

        for cid, rle in cands:
            m = shrink_frac(decode_rle(rle).astype(bool), a.shrink)
            variants = offset_variants(m)
            if variants is None:
                continue
            base_box, base_sub = variants[0]
            # best ground-truth partner per entry, chosen at the deployed geometry
            best = {}
            for fid, eid, gbox, gsub in gt_masks:
                v, _, _ = iou_boxed(base_box, base_sub, gbox, gsub)
                if v > best.get(eid, (0.0, None))[0]:
                    best[eid] = (v, (fid, gbox, gsub))
            for eid, (v0, partner) in best.items():
                if partner is None or v0 <= 0.02:
                    continue
                fid, gbox, gsub = partner
                ious = {}
                for k, (cbox, csub) in variants.items():
                    ious[k], _, _ = iou_boxed(cbox, csub, gbox, gsub)
                rows.append(dict(stem=stem, cand_id=cid, entry_id=eid,
                                 gt_id=fid, base=v0,
                                 **{f"o{k}": ious[k] for k in OFFSETS}))
        if (n + 1) % 100 == 0:
            log(f"  {n+1}/{len(cand_by_stem)} stems  ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "nearmiss_offsets.parquet", index=False)
    log(f"\nmeasured {len(df)} (candidate, entry) pairs with any overlap "
        f"({time.time()-t0:.0f}s)")

    near = df[(df["base"] > BAND[0]) & (df["base"] < BAND[1])]
    matched = df[df["base"] > BAND[1]]
    log(f"  matched at the deployed geometry : {len(matched)}")
    log(f"  near-miss band {BAND[0]}-{BAND[1]}         : {len(near)}")
    log(f"  below {BAND[0]}                        : "
        f"{len(df) - len(near) - len(matched)}")

    log("\nglobal offset, scored net over both populations:")
    log(f"{'k':>3s} {'crossings':>10s} {'losses':>7s} {'net TP':>7s} "
        f"{'d numerator':>12s}")
    base_num = float(matched["base"].sum())
    for k in OFFSETS:
        c = df[f"o{k}"]
        cross = int(((df["base"] < BAND[1]) & (c > BAND[1])).sum())
        lost = int(((df["base"] > BAND[1]) & (c <= BAND[1])).sum())
        num_k = float(c[c > BAND[1]].sum())
        log(f"{k:+3d} {cross:10d} {lost:7d} {cross-lost:+7d} "
            f"{num_k - base_num:+12.2f}")

    per_best = df[[f"o{k}" for k in OFFSETS]].max(axis=1)
    orc_cross = int(((df["base"] < BAND[1]) & (per_best > BAND[1])).sum())
    log(f"\nper-instance oracle offset: {orc_cross} crossings "
        f"({100*orc_cross/max(len(near),1):.1f}% of the near-miss band)")
    argk = df[[f"o{k}" for k in OFFSETS]].idxmax(axis=1).str[1:].astype(int)
    conv = argk[(df["base"] < BAND[1]) & (per_best > BAND[1])]
    log("  offsets those crossings need:")
    log("   " + conv.value_counts().sort_index().to_string().replace("\n", "\n   "))

    json.dump(dict(n_pairs=len(df), n_near=len(near), n_matched=len(matched),
                   oracle_crossings=orc_cross,
                   global_net={str(k): int(((df["base"] < BAND[1]) &
                                            (df[f"o{k}"] > BAND[1])).sum() -
                                           ((df["base"] > BAND[1]) &
                                            (df[f"o{k}"] <= BAND[1])).sum())
                               for k in OFFSETS}),
              open(OUT / "nearmiss_offsets.json", "w"), indent=1)
    log(f"\nwrote {OUT/'nearmiss_offsets.parquet'} and .json")


if __name__ == "__main__":
    main()
