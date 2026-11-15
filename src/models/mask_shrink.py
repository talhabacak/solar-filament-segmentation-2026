"""Do our masks run fat? Measure PQ under morphological shrinking.

Visual review of the near-miss and worst-TP galleries showed one dominant
systematic error: the predicted outline encircles the ground truth almost
everywhere. The FP profile had measured the same thing without naming it —
kept-candidate area p50 1486 px vs GT 1228 px, i.e. ~21% fat. For a long thin
filament a two-pixel excess per side takes a width-10 structure to 14 and caps
IoU at ~0.71, and 25.5% of all GT sits in the 0.1-0.5 near-miss band.

This sweeps erosion (and a mild opening) over the operating-point masks and
rescoring with the exact scorer, per fold and pooled.

Usage: python src/models/mask_shrink.py --folds 0,1,2,3,4
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
from filament.rle import decode_rle, encode_mask

CACHE = ROOT / "outputs" / "oof_cache"
R_MAX = 0.97
K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def log(*a):
    print(*a, flush=True)


def shrink(mask, iters, min_keep=64):
    """Erode `iters` times; never let an instance vanish (fall back one step)."""
    if iters <= 0:
        return mask
    cur = mask.astype(np.uint8)
    for _ in range(iters):
        nxt = cv2.erode(cur, K3)
        if nxt.sum() < min_keep:
            return cur.astype(bool)
        cur = nxt
    return cur.astype(bool)


def shrink_frac(mask, frac, min_keep=64):
    """Remove the outermost `frac` of the area, continuously.

    Integer erosion jumps a full pixel per step (measured: 1 step +0.0053 PQ,
    2 steps -0.049), so the optimum lies between. The distance transform gives
    sub-pixel control: keep pixels whose distance to the boundary exceeds the
    (frac) quantile of that distance.
    """
    if frac <= 0:
        return mask
    # Computed on the bounding box with a one-pixel zero margin. Every pixel outside the box is
    # zero in the full frame too, and with the 3x3 chamfer metric the nearest zero to any inside
    # pixel lies on that margin, so the distances inside are the same; OpenCV accumulates the
    # 3x3 transform in fixed-point, so the values are the same to the bit. Held to bit-identity
    # by outputs/eda/_verify_lossless_fusion.py.
    from maskops import bbox, pad_box
    box = bbox(mask)
    if box is None:
        return mask
    y0, y1, x0, x1 = pad_box(box, 1, mask.shape)
    m = mask[y0:y1, x0:x1].astype(np.uint8)
    if m.sum() < min_keep:
        return mask
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    vals = dist[m > 0]
    t = np.quantile(vals, frac)
    win = dist > t
    if win.sum() < min_keep:
        return mask
    out = np.zeros(mask.shape, dtype=bool)
    out[y0:y1, x0:x1] = win
    return out


def shrink_true_frac(mask, frac, min_keep=64):
    """Remove exactly `frac` of the area. Unlike `shrink_frac`, the parameter means it.

    `shrink_frac` thresholds a tied field: for a thin structure a median 19% of
    pixels sit at the minimum distance (the outer ring, all at dist = 1), so
    np.quantile(v, f) returns that minimum for EVERY f up to 0.19 and `dist > t`
    deletes the whole ring. Measured over 382 fold-0 candidates, f = 0.025, 0.05
    and 0.10 all remove the same 19.0%. That function is one-pixel integer
    erosion wearing a fractional parameter, which is why the ledger's "2.5%
    against 5% shrink" comparison (+0.0004) compared an operation with itself,
    and why the sub-pixel region -- where the erosion sweep says the optimum lies,
    one step +0.0053 PQ against two steps -0.049 -- was never reachable.

    Here the field is made continuous (exact distance transform, then a mild
    blur) and pixels are removed BY RANK, so exactly frac of them go. Ties at the
    cut are broken deterministically rather than taken wholesale. Work happens in
    a padded bounding box: a full 2048^2 float field per candidate is slow and,
    with several jobs resident, enough to exhaust memory.
    """
    if frac <= 0:
        return mask
    m = mask.astype(bool)
    n = int(m.sum())
    if n < min_keep:
        return mask
    ys, xs = np.nonzero(m)
    P = 4
    y0, y1 = max(0, ys.min() - P), min(m.shape[0], ys.max() + P + 1)
    x0, x1 = max(0, xs.min() - P), min(m.shape[1], xs.max() + P + 1)
    u = m[y0:y1, x0:x1].astype(np.uint8)
    d = cv2.distanceTransform(u, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    d = cv2.GaussianBlur(d, (0, 0), 1.0) * u
    k = int(round(frac * n))
    if k <= 0 or n - k < min_keep:
        return mask
    v = d[u > 0]
    thr = float(np.partition(v, k)[k])
    sub = (d > thr) & (u > 0)
    if int(sub.sum()) > n - k:
        tied = np.flatnonzero(((d == thr) & (u > 0)).ravel())
        o = sub.ravel().copy()
        o[tied[:int(sub.sum() - (n - k))]] = False
        sub = o.reshape(sub.shape)
    if int(sub.sum()) < min_keep:
        return mask
    out = np.zeros_like(m)
    out[y0:y1, x0:x1] = sub
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0")
    ap.add_argument("--tag", default="_e013")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--min-area", type=int, default=128)
    ap.add_argument("--iters", default="0,1,2,3")
    ap.add_argument("--fracs", default="", help="continuous shrink fractions, e.g. 0.05,0.1")
    a = ap.parse_args()

    gt_all = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    results = {}

    modes = [("erode", int(x)) for x in a.iters.split(",") if x] + \
            [("frac", float(x)) for x in a.fracs.split(",") if x]
    for mode, it in modes:
        rows, gt_rows = [], []
        t0 = time.time()
        for fold in [int(x) for x in a.folds.split(",")]:
            det = pd.read_parquet(CACHE / f"fold{fold}_preds{a.tag}.parquet")
            det = det[(det["conf"] >= a.conf) & (det["r_norm"] <= R_MAX)]
            if "view" in det.columns:
                det = det[det["view"] == "o"]
            val_files = sorted(folds.loc[folds["fold"] == fold, "file_name"])
            gt_rows.append(gt_all[gt_all["file_name"].isin(val_files)])
            for stem, g in det.groupby("stem"):
                taken = np.zeros((IMG_H, IMG_W), dtype=bool)
                k = 0
                for r in g.sort_values("conf", ascending=False).itertuples():
                    m = decode_rle(r.rle).astype(bool) & ~taken
                    if m.sum() < a.min_area:
                        continue
                    taken |= m                      # claim BEFORE shrinking, so
                    m = shrink(m, it) if mode == "erode" else shrink_frac(m, it)
                    if m.sum() < a.min_area:        # shrinking never lets a weaker
                        continue                    # mask reclaim those pixels
                    k += 1
                    rows.append((f"{stem}_{k}", encode_mask(m.astype(np.uint8))))
        gt_sub = pd.concat(gt_rows)[["filament_id", "segmentation_rle"]]
        pred_df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
        r = score(gt_sub, pred_df)
        results[f"{mode}{it}"] = {k2: v for k2, v in r.items()
                                  if k2 in ("pq", "sq", "rq", "tp", "fp", "fn")}
        log(f"{mode}={it}: PQ={r['pq']:.4f} SQ={r['sq']:.4f} RQ={r['rq']:.4f} "
            f"TP={r['tp']} FP={r['fp']} FN={r['fn']}  ({time.time()-t0:.0f}s)")

    dest = ROOT / "outputs" / "eda" / f"mask_shrink{a.tag}.json"
    with open(dest, "w") as f:
        json.dump(results, f, indent=1)
    log(f"saved -> {dest}")


if __name__ == "__main__":
    main()


def mean_width_px(mask):
    """area / (perimeter/2): the mean width of an elongated structure."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    per = sum(cv2.arcLength(c, True) for c in cs)
    return float(mask.sum()) / max(per / 2.0, 1.0)


def shrink_width_gated(mask, max_width=12.0, nominal=0.05):
    """Production one-pixel ring erosion, applied ONLY to masks thinner than max_width.

    Measured on fold 0 (radial_denoise_profile.py): before any shrink the model
    over-covers thin ground truth and under-covers thick -- log2(pred/gt) is
    +0.43 at 7 px mean width, +0.09 at 11 px, -0.14 at 19 px. A fixed ~1 px
    boundary error is a large share of a thin mask and a negative share of a
    thick one once thresholded. The production ring erosion removes ~29% of a
    thin mask (which cancels the +35% over-coverage almost exactly) but also
    ~11% of a thick mask that was already 9% too small. So: erode the thin ones,
    leave the thick ones alone. The erosion itself is the production call, so
    the thin branch is bit-identical to what ships.
    """
    if mean_width_px(mask) < max_width:
        return shrink_frac(mask, nominal)
    return mask.copy()
