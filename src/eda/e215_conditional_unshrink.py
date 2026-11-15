"""E215 -- is the uniform one-pixel ring erosion wrong for THIN candidates?

Production removes the outer ring of every candidate mask (`mask_shrink.shrink_frac` at 0.05 is
integer erosion in disguise: it deletes the whole dist == 1 ring, ~19 % of a thin mask's area).
That was measured as a single global choice (+0.0046 against no shrink) because the detector
masks, upsampled from 448 px prototypes, run fat. But the error atlas says the opposite for the
loss: in the shape band (best pool IoU in (0.05, 0.5]) candidates are too SMALL, 838 GT copies
(10 % of all GT) sit at best IoU 0.40-0.50, and thin filaments are matched far less often
(width <= 6 px: 36-50 % against 59-65 %). A ring is a larger share of a thin mask, so one global
erosion may be right for wide candidates and wrong for thin ones.

Hypothesis (single variable): restoring the ring on kept candidates thinner than W raises pooled
PQ. The KEEP DECISION is held fixed at sub036's out-of-fold selection (conf >= 0.40, bag5 p >=
0.60), so the selector, its features and its threshold are untouched -- the transform acts after
selection and would deploy without a refit.

Ring restore = one dilation with the 3x3 cross (the ring shrink_frac removes is the set of pixels
4-adjacent to background), restricted to pixels no other kept mask owns; when two restored rings
compete, the candidate with the higher selector score claims first. Masks stay disjoint.

Width = area / skeleton length of the kept (shrunk) mask, the selector's own width_proxy.

Pre-registered reading: W is chosen on four folds and applied to the fifth (cross-fitted), so
the reported delta carries no selection over W. Promising iff the cross-fitted delta is positive,
>= 4/5 folds agree, and the paired image-bootstrap 95 % CI excludes zero. Equivalence band
0.005 (project-measurement-protocol). The arm 'all' (restore every ring) must come out negative
-- it re-measures the old +0.0046 shrink gain and guards the implementation.

Usage: python src/eda/e215_conditional_unshrink.py [--reps 4000]
Output: outputs/eda/e215_conditional_unshrink.json, outputs/eda/e215_per_image.parquet
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
for p in ("src", "src/eda", "src/models"):
    sys.path.insert(0, str(ROOT / p))
from filament.rle import decode_rle, encode_mask  # noqa: E402
from e115c_threshold_ci import per_image, pq  # noqa: E402

OUT = ROOT / "outputs" / "eda"
CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
WIDTHS = [0, 3, 4, 5, 6, 7, 8, 10, 12, 1e9]          # 0 = deployed, 1e9 = restore every ring


def log(*a):
    print(*a, flush=True)


def width_of(m):
    from skimage.morphology import skeletonize
    ys, xs = np.nonzero(m)
    crop = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return crop.sum() / max(int(skeletonize(crop).sum()), 1)


def restored_rings(masks, order):
    """For each kept mask, the ring pixels it may take back: 4-adjacent dilation, outside every
    kept mask, first claim by selector score."""
    owned = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        owned |= m
    claimed = np.zeros_like(owned)
    rings = [None] * len(masks)
    for i in order:
        m = masks[i]
        ys, xs = np.nonzero(m)
        y0, y1 = max(0, ys.min() - 2), min(m.shape[0], ys.max() + 3)
        x0, x1 = max(0, xs.min() - 2), min(m.shape[1], xs.max() + 3)
        sub = m[y0:y1, x0:x1].astype(np.uint8)
        ring = cv2.dilate(sub, CROSS).astype(bool) & ~owned[y0:y1, x0:x1] & ~claimed[y0:y1, x0:x1]
        claimed[y0:y1, x0:x1] |= ring
        rings[i] = (y0, y1, x0, x1, ring)
    return rings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4000)
    a_ = ap.parse_args()
    t0 = time.time()

    a = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet",
                        columns=["stem", "cand_id", "rle", "conf", "from_det", "from_sem"])
    a = a.merge(pd.read_parquet(OUT / "bag5_oof_scores.parquet"), on="cand_id")
    folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    keep = a[(a["conf"] >= 0.40) & (a["p"] >= 0.60)].copy()
    log(f"kept {len(keep)} candidates over {keep['stem'].nunique()} images")

    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    stems = np.array(sorted(fold_of))
    gt = gt[gt["stem"].isin(set(stems))]
    ncopy = gt.groupby("stem")["entry_id"].nunique().reindex(stems).fillna(1).to_numpy()
    fold = np.array([fold_of[s] for s in stems])

    # one pass over images: width of every kept mask and its restorable ring
    rows = []
    for n, (stem, g) in enumerate(keep.groupby("stem")):
        g = g.reset_index(drop=True)
        masks = [decode_rle(r).astype(bool) for r in g["rle"]]
        order = np.argsort(-g["p"].to_numpy(), kind="stable")
        rings = restored_rings(masks, order)
        for i, m in enumerate(masks):
            y0, y1, x0, x1, ring = rings[i]
            grown = m.copy()
            grown[y0:y1, x0:x1] |= ring
            rows.append(dict(cand_id=g.at[i, "cand_id"], stem=stem, width=width_of(m),
                             area=int(m.sum()), ring_px=int(ring.sum()),
                             from_det=bool(g.at[i, "from_det"]), from_sem=bool(g.at[i, "from_sem"]),
                             rle=g.at[i, "rle"], rle_grown=encode_mask(grown)))
        if n % 100 == 0:
            log(f"  {n} images, {time.time() - t0:.0f}s")
    k = pd.DataFrame(rows)
    log(f"width of kept masks: p10 {k.width.quantile(.1):.2f} p50 {k.width.median():.2f} "
        f"p90 {k.width.quantile(.9):.2f}; ring/area median {(k.ring_px / k.area).median():.3f}")

    parts = {}
    for W in WIDTHS:
        sub = k[["cand_id", "stem"]].copy()
        sub["rle"] = np.where(k["width"] < W, k["rle_grown"], k["rle"])
        st = per_image(sub, gt, stems, ncopy)[0]
        parts[W] = np.stack([st["tp"], st["fp"], st["fn"], st["iou"]], 1).astype(float)
        tot = parts[W].sum(0)
        log(f"  W < {W:>6g}: grown {int((k.width < W).sum()):5d}  PQ {pq(*tot):.4f}  "
            f"TP {int(tot[0])} FP {int(tot[1])} FN {int(tot[2])}  SQ {tot[3] / tot[0]:.4f}")

    base = parts[0]
    res = {"pooled": {str(W): float(pq(*parts[W].sum(0))) for W in WIDTHS}, "per_fold": {}}
    for W in WIDTHS:
        res["per_fold"][str(W)] = [float(pq(*parts[W][fold == F].sum(0)) - pq(*base[fold == F].sum(0)))
                                   for F in range(5)]

    # cross-fitted W: chosen on four folds, read on the fifth
    chosen, cf = [], np.zeros_like(base)
    for F in range(5):
        tr = fold != F
        best = max(WIDTHS, key=lambda W: pq(*parts[W][tr].sum(0)))
        chosen.append(best)
        cf[fold == F] = parts[best][fold == F]
    d_cf = pq(*cf.sum(0)) - pq(*base.sum(0))
    folds_pos = sum(pq(*cf[fold == F].sum(0)) > pq(*base[fold == F].sum(0)) for F in range(5))

    rng = np.random.default_rng(0)
    boots = np.empty(a_.reps)
    for r in range(a_.reps):
        idx = rng.choice(len(stems), len(stems), replace=True)
        boots[r] = pq(*cf[idx].sum(0)) - pq(*base[idx].sum(0))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    res["crossfit"] = dict(chosen_W=[float(c) for c in chosen], delta=float(d_cf), folds_positive=int(folds_pos),
                           ci95=[float(lo), float(hi)], p_gt0=float((boots > 0).mean()))
    log(f"\ncross-fitted: chosen W {chosen}  delta {d_cf:+.4f}  folds + {folds_pos}/5  "
        f"CI [{lo:+.4f}, {hi:+.4f}]  P(>0) {(boots > 0).mean():.3f}")
    promising = d_cf > 0 and folds_pos >= 4 and lo > 0
    res["verdict"] = "promising" if promising else ("flat" if (lo > -0.005 and hi < 0.005) else "not established")
    log("verdict:", res["verdict"])

    pd.DataFrame({"stem": stems, "fold": fold,
                  **{f"{c}_{W:g}": parts[W][:, j] for W in WIDTHS for j, c in enumerate(("tp", "fp", "fn", "iou"))}}
                 ).to_parquet(OUT / "e215_per_image.parquet", index=False)
    k.drop(columns=["rle", "rle_grown"]).to_parquet(OUT / "e215_kept_widths.parquet", index=False)
    json.dump(res, open(OUT / "e215_conditional_unshrink.json", "w"), indent=1)
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
