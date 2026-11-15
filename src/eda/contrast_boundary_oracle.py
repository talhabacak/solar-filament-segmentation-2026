"""Relative-contrast boundary oracle (physics -> annotation convention; saf_bilim ch. 04/06).

Physics: at GONG's ~2" resolution a filament is a PSF-merged bundle of
150-450 km threads; the visible ribbon has no sharp physical edge, its cross
profile is a smooth darkness dip. A human annotator therefore draws the polygon
where the darkness has fallen to some fraction of the core darkness -- a
RELATIVE-contrast contour -- whereas our masks are iso-probability contours of a
network (p >= 0.3, then a 5% shrink). Hard photometric snapping (Otsu) lost
~0.07 PQ in the ledger, but that moved boundaries to absolute intensity edges
in a textured image. This is the soft, envelope-constrained version:

  for every deployed mask M (E115b labelled pool, honest OOF selector, policy
  0.40/0.60): envelope E = dilate(M, 2r+1); background B = median intensity of
  the ring E \\ dilate(M, 3); core C = 5th percentile of the smoothed intensity
  inside M; darkness D = (B - I_smooth) / (B - C) clipped to [0, 1];
  the re-cut mask at fraction f is  M_f = E & (D >= f)  (largest component
  touching M, holes filled, empty -> keep M).

Oracle question, per fraction f in {0.25 .. 0.75}: does replacing every
deployed mask by M_f raise the exact PQ? Reported per fold and pooled, with the
best f chosen on the OTHER folds (nested), so the number is an honest
post-processing estimate, not a fit. Also the matched-row IoU distribution at
each f, so a gain (or loss) can be read as boundary movement.

PRE-REGISTERED: a nested-f gain >= +0.003 pooled with >= 4/5 folds positive
opens a boundary post-processing lane (no GPU); anything less closes the
relative-contrast idea alongside Otsu snapping.

Usage: python src/eda/contrast_boundary_oracle.py
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from filament.pq import score  # noqa: E402
from filament.rle import decode_rle, encode_mask  # noqa: E402

OUT = ROOT / "outputs" / "eda"
IMG = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
CONF, PCUT = 0.40, 0.60
FRACS = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
R_ENV, SIGMA = 6, 1.5
K_ENV = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * R_ENV + 1, 2 * R_ENV + 1))
K3 = np.ones((3, 3), np.uint8)


def log(*a):
    print(*a, flush=True)


def recut(Is, m, f):
    """Relative-contrast re-cut of mask m on the smoothed image Is at fraction f."""
    ys, xs = np.nonzero(m)
    y0, y1 = max(ys.min() - R_ENV - 2, 0), min(ys.max() + R_ENV + 3, m.shape[0])
    x0, x1 = max(xs.min() - R_ENV - 2, 0), min(xs.max() + R_ENV + 3, m.shape[1])
    mc, Ic = m[y0:y1, x0:x1].astype(np.uint8), Is[y0:y1, x0:x1]
    env = cv2.dilate(mc, K_ENV).astype(bool)
    near = cv2.dilate(mc, K3).astype(bool)
    ring = env & ~near
    if ring.sum() < 20:
        return m
    B = float(np.median(Ic[ring]))
    C = float(np.percentile(Ic[mc.astype(bool)], 5))
    if B - C < 2:
        return m
    D = np.clip((B - Ic) / (B - C), 0, 1)
    cand = (env & (D >= f)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    if n <= 1:
        return m
    # keep components that overlap the original mask; fill holes
    keep = np.zeros_like(cand, bool)
    for i in range(1, n):
        comp = lab == i
        if (comp & mc.astype(bool)).sum() > 0:
            keep |= comp
    if keep.sum() < 64:
        return m
    ff = keep.astype(np.uint8)
    cnts, _ = cv2.findContours(ff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(ff)
    cv2.drawContours(filled, cnts, -1, 1, -1)
    out = np.zeros_like(m)
    out[y0:y1, x0:x1] = filled.astype(bool)
    return out


def main():
    cand = pd.read_parquet(OUT / "e115b_labelled_pool.parquet").merge(pd.read_parquet(OUT / "e115b_p_oof.parquet"), on="cand_id")
    keep = cand[(cand["conf"] >= CONF) & (cand["p_oof"] >= PCUT)].copy()
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    keep["fold"] = keep["stem"].map(fold_of).astype(int)
    stems = sorted(set(keep["stem"]))
    log(f"kept {len(keep)} masks over {len(stems)} images; fractions {FRACS}")

    rles = {f: [] for f in ["prod"] + FRACS}
    ids = []
    t0 = time.time()
    for n, stem in enumerate(stems):
        img = cv2.imread(str(IMG / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        Is = cv2.GaussianBlur(img, (0, 0), SIGMA)
        kp = keep[keep["stem"] == stem]
        for cid, rle in zip(kp["cand_id"], kp["rle"]):
            m = decode_rle(rle).astype(bool)
            ids.append(cid)
            rles["prod"].append(rle)
            for f in FRACS:
                rles[f].append(encode_mask(recut(Is, m, f).astype(np.uint8)))
        if n % 100 == 0:
            log(f"  {n}/{len(stems)} images ({time.time()-t0:.0f}s)")
    fold_arr = keep.set_index("cand_id").loc[ids, "fold"].to_numpy()
    stem_arr = keep.set_index("cand_id").loc[ids, "stem"].to_numpy()
    g = gt[gt["stem"].isin(stems)]

    def sc(arm, fold=None):
        sel = np.ones(len(ids), bool) if fold is None else (fold_arr == fold)
        pred = pd.DataFrame({"filament_id": np.array(ids)[sel], "segmentation_rle": np.array(rles[arm])[sel]})
        gg = g if fold is None else g[g["stem"].map(fold_of) == fold]
        r = score(gg[["filament_id", "segmentation_rle"]], pred, return_details=True)
        return r

    res = {}
    log("\narm      pooled PQ    TP    FP    FN   matched-IoU mean | per-fold PQ")
    for arm in ["prod"] + FRACS:
        r = sc(arm)
        per = [sc(arm, f)["pq"] for f in range(5)]
        miou = float(r["matches"]["iou"].mean()) if len(r["matches"]) else float("nan")
        res[str(arm)] = dict(pq=r["pq"], tp=r["tp"], fp=r["fp"], fn=r["fn"], matched_iou=miou, per_fold=per)
        log(f"{str(arm):8s} {r['pq']:.4f}  {r['tp']:5d} {r['fp']:5d} {r['fn']:5d}   {miou:.4f}          | " + " ".join(f"{x:.4f}" for x in per))
    # nested choice of f: for each held-out fold pick the f best on the other four
    prod_per = res["prod"]["per_fold"]
    nested = []
    for f in range(5):
        others = [k for k in range(5) if k != f]
        best = max(FRACS, key=lambda fr: sum(res[str(fr)]["per_fold"][k] - prod_per[k] for k in others))
        nested.append((f, best, res[str(best)]["per_fold"][f] - prod_per[f]))
    d = [x[2] for x in nested]
    log("\nnested (f chosen on the other folds): " + " ".join(f"fold{f}:f={b} {x:+.4f}" for f, b, x in nested))
    log(f"nested mean delta {np.mean(d):+.4f}, folds positive {sum(x > 0 for x in d)}/5")
    verdict = ("OPEN a boundary post-processing lane" if np.mean(d) >= 0.003 and sum(x > 0 for x in d) >= 4
               else "CLOSED: relative-contrast boundary does not beat the network+shrink boundary")
    log(f"VERDICT: {verdict}")
    res["nested"] = dict(per_fold=nested, mean=float(np.mean(d)), verdict=verdict)
    json.dump(res, open(OUT / "contrast_boundary_oracle.json", "w"), indent=1, default=float)
    log(f"wrote {OUT / 'contrast_boundary_oracle.json'}")


if __name__ == "__main__":
    main()
