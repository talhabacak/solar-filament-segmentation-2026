"""Experiment C: fuse the detector's soft evidence into the semantic probability map.

Precondition (measured): the two lanes' error profiles differ -- the two-lane
fusion was the single change the pipeline never absorbed (LB 0.36 -> 0.37), and
the 2048 detector carries SQ 0.6711 against the semantic lane's ~0.67. Today the
lanes only meet at the CANDIDATE level (instance masks into a shared pool). This
tests meeting them earlier, at the PIXEL level: a detector-confidence raster
averaged into the semantic map before components are extracted, so a filament
the detector believes in but the semantic map holds at 0.4 can cross the
component threshold -- and vice versa, semantic evidence can complete a detector
mask's missing tail.

Composition: det_map(x) = max over fold-0 OOF candidates covering x of that
candidate's confidence (max, not sum -- overlapping proposals of one object
should not inflate). fused = (1-w)*sem + w*det for w in a small grid, then
connected components with the tiled-lane sweep's knobs, scored exactly.

Baselines on the same fold for context: semantic-only CC 0.4277 (tiled gate's
best), the candidate-level pipeline 0.4383 (fold-0 fused+selector).

Usage: python src/eda/det_prob_fusion.py
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament.pq import score
from filament.rle import decode_rle, encode_mask

SEM = ROOT / "outputs" / "semantic"
OOF = ROOT / "outputs" / "oof_cache"
OUT = ROOT / "outputs" / "eda"


def log(*a):
    print(*a, flush=True)


def instances(prob, thr, min_area, close_k):
    b = (prob > thr).astype(np.uint8)
    if close_k:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    return [(lab == i) for i in range(1, n)
            if stats[i, cv2.CC_STAT_AREA] >= min_area]


def main():
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    f0 = sorted(f.rsplit(".", 1)[0]
                for f in folds.loc[folds["fold"] == 0, "file_name"].unique())
    z = np.load(SEM / "fold0_probs_cldice.npz")
    det = pd.read_parquet(OOF / "fold0_preds_e016.parquet")
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt = gt[gt["file_name"].isin({s + ".jpeg" for s in f0})]
    log(f"fold 0: {len(f0)} images, {len(det)} detector candidates, "
        f"{len(gt)} GT rows")

    t0 = time.time()
    det_maps = {}
    for stem, g in det.groupby("stem"):
        m = np.zeros((2048, 2048), np.float32)
        for r in g.itertuples():
            mm = decode_rle(r.rle).astype(bool)
            np.maximum(m, np.float32(r.conf) * mm, out=m)
        det_maps[stem] = m
    log(f"detector rasters built ({time.time()-t0:.0f}s)")

    grid_w = (0.0, 0.3, 0.5)
    grid = [(0.5, 400, 9), (0.6, 400, 9), (0.4, 400, 9), (0.5, 200, 9)]
    log(f"\n{'w_det':>5s} {'thr':>4s} {'area':>5s} {'close':>5s} "
        f"{'PQ':>7s} {'SQ':>7s} {'RQ':>7s} {'TP':>5s} {'FP':>5s} {'FN':>5s}")
    best = None
    for w in grid_w:
        for thr, ma, ck in grid:
            preds = []
            for stem in f0:
                sem = z[stem].astype(np.float32) / 255.0
                fusedp = (1 - w) * sem + w * det_maps.get(stem, 0.0)
                for i, mm in enumerate(instances(fusedp, thr, ma, ck)):
                    preds.append((f"{stem}_{i+1}",
                                  encode_mask(mm.astype(np.uint8))))
            pdf = pd.DataFrame(preds, columns=["filament_id", "segmentation_rle"])
            s = score(gt[["filament_id", "segmentation_rle"]], pdf)
            log(f"{w:5.1f} {thr:4.1f} {ma:5d} {ck:5d} {s['pq']:7.4f} "
                f"{s['sq']:7.4f} {s['rq']:7.4f} {s['tp']:5d} {s['fp']:5d} "
                f"{s['fn']:5d}")
            if best is None or s["pq"] > best[0]:
                best = (s["pq"], w, thr, ma, ck)
    log(f"\nbest: PQ {best[0]:.4f} at w_det={best[1]}, thr={best[2]}, "
        f"area={best[3]}, close={best[4]}")
    log("context: semantic-only CC best on this fold was 0.4277 (w_det=0 rows "
        "reproduce it modulo TTA); candidate-level pipeline 0.4383.")


if __name__ == "__main__":
    main()
