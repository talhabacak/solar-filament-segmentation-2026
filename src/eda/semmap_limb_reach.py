"""Reach by radius band for a semantic probability map, against a baseline map.

The instance-quality gate (gate_instance_quality.py) reports reach for three
populations but not by radius. The limb hypotheses (ledger §16, RDN run) need
exactly that: for every GT row, does the map at the production threshold
(p >= 0.3, components >= 64 px) contain a connected component matching it at
IoU > 0.5, by band -- for the baseline map and the candidate map, on the same
rows. No selector, no fusion: this isolates what the semantic lane offers.

Pre-registered limb gate for RDN (docs/23 §11 test-time version: +11 rows):
reach at r > 0.8 must rise by >= +10 rows, with the disc (r <= 0.8) not
falling by more than the limb gains.

Usage: python src/eda/semmap_limb_reach.py --base fold0_probs_cldice.npz --cand fold0_probs_rdn08.npz
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from filament.rle import decode_rle

OUT = ROOT / "outputs" / "eda"
SEM = ROOT / "outputs" / "semantic"
DISK_C, DISK_R = 1023.5, 900.0
BANDS = [(0.0, 0.5), (0.5, 0.8), (0.8, 0.9), (0.9, 1.01)]
THR, MIN_AREA = 0.3, 64


def log(*a):
    print(*a, flush=True)


def components(prob):
    if prob.max() > 1.5:
        prob = prob.astype(np.float32) / 255.0
    if prob.shape[0] != 2048:
        prob = cv2.resize(prob, (2048, 2048), interpolation=cv2.INTER_LINEAR)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((prob >= THR).astype(np.uint8), connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    return lab, areas


def best_iou(mask, lab, areas):
    ids, cnt = np.unique(lab[mask], return_counts=True)
    best = 0.0
    ga = mask.sum()
    for i, c in zip(ids, cnt):
        if i == 0 or areas[i] < MIN_AREA:
            continue
        best = max(best, c / (ga + areas[i] - c))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cand", required=True)
    a = ap.parse_args()
    zb, zc = np.load(SEM / a.base), np.load(SEM / a.cand)
    stems = sorted(set(zb.files) & set(zc.files))
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt = gt[gt["stem"].isin(stems)]
    log(f"{len(stems)} images, {len(gt)} GT rows")
    rows = []
    for k, stem in enumerate(stems):
        lb, ab = components(zb[stem])
        lc, ac = components(zc[stem])
        for r in gt[gt["stem"] == stem].itertuples():
            m = decode_rle(r.segmentation_rle).astype(bool)
            ys, xs = np.nonzero(m)
            if not len(xs):
                continue
            rr = float(np.hypot(xs.mean() - DISK_C, ys.mean() - DISK_C) / DISK_R)
            rows.append(dict(fid=r.filament_id, r=rr, base=best_iou(m, lb, ab), cand=best_iou(m, lc, ac)))
        if (k + 1) % 40 == 0:
            log(f"  {k+1}/{len(stems)}")
    t = pd.DataFrame(rows)
    log(f"\n  {'band':<10}{'rows':>6}{'base reach':>12}{'cand reach':>12}{'Δ rows':>8}{'base IoU':>10}{'cand IoU':>10}")
    tot_l = 0
    for lo, hi in BANDS + [(0.0, 0.8), (0.8, 1.01), (0.0, 1.01)]:
        g = t[(t["r"] > lo) & (t["r"] <= hi)]
        b, c = int((g["base"] > 0.5).sum()), int((g["cand"] > 0.5).sum())
        log(f"  ({lo},{hi}]".ljust(10) + f"{len(g):>6}{b:>12}{c:>12}{c-b:>+8}{g['base'].median():>10.3f}{g['cand'].median():>10.3f}")
        if (lo, hi) == (0.8, 1.01):
            tot_l = c - b
    disc = t[t["r"] <= 0.8]
    dd = int((disc["cand"] > 0.5).sum()) - int((disc["base"] > 0.5).sum())
    log(f"\n  limb gate: r > 0.8 Δ {tot_l:+d} rows (>= +10?) with disc Δ {dd:+d} -> "
        f"{'PASS' if tot_l >= 10 and dd >= -tot_l else 'FAIL'}")
    t.to_parquet(OUT / f"semmap_limb_reach_{Path(a.cand).stem}.parquet", index=False)


if __name__ == "__main__":
    main()
