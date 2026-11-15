"""Boundary-band probability profile of every candidate-bank variant (chatgpt_science S002).

The candidate bank's legal oracle is large but no ranker has been able to tell
which variant of a cluster is the better mask: shape and aggregate features gave
OOF Spearman(pred delta, delta) = 0.145 (E096 v1). The information that was
never given to the ranker is what the semantic field says exactly AT the
variant's boundary: a variant that stops where the probability is still high
is under-segmented, one that reaches into probability ~0 is over-segmented.

For each variant mask m and the production field P (0.5 blob + 0.5 clDice,
the map the pool is built from):

  inner   1-px band just inside the boundary:  m & ~erode(m, 3x3)
  outer1  1-px band just outside:              dilate(m, 3x3) & ~m
  outer2  the next 2 px:                        dilate(m, 7x7) & ~dilate(m, 3x3)

  per band: mean, p10, p50, p90 of P
  inner band: fraction of pixels with P > 0.3 / 0.5 / 0.7
  in_out_diff = inner_mean - outer1_mean;  in_out_ratio = inner_mean / (outer1_mean + 0.05)
  edge_drop   = inner_mean - outer2_mean

Writes outputs/eda/variant_bank_fold{k}_ring.parquet indexed like the bank, so
candidate_bank_ranker.py --extra ring can join it. No decision here; the gate
is the ranker's OOF Spearman >= 0.35 (pre-registered in the roadmap).

Usage: python src/eda/candidate_bank_ring_features.py --fold 0
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from filament.rle import decode_rle  # noqa: E402

OUT = ROOT / "outputs" / "eda"
SEM = ROOT / "outputs" / "semantic"
K3 = np.ones((3, 3), np.uint8)
K7 = np.ones((7, 7), np.uint8)


def log(*a):
    print(*a, flush=True)


def band_stats(P, band, prefix):
    v = P[band]
    if v.size == 0:
        return {f"{prefix}_mean": np.nan, f"{prefix}_p10": np.nan, f"{prefix}_p50": np.nan, f"{prefix}_p90": np.nan}
    q = np.percentile(v, [10, 50, 90])
    return {f"{prefix}_mean": float(v.mean()), f"{prefix}_p10": float(q[0]), f"{prefix}_p50": float(q[1]), f"{prefix}_p90": float(q[2])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--field", default="blob5ens")
    a = ap.parse_args()
    vb = pd.read_parquet(OUT / f"variant_bank_fold{a.fold}.parquet")
    Z = np.load(SEM / f"fold{a.fold}_probs_{a.field}.npz")
    log(f"fold {a.fold}: {len(vb)} variants over {vb['stem'].nunique()} stems; field {a.field}")
    rows, t0 = {}, time.time()
    for n, (stem, g) in enumerate(vb.groupby("stem")):
        P = Z[stem].astype(np.float32) / 255.0
        for idx, rle in zip(g.index, g["rle"]):
            m = decode_rle(rle).astype(np.uint8)
            ys, xs = np.nonzero(m)
            y0, y1, x0, x1 = max(ys.min() - 8, 0), min(ys.max() + 9, 2048), max(xs.min() - 8, 0), min(xs.max() + 9, 2048)
            mm, PP = m[y0:y1, x0:x1], P[y0:y1, x0:x1]
            er = cv2.erode(mm, K3)
            d3 = cv2.dilate(mm, K3)
            d7 = cv2.dilate(mm, K7)
            inner = (mm & ~er).astype(bool)
            outer1 = (d3 & ~mm).astype(bool)
            outer2 = (d7 & ~d3).astype(bool)
            r = {}
            r.update(band_stats(PP, inner, "ring_in"))
            r.update(band_stats(PP, outer1, "ring_out1"))
            r.update(band_stats(PP, outer2, "ring_out2"))
            vi = PP[inner]
            r["ring_in_frac30"] = float((vi > 0.3).mean()) if vi.size else np.nan
            r["ring_in_frac50"] = float((vi > 0.5).mean()) if vi.size else np.nan
            r["ring_in_frac70"] = float((vi > 0.7).mean()) if vi.size else np.nan
            r["ring_in_out_diff"] = r["ring_in_mean"] - r["ring_out1_mean"]
            r["ring_in_out_ratio"] = r["ring_in_mean"] / (r["ring_out1_mean"] + 0.05)
            r["ring_edge_drop"] = r["ring_in_mean"] - r["ring_out2_mean"]
            rows[idx] = r
        if n % 25 == 0:
            log(f"  {n}/{vb['stem'].nunique()} stems ({time.time()-t0:.0f}s)")
    df = pd.DataFrame.from_dict(rows, orient="index").reindex(vb.index)
    df.to_parquet(OUT / f"variant_bank_fold{a.fold}_ring.parquet")
    log(f"wrote {len(df)} rows x {df.shape[1]} ring features ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
