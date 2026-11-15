"""P1: one confidence gate is applied to two different confidence SCALES.

The production policy keeps a candidate iff conf >= 0.40 AND p_sel >= 0.35. But
`conf` means two different things: for detector candidates it is YOLO's box
confidence; for semantic candidates it is the mean probability inside the mask.
These are not on a common scale, yet one threshold gates both -- and the FN
drop-reason analysis already counted 315 positives killed by the conf gate ALONE
(p_sel would have kept them).

This sweeps the gates PER SOURCE on the production (shrunk) pool with the OOF
selector probabilities, scored with fast_pq (valid: the pool is de-overlapped/
disjoint). Cross-fitted confirmation follows only if the in-sample gain clears
the noise floor.

Usage: python src/eda/per_source_gates.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament.rle import decode_rle, encode_mask
from mask_shrink import shrink_frac
from selector import fast_pq
from selector_v5 import BASE, SRC, CROSS, oof_probs

OUT = ROOT / "outputs" / "eda"


def log(*a):
    print(*a, flush=True)


def main():
    cand = pd.read_parquet(ROOT / "outputs" / "selector" / "fused_pool_all.parquet")
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    total_gt = len(gt[gt["file_name"].isin(set(cand["stem"] + ".jpeg"))])

    t0 = time.time()
    rles, areas = [], []
    for r in cand.itertuples():
        m = shrink_frac(decode_rle(r.rle).astype(bool), 0.05)
        rles.append(encode_mask(m.astype(np.uint8)))
        areas.append(int(m.sum()))
    cand["rle"], cand["area"] = rles, areas
    cand["log10_area"] = np.log10(np.maximum(cand["area"], 1))
    cand["y"] = (cand["weight"] >= 1).astype(int)
    cand["p_sel"] = oof_probs(cand, BASE + SRC + CROSS)
    log(f"pool ready ({time.time()-t0:.0f}s); {len(cand)} candidates, "
        f"total GT {total_gt}")

    # the pool marks provenance with from_det (cluster contains a detector member)
    is_det = cand["from_det"].astype(bool)
    log(f"from_det: {int(is_det.sum())} detector-backed, "
        f"{int((~is_det).sum())} semantic-only")

    base = fast_pq(cand[(cand["conf"] >= 0.40) & (cand["p_sel"] >= 0.35)], total_gt)
    log(f"\nproduction policy       : PQ {base['pq']:.4f}  "
        f"TP {base['tp']} FP {base['fp']} FN {base['fn']}")

    log("\nper-source gate sweep (p_sel >= 0.35 fixed):")
    log(f"{'c_det':>6s} {'c_sem':>6s} {'PQ':>7s} {'dPQ':>8s} {'TP':>5s} {'FP':>5s}")
    best = (base["pq"], 0.40, 0.40)
    for c_det in (0.20, 0.30, 0.40, 0.50):
        for c_sem in (0.00, 0.20, 0.30, 0.40, 0.50):
            keep = cand[(cand["p_sel"] >= 0.35)
                        & (np.where(is_det, cand["conf"] >= c_det,
                                    cand["conf"] >= c_sem))]
            q = fast_pq(keep, total_gt)
            flag = ""
            if q["pq"] > best[0]:
                best = (q["pq"], c_det, c_sem)
                flag = "  <-"
            log(f"{c_det:6.2f} {c_sem:6.2f} {q['pq']:7.4f} "
                f"{q['pq']-base['pq']:+8.4f} {q['tp']:5d} {q['fp']:5d}{flag}")

    # and the p_sel axis per source, at the best conf gates
    log(f"\np_sel per source at c_det={best[1]}, c_sem={best[2]}:")
    for p_det in (0.30, 0.35, 0.40):
        for p_sem in (0.25, 0.30, 0.35, 0.40):
            keep = cand[np.where(is_det,
                                 (cand["conf"] >= best[1]) & (cand["p_sel"] >= p_det),
                                 (cand["conf"] >= best[2]) & (cand["p_sel"] >= p_sem))]
            q = fast_pq(keep, total_gt)
            log(f"  p_det {p_det:.2f} p_sem {p_sem:.2f}: PQ {q['pq']:.4f} "
                f"({q['pq']-base['pq']:+.4f})")

    log(f"\nbest per-source gates: PQ {best[0]:.4f} "
        f"(c_det {best[1]}, c_sem {best[2]})  vs production {base['pq']:.4f}")
    log("NOTE: in-sample sweep. If the gain clears ~+0.004, rerun cross-fitted "
        "(fold-held-out policy choice) before believing it.")


if __name__ == "__main__":
    main()
