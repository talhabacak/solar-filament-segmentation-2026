"""E143 -- is the selector calibrated across candidate size, and what is the exact keep rule?

E142 measured where the deployed set loses true candidates it already has:

    area quintile 0 (1-277 px)      78.0% of the true candidates are dropped
    area quintile 1 (278-505 px)    50.1%
    area quintile 4 (1768+ px)       9.9%

Small is dropped seven times more often than large. Two explanations, and they
demand opposite actions, so they have to be separated before anything is built.

1. It is CORRECT. Derive the metric's own keep rule. PQ = sum(IoU)/(TP + FP/2 + FN/2).
   Adding one candidate with match probability q and expected IoU u moves the
   numerator by q*u and the denominator by 1/2 either way: a true one turns an FN
   (1/2) into a TP (1), a false one adds an FP (1/2). So the candidate pays iff

       q * u  >  PQ / 2

   which is exactly the quantity the two-stage selector already scores, and it
   carries no size term. Small objects earn the same TP credit as large ones but
   have lower expected IoU, so a calibrated selector SHOULD drop more of them.

2. It is MISCALIBRATION. If the regressor systematically under-predicts IoU for
   small candidates, they are dropped below their true worth and a per-size
   recalibration is a free gain.

This bins candidates by size quintile and predicted score, then compares the
predicted q*u against the realised one out of fold. Under (1) the curves lie on
the diagonal in every size bin; under (2) the small bins sit above it. The
empirical optimum of a global raw-score cut is reported next to the derived
PQ/2, as a check on the derivation itself.

Usage: python src/eda/size_calibration.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/eda", "src/models"):
    sys.path.insert(0, str(ROOT / p))
from filament.pq import score  # noqa: E402

OUT = ROOT / "outputs" / "eda"
GATE = (0.40, 0.60)


def log(*a):
    print(*a, flush=True)


def main():
    a = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    a = a.merge(pd.read_parquet(OUT / "t020pox_oof_scores.parquet"), on="cand_id", how="left")
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    g = gt[gt["stem"].isin(set(a["stem"]))]
    raw_col = "raw" if "raw" in a.columns else None
    if raw_col is None:
        log("no raw score column; the quantile p is a monotone transform of it, so the")
        log("calibration is read against p and the break-even is located empirically")
    a["realised"] = a["y"] * a["iou_sum"]
    a["q"] = pd.qcut(a["log10_area"], 5, labels=False, duplicates="drop")

    log("realised iou_sum per candidate, by size quintile and selector score decile")
    log("(a calibrated selector gives the same realised value at the same score,")
    log(" whatever the size; a size effect at fixed score is miscalibration)\n")
    dec = pd.qcut(a["p"], 10, labels=False, duplicates="drop")
    hdr = "  score decile " + "".join(f"{i:>7d}" for i in range(10))
    log(hdr)
    tab = {}
    for qq in range(5):
        row = []
        for d in range(10):
            m = (a["q"] == qq) & (dec == d)
            row.append(a.loc[m, "realised"].mean() if m.sum() >= 20 else np.nan)
        tab[qq] = row
        log(f"  size q{qq}      " + "".join(f"{v:7.3f}" if np.isfinite(v) else "      ." for v in row))
    log("\n  all sizes    " + "".join(
        f"{a.loc[dec == d, 'realised'].mean():7.3f}" for d in range(10)))

    # the metric's own break-even, and where a global cut actually peaks
    deployed = a[(a["conf"] >= GATE[0]) & (a["p"] >= GATE[1])]
    base = score(g[["filament_id", "segmentation_rle"]],
                 deployed[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"}))
    log(f"\ndeployed PQ {base['pq']:.4f}  ->  derived break-even q*u = PQ/2 = {base['pq'] / 2:.4f}")
    log("\nrealised value of the candidates each score decile would add:")
    for d in range(10):
        m = dec == d
        log(f"  decile {d}: mean realised iou_sum {a.loc[m, 'realised'].mean():.4f}  "
            f"({'PAYS' if a.loc[m, 'realised'].mean() > base['pq'] / 2 else 'costs'}), "
            f"n {int(m.sum()):5d}, deployed keeps {int((m & (a['p'] >= GATE[1]) & (a['conf'] >= GATE[0])).sum()):5d}")

    # per-size break-even: does any size bin have deciles that pay but are dropped?
    log("\ncandidates that PAY by the derived rule but the gate drops "
        "(realised mean > PQ/2 within the size-by-decile cell):")
    tot = 0
    for qq in range(5):
        for d in range(10):
            m = (a["q"] == qq) & (dec == d)
            if m.sum() < 20:
                continue
            dropped = m & ~((a["p"] >= GATE[1]) & (a["conf"] >= GATE[0]))
            if a.loc[m, "realised"].mean() > base["pq"] / 2 and dropped.sum() > 0:
                log(f"  size q{qq} decile {d}: realised {a.loc[m, 'realised'].mean():.3f}, "
                    f"{int(dropped.sum())} dropped")
                tot += int(dropped.sum())
    log(f"  total {tot} candidates dropped from cells that pay on average")
    json.dump(dict(deployed_pq=base["pq"], break_even=base["pq"] / 2,
                   table={str(k): [None if not np.isfinite(x) else float(x) for x in v] for k, v in tab.items()},
                   droppable=tot), open(OUT / "size_calibration.json", "w"), indent=1, default=float)
    log(f"\nwrote {OUT / 'size_calibration.json'}")


if __name__ == "__main__":
    main()
