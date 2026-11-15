"""E128 -- is a false positive or a missed filament more expensive, empirically?

The algebra of PQ = sum(IoU) / (TP + FP/2 + FN/2) says a candidate is worth
admitting when its match probability p satisfies

    p * IoU  >  0.5 * PQ    ->    p* = 0.5 * PQ / IoU,

which at PQ 0.436 and matched IoU 0.679 is p* = 0.32: one recovered filament is
worth about two false positives. But that derivation assumes an admitted
candidate is free of side effects, and it is not -- predictions must be pixel
disjoint, a new mask carves its neighbours, and only one prediction may match a
given ground-truth instance, so admitting a near-duplicate can convert an
existing true positive into a false positive.

This measures the real thing. Starting from the deployed keep set (honest
out-of-fold pool, conf >= 0.40, p_oof >= 0.60), each band of the selector score
is added (below the threshold) or removed (above it) and the exact scorer is
re-run over all 703 images. The reported dPQ per band is the true marginal value
of admitting those candidates, including every side effect, and the band where
it crosses zero is the empirical break-even -- the answer to "which hurts more,
a false positive or a miss, at our operating point".

Usage: python src/eda/marginal_band_value.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "eda"))
from filament.pq import score  # noqa: E402

OUT = ROOT / "outputs" / "eda"
CONF, PCUT = 0.40, 0.60
BANDS = [(0.40, 0.45), (0.45, 0.50), (0.50, 0.55), (0.55, 0.60),
         (0.60, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 0.80)]


def log(*a):
    print(*a, flush=True)


def main():
    cand = pd.read_parquet(OUT / "e115b_labelled_pool.parquet").merge(
        pd.read_parquet(OUT / "e115b_p_oof.parquet"), on="cand_id")
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt = gt[gt["stem"].isin(set(cand["stem"]))]
    gsub = gt[["filament_id", "segmentation_rle"]]

    def sc(df):
        return score(gsub, df[["cand_id", "rle"]].rename(
            columns={"cand_id": "filament_id", "rle": "segmentation_rle"}))

    keep = cand[(cand["conf"] >= CONF) & (cand["p_oof"] >= PCUT)]
    base = sc(keep)
    log(f"deployed keep set: {len(keep)} masks, PQ {base['pq']:.4f} "
        f"(TP {base['tp']} FP {base['fp']} FN {base['fn']})\n")
    log(f"{'band':14s} {'n':>6s} {'true':>6s} {'purity':>7s} {'dPQ':>9s} {'dTP':>6s} {'dFP':>6s} {'per mask':>9s}")
    res = {}
    for lo, hi in BANDS:
        band = cand[(cand["conf"] >= CONF) & (cand["p_oof"] >= lo) & (cand["p_oof"] < hi)]
        if band.empty:
            continue
        added = hi <= PCUT
        trial = pd.concat([keep, band]) if added else keep[~keep["cand_id"].isin(band["cand_id"])]
        r = sc(trial)
        d = r["pq"] - base["pq"]
        # a band above the threshold is REMOVED, so report the value of KEEPING it
        d_keep = d if added else -d
        purity = float(band["y"].mean())
        res[f"{lo}-{hi}"] = dict(n=len(band), purity=purity, d_pq_if_kept=float(d_keep),
                                 d_tp=int(r["tp"] - base["tp"]), d_fp=int(r["fp"] - base["fp"]), added=bool(added))
        log(f"{lo:.2f}-{hi:.2f} {'add' if added else 'drop':4s} {len(band):6d} {int(band['y'].sum()):6d} "
            f"{purity:7.3f} {d_keep:+9.4f} {r['tp']-base['tp']:+6d} {r['fp']-base['fp']:+6d} "
            f"{d_keep/max(len(band),1)*1000:+9.4f}")
    log("\n'per mask' is dPQ x 1000 per candidate in the band; the band where dPQ crosses")
    log("zero is the empirical break-even purity -- above it a mask pays, below it a")
    log("false positive costs more than the filament it might recover.")
    json.dump(dict(base=dict(pq=base["pq"], tp=base["tp"], fp=base["fp"], fn=base["fn"]), bands=res),
              open(OUT / "marginal_band_value.json", "w"), indent=1, default=float)
    log(f"wrote {OUT / 'marginal_band_value.json'}")


if __name__ == "__main__":
    main()
