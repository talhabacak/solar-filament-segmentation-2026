"""Rebuild the de-overlapped candidate RLEs for the pooled OOF candidate set.

Exactly reproduces src/models/pooled_oof.py::build_all_candidates() ordering and
cand_id assignment, but skips the (expensive) shape features -- those already live
in outputs/selector/pooled_candidates(_v4).parquet.

Output: outputs/eda/cand_rle_deov.parquet  (cand_id, stem, fold, rle_deov)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament import IMG_H, IMG_W
from filament.rle import decode_rle, encode_mask

R_MAX = 0.97
CACHE_DIR = ROOT / "outputs" / "oof_cache"
OUT = ROOT / "outputs" / "eda" / "cand_rle_deov.parquet"


def log(*a):
    print(*a, flush=True)


def main():
    if OUT.exists():
        log(f"{OUT} already exists -- nothing to do")
        return
    t0 = time.time()
    rows = []
    for fold in range(5):
        preds = pd.read_parquet(CACHE_DIR / f"fold{fold}_preds_e013.parquet")
        preds = preds[(preds["conf"] >= 0.05) & (preds["r_norm"] <= R_MAX)]
        for stem, g in preds.groupby("stem"):
            g = g.sort_values("conf", ascending=False)
            taken = np.zeros((IMG_H, IMG_W), dtype=bool)
            k = 0
            for r in g.itertuples():
                mm = decode_rle(r.rle).astype(bool) & ~taken
                if not mm.any():
                    del mm
                    continue
                taken |= mm
                k += 1
                rows.append(dict(cand_id=f"{stem}_{k}", stem=stem, fold=fold,
                                 area_deov=int(mm.sum()),
                                 rle_deov=encode_mask(mm.astype(np.uint8))))
                del mm
            del taken
        log(f"fold {fold}: cumulative {len(rows)} candidates "
            f"({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    log(f"wrote {OUT}  rows={len(df)}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
