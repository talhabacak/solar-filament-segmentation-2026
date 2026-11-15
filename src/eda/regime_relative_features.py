"""E207 -- give the selector the candidate's evidence RELATIVE TO ITS OWN FRAME.

E205b located the loss precisely: across hf quartiles SQ is flat and FP/GT is flat, the whole
degradation is recall, and about half of it is the gate discarding real candidates (selector
0.114 -> 0.144) rather than the model failing to propose them (never 0.036 -> 0.048). The same
signature appears by epoch (44b: in 2019-2022 the gate drops 16.3% of consensus-confirmed copies
against 6.6% earlier). One sentence covers both: the gate judges a candidate on an ABSOLUTE
evidence scale calibrated on the pooled training set, while the scale itself moves with the frame.

The deployed 42 carry exactly one frame-relative quantity, `rank_conf`, and three frame-level
context scalars (n_cands_img, img_cand_density, img_sem_area_frac). Everything that actually
drives the score -- conf, det_conf, sem_prob_mean, sem_contrast, sem_ring_mean, member_iou_mean --
enters in absolute units. This block re-expresses each of them against its own frame, so a
candidate that is strong FOR THIS IMAGE is not punished for coming from a faint or a busy one:

    <f>_pct   the candidate's percentile of f among the candidates of its own frame
    <f>_z     (f - frame mean) / frame sd

Percentile and z-score are kept together on purpose: the percentile is robust and ignores how
far apart the candidates are, the z-score keeps that spread, and which of the two matters is an
empirical question this block lets the learner settle.

Usage: python src/eda/regime_relative_features.py           (writes regime_features_t020.parquet)
       python src/eda/regime_relative_features.py --test    (the dumped test pool)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "eda"

# the absolute-scale quantities the two-stage score leans on (permutation importance, E201:
# det_conf is 2nd, sem_ring_mean 5th, img_sem_area_frac 6th)
SRC = ["conf", "det_conf", "sem_conf", "sem_prob_mean", "sem_prob_p90", "sem_contrast",
       "sem_ring_mean", "member_iou_mean", "area", "width_proxy"]
REG = [f"{c}_{k}" for c in SRC for k in ("pct", "z")]


def build(pool):
    out = {"cand_id": pool["cand_id"].to_numpy()}
    g = pool.groupby("stem")
    for c in SRC:
        # percentile within the frame; a frame with one candidate gets 0.5, not 0 or 1
        r = g[c].rank(pct=True, method="average")
        n = g[c].transform("size")
        out[f"{c}_pct"] = np.where(n > 1, r, 0.5)
        mu, sd = g[c].transform("mean"), g[c].transform("std")
        out[f"{c}_z"] = np.where(sd > 1e-9, (pool[c] - mu) / sd.replace(0, np.nan), 0.0)
    return pd.DataFrame(out).fillna({f"{c}_z": 0.0 for c in SRC})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--pool", default="")
    a = ap.parse_args()
    name = a.pool or ("test_pool_t020pox" if a.test else "fused_pool_all_blob5_t020_labelled")
    pool = pd.read_parquet(OUT / f"{name}.parquet")
    missing = [c for c in SRC if c not in pool.columns]
    assert not missing, f"havuzda yok: {missing}"
    d = build(pool)
    dest = OUT / ("regime_features_test_t020.parquet" if a.test else "regime_features_t020.parquet")
    d.to_parquet(dest, index=False)
    print(f"{name}: {len(d)} satir, {len(REG)} ozellik -> {dest.name}")
    print(f"  kare basina aday: medyan {pool.groupby('stem').size().median():.0f}")
    for c in SRC[:4]:
        print(f"  {c:16s} mutlak medyan {pool[c].median():.3f}   "
              f"pct medyan {d[c+'_pct'].median():.3f}   z sd {d[c+'_z'].std():.3f}")


if __name__ == "__main__":
    main()
