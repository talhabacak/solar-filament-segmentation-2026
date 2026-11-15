#!/usr/bin/env bash
# Does the fusion vote threshold decide the 72 rows fusion destroys?
#
# Measured on fold 0 against the CORRECTED unreachable population (414 rows,
# recomputed by src/eda/true_unreachable.py): of the 147 rows the raw e013
# detector reaches but the built pool does not, 75 die at greedy de-overlap and
# 72 are already gone before it. Those 72 sit at median IoU 0.437 and p90 0.490
# in the de-overlap-free pool and NONE of them lack an overlapping candidate --
# clustering finds them and the confidence-weighted vote pushes the fused mask
# under the threshold.
#
# The vote is a single hardcoded 0.5 in _fuse_body: a pixel survives only where
# members carrying more than half the confidence agree, which pulls the fused
# mask toward the INTERSECTION. Lower grows it toward the union.
#
# The control is rebuilt rather than reused: the flag touched _fuse_body, and a
# vote of 0.5 should reproduce the old pool exactly. If it does not, the
# comparison is measuring the edit rather than the threshold.
#
# This reports pool REACHABILITY, which is a ceiling and not a result. The
# ledger's own rule is that raising a ceiling has never once raised realised PQ
# here, so an arm only earns a harness run if reach moves materially.
#
# Usage: bash scripts/fusion_vote_sweep.sh
set -u
cd "$(dirname "$0")/.."
LOG=outputs/eda/fusion_vote.log
: > "$LOG"

for V in 0.50 0.35 0.65; do
  TAG=$(echo "$V" | tr -d '.')
  echo "=== vote $V  $(date +%H:%M:%S) ===" >> "$LOG"
  python src/models/build_fused_pool.py --folds 0 --vote "$V" \
      --out "fused_pool_f0_vote${TAG}.parquet" >> "$LOG" 2>&1
  python - "$V" "fused_pool_f0_vote${TAG}.parquet" <<'PY' >> "$LOG" 2>&1
import sys, time
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))
from filament.rle import decode_rle

vote, pool_name = sys.argv[1], sys.argv[2]
folds = pd.read_parquet("data/splits/folds_v1.parquet")
f0 = set(folds.loc[folds["fold"] == 0, "file_name"].str.replace(".jpeg", "", regex=False))
gt = pd.read_parquet("outputs/gt_cache/gt_full.parquet")
gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
gt = gt[gt["stem"].isin(f0)]
pool = pd.read_parquet(f"outputs/selector/{pool_name}")

def boxed(m):
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return None
    return ((int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1),
            m[ys.min():ys.max() + 1, xs.min():xs.max() + 1], int(m.sum()))

def iou(a, b):
    (ay0, ay1, ax0, ax1), asub, asum = a
    (by0, by1, bx0, bx1), bsub, bsum = b
    y0, y1 = max(ay0, by0), min(ay1, by1)
    x0, x1 = max(ax0, bx0), min(ax1, bx1)
    if y0 >= y1 or x0 >= x1:
        return 0.0
    i = int(np.logical_and(asub[y0-ay0:y1-ay0, x0-ax0:x1-ax0],
                           bsub[y0-by0:y1-by0, x0-bx0:x1-bx0]).sum())
    u = asum + bsum - i
    return i / u if u else 0.0

by = {s: list(g["rle"]) for s, g in pool.groupby("stem")}
best, areas = [], []
for stem, g in gt.groupby("stem"):
    cs = [boxed(decode_rle(r).astype(bool)) for r in by.get(stem, [])]
    cs = [c for c in cs if c is not None]
    areas.extend(c[2] for c in cs)
    for r in g.itertuples():
        gm = boxed(decode_rle(r.segmentation_rle).astype(bool))
        best.append(max((iou(gm, c) for c in cs), default=0.0) if gm else 0.0)
b = np.array(best)
print(f"VOTE {vote}: candidates {len(pool)}  median candidate area {np.median(areas):.0f}px")
print(f"  reach IoU>0.5 : {(b > 0.5).sum()}/{len(b)} ({100*(b>0.5).mean():.2f}%)")
print(f"  mean best IoU : {b.mean():.4f}   near-miss band 0.3-0.5: {((b>0.3)&(b<=0.5)).sum()}")
PY
done

echo "=== summary ===" >> "$LOG"
grep -E "^VOTE|reach IoU|mean best" "$LOG" >> "$LOG"
echo "sweep done $(date +%H:%M:%S)" >> "$LOG"
