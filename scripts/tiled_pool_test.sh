#!/usr/bin/env bash
# Fourth-source test, paired against the same fold-0 harness that measured M2F.
#
# The precondition is measured, not assumed. Of fold-0's 440 never-proposed GT
# rows the native-resolution tiled lane reaches 106 (24.1%) at IoU > 0.5, close
# to the query lane's 121 that went on to add +0.0025 -- and, decisively, only
# 55 of those are the same rows. Tiled-only 51, M2F-only 66, union 172 (39.1%).
# Two sources that miss different things are the one configuration that has ever
# added at the pipeline level here, so the third arm is the point of the test.
#
# Arms: base (e013 detectors alone) is already measured at 0.4343 cross-fitted
# and M2F at 0.4368; this adds tiled alone and all three together, on the same
# pool builder, the same shrink and the same cross-fitted selector, so any
# difference is attributable to the sources.
#
# Usage: bash scripts/tiled_pool_test.sh
set -u
cd "$(dirname "$0")/.."
LOG=outputs/eda/tiled_pool.log
mkdir -p outputs/eda
: > "$LOG"

echo "=== concat $(date +%H:%M:%S) ===" >> "$LOG"
python - <<'PY' >> "$LOG" 2>&1
import pandas as pd
a = pd.read_parquet("outputs/oof_cache/fold0_preds_e013.parquet")
m = pd.read_parquet("outputs/oof_cache/fold0_preds_m2f.parquet")
t = pd.read_parquet("outputs/oof_cache/fold0_preds_tiled.parquet")
# the intersection drops e013's rle_lo/rle_hi, which build_fused_pool never
# reads -- checked, so the arms stay comparable to the M2F run that did the same
c2 = [c for c in a.columns if c in t.columns]
pd.concat([a[c2], t[c2]], ignore_index=True).to_parquet(
    "outputs/oof_cache/fold0_preds_e013tiled.parquet", index=False)
c3 = [c for c in a.columns if c in m.columns and c in t.columns]
pd.concat([a[c3], m[c3], t[c3]], ignore_index=True).to_parquet(
    "outputs/oof_cache/fold0_preds_e013m2ftiled.parquet", index=False)
print(f"e013 {len(a)} + m2f {len(m)} + tiled {len(t)}")
PY

echo "=== pools $(date +%H:%M:%S) ===" >> "$LOG"
python src/models/build_fused_pool.py --folds 0 --det-tag _e013tiled \
    --out fused_pool_f0_tiled.parquet >> "$LOG" 2>&1
python src/models/build_fused_pool.py --folds 0 --det-tag _e013m2ftiled \
    --out fused_pool_f0_all3.parquet >> "$LOG" 2>&1

echo "=== harness tiled $(date +%H:%M:%S) ===" >> "$LOG"
python src/models/fused_shrink_eval.py --pool fused_pool_f0_tiled.parquet \
    --tag _f0tiled >> "$LOG" 2>&1
echo "=== harness all3 $(date +%H:%M:%S) ===" >> "$LOG"
python src/models/fused_shrink_eval.py --pool fused_pool_f0_all3.parquet \
    --tag _f0all3 >> "$LOG" 2>&1

echo "=== verdict ===" >> "$LOG"
python - <<'PY' >> "$LOG" 2>&1
import json
rows = [("base (e013 only)", "_f0base"), ("+ M2F", "_f0m2f"),
        ("+ tiled", "_f0tiled"), ("+ M2F + tiled", "_f0all3")]
base = None
for name, tag in rows:
    try:
        d = json.load(open(f"outputs/selector/fused_shrink_results{tag}.json"))
    except FileNotFoundError:
        print(f"{name:22s} MISSING"); continue
    pq, orc, auc = d["crossfit_abs_pq"], d["oracle"]["pq"], d["auc"]
    if base is None:
        base = pq
    print(f"{name:22s} crossfit {pq:.4f} ({pq-base:+.4f})  "
          f"oracle {orc:.4f}  auc {auc:.4f}")
PY
echo "chain done $(date +%H:%M:%S)" >> "$LOG"
