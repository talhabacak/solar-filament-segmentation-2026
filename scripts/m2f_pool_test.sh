#!/usr/bin/env bash
# Third-source test, paired: fold-0 pool with e013 detectors alone vs with the
# M2F query candidates added as extra detector-type members. Same machinery, same
# harness, difference attributable to the new source.
set -u
cd "$(dirname "$0")/.."
LOG=outputs/eda/m2f_pool.log
mkdir -p outputs/eda

echo "=== cache m2f preds $(date +%H:%M:%S) ===" >> "$LOG"
python src/models/cache_m2f_preds.py \
    --weights outputs/runs_m2f/m2f_swint_f0_1024/last.pt --fold 0 >> "$LOG" 2>&1

echo "=== concat $(date +%H:%M:%S) ===" >> "$LOG"
python - <<'PY' >> "$LOG" 2>&1
import pandas as pd
a = pd.read_parquet("outputs/oof_cache/fold0_preds_e013.parquet")
b = pd.read_parquet("outputs/oof_cache/fold0_preds_m2f.parquet")
cols = [c for c in a.columns if c in b.columns]
pd.concat([a[cols], b[cols]], ignore_index=True).to_parquet(
    "outputs/oof_cache/fold0_preds_e013m2f.parquet", index=False)
print(f"concat: {len(a)} e013 + {len(b)} m2f")
PY

echo "=== pools $(date +%H:%M:%S) ===" >> "$LOG"
python src/models/build_fused_pool.py --folds 0 --out fused_pool_f0_base.parquet >> "$LOG" 2>&1
python src/models/build_fused_pool.py --folds 0 --det-tag _e013m2f \
    --out fused_pool_f0_m2f.parquet >> "$LOG" 2>&1

echo "=== harness base $(date +%H:%M:%S) ===" >> "$LOG"
python src/models/fused_shrink_eval.py --pool fused_pool_f0_base.parquet --tag _f0base >> "$LOG" 2>&1
echo "=== harness m2f $(date +%H:%M:%S) ===" >> "$LOG"
python src/models/fused_shrink_eval.py --pool fused_pool_f0_m2f.parquet --tag _f0m2f >> "$LOG" 2>&1
echo "chain done $(date +%H:%M:%S)" >> "$LOG"
