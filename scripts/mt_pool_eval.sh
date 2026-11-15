#!/usr/bin/env bash
# When the multi-threshold pool finishes building, run the standard shrink+selector
# harness on it and drop the result next to the baseline for comparison.
set -u
cd "$(dirname "$0")/.."
LOG=outputs/eda/pool_mt.log
while ! grep -q "saved\|wrote\|->" <(tail -n 3 "$LOG" 2>/dev/null) && ! ls outputs/selector/fused_pool_mt.parquet >/dev/null 2>&1; do sleep 120; done
sleep 10
echo "=== eval start $(date +%H:%M:%S) ===" >> outputs/eda/mt_eval.log
python src/models/fused_shrink_eval.py --pool fused_pool_mt.parquet --tag _mt \
    >> outputs/eda/mt_eval.log 2>&1
echo "eval exit=$? $(date +%H:%M:%S)" >> outputs/eda/mt_eval.log
