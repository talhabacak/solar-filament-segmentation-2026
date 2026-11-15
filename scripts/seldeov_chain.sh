#!/usr/bin/env bash
# Run experiment A when the no-deoverlap pool finishes building.
set -u
cd "$(dirname "$0")/.."
while ! ls outputs/selector/fused_pool_nodeov.parquet >/dev/null 2>&1; do sleep 60; done
sleep 15
python src/models/eval_seldeov.py > outputs/eda/eval_seldeov.log 2>&1
echo "done $(date +%H:%M:%S)" >> outputs/eda/eval_seldeov.log
