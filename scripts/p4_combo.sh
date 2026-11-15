#!/usr/bin/env bash
# P4: the composition of two half-failures whose failure modes cancel.
# multithresh candidates died to BUILD-time de-overlap (pixel theft between
# nested masks); selection-time de-overlap (A) ran on the OLD candidate set.
# no-deoverlap + multithresh together: nested candidates stay whole, faint
# clean components enter the pool, carving happens only among the ~7 selected.
set -u
cd "$(dirname "$0")/.."
LOG=outputs/eda/p4_combo.log
mkdir -p outputs/eda
echo "=== build $(date +%H:%M:%S) ===" >> "$LOG"
python src/models/build_fused_pool.py --no-deoverlap --multithresh \
    --out fused_pool_nodeov_mt.parquet >> "$LOG" 2>&1
echo "=== eval $(date +%H:%M:%S) ===" >> "$LOG"
python src/models/eval_seldeov.py --pool fused_pool_nodeov_mt.parquet >> "$LOG" 2>&1
echo "chain done $(date +%H:%M:%S)" >> "$LOG"
