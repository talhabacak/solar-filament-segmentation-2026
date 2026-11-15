#!/usr/bin/env bash
# Rebuild the whole selector pipeline on the 2048 detector lane:
#   1. cache each 2048 fold model's OOF predictions (own held-out images, conf
#      0.05 -- the same raw-pool convention as the e013 caches)
#   2. build the fused pool with those detectors (semantic lane unchanged)
#   3. run the shrink+selector harness, tagged _e016 so nothing production is
#      touched
# The comparison that decides adoption: cross-fitted PQ vs the 0.4286 baseline.
set -u
cd "$(dirname "$0")/.."
LOG=outputs/eda/e016_pool.log
mkdir -p outputs/eda

for F in 0 1 2 3 4; do
  echo "--- cache fold $F $(date +%H:%M:%S)" >> "$LOG"
  python src/models/sweep_oof.py cache \
      --weights "outputs/runs/e016_yolo11m_f${F}_2048/weights/best.pt" \
      --fold "$F" --imgsz 2048 --tag _e016 >> "$LOG" 2>&1
done

echo "--- pool build $(date +%H:%M:%S)" >> "$LOG"
python src/models/build_fused_pool.py --det-tag _e016 \
    --out fused_pool_e016.parquet >> "$LOG" 2>&1

echo "--- harness $(date +%H:%M:%S)" >> "$LOG"
python src/models/fused_shrink_eval.py --pool fused_pool_e016.parquet \
    --tag _e016 >> "$LOG" 2>&1
echo "chain done $(date +%H:%M:%S)" >> "$LOG"
