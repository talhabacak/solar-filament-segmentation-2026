#!/usr/bin/env bash
# Spine-auxiliary sweep: two weights against the heavy-60 baseline (0.7028).
# 0.3 is the clDice weight we already trust for an auxiliary term; 0.1 guards
# against the aux task dominating a 13M encoder.
set -u
cd "$(dirname "$0")/.."
LOG=outputs/eda/spine_aux.log
mkdir -p outputs/eda
for W in 0.3 0.1; do
  echo "--- spine-aux w=$W start $(date +%H:%M:%S)" >> "$LOG"
  python src/models/train_spine_aux.py --fold 0 --epochs 60 --spine-w "$W" \
      --name "spineaux_f0_768_w${W}" > /dev/null 2>&1
  echo "w=$W exit=$? $(date +%H:%M:%S)" >> "$LOG"
done
echo "chain done $(date +%H:%M:%S)" >> "$LOG"
