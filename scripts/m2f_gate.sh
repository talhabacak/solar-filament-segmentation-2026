#!/usr/bin/env bash
# Run the M2F gate when training finishes (log sentinel, as always on Windows).
set -u
cd "$(dirname "$0")/.."
RUN=outputs/runs_m2f/m2f_swint_f0_1024
while ! grep -q "^done ->" "$RUN/train.log" 2>/dev/null; do sleep 180; done
sleep 20
python src/models/eval_m2f.py --weights "$RUN/last.pt" > outputs/eda/eval_m2f.log 2>&1
echo "gate done $(date +%H:%M:%S)" >> outputs/eda/eval_m2f.log
