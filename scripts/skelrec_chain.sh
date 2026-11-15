#!/usr/bin/env bash
# Skeleton Recall Loss arms, plus the spine-aux re-run its corrupted first
# attempt invalidated. Baseline to beat: heavy-60 + clDice = 0.7028.
set -u
cd "$(dirname "$0")/.."
LOG=outputs/eda/skelrec.log
mkdir -p outputs/eda

echo "--- skelrec w=1.0 replacing clDice $(date +%H:%M:%S)" >> "$LOG"
python src/models/train_skelrecall.py --fold 0 --skel-w 1.0 --cldice 0.0 \
    --name skelrec_f0_w1.0_nocld > /dev/null 2>&1
echo "exit=$? $(date +%H:%M:%S)" >> "$LOG"

echo "--- skelrec w=0.1 stacked on clDice $(date +%H:%M:%S)" >> "$LOG"
python src/models/train_skelrecall.py --fold 0 --skel-w 0.1 --cldice 0.3 \
    --name skelrec_f0_w0.1_cld0.3 > /dev/null 2>&1
echo "exit=$? $(date +%H:%M:%S)" >> "$LOG"

echo "--- spine-aux w=0.3 RE-RUN with corrected target $(date +%H:%M:%S)" >> "$LOG"
python src/models/train_spine_aux.py --fold 0 --spine-w 0.3 \
    --name spineaux_f0_768_w0.3_fixed > /dev/null 2>&1
echo "exit=$? $(date +%H:%M:%S)" >> "$LOG"
echo "chain done $(date +%H:%M:%S)" >> "$LOG"
