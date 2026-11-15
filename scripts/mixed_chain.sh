#!/usr/bin/env bash
# Transductive lane: sanity arm first (fold 0, honest val must hold ~0.70), then
# the submission member (all 707 + test pseudo, fixed 60 epochs).
set -u
cd "$(dirname "$0")/.."
LOG=outputs/eda/mixed_chain.log
mkdir -p outputs/eda

echo "=== mixed f0 start $(date +%H:%M:%S) ===" >> "$LOG"
python src/models/train_mixed.py --fold 0 --name mixed_f0_768 > /dev/null 2>&1
echo "f0 exit=$? $(date +%H:%M:%S)" >> "$LOG"

echo "=== mixed all707 start $(date +%H:%M:%S) ===" >> "$LOG"
python src/models/train_mixed.py --all-data --name mixed_all707 > /dev/null 2>&1
echo "all707 exit=$? $(date +%H:%M:%S)" >> "$LOG"
echo "chain done $(date +%H:%M:%S)" >> "$LOG"
