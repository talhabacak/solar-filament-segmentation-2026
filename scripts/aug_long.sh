#!/usr/bin/env bash
# heavy-60 recipe (our strongest single-model config, 0.7028 fold-0) on folds 1-4,
# completing the five-fold set. Enables an honest per-fold ensemble upgrade of the
# semantic lane and a stronger production ensemble either way.
set -u
cd "$(dirname "$0")/.."
LOG=outputs/eda/heavy_folds.log
for F in 1 2 3 4; do
  echo "--- heavy fold $F start $(date +%H:%M:%S)" >> "$LOG"
  python src/models/finetune_aug.py \
      --aug heavy --fold "$F" --imgsz 768 --epochs 60 \
      --batch 2 --accum 1 --lr 3e-4 --pct-start 0.1 --wd 1e-4 \
      --cldice 0.3 --cldice-warmup 5 \
      --name "aug_heavy_f${F}_768_60ep" > /dev/null 2>&1
  echo "heavy fold $F exit=$? $(date +%H:%M:%S)" >> "$LOG"
done
echo "chain done $(date +%H:%M:%S)" >> "$LOG"
