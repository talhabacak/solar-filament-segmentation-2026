#!/usr/bin/env bash
# The winning recipe at the winning resolution, plus the "is 60 enough?" check.
#
# The 2x2 came out with a clear interaction rather than a main effect:
#
#              30 ep     60 ep
#   light      0.6995    0.6972   degrades  (reproduces the ledger's negative)
#   heavy      0.6975    0.7028   improves
#
# So "100 epochs degrades" was never a property of the schedule -- it was light
# augmentation being too thin to fill it. heavy-60's 0.7028 beats the previous
# all-time best (0.70185 at 1536) using 2.25x less compute, which makes the
# obvious next question whether the two gains compose.
#
# 1536 needs gradient checkpointing: 1024 at batch 2 already peaks at 12.3 GB and
# 1536 is 2.25x the pixels, so batch 2 would want roughly 27 GB of a 16.3 GB card.
#
# heavy-90 at 768 asks whether 60 was still short. The tail slope fell from
# +0.00083 at 30 epochs to +0.00008 at 60, so this is expected to be small -- it
# is run to bound the axis, not because a gain is likely.
set -u
cd "$(dirname "$0")/.."
LOG=outputs/eda/heavy1536.log
mkdir -p outputs/eda

echo "=== heavy 1536 60ep start $(date +%H:%M:%S) ===" >> "$LOG"
python src/models/finetune_aug.py \
    --aug heavy --fold 0 --imgsz 1536 --epochs 60 \
    --batch 2 --accum 1 --grad-ckpt \
    --lr 3e-4 --pct-start 0.1 --wd 1e-4 \
    --cldice 0.3 --cldice-warmup 5 \
    --name "aug_heavy_f0_1536_60ep" > /dev/null 2>&1
echo "heavy 1536 exit=$? $(date +%H:%M:%S)" >> "$LOG"

echo "=== heavy 768 90ep start $(date +%H:%M:%S) ===" >> "$LOG"
python src/models/finetune_aug.py \
    --aug heavy --fold 0 --imgsz 768 --epochs 90 \
    --batch 2 --accum 1 --lr 3e-4 --pct-start 0.1 --wd 1e-4 \
    --cldice 0.3 --cldice-warmup 5 \
    --name "aug_heavy_f0_768_90ep" > /dev/null 2>&1
echo "heavy 768 90ep exit=$? $(date +%H:%M:%S)" >> "$LOG"
echo "chain done $(date +%H:%M:%S)" >> "$LOG"
