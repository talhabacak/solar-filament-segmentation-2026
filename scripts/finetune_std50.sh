#!/usr/bin/env bash
# Fine-tune the finished 50-epoch GONG pretraining under Talha's new standard.
#
# Standard as specified: 30 epochs, lr 3e-5 -> 1e-6 cosine, patience 8, AdamW,
# wd my call (0.005, matching the pretraining stage so both stages are the same
# recipe), plus every adopted ablation -- clDice 0.3 after 5 epochs, and
# all-copies, which is already how the semantic lane reads `entries`.
#
# Two learning rates, not one. The spec says 3e-5, but that exact value measured
# 0.6902 against 1e-4's 0.6996 on the previous GONG checkpoint -- a 30-epoch
# cosine from 3e-5 barely moves a pretrained representation. Running both keeps
# the spec honoured and the comparison on the record instead of me quietly
# substituting my own number.
#
# Sequential. Each 768 run peaks at 7.0 GB, so two would sit at ~14 of 16.3 GB --
# closer to the WDDM spill margin that cost us hours last night than 22 minutes
# of wall clock is worth.
set -u
cd "$(dirname "$0")/.."
CK=outputs/runs_pretrain/pre_gong_b3_768_std50/best.pt
LOG=outputs/eda/ft_std50.log
mkdir -p outputs/eda

for LR in 3e-5 1e-4; do
  echo "=== 768 lr=$LR start $(date +%H:%M:%S) ===" >> "$LOG"
  python src/models/train_semantic_arch.py \
      --arch unet --encoder timm-efficientnet-b3 --in-ch 3 \
      --fold 0 --imgsz 768 --batch 2 --epochs 30 \
      --init "$CK" --cosine --lr "$LR" --min-lr 1e-6 --wd 0.005 \
      --patience 8 --cldice 0.3 --cldice-warmup 5 \
      --name "sem_b3_f0_768_std50_lr$LR" > /dev/null 2>&1
  echo "768 lr=$LR exit=$? $(date +%H:%M:%S)" >> "$LOG"
done
echo "chain done $(date +%H:%M:%S)" >> "$LOG"
