#!/usr/bin/env bash
# Fine-tune augmentation study: light (control) / medium / heavy, 30 epochs each.
#
# Recipe is the REFERENCE run's, not the new pretraining standard: these start
# from ImageNet, and lr 3e-5 is a fine-tune-from-GONG number that measured 0.6902
# vs 0.6996 at 1e-4. Matching sem_b3_f0_768_cldice (OneCycle 3e-4, pct_start 0.1,
# clDice 0.3 after 5 epochs) makes the arms readable against its 0.6981.
#
# batch 2, accum 1 -- the reference's exact setting. A first pass used accum 2 on
# the assumption that the reference ran batch 4; its own log says batch=2 at the
# same 7.0 GB peak, so accumulation bought no VRAM and merely halved the number
# of optimiser steps. That penalises the heavy arm specifically, since more
# augmentation needs MORE training, not less. Matching exactly also makes the
# `light` arm a reproduction check: it should land near 0.6981.
#
# Sequential, not parallel: the 50-epoch GONG pretraining already holds 9.2 GB of
# the 16.3 GB card, and two arms at once would spill to host RAM under WDDM and
# run 40x slower.
set -u
cd "$(dirname "$0")/.."
LOG=outputs/eda/aug_chain.log
mkdir -p outputs/runs_aug outputs/eda

for AUG in light medium heavy; do
  echo "=== $AUG start $(date +%H:%M:%S) ===" >> "$LOG"
  python src/models/finetune_aug.py \
      --aug "$AUG" --fold 0 --imgsz 768 --epochs 30 \
      --batch 2 --accum 1 --lr 3e-4 --pct-start 0.1 --wd 1e-4 \
      --cldice 0.3 --cldice-warmup 5 \
      --name "aug_${AUG}_f0_768" > /dev/null 2>&1
  echo "$AUG exit=$?" >> "$LOG"
done
echo "chain done $(date +%H:%M:%S)" >> "$LOG"
