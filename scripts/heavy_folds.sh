#!/usr/bin/env bash
# Folds 1-4 of the heavy-60 recipe, so the pixel gain can be converted to PQ.
#
# Why this and not another single-fold variant. heavy-60 measured 0.7028 pixel-Dice
# against a matched 0.6995, but the competition metric is PQ, and this project has
# a documented pattern of pixel- and oracle-level gains that the learner never
# captures (six structural changes, docs/15 section 4). Until the OOF pool gives a
# PQ, +0.0033 is a promise rather than a result.
#
# It is also the prerequisite for ANY leaderboard probe: submit_v2 builds the
# semantic lane from five folds (currently hardcoded to sem_b3_f{0..4}_1536_cldice),
# so no new semantic model can reach the leaderboard on fold 0 alone.
#
# 768 rather than 1536 for these four: heavy-60 at 768 (0.7028) already beat the
# 1536 light record (0.70185), and four 1536 folds would cost 15.6 h against 3 h.
# If the 1536 fold-0 run shows the recipe and the resolution compose, these get
# rerun at 1536 -- but that decision needs its number first.
#
# Waits on the heavy_1536 chain's log sentinel. NOT on pgrep: Git Bash has no
# pgrep, and a `while pgrep ...` guard silently evaluates false, which already
# cost us one accidental parallel run.
set -u
cd "$(dirname "$0")/.."
LOG=outputs/eda/heavy_folds.log
WAITON=outputs/eda/heavy1536.log
mkdir -p outputs/eda

while ! grep -q "^chain done" "$WAITON" 2>/dev/null; do sleep 120; done
echo "=== heavy folds 1-4 start $(date +%H:%M:%S) ===" >> "$LOG"

for F in 1 2 3 4; do
  echo "--- fold $F start $(date +%H:%M:%S)" >> "$LOG"
  python src/models/finetune_aug.py \
      --aug heavy --fold "$F" --imgsz 768 --epochs 60 \
      --batch 2 --accum 1 --lr 3e-4 --pct-start 0.1 --wd 1e-4 \
      --cldice 0.3 --cldice-warmup 5 \
      --name "aug_heavy_f${F}_768_60ep" > /dev/null 2>&1
  echo "fold $F exit=$? $(date +%H:%M:%S)" >> "$LOG"
done
echo "chain done $(date +%H:%M:%S)" >> "$LOG"
