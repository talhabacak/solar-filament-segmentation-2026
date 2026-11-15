#!/usr/bin/env bash
# Folds 1-4 of yolo11m @2048 (the e016 recipe) -- the honest-band detector upgrade.
#
# Rationale: at the verified operating point (conf 0.3, NMS iou 0.0), 2048 beats
# 1792 on fold 0 (PQ 0.4235 vs 0.4193, SQ 0.6711 vs 0.6656). The earlier
# "resolution closed" reading was operating-point-dependent. Five 2048 folds let
# the fused pool + selector be rebuilt on higher-SQ candidates, which is the
# path to the top of the honest band (~0.43-0.45 local, 0.40+ LB).
set -u
cd "$(dirname "$0")/.."
LOG=outputs/eda/e016_folds.log
mkdir -p outputs/eda
for F in 1 2 3 4; do
  echo "--- fold $F start $(date +%H:%M:%S)" >> "$LOG"
  python src/models/train_yolov8x.py \
      --model yolo11m-seg.pt --fold "$F" --imgsz 2048 --epochs 40 \
      --patience 15 --batch 2 \
      --name "e016_yolo11m_f${F}_2048" > /dev/null 2>&1
  echo "fold $F exit=$? $(date +%H:%M:%S)" >> "$LOG"
done
echo "chain done $(date +%H:%M:%S)" >> "$LOG"
