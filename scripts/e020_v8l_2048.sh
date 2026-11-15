#!/usr/bin/env bash
# Replicate HDJoJo's CURRENT recipe with one honest difference: fold-0 split so it
# can be scored locally. Their updated notebook (re-run 08-25, after their 0.54)
# names the model magfilo_260825_yolov8l_seg_imgsz2048_nc1.pt -- yolov8l, native
# 2048, nc=1, same conf 0.3 / iou 0.0 / max_det 100 inference as before.
#
# Everything except backbone+imgsz follows e013's args (our production recipe).
# Waits for the e016 scoring probe to release the GPU first.
set -u
cd "$(dirname "$0")/.."
LOG=outputs/eda/e020.log
while ! grep -q "conf 0.40" outputs/eda/pq_e016.log 2>/dev/null; do sleep 60; done
echo "=== e020 yolov8l 2048 start $(date +%H:%M:%S) ===" >> "$LOG"
python src/models/train_yolov8x.py \
    --model yolov8l-seg.pt --fold 0 --imgsz 2048 --epochs 40 --patience 15 \
    --batch 2 --name e020_yolov8l_f0_2048 > /dev/null 2>&1
echo "e020 exit=$? $(date +%H:%M:%S)" >> "$LOG"
echo "=== fold-0 PQ eval $(date +%H:%M:%S) ===" >> "$LOG"
python src/eda/pq_single_yolo.py \
    --weights outputs/runs/e020_yolov8l_f0_2048/weights/best.pt \
    --imgsz 2048 >> "$LOG" 2>&1
echo "chain done $(date +%H:%M:%S)" >> "$LOG"
