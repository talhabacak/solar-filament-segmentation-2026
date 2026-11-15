#!/usr/bin/env bash
# Stage 2 of the crack idea on the winning backbone: fine-tune the crack-pretrained
# yolo11m on filaments with the exact e013 recipe, then score fold-0 PQ.
#
# Waits on the log sentinel ("epochs completed in") -- not a PID, for the Windows
# reasons in MISTAKES.md -- and tolerates the pretraining being resumed after the
# reboot: the sentinel appears exactly once whenever the run truly ends.
set -u
cd "$(dirname "$0")/.."
LOG=outputs/eda/crack_11m_chain.log
WAITON=outputs/eda/crack_pre_resume.log
mkdir -p outputs/eda

while ! tr '\r' '\n' < "$WAITON" 2>/dev/null | grep -q "epochs completed in"; do sleep 180; done
echo "crack pretrain finished $(date +%H:%M:%S)" >> "$LOG"
sleep 20

echo "=== e019 yolo11m crack-init fine-tune start $(date +%H:%M:%S) ===" >> "$LOG"
python src/models/train_yolov8x.py \
    --model outputs/runs/crack_pre_yolo11m_640/weights/best.pt \
    --fold 0 --imgsz 1792 --epochs 40 --patience 15 --batch 2 \
    --name e019_yolo11m_crack_f0_1792 > /dev/null 2>&1
echo "e019 exit=$? $(date +%H:%M:%S)" >> "$LOG"

echo "=== fold-0 PQ eval $(date +%H:%M:%S) ===" >> "$LOG"
python src/eda/pq_single_yolo.py \
    --weights outputs/runs/e019_yolo11m_crack_f0_1792/weights/best.pt >> "$LOG" 2>&1
echo "chain done $(date +%H:%M:%S)" >> "$LOG"
