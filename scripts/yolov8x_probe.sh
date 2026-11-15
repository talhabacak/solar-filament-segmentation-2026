#!/usr/bin/env bash
# Is the public 0.40 recipe's edge the BACKBONE? One variable: yolo11m -> yolov8x.
#
# What the operating-point test already settled: their post-processing is not the
# edge. Applied to our own five e013 folds on the full 707-image OOF, their exact
# settings give PQ 0.4023 against our 0.4178 single-model and 0.4286 pipeline. Our
# post-processing is 0.026 BETTER, so their +0.02 on the board has to come from
# the model.
#
# Their settings are nonetheless well chosen -- NMS iou 0.0 is worth +0.025 over
# the ultralytics default 0.7 (0.4023 vs 0.3771), because the metric needs
# pixel-disjoint instances and an overlapping duplicate can never be a second TP.
#
# So this changes exactly one thing against e013: the backbone. Same fold-0 split,
# same imgsz 1792, same nc=1/single_cls, same mask_ratio and overlap_mask, same
# epochs and patience. If yolov8x closes the gap, capacity was the answer after
# all and our "capacity axis is dead" reading (from yolo11m vs yolo11l, a much
# narrower range) needs revising.
#
# Waits for the tiled run's log sentinel, then runs the tiled gate, then trains.
set -u
cd "$(dirname "$0")/.."
LOG=outputs/eda/yolov8x.log
TILED=outputs/runs_tiled/tiled_b3_f0_1056
mkdir -p outputs/eda

while ! grep -q "^done\." "$TILED/train.log" 2>/dev/null; do sleep 120; done
echo "tiled finished $(date +%H:%M:%S); running the gate" >> "$LOG"
python src/models/eval_tiled.py --weights "$TILED/best.pt" --fold 0 --tta 8 \
    > outputs/eda/eval_tiled.log 2>&1
echo "gate exit=$? $(date +%H:%M:%S)" >> "$LOG"

echo "=== yolov8x-seg f0 1792 start $(date +%H:%M:%S) ===" >> "$LOG"
python - <<'PY' >> "$LOG" 2>&1
from pathlib import Path
from ultralytics import YOLO
ROOT = Path(".").resolve()
m = YOLO("yolov8x-seg.pt")
m.train(
    data=str(ROOT / "data" / "yolo_v1" / "fold0.yaml"),
    epochs=40, patience=15, batch=2, imgsz=1792,
    single_cls=True, overlap_mask=False, mask_ratio=4,
    lr0=0.01, lrf=0.01, close_mosaic=10, amp=True, workers=4,
    project=str(ROOT / "outputs" / "runs"), name="e017_yolov8x_f0_1792",
    exist_ok=True, verbose=False, plots=False,
)
PY
echo "yolov8x exit=$? $(date +%H:%M:%S)" >> "$LOG"
