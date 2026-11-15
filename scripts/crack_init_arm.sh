#!/usr/bin/env bash
# Second arm: same yolov8x, initialised from crack segmentation instead of COCO.
#
# Talha's idea, and it is the best-motivated initialisation we have tried. Cracks
# are thin, curvilinear, branching, dark-on-light structures of variable width --
# structurally the closest thing to a filament we have found, and unlike every
# previous attempt this one carries REAL human instance labels of such structures
# rather than pseudo-labels from our own teacher (GONG) or a different modality
# with no thin-structure prior (Sentinel-1).
#
# The checkpoint is OpenSistemas/YOLOv8-crack-seg yolov8x, AGPL-3.0, trained from
# COCO for 100 epochs at 640 on the public-domain Crack-seg set (3717 train
# images). Its head is already nc=1 {0: 'crack'}, matching our single_cls export,
# so nothing has to be surgically resized -- which is what invalidated the first
# Sentinel-1 result.
#
# Rules: permitted. The host's ban (discussion 729396) is on external GROUND
# TRUTH for filaments; crack labels are a different object class in a different
# domain, and Sec 2.6.b explicitly allows "external data and models" that are
# freely and equally accessible. Same category as the GONG archive images and
# ImageNet weights. The weights live in data/external/, which is gitignored.
#
# Expectation, calibrated on our own record rather than hope: the entire
# initialisation axis is worth about +0.017 (ImageNet vs random), and four
# different pretrainings have failed to beat ImageNet -- GONG +0.0015, GONG at 50
# epochs -0.0066, Sentinel-1 -0.0096. So this is a mechanism worth testing, not a
# lever that closes a 0.15 gap.
#
# One variable against e017: the initial weights. Same fold, imgsz, epochs,
# patience, batch, mask_ratio, overlap_mask, lr schedule.
set -u
cd "$(dirname "$0")/.."
LOG=outputs/eda/crack_arm.log
WAITON=outputs/eda/yolov8x_train.log
mkdir -p outputs/eda

# Ultralytics prints "N epochs completed in ..." on both normal and early-stopped
# finishes. Waiting on the log rather than a PID, for the Windows reasons already
# logged: ps -W reports WINPIDs bash's kill does not accept, and finished runs can
# linger with their worker processes alive.
while ! grep -q "epochs completed in" "$WAITON" 2>/dev/null; do sleep 120; done
echo "e017 finished $(date +%H:%M:%S)" >> "$LOG"
sleep 30

echo "=== e018 crack-init start $(date +%H:%M:%S) ===" >> "$LOG"
python src/models/train_yolov8x.py \
    --model data/external/crack_weights/yolov8x_crack.pt \
    --fold 0 --imgsz 1792 --epochs 40 --patience 15 --batch 2 \
    --name e018_yolov8x_crack_f0_1792 >> "$LOG" 2>&1
echo "e018 exit=$? $(date +%H:%M:%S)" >> "$LOG"
