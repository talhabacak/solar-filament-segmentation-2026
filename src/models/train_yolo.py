"""E010: single-stage YOLO-seg baseline training (fold-aware).

Physics-aware augmentation policy: no mosaic/mixup/copy-paste (destroys full-disk
radial context), small rotations only, horizontal flip allowed (single-class masks
are chirality-agnostic), no HSV (grayscale), mild scale jitter.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11m-seg.pt")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--name", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--translate", type=float, default=0.05,
                    help="E185: 0.0 removes the disk-centre jitter the fixed geometry forbids")
    ap.add_argument("--scale", type=float, default=0.10,
                    help="E185: 0.0 removes the disk-radius jitter (the real spread is 0.3%)")
    ap.add_argument("--data", default="yolo_v1", help="dataset dir under data/ (label variant)")
    ap.add_argument("--mask-ratio", type=int, default=4,
                    help="seg target grid = imgsz/mask_ratio; 4 caps GT fidelity at IoU~0.63")
    ap.add_argument("--optimizer", default="auto")
    ap.add_argument("--lr0", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=0.0005)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--box", type=float, default=7.5, help="box+seg loss gain")
    ap.add_argument("--cos-lr", action="store_true")
    a = ap.parse_args()

    from ultralytics import YOLO
    name = a.name or f"e010_{Path(a.model).stem}_f{a.fold}_{a.imgsz}"
    model = YOLO(a.model)
    extra = {}
    if a.lr0 is not None:
        extra["lr0"] = a.lr0
    model.train(
        data=str(ROOT / "data" / a.data / f"fold{a.fold}.yaml"),
        imgsz=a.imgsz, epochs=a.epochs, batch=a.batch,
        single_cls=True, amp=True, cache=False,
        project=str(ROOT / "outputs" / "runs"), name=name,
        seed=2026, deterministic=False, patience=15,
        overlap_mask=False,          # instances must stay separate
        mask_ratio=a.mask_ratio, optimizer=a.optimizer, cos_lr=a.cos_lr,
        weight_decay=a.weight_decay, dropout=a.dropout, box=a.box, **extra,
        mosaic=0.0, mixup=0.0, copy_paste=0.0, erasing=0.0,
        # E185 (Talha, 2026-09-13): translate and scale contradict the policy the semantic lane
        # follows. The disk sits at a FIXED centre with a measured radius of 904.1 +/- 2.8 px --
        # a 0.3% spread -- so translate 0.05 (+/-90 px at 1792) and scale 0.10 teach invariance
        # to a variation the data does not contain, and the rest of the pipeline (r_norm
        # features, the R_MAX 0.97 cut, the disk-geometry registration) assumes that fixed frame.
        degrees=10.0, translate=a.translate, scale=a.scale, shear=0.0, perspective=0.0,
        fliplr=0.5, flipud=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,
        plots=a.plots, val=True, workers=a.workers,
    )

if __name__ == "__main__":
    main()
