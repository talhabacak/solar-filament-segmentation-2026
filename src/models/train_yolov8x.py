"""yolov8x-seg at 1792 -- one variable against e013, the backbone.

Why this run exists. The public inference notebook that scores 0.40 is a single
yolov8x-seg at 1792 with no folds, no ensemble and no selector. Applying its exact
operating point to our own five e013 yolo11m folds gives PQ 0.4023 on the full
707-image OOF, against our 0.4178 single-model and 0.4286 pipeline -- so their
post-processing is 0.026 WORSE than ours and cannot be the source of their edge.
That leaves the model, and two candidates: the backbone (71M vs our 27M) and
training on all 707 images instead of a fold's 80%.

This isolates the first. Everything else matches e013's args.yaml exactly: same
fold-0 split, imgsz 1792, single_cls, overlap_mask False, mask_ratio 4, lr0 0.01,
lrf 0.01, close_mosaic 10, 40 epochs, patience 15. Only `model` changes.

If it closes the gap, our "capacity axis is dead" reading needs revising -- that
was measured on yolo11m vs yolo11l, 27M vs 32M, a far narrower range than 27M vs
71M.

Must be a FILE, not a heredoc. Ultralytics spawns dataloader workers, and on
Windows spawn re-imports the main module: run from stdin and every worker dies
with OSError 22 trying to read '<stdin>' as a path. The __main__ guard is needed
for the same reason.

Usage: python src/models/train_yolov8x.py [--model yolov8x-seg.pt] [--fold 0]
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8x-seg.pt")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=1792)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    name = a.name or f"e017_{Path(a.model).stem}_f{a.fold}_{a.imgsz}"

    from ultralytics import YOLO
    YOLO(a.model).train(
        data=str(ROOT / "data" / "yolo_v1" / f"fold{a.fold}.yaml"),
        epochs=a.epochs, patience=a.patience, batch=a.batch, imgsz=a.imgsz,
        # every one of these mirrors e013_yolo11m_f0_1792/args.yaml
        single_cls=True, overlap_mask=False, mask_ratio=4,
        lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
        warmup_epochs=3.0, close_mosaic=10, cos_lr=False, amp=True,
        workers=a.workers, project=str(ROOT / "outputs" / "runs"), name=name,
        exist_ok=True, verbose=True, plots=False, seed=0,
    )


if __name__ == "__main__":
    main()
