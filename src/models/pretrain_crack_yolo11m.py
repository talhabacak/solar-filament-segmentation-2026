"""Stage 1 of Talha's crack idea on the WINNING backbone: yolo11m on crack-seg.

Why this exists. The crack initialisation is the first pretraining in the whole
project to beat its matched control on real PQ (+0.0055 within yolov8x), but the
yolov8x backbone it came attached to costs more than the init gains (yolo11m
COCO 0.4193 > yolov8x crack 0.4118 on fold-0 PQ). The public repo only ships
YOLOv8 weights, so transferring the mechanism to yolo11m means redoing the crack
pretraining ourselves: COCO -> crack-seg (this run) -> filaments (e019).

Mirrors the OpenSistemas recipe where it is known: 100 epochs at 640 from COCO
weights, default hyperparameters, single class. 3717 train images at 640 is
fast -- roughly an hour.

Usage: python src/models/pretrain_crack_yolo11m.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    from ultralytics import YOLO
    YOLO("yolo11m-seg.pt").train(
        data=str(ROOT / "data" / "external" / "crack_seg" / "crack_local.yaml"),
        epochs=100, patience=20, batch=16, imgsz=640,
        project=str(ROOT / "outputs" / "runs"), name="crack_pre_yolo11m_640",
        exist_ok=True, verbose=False, plots=False, seed=0,
    )


if __name__ == "__main__":
    main()
