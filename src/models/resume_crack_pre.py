"""Resume the crack pretraining after the reboot -- ultralytics restores the
optimizer, LR schedule and epoch counter from last.pt when resume=True."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    from ultralytics import YOLO
    YOLO(str(ROOT / "outputs" / "runs" / "crack_pre_yolo11m_640" /
             "weights" / "last.pt")).train(resume=True)


if __name__ == "__main__":
    main()
