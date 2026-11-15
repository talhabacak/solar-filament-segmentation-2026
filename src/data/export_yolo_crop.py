"""Export disk-ROI-cropped YOLO dataset: constant crop [112:1936] -> 1824x1824.

Disk geometry is FIXED across all 887 images (center 1023.5, R=900; annotations
end at r<=0.966, mask extent box ~1764px), so a constant crop is safe and exactly
invertible (+112 offset). Images saved as PNG (lossless — avoid double-JPEG).
Labels: renormalize any source label set (yolo_v1 medoid / yolo_v2_* consensus).

Usage: python src/data/export_yolo_crop.py --src data/yolo_v1 --out data/yolo_v1_crop
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

CROP0 = 112
CROP1 = 1936
SIZE = CROP1 - CROP0  # 1824


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/yolo_v1")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    src = ROOT / a.src
    out = ROOT / (a.out or (a.src + "_crop"))
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "lists").mkdir(parents=True, exist_ok=True)

    stems = []
    for img_path in sorted((src / "images").glob("*.jpeg")):
        stem = img_path.stem
        stems.append(stem)
        dst = out / "images" / f"{stem}.png"
        if not dst.exists():
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(str(dst), img[CROP0:CROP1, CROP0:CROP1])
        # renormalize labels: x' = (x*2048 - CROP0)/SIZE
        lab = src / "labels" / f"{stem}.txt"
        lines_out = []
        for line in lab.read_text().splitlines():
            parts = line.split()
            cls, coords = parts[0], np.array(parts[1:], dtype=np.float64)
            coords = (coords * 2048.0 - CROP0) / SIZE
            coords = np.clip(coords, 0.0, 1.0)
            lines_out.append(cls + " " + " ".join(f"{v:.6f}" for v in coords))
        (out / "labels" / f"{stem}.txt").write_text("\n".join(lines_out), encoding="utf-8")

    # fold lists from source lists (swap dir + extension)
    for lst in sorted((src / "lists").glob("*.txt")):
        names = [Path(p).stem for p in lst.read_text().splitlines() if p.strip()]
        (out / "lists" / lst.name).write_text(
            "\n".join(str(out / "images" / f"{n}.png") for n in names), encoding="utf-8")
    for k in range(5):
        (out / f"fold{k}.yaml").write_text(
            f"path: {out.as_posix()}\ntrain: lists/train_f{k}.txt\nval: lists/val_f{k}.txt\n"
            f"names:\n  0: filament\n", encoding="utf-8")
    print(f"cropped {len(stems)} images ({SIZE}x{SIZE} PNG) -> {out}")


if __name__ == "__main__":
    main()
