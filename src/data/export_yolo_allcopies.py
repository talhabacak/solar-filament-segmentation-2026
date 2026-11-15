"""Export YOLO dataset with ALL annotator copies as separate samples (1154 rows).

Rationale: the official metric scores one prediction set against EVERY annotator
copy, so the target distribution the model should learn is the annotator
distribution itself — not a single collapsed target. Training on all copies shows
the model that distribution directly (a principled label-noise augmentation),
whereas medoid/consensus collapse it.

Each COCO image entry becomes one training sample: image file hardlinked as
"<entryId>-<stem>.jpeg", label file from that entry's annotations only.
Folds still group by PHYSICAL file (no copy of an image can span folds).
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"
COCO_PATH = DATA / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
OUT = ROOT / "data" / "yolo_v3_allcopies"


def main():
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    (OUT / "lists").mkdir(parents=True, exist_ok=True)

    with open(COCO_PATH) as f:
        coco = json.load(f)
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)

    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"], folds["fold"]))

    by_fold = defaultdict(list)
    n = 0
    for im in coco["images"]:
        entry_id, fn = im["id"], im["file_name"]
        name = f"{entry_id}.jpeg"                       # entry id already has the stem
        dst = OUT / "images" / name
        if not dst.exists():
            try:
                os.link(DATA / "train" / "train_images" / fn, dst)
            except OSError:
                import shutil
                shutil.copy2(DATA / "train" / "train_images" / fn, dst)
        lines = []
        for a in anns[entry_id]:
            for poly in a["segmentation"]:
                xy = np.clip(np.asarray(poly, dtype=np.float64) / 2048.0, 0.0, 1.0)
                lines.append("0 " + " ".join(f"{v:.6f}" for v in xy))
        (OUT / "labels" / f"{entry_id}.txt").write_text("\n".join(lines), encoding="utf-8")
        by_fold[fold_of[fn]].append(str(dst))
        n += 1

    for k in range(5):
        tr = sorted(p for f, ps in by_fold.items() if f != k for p in ps)
        va = sorted(by_fold[k])
        (OUT / "lists" / f"train_f{k}.txt").write_text("\n".join(tr), encoding="utf-8")
        (OUT / "lists" / f"val_f{k}.txt").write_text("\n".join(va), encoding="utf-8")
        (OUT / f"fold{k}.yaml").write_text(
            f"path: {OUT.as_posix()}\ntrain: lists/train_f{k}.txt\nval: lists/val_f{k}.txt\n"
            f"names:\n  0: filament\n", encoding="utf-8")
    print(f"exported {n} annotator-copy samples "
          f"(fold sizes {[len(by_fold[k]) for k in range(5)]}) -> {OUT}")


if __name__ == "__main__":
    main()
