"""Export competition data to Ultralytics YOLO-seg format (fold-aware).

Target strategy per physical image: 'medoid' = the annotator copy whose union
mask has the highest mean semantic IoU against the other copies (single-copy
files: that copy). Single class 0 = filament. Images are NTFS-hardlinked (no
duplication); per-fold train/val path lists + yaml are generated.

Usage: python src/data/export_yolo.py [--strategy medoid] [--out data/yolo_v1]
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pycocotools import mask as mask_util

DATA = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"
COCO_PATH = DATA / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
SCALE = 512  # medoid computation resolution


def union_mask_512(anns):
    rles = []
    for a in anns:
        rles.extend(mask_util.frPyObjects(a["segmentation"], 2048, 2048))
    m = mask_util.decode(mask_util.merge(rles))
    return m[::4, ::4]  # cheap 512 downsample fine for medoid IoU


def pick_medoid(entries_anns):
    """entries_anns: {entry_id: [ann, ...]} -> medoid entry_id."""
    ids = sorted(entries_anns)
    if len(ids) == 1:
        return ids[0]
    masks = {e: union_mask_512(entries_anns[e]).astype(bool) for e in ids}
    best, best_score = ids[0], -1.0
    for e in ids:
        ious = []
        for o in ids:
            if o == e:
                continue
            inter = (masks[e] & masks[o]).sum()
            union = (masks[e] | masks[o]).sum()
            ious.append(inter / union if union else 0.0)
        s = float(np.mean(ious))
        if s > best_score:
            best, best_score = e, s
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="medoid", choices=["medoid"])
    ap.add_argument("--out", default="data/yolo_v1")
    args = ap.parse_args()

    out = ROOT / args.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "lists").mkdir(parents=True, exist_ok=True)

    with open(COCO_PATH) as f:
        coco = json.load(f)
    anns_by_entry = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_entry[a["image_id"]].append(a)
    entries_by_file = defaultdict(dict)
    for im in coco["images"]:
        entries_by_file[im["file_name"]][im["id"]] = anns_by_entry[im["id"]]

    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"], folds["fold"]))

    chosen = {}
    for fn, entries in entries_by_file.items():
        medoid = pick_medoid(entries)
        chosen[fn] = medoid
        stem = fn.rsplit(".", 1)[0]
        # hardlink image
        dst = out / "images" / fn
        if not dst.exists():
            try:
                os.link(DATA / "train" / "train_images" / fn, dst)
            except OSError:
                import shutil
                shutil.copy2(DATA / "train" / "train_images" / fn, dst)
        # label file: one line per instance, class 0 + normalized polygon
        lines = []
        for a in entries[medoid]:
            for poly in a["segmentation"]:
                xy = np.asarray(poly, dtype=np.float64) / 2048.0
                xy = np.clip(xy, 0.0, 1.0)
                lines.append("0 " + " ".join(f"{v:.6f}" for v in xy))
        (out / "labels" / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")

    pd.Series(chosen, name="medoid_entry").rename_axis("file_name").to_frame().to_parquet(
        out / "medoid_choice.parquet")

    for k in range(5):
        tr = [str(out / "images" / fn) for fn, f in fold_of.items() if f != k]
        va = [str(out / "images" / fn) for fn, f in fold_of.items() if f == k]
        (out / "lists" / f"train_f{k}.txt").write_text("\n".join(sorted(tr)), encoding="utf-8")
        (out / "lists" / f"val_f{k}.txt").write_text("\n".join(sorted(va)), encoding="utf-8")
        (out / f"fold{k}.yaml").write_text(
            f"path: {out.as_posix()}\n"
            f"train: lists/train_f{k}.txt\n"
            f"val: lists/val_f{k}.txt\n"
            f"names:\n  0: filament\n", encoding="utf-8")

    n_multi = sum(1 for fn, e in entries_by_file.items() if len(e) > 1)
    print(f"exported {len(chosen)} images ({n_multi} multi-copy, medoid chosen), "
          f"labels + 5 fold yamls -> {out}")


if __name__ == "__main__":
    main()
