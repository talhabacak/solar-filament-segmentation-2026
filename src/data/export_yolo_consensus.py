"""Export YOLO-seg labels with CONSENSUS targets from cross-annotator clusters.

Modes (multi-copy files only; 411 single-copy files keep their original set):
  majority : keep clusters with support >= 2; mask = pixelwise mean of per-copy
             union >= 0.5 (matches the ceiling-analysis consensus construction)
  union    : keep ALL clusters (support >= 1), same consensus mask (support-1
             clusters reduce to the single member's mask)

Masks are converted to YOLO polygons via the largest external contour
(MAGFiLO GT rule: one piece, no holes).

Usage: python src/data/export_yolo_consensus.py --mode majority --out data/yolo_v2_majority
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament import IMG_H, IMG_W
from filament.rle import rles_to_layers

DATA = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"


def mask_to_yolo_polys(mask):
    """Largest external contour -> normalized flat polygon (list of floats)."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_TC89_L1)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 4 or len(c) < 3:
        return None
    pts = c.reshape(-1, 2).astype(np.float64) / 2048.0
    return np.clip(pts, 0.0, 1.0).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="majority", choices=["majority", "union"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = ROOT / (a.out or f"data/yolo_v2_{a.mode}")
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "lists").mkdir(parents=True, exist_ok=True)

    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt_by_ann = gt.set_index("ann_id")
    clusters = pd.read_parquet(ROOT / "outputs" / "eda" / "annotator_clusters.parquet")
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"], folds["fold"]))

    copies = gt.groupby("file_name")["entry_id"].nunique()
    multi_files = set(copies[copies >= 2].index)

    n_inst = {"single": 0, "consensus": 0}
    for fn in sorted(fold_of):
        stem = fn.rsplit(".", 1)[0]
        dst = out / "images" / fn
        if not dst.exists():
            try:
                os.link(DATA / "train" / "train_images" / fn, dst)
            except OSError:
                import shutil
                shutil.copy2(DATA / "train" / "train_images" / fn, dst)

        lines = []
        if fn not in multi_files:
            sub = gt[gt["file_name"] == fn]
            for r in sub.itertuples():
                seg = None  # original polygon: rasterize->contour keeps parity with consensus path
                m = rles_to_layers([r.segmentation_rle])[0]
                seg = mask_to_yolo_polys(m)
                if seg is not None:
                    lines.append("0 " + " ".join(f"{v:.6f}" for v in seg))
                    n_inst["single"] += 1
        else:
            cl = clusters[clusters["file_name"] == fn]
            if a.mode == "majority":
                cl = cl[cl["support"] >= 2]
            for r in cl.itertuples():
                ann_ids = r.member_ann_ids.split(",")
                per_copy = defaultdict(lambda: np.zeros((IMG_H, IMG_W), dtype=bool))
                for aid in ann_ids:
                    row = gt_by_ann.loc[aid]
                    per_copy[row["entry_id"]] |= rles_to_layers([row["segmentation_rle"]])[0].astype(bool)
                acc = np.zeros((IMG_H, IMG_W), dtype=np.float32)
                for m in per_copy.values():
                    acc += m
                cons = (acc / len(per_copy)) >= 0.5
                if not cons.any():
                    cons = acc > 0
                seg = mask_to_yolo_polys(cons)
                if seg is not None:
                    lines.append("0 " + " ".join(f"{v:.6f}" for v in seg))
                    n_inst["consensus"] += 1
        (out / "labels" / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")

    for k in range(5):
        tr = [str(out / "images" / fn) for fn, f in fold_of.items() if f != k]
        va = [str(out / "images" / fn) for fn, f in fold_of.items() if f == k]
        (out / "lists" / f"train_f{k}.txt").write_text("\n".join(sorted(tr)), encoding="utf-8")
        (out / "lists" / f"val_f{k}.txt").write_text("\n".join(sorted(va)), encoding="utf-8")
        (out / f"fold{k}.yaml").write_text(
            f"path: {out.as_posix()}\ntrain: lists/train_f{k}.txt\nval: lists/val_f{k}.txt\n"
            f"names:\n  0: filament\n", encoding="utf-8")

    print(f"mode={a.mode}: {n_inst['single']} single-copy + {n_inst['consensus']} consensus "
          f"instances -> {out}")


if __name__ == "__main__":
    main()
