"""Pre-render the semantic lane's inputs so training stops paying for them.

Per sample the live dataset was decoding a 2048x2048 JPEG, rasterising ~7 RLE
masks at 2048x2048 (4 MB each) and summing them, then warping and resizing —
about 100 ms of CPU and ~30 MB of churn per sample, times 4 workers. This writes
that work to disk once:

  images_{S}.npy   uint8 memmap (n_files, S, S)      resized full-disk images
  masks_{S}.npy    uint8 memmap (n_entries, packed)  bit-packed union of ALL instances
  masks_lr_{S}.npy uint8 memmap (n_entries, packed)  bit-packed union of Left/Right only
  index.json       file/entry order + shapes

The Left/Right mask exists so training can down-weight Unidentifiable-only
pixels. Measured reasons to treat them as weaker supervision: U is 37.5% of
annotations but 61% of never-found instances, its recall is the worst (0.482 vs
0.66/0.62), its clusters are skipped by at least one annotator 64.8% of the time
(vs 37-39%), and the MAGFiLO protocol explicitly told annotators to minimise
detail on U (measured boundary complexity 3.16 vs 7.1).

Augmentation then happens at S (1536) instead of 2048, which is another 1.8x.

Usage: python src/data/build_semantic_cache.py --size 1536
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament.rle import rles_to_layers

DATA = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
OUT = ROOT / "data" / "semantic_cache"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=1536)
    a = ap.parse_args()
    S = a.size
    OUT.mkdir(parents=True, exist_ok=True)

    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    files = sorted(gt["file_name"].unique())
    entries = gt[["entry_id", "file_name"]].drop_duplicates().sort_values("entry_id")
    entry_ids = entries["entry_id"].tolist()
    by_entry = {e: g["segmentation_rle"].tolist() for e, g in gt.groupby("entry_id")}
    lr_by_entry = {e: g.loc[g["category_id"].isin([1, 2]), "segmentation_rle"].tolist()
                   for e, g in gt.groupby("entry_id")}

    img_path = OUT / f"images_{S}.npy"
    msk_path = OUT / f"masks_{S}.npy"
    imgs = np.lib.format.open_memmap(img_path, mode="w+", dtype=np.uint8,
                                     shape=(len(files), S, S))
    packed_w = (S + 7) // 8 * S
    msks = np.lib.format.open_memmap(msk_path, mode="w+", dtype=np.uint8,
                                     shape=(len(entry_ids), packed_w))
    lr_path = OUT / f"masks_lr_{S}.npy"
    msks_lr = np.lib.format.open_memmap(lr_path, mode="w+", dtype=np.uint8,
                                        shape=(len(entry_ids), packed_w))

    t0 = time.time()
    for i, fn in enumerate(files):
        img = cv2.imread(str(DATA / fn), cv2.IMREAD_GRAYSCALE)
        imgs[i] = cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA)
        if (i + 1) % 100 == 0:
            print(f"  images {i+1}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    imgs.flush()

    def render(rles):
        if not rles:
            return np.zeros((S, S), np.uint8)
        m = (rles_to_layers(rles).sum(axis=0) > 0).astype(np.float32)
        return (cv2.resize(m, (S, S), interpolation=cv2.INTER_AREA) > 0.5).astype(np.uint8)

    for j, e in enumerate(entry_ids):
        msks[j] = np.packbits(render(by_entry[e]), axis=-1).ravel()
        msks_lr[j] = np.packbits(render(lr_by_entry[e]), axis=-1).ravel()
        if (j + 1) % 100 == 0:
            print(f"  masks {j+1}/{len(entry_ids)} ({time.time()-t0:.0f}s)", flush=True)
    msks.flush()
    msks_lr.flush()

    with open(OUT / f"index_{S}.json", "w") as f:
        json.dump({"size": S, "files": files, "entry_ids": entry_ids,
                   "entry_file": dict(zip(entries["entry_id"], entries["file_name"]))}, f)
    print(f"cached {len(files)} images + {len(entry_ids)} masks at {S}px "
          f"({img_path.stat().st_size/2**30:.2f} + {msk_path.stat().st_size/2**30:.2f} GB) "
          f"in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
