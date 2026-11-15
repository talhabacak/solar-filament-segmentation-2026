"""Native-resolution cache for the tiled lane: 2048^2 images, masks, foreground anchors.

The existing semantic cache stores images already downsampled to 768/1536, which
is exactly the ceiling docs/17 identifies. This one keeps native pixels and lets
the sampler cut tiles out of them.

Three arrays, and one of them is the whole point:

  images    (n_img, 2048, 2048) uint8 -- 2.96 GB. Decoding a 2048^2 JPEG costs
            about 60 ms, and a tiled epoch touches every image several times, so
            it is decoded once here rather than every epoch.
  masks     (n_entry, 2048, 256) uint8, bit-packed along width -- 605 MB instead
            of 4.8 GB. Addressed per ENTRY, not per image: the metric scores a
            prediction against each annotator's copy separately, and training on
            all copies measured +0.0051 (docs/15). Unioning them is a variant to
            test on this lane, not the default to assume.
  fg        up to FG_POINTS foreground coordinates per entry. The union of
            annotators covers 0.302% of a frame, so a uniformly placed 1056px
            tile is empty about as often as not and the model would spend its
            epochs confirming that sky is not a filament. Anchoring a share of
            crops on a real filament pixel is what makes the sampling affordable.

Usage: python src/data/build_tile_cache.py
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament import IMG_H, IMG_W
from filament.rle import decode_rle

OUT = ROOT / "data" / "tile_cache"
IMDIR = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
FULL = IMG_H
PACKED_W = (IMG_W + 7) // 8
FG_POINTS = 256


def log(*a):
    print(*a, flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    entries = (gt[["entry_id", "file_name"]].drop_duplicates()
               .sort_values("entry_id").reset_index(drop=True))
    files = sorted(entries["file_name"].unique())
    frow = {f: i for i, f in enumerate(files)}
    log(f"{len(files)} images, {len(entries)} annotator entries")

    imgs = np.lib.format.open_memmap(OUT / f"images_{FULL}.npy", mode="w+",
                                     dtype=np.uint8, shape=(len(files), FULL, FULL))
    t0 = time.time()
    paths = {p.name: p for p in IMDIR.rglob("*.jpeg")}
    for i, fn in enumerate(files):
        im = cv2.imread(str(paths[fn]), cv2.IMREAD_GRAYSCALE)
        if im.shape != (FULL, FULL):
            raise SystemExit(f"{fn}: unexpected shape {im.shape}")
        imgs[i] = im
        if (i + 1) % 200 == 0:
            log(f"  images {i+1}/{len(files)} ({time.time()-t0:.0f}s)")
    imgs.flush()
    del imgs

    msks = np.lib.format.open_memmap(OUT / f"masks_{FULL}.npy", mode="w+",
                                     dtype=np.uint8,
                                     shape=(len(entries), FULL, PACKED_W))
    fg = {}
    rng = np.random.default_rng(0)
    by_entry = dict(list(gt.groupby("entry_id")))
    t0 = time.time()
    for j, eid in enumerate(entries["entry_id"]):
        m = None
        for r in by_entry[eid]["segmentation_rle"]:
            d = decode_rle(r).astype(bool)
            m = d if m is None else (m | d)
        msks[j] = np.packbits(m, axis=-1)
        ys, xs = np.nonzero(m)
        if len(ys):
            k = rng.choice(len(ys), min(FG_POINTS, len(ys)), replace=False)
            fg[str(eid)] = np.stack([ys[k], xs[k]]).astype(np.int16)
        else:
            fg[str(eid)] = np.zeros((2, 0), np.int16)
        if (j + 1) % 300 == 0:
            log(f"  masks {j+1}/{len(entries)} ({time.time()-t0:.0f}s)")
    msks.flush()
    del msks

    np.savez_compressed(OUT / f"fg_{FULL}.npz", **fg)
    json.dump(dict(files=files,
                   entry_ids=[str(e) for e in entries["entry_id"]],
                   entry_file=[frow[f] for f in entries["file_name"]],
                   full=FULL, packed_w=PACKED_W, fg_points=FG_POINTS),
              open(OUT / f"index_{FULL}.json", "w"))

    gb = sum(p.stat().st_size for p in OUT.glob(f"*_{FULL}.*")) / 2 ** 30
    log(f"\nwrote -> {OUT}  ({gb:.2f} GB)")
    log(f"  images_{FULL}.npy   ({len(files)}, {FULL}, {FULL}) uint8")
    log(f"  masks_{FULL}.npy    ({len(entries)}, {FULL}, {PACKED_W}) uint8 bit-packed")
    log(f"  fg_{FULL}.npz       up to {FG_POINTS} anchor points per entry")


if __name__ == "__main__":
    main()
