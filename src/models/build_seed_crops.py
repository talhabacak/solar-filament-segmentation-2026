"""Crops for a seed-refiner: segment from a BOX, not from an existing mask.

Our own crop refiner failed (+0.0038 then overfitting) and the reason is visible
in hindsight: it was handed the pipeline's mask as an input channel and asked
for a small correction, so the cheapest thing it could learn was to copy the
input. A community notebook in docs/ (yolo11s detect -> crop -> U-Net, local PQ
~0.409) uses the same crop idea with one decisive difference — the seed is a
FILLED BOX. The network gets localisation and nothing else, and has to produce
the whole mask, which is a real learning signal and carries no mask bias.

Their settings, taken as-is so this is a fair reproduction rather than a
reinterpretation:

    CONTEXT   1.5      square crop side = 1.5 * max(box_w, box_h)
    MIN_CROP  96
    PAD_RATIO 0.40     the seed box is the box padded by this
    CROP      512      the crop is resized to this, so scale is normalised
    channels  image, CLAHE(2.0, 8x8), seed

Scale normalisation is the part our version lacked: a fixed 320 px window at
native resolution shows a 70 px filament and a 400 px one at completely
different apparent sizes, while this makes every filament fill the frame
similarly.

Stores the image crop and the target, plus the GT box in crop coordinates so the
seed can be re-jittered every epoch rather than baked in. CLAHE is cheap and is
computed at train time.

Usage: python src/models/build_seed_crops.py
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament import IMG_H, IMG_W

ANN = (ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" /
       "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
DATA = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
OUT = ROOT / "data" / "seed_crops"

CROP = 512
CONTEXT = 1.5
MIN_CROP = 96


def log(*a):
    print(*a, flush=True)


def square_bounds(bbox, h=IMG_H, w=IMG_W, context=CONTEXT, minimum=MIN_CROP):
    """Square window around a box, clamped into the frame (notebook's rule)."""
    x, y, bw, bh = bbox
    cx, cy = x + bw / 2, y + bh / 2
    side = max(int(math.ceil(max(bw, bh) * context)), minimum)
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + side, y0 + side
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > w:
        x0 -= x1 - w
        x1 = w
    if y1 > h:
        y0 -= y1 - h
        y1 = h
    return max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)


def ann_mask(ann, h=IMG_H, w=IMG_W):
    from pycocotools import mask as mask_utils
    seg = ann.get("segmentation")
    if seg:
        rles = mask_utils.frPyObjects(seg, h, w)
        rle = mask_utils.merge(rles) if isinstance(rles, list) else rles
        m = mask_utils.decode(rle)
        return (np.any(m, axis=2) if m.ndim == 3 else m).astype(np.uint8)
    x, y, bw, bh = map(int, ann["bbox"])
    m = np.zeros((h, w), np.uint8)
    m[y:y + bh, x:x + bw] = 1
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", type=int, default=CROP)
    ap.add_argument("--context", type=float, default=CONTEXT,
                    help="square crop side = context * max(box_w, box_h). The "
                         "notebook uses 1.5, but at 1.5 only 79%% of matched "
                         "detections have the whole true filament inside the "
                         "crop and the worst decile keeps 62%% of it, so the "
                         "model cannot see pixels it is asked to predict.")
    a = ap.parse_args()
    S = a.crop
    out = OUT if a.context == CONTEXT else Path(str(OUT) + f"_c{a.context:g}")
    out.mkdir(parents=True, exist_ok=True)

    d = json.load(open(ANN))
    by_image = {}
    for ann in d["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)
    images = {im["id"]: im for im in d["images"]}
    n = len(d["annotations"])
    log(f"{n} annotations over {len(images)} entries")

    imgs = np.lib.format.open_memmap(out / f"img_{S}.npy", mode="w+",
                                     dtype=np.uint8, shape=(n, S, S))
    tgts = np.lib.format.open_memmap(out / f"tgt_{S}.npy", mode="w+",
                                     dtype=np.uint8, shape=(n, S, S))
    meta, k, t0 = [], 0, time.time()
    cache_fn, cache_img = None, None
    for image_id, anns in by_image.items():
        fn = images[image_id]["file_name"]
        if fn != cache_fn:
            cache_img = cv2.imread(str(DATA / fn), cv2.IMREAD_GRAYSCALE)
            cache_fn = fn
        if cache_img is None:
            continue
        for ann in anns:
            x0, y0, x1, y1 = square_bounds(ann["bbox"], context=a.context)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            sub = cache_img[y0:y1, x0:x1]
            m = ann_mask(ann)[y0:y1, x0:x1]
            imgs[k] = cv2.resize(sub, (S, S), interpolation=cv2.INTER_AREA)
            tgts[k] = (cv2.resize(m, (S, S), interpolation=cv2.INTER_NEAREST) > 0)
            bx, by, bw, bh = ann["bbox"]
            sx, sy = S / (x1 - x0), S / (y1 - y0)
            meta.append(dict(idx=k, ann_id=ann["id"], image_id=image_id,
                             file_name=fn, category_id=int(ann["category_id"]),
                             x0=x0, y0=y0, x1=x1, y1=y1,
                             bx=(bx - x0) * sx, by=(by - y0) * sy,
                             bw=bw * sx, bh=bh * sy))
            k += 1
        if k % 1000 < len(anns):
            log(f"  {k}/{n} crops ({time.time()-t0:.0f}s)")
    imgs.flush()
    tgts.flush()
    pd.DataFrame(meta).to_parquet(out / f"meta_{S}.parquet", index=False)
    with open(out / f"index_{S}.json", "w") as f:
        json.dump(dict(n=k, crop=S, context=a.context, min_crop=MIN_CROP), f, indent=1)
    log(f"wrote {k} crops at {S}px context {a.context:g} -> {out} "
        f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
