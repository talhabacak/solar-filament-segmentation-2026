"""COCO RLE utilities matching the competition submission format exactly.

Format (confirmed from the official self-evaluation notebook):
    segmentation_rle = compressed COCO RLE "counts" as an ASCII string,
    decoded with {"size": [2048, 2048], "counts": counts} (Fortran order).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

from . import IMG_H, IMG_W


def encode_mask(mask: np.ndarray) -> str:
    """Encode a binary HxW mask to a compressed COCO RLE counts string."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    rle = mask_util.encode(np.asfortranarray(mask))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return counts


def decode_rle(counts: str | bytes, height: int = IMG_H, width: int = IMG_W) -> np.ndarray:
    """Decode a compressed COCO RLE counts string to a uint8 HxW mask."""
    if isinstance(counts, str):
        counts = counts.encode("ascii")
    return mask_util.decode({"size": [height, width], "counts": counts})


def rles_to_layers(rles: list, height: int = IMG_H, width: int = IMG_W) -> np.ndarray:
    """Decode a list of counts strings to an (n, H, W) uint8 stack.

    Mirrors the official notebook's rles_to_layers (which returns float32; we
    keep uint8 here and cast where float behaviour must be mirrored).
    """
    if not rles:
        return np.zeros((0, height, width), dtype=np.uint8)
    dicts = [
        {"size": [height, width], "counts": c.encode("ascii") if isinstance(c, str) else c}
        for c in rles
    ]
    masks = mask_util.decode(dicts)  # (H, W, n)
    return masks.transpose(2, 0, 1)


def overlap_pixel_count(masks: list[np.ndarray] | np.ndarray) -> int:
    """Number of pixels covered by 2+ masks. Must be 0 for a legal submission image."""
    if len(masks) < 2:
        return 0
    total = np.zeros((IMG_H, IMG_W), dtype=np.uint16)
    for m in masks:
        total += m.astype(np.uint16)
    return int((total > 1).sum())


def write_submission(preds: dict[str, list[np.ndarray]], path: str,
                     validate: bool = True) -> pd.DataFrame:
    """Write a submission CSV.

    preds: {image_stem: [mask, ...]}  (masks binary 2048x2048, pixel-disjoint per image)
    Rows: filament_id = f"{stem}_{k}" (k from 1), segmentation_rle = counts string.
    """
    rows = []
    for stem, masks in preds.items():
        if validate:
            ov = overlap_pixel_count(masks)
            if ov:
                raise ValueError(f"{stem}: {ov} overlapping pixels between predicted masks")
        for k, m in enumerate(masks, start=1):
            if validate:
                assert m.shape == (IMG_H, IMG_W), f"{stem}: mask shape {m.shape}"
                dec = decode_rle(encode_mask(m))
                assert np.array_equal(dec, m.astype(np.uint8)), f"{stem}: RLE round-trip failed"
            rows.append({"filament_id": f"{stem}_{k}", "segmentation_rle": encode_mask(m)})
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    df.to_csv(path, index=False)
    return df


def read_submission(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str)
