"""Ground-truth dataframe construction from the competition COCO JSON.

Produces gt_df in the official scorer format:
    filament_id      = "<cocoImageId>_<k>"   (cocoImageId is "<annotatorEntry>-<stem>")
    segmentation_rle = compressed COCO RLE counts (ascii) of the rasterized polygon
Rasterization: pycocotools frPyObjects -> merge -> RLE at 2048x2048 (COCO convention).
"""

from __future__ import annotations

import json

import pandas as pd
from pycocotools import mask as mask_util
from tqdm import tqdm

from . import IMG_H, IMG_W


def ann_to_rle_counts(segmentation: list, height: int = IMG_H, width: int = IMG_W) -> str:
    rles = mask_util.frPyObjects(segmentation, height, width)
    rle = mask_util.merge(rles)
    counts = rle["counts"]
    return counts.decode("ascii") if isinstance(counts, bytes) else counts


def build_gt_df(coco_json_path: str, entry_ids: list | None = None,
                file_names: list | None = None, progress: bool = True) -> pd.DataFrame:
    """Build the official-format GT dataframe (optionally subset by entry or file)."""
    with open(coco_json_path) as f:
        coco = json.load(f)

    images = coco["images"]
    if entry_ids is not None:
        keep = set(entry_ids)
        images = [im for im in images if im["id"] in keep]
    if file_names is not None:
        keep_f = set(file_names)
        images = [im for im in images if im["file_name"] in keep_f]
    keep_img_ids = {im["id"] for im in images}
    file_by_id = {im["id"]: im["file_name"] for im in images}

    anns_by_img: dict[str, list] = {}
    for a in coco["annotations"]:
        if a["image_id"] in keep_img_ids:
            anns_by_img.setdefault(a["image_id"], []).append(a)

    rows = []
    iterator = anns_by_img.items()
    if progress:
        iterator = tqdm(iterator, total=len(anns_by_img), desc="rasterize GT")
    for img_id, anns in iterator:
        for k, a in enumerate(anns, start=1):
            rows.append(dict(
                filament_id=f"{img_id}_{k}",
                segmentation_rle=ann_to_rle_counts(a["segmentation"]),
                entry_id=img_id,
                file_name=file_by_id[img_id],
                category_id=a["category_id"],
                ann_id=a["id"],
                area=a["area"],
            ))
    return pd.DataFrame(rows)
