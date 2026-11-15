"""Core package for the Solar Filament Segmentation Challenge 2026.

Modules:
    rle  -- COCO RLE encode/decode, disjointness validation, submission I/O
    pq   -- exact local reimplementation of the official Panoptic Quality scorer
    gt   -- ground-truth dataframe construction from the competition COCO JSON
"""

IMG_H = 2048
IMG_W = 2048
