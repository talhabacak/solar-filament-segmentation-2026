"""Bounding-box-local versions of the full-frame mask operations the fusion step repeats.

A candidate mask is a 2048x2048 array of which a few thousand pixels are set, and the fusion and
featurisation code was written against the full frame: np.nonzero over four million pixels for
a centroid, a 31x31 dilation of the whole frame for a photometric ring, a distance transform of
the whole frame for a one-pixel shrink, a stack of every candidate's full frame for an IoU
matrix. Each of these depends only on the pixels inside the mask's bounding box plus a margin,
so restricting the computation to that window gives the same numbers.

"The same numbers" is meant literally, because the selector bundle was fitted on features from
the full-frame code and the submitted CSV is the artefact of record: every function here is
held to bit-identity against the full-frame original (outputs/eda/_verify_lossless_fusion.py
rebuilds the test pool and compares all 42 features and every RLE). The rules that make that
true are stated at each function.
"""
import numpy as np


def bbox(mask):
    """(y0, y1, x0, x1), exclusive ends, of a 2-D bool mask; None when the mask is empty."""
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def nonzero_boxed(mask, box=None):
    """np.nonzero(mask), computed on the bounding-box crop.

    np.nonzero walks the array in C order, and so does the crop; adding the offsets back in
    place keeps the int64 dtype, so the arrays are equal element for element and any statistic
    taken over them (means, covariances, percentiles) sees the same sequence.
    """
    if box is None:
        box = bbox(mask)
    if box is None:
        return np.nonzero(mask)
    y0, y1, x0, x1 = box
    ys, xs = np.nonzero(mask[y0:y1, x0:x1])
    ys += y0
    xs += x0
    return ys, xs


def pad_box(box, pad, shape):
    y0, y1, x0, x1 = box
    return max(y0 - pad, 0), min(y1 + pad, shape[0]), max(x0 - pad, 0), min(x1 + pad, shape[1])


def iou_matrix_boxed(masks, boxes):
    """filament.pq._entry_iou_matrix(M, M) for a list of full-frame bool masks, pairwise on the
    overlap of their bounding boxes.

    The original forms intersection and areas as float32 matrix products of 0/1 values, all of
    which are exact integers below 2^24, then computes union = a_i + a_j - inter and iou =
    inter / union in float32. Building the same float32 arrays from integer counts and applying
    the same expressions reproduces every value, including the division's rounding.
    """
    n = len(masks)
    areas = np.zeros(n, dtype=np.float32)
    inter = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        if boxes[i] is not None:
            y0, y1, x0, x1 = boxes[i]
            areas[i] = np.count_nonzero(masks[i][y0:y1, x0:x1])
        inter[i, i] = areas[i]
    for i in range(n):
        if boxes[i] is None:
            continue
        ay0, ay1, ax0, ax1 = boxes[i]
        for j in range(i + 1, n):
            if boxes[j] is None:
                continue
            by0, by1, bx0, bx1 = boxes[j]
            y0, y1, x0, x1 = max(ay0, by0), min(ay1, by1), max(ax0, bx0), min(ax1, bx1)
            if y0 >= y1 or x0 >= x1:
                continue
            c = np.count_nonzero(masks[i][y0:y1, x0:x1] & masks[j][y0:y1, x0:x1])
            inter[i, j] = c
            inter[j, i] = c
    union = areas.reshape(-1, 1) + areas.reshape(1, -1) - inter
    return np.where(union == 0, np.float32(0.0), inter / union).astype(np.float32)


def union_box(boxes):
    bs = [b for b in boxes if b is not None]
    if not bs:
        return None
    return (min(b[0] for b in bs), max(b[1] for b in bs), min(b[2] for b in bs), max(b[3] for b in bs))
