"""Exact local reimplementation of the official competition PQ scorer.

Semantics mirrored bit-for-bit from outputs/research/official_pq_scorer_extracted.py
(organizer's self-evaluation notebook, last run 2026-08-09):

- GT filament_id: "<annotatorEntryId>-<imageStem>_<k>"; pred filament_id "<imageStem>_<k>".
- GT grouped per annotator-image entry (text before the first '_').
- Per entry: (n_gt, n_pred) IoU matrix at 2048x2048, computed in float32 exactly like
  the official torch matmul path (intersections/areas are integers < 2**24 so the
  float32 arithmetic is exact up to the final division, which we also do in float32).
- hit = iou > 0.5 (STRICT). TP = every hit pair, its float32 IoU summed into the
  numerator. FP = prediction column with no hit; FN = GT row with no hit.
- Entries with n_pred == 0 contribute all their GT as FN.
- GLOBAL pooling over all entries, single final division:
      PQ = sum(TP IoU) / (TP + 0.5*FP + 0.5*FN);  0.0 if denominator == 0.
- Predictions for images that never occur in GT are silently ignored.

Memory note: instead of flattening full 2048x2048 layers (official), we restrict the
matmul to the union of active pixels of the entry's masks — numerically identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import IMG_H, IMG_W
from .rle import rles_to_layers

IOU_THRESHOLD = 0.5


def _entry_iou_matrix(gt_layers: np.ndarray, pred_layers: np.ndarray) -> np.ndarray:
    """Float32 IoU matrix (n_gt, n_pred), bitwise-equal to the official torch path."""
    n_gt = gt_layers.shape[0]
    n_pred = pred_layers.shape[0]
    if n_gt == 0 or n_pred == 0:
        return np.zeros((n_gt, n_pred), dtype=np.float32)

    gt_flat = gt_layers.reshape(n_gt, -1)
    pred_flat = pred_layers.reshape(n_pred, -1)

    # Restrict to active pixels: zero columns contribute nothing to the matmul.
    active = (gt_flat.any(axis=0)) | (pred_flat.any(axis=0))
    gt_act = gt_flat[:, active].astype(np.float32)
    pred_act = pred_flat[:, active].astype(np.float32)

    intersection = gt_act @ pred_act.T                              # exact ints in f32
    gt_areas = gt_act.sum(axis=1, dtype=np.float32).reshape(-1, 1)
    pred_areas = pred_act.sum(axis=1, dtype=np.float32).reshape(1, -1)
    union = gt_areas + pred_areas - intersection                     # exact ints in f32

    iou = np.where(union == 0, np.float32(0.0), intersection / union)
    return iou.astype(np.float32)


def score(gt_df: pd.DataFrame, pred_df: pd.DataFrame,
          height: int = IMG_H, width: int = IMG_W,
          return_details: bool = False, gt_cache: dict | None = None):
    """Compute the official PQ of pred_df against gt_df.

    Both dataframes need columns filament_id, segmentation_rle (official format).
    Returns a dict with pq/sq/rq/tp/fp/fn (+ per-entry DataFrame and matched pairs
    when return_details=True).

    gt_cache: optional dict memoizing decoded GT layers across repeated calls with
    the same gt_df (postprocess sweeps). Purely a speed optimization — semantics
    are untouched; do not share a cache across different gt_df contents.
    """
    gt_entry_ids = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_image_ids = pred_df["filament_id"].str.split("_", n=1).str[0]

    tp_iou_scores: list[float] = []
    fp_count = 0
    fn_count = 0
    detail_rows = []
    matched_pairs = []

    for annotator_image in gt_entry_ids.unique():
        image_id = annotator_image.split("-", maxsplit=1)[1]

        gt_sel = gt_df.loc[gt_entry_ids == annotator_image]
        pred_sel = pred_df.loc[pred_image_ids == image_id]
        n_gt = len(gt_sel)
        n_pred = len(pred_sel)

        if n_gt == 0:            # unreachable via GT-derived entries; official parity
            fp_count += n_pred
            continue
        if n_pred == 0:
            fn_count += n_gt
            if return_details:
                detail_rows.append(dict(annotator_image=annotator_image, n_gt=n_gt,
                                        n_pred=0, tp=0, fp=0, fn=n_gt, tp_iou_sum=0.0))
            continue

        if gt_cache is not None and annotator_image in gt_cache:
            packed, shape = gt_cache[annotator_image]
            gt_layers = np.unpackbits(packed)[:np.prod(shape)].reshape(shape)
        else:
            gt_layers = rles_to_layers(gt_sel["segmentation_rle"].tolist(), height, width)
            if gt_cache is not None:
                gt_cache[annotator_image] = (np.packbits(gt_layers), gt_layers.shape)
        pred_layers = rles_to_layers(pred_sel["segmentation_rle"].tolist(), height, width)

        iou = _entry_iou_matrix(gt_layers, pred_layers)
        hit = iou > IOU_THRESHOLD

        # Official extends iou[hit] in row-major order.
        entry_tp_ious = iou[hit].tolist()
        tp_iou_scores.extend(entry_tp_ious)

        entry_fp = int((hit.sum(axis=0) == 0).sum())
        entry_fn = int((hit.sum(axis=1) == 0).sum())
        fp_count += entry_fp
        fn_count += entry_fn

        if return_details:
            detail_rows.append(dict(annotator_image=annotator_image, n_gt=n_gt,
                                    n_pred=n_pred, tp=len(entry_tp_ious),
                                    fp=entry_fp, fn=entry_fn,
                                    tp_iou_sum=float(sum(entry_tp_ious))))
            gi, pj = np.nonzero(hit)
            for a, b in zip(gi, pj):
                matched_pairs.append(dict(
                    annotator_image=annotator_image,
                    gt_filament_id=gt_sel["filament_id"].iloc[int(a)],
                    pred_filament_id=pred_sel["filament_id"].iloc[int(b)],
                    iou=float(iou[a, b])))

    tp_count = len(tp_iou_scores)
    denominator = tp_count + 0.5 * fp_count + 0.5 * fn_count
    pq = sum(tp_iou_scores) / denominator if denominator > 0 else 0.0
    sq = (sum(tp_iou_scores) / tp_count) if tp_count > 0 else 0.0
    rq = (tp_count / denominator) if denominator > 0 else 0.0

    out = dict(pq=pq, sq=sq, rq=rq, tp=tp_count, fp=fp_count, fn=fn_count)
    if return_details:
        out["entries"] = pd.DataFrame(detail_rows)
        out["matches"] = pd.DataFrame(matched_pairs)
    return out
