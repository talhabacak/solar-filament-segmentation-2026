"""Edge-case tests for the local PQ scorer (official semantics)."""

import numpy as np
import pandas as pd
import pytest

from filament import IMG_H, IMG_W
from filament.rle import encode_mask, decode_rle, overlap_pixel_count
from filament.pq import score


def make_mask(boxes):
    """boxes: list of (r0, r1, c0, c1) half-open rectangles."""
    m = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    for r0, r1, c0, c1 in boxes:
        m[r0:r1, c0:c1] = 1
    return m


def df(rows):
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


A = make_mask([(0, 10, 0, 30)])          # 300 px
A_HALF = make_mask([(0, 10, 10, 40)])    # 300 px, overlap 200 -> IoU exactly 0.5
A_JUST_ABOVE = make_mask([(0, 10, 0, 31)])  # 310px, inter 300, union 310 -> IoU ~0.968
B = make_mask([(100, 120, 100, 160)])    # separate filament
EMPTY = np.zeros((IMG_H, IMG_W), dtype=np.uint8)


def test_exact_match_pq_1():
    gt = df([("01-IMG_1", encode_mask(A)), ("01-IMG_2", encode_mask(B))])
    pred = df([("IMG_1", encode_mask(A)), ("IMG_2", encode_mask(B))])
    r = score(gt, pred)
    assert r["pq"] == 1.0 and r["tp"] == 2 and r["fp"] == 0 and r["fn"] == 0


def test_iou_exactly_half_is_not_tp():
    # inter=200, union=400 -> IoU = 0.5 exactly; strict > means NOT a TP
    gt = df([("01-IMG_1", encode_mask(A))])
    pred = df([("IMG_1", encode_mask(A_HALF))])
    r = score(gt, pred)
    assert r["tp"] == 0 and r["fp"] == 1 and r["fn"] == 1
    assert r["pq"] == 0.0


def test_iou_just_above_half_is_tp():
    gt = df([("01-IMG_1", encode_mask(A))])
    pred = df([("IMG_1", encode_mask(A_JUST_ABOVE))])
    r = score(gt, pred)
    assert r["tp"] == 1 and r["fp"] == 0 and r["fn"] == 0
    assert r["pq"] == pytest.approx(300 / 310, abs=1e-6)


def test_empty_pred_mask_is_fp():
    gt = df([("01-IMG_1", encode_mask(A))])
    pred = df([("IMG_1", encode_mask(A)), ("IMG_2", encode_mask(EMPTY))])
    r = score(gt, pred)
    assert r["tp"] == 1 and r["fp"] == 1 and r["fn"] == 0
    assert r["pq"] == pytest.approx(1.0 / 1.5)


def test_zero_pred_entry_all_fn():
    gt = df([("01-IMG_1", encode_mask(A)), ("01-IMG_2", encode_mask(B))])
    pred = df([("OTHER_1", encode_mask(A))])  # OTHER not in GT -> ignored
    r = score(gt, pred)
    assert r["tp"] == 0 and r["fp"] == 0 and r["fn"] == 2
    assert r["pq"] == 0.0


def test_multi_annotator_global_pooling():
    # copy 01 has {A}; copy 02 has {A, B}; prediction = {A}
    # entry 01: TP(1.0); entry 02: TP(1.0) + FN(B)
    # PQ = 2.0 / (2 + 0.5*0 + 0.5*1) = 0.8
    gt = df([
        ("01-IMG_1", encode_mask(A)),
        ("02-IMG_1", encode_mask(A)),
        ("02-IMG_2", encode_mask(B)),
    ])
    pred = df([("IMG_1", encode_mask(A))])
    r = score(gt, pred)
    assert r["tp"] == 2 and r["fn"] == 1 and r["fp"] == 0
    assert r["pq"] == pytest.approx(2.0 / 2.5)


def test_rle_roundtrip_random_and_edges():
    rng = np.random.default_rng(0)
    for _ in range(5):
        m = (rng.random((IMG_H, IMG_W)) > 0.999).astype(np.uint8)
        assert np.array_equal(decode_rle(encode_mask(m)), m)
    assert np.array_equal(decode_rle(encode_mask(EMPTY)), EMPTY)
    full = np.ones((IMG_H, IMG_W), dtype=np.uint8)
    assert np.array_equal(decode_rle(encode_mask(full)), full)


def test_overlap_validator_catches_one_pixel():
    m1 = make_mask([(0, 10, 0, 10)])
    m2 = make_mask([(9, 20, 9, 20)])  # shares pixel (9,9)
    assert overlap_pixel_count([m1, m2]) == 1
    assert overlap_pixel_count([m1, make_mask([(50, 60, 50, 60)])]) == 0
