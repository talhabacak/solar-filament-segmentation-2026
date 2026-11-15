"""Adversarial equivalence test: local scorer vs the verbatim official code.

Generates randomized synthetic GT/prediction sets (in the official string format)
and asserts the local PQ equals the official notebook's PQ to <= 1e-9, including
engineered near-0.5-IoU borderline pairs, empty masks, multi-annotator copies,
and predictions for unknown images.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from filament import IMG_H, IMG_W
from filament.rle import encode_mask
from filament.pq import score

OFFICIAL_PATH = (Path(__file__).resolve().parents[1]
                 / "outputs" / "research" / "official_pq_scorer_extracted.py")


@pytest.fixture(scope="module")
def official():
    spec = importlib.util.spec_from_file_location("official_pq", OFFICIAL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def random_blob_mask(rng, region=256):
    """Small random blob placed inside a random location of the 2048 canvas."""
    m = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    r0 = int(rng.integers(0, IMG_H - region))
    c0 = int(rng.integers(0, IMG_W - region))
    n_rects = int(rng.integers(1, 4))
    for _ in range(n_rects):
        h = int(rng.integers(4, 60))
        w = int(rng.integers(4, 60))
        rr = int(rng.integers(0, region - h))
        cc = int(rng.integers(0, region - w))
        m[r0 + rr:r0 + rr + h, c0 + cc:c0 + cc + w] = 1
    return m


def perturb(rng, mask):
    """Shifted/eroded copy engineered to straddle IoU ~0.5."""
    shift = int(rng.integers(0, 12))
    axis = int(rng.integers(0, 2))
    out = np.roll(mask, shift, axis=axis)
    if rng.random() < 0.5:
        rows, cols = np.nonzero(out)
        if len(rows) > 10:
            drop = rng.random(len(rows)) < rng.uniform(0.0, 0.5)
            out = out.copy()
            out[rows[drop], cols[drop]] = 0
    return out


def gen_case(rng):
    gt_rows, pred_rows = [], []
    n_images = int(rng.integers(1, 5))
    for i in range(n_images):
        stem = f"IMG{i}"
        base_masks = [random_blob_mask(rng) for _ in range(int(rng.integers(1, 6)))]
        n_copies = int(rng.integers(1, 4))
        for c in range(n_copies):
            entry = f"{c:02d}{i}-{stem}"
            k = 0
            for bm in base_masks:
                if rng.random() < 0.85:      # annotator may skip a filament
                    k += 1
                    m = perturb(rng, bm) if rng.random() < 0.4 else bm
                    gt_rows.append((f"{entry}_{k}", encode_mask(m)))
            if k == 0:                        # entries must have >= 1 GT row
                gt_rows.append((f"{entry}_1", encode_mask(base_masks[0])))
        if rng.random() < 0.85:               # sometimes no predictions at all
            k = 0
            for bm in base_masks:
                if rng.random() < 0.8:
                    k += 1
                    m = perturb(rng, bm) if rng.random() < 0.6 else bm
                    pred_rows.append((f"{stem}_{k}", encode_mask(m)))
            if rng.random() < 0.2:            # empty predicted mask (all zeros)
                k += 1
                pred_rows.append((f"{stem}_{k}",
                                  encode_mask(np.zeros((IMG_H, IMG_W), np.uint8))))
    if rng.random() < 0.3:                    # prediction for an unknown image
        pred_rows.append(("GHOST_1", encode_mask(random_blob_mask(rng))))
    if not pred_rows:                          # scorer needs a non-empty pred df
        pred_rows.append(("GHOST_1", encode_mask(random_blob_mask(rng))))
    gt = pd.DataFrame(gt_rows, columns=["filament_id", "segmentation_rle"])
    pred = pd.DataFrame(pred_rows, columns=["filament_id", "segmentation_rle"])
    return gt, pred


N_TRIALS = 100


def test_randomized_equivalence(official):
    rng = np.random.default_rng(20260819)
    max_diff = 0.0
    for trial in range(N_TRIALS):
        gt, pred = gen_case(rng)
        overlap_df = official.get_overlap_df(gt, pred)
        pq_official = official.get_pq_score(overlap_df)
        pq_local = score(gt, pred)["pq"]
        diff = abs(pq_local - pq_official)
        max_diff = max(max_diff, diff)
        assert diff <= 1e-9, (
            f"trial {trial}: local {pq_local!r} != official {pq_official!r}")
    print(f"max |PQ diff| over {N_TRIALS} trials: {max_diff:.2e}")


def test_engineered_half_iou_equivalence(official):
    m1 = np.zeros((IMG_H, IMG_W), np.uint8); m1[0:10, 0:30] = 1        # 300 px
    m2 = np.zeros((IMG_H, IMG_W), np.uint8); m2[0:10, 10:40] = 1       # IoU = 0.5 exact
    m3 = np.zeros((IMG_H, IMG_W), np.uint8); m3[0:10, 0:31] = 1        # IoU > 0.5
    gt = pd.DataFrame([("01-X_1", encode_mask(m1))],
                      columns=["filament_id", "segmentation_rle"])
    for pred_mask, want_tp in [(m2, 0), (m3, 1)]:
        pred = pd.DataFrame([("X_1", encode_mask(pred_mask))],
                            columns=["filament_id", "segmentation_rle"])
        pq_official = official.get_pq_score(official.get_overlap_df(gt, pred))
        r = score(gt, pred)
        assert abs(r["pq"] - pq_official) <= 1e-12
        assert r["tp"] == want_tp
