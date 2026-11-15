"""Tubed-skeleton targets for Skeleton Recall Loss, from the HUMAN spine.

Design taken from Skeleton Recall Loss (Kirchhoff et al., ECCV 2024), which
reports beating clDice on five benchmarks at 2% VRAM overhead instead of 52%,
and specifically avoids the "perforated and jagged" skeletons that
differentiable soft-skeleton approximations produce. Their skeleton comes from
skeletonising the ground-truth mask; ours can come from the annotators'
own centreline, which the organizers have confirmed is permitted supervision
and which we measured to be a more stable annotation than the boundary
(cross-annotator ribbon IoU 0.708 vs mask IoU 0.629).

Their recipe: binarise, skeletonise, dilate with a radius-2 diamond kernel to
form a tube, and precompute on CPU. We substitute the human polyline for the
skeletonisation step and keep the rest.

One precondition has to hold before the substitution is safe. The loss rewards
the mask prediction for covering the tube, so any tube pixel lying OUTSIDE the
ground-truth mask actively pushes the prediction out of the label — the
opposite of what we want. This script therefore measures containment first and
intersects the tube with the mask, reporting how much that clips.

Usage: python src/data/build_tube_cache.py --size 768
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

ANN = (ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train"
       / "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
CACHE = ROOT / "data" / "semantic_cache"
FULL = 2048
DILATE_R = 2          # the paper's diamond radius, at the training resolution


def log(*a):
    print(*a, flush=True)


def unpack(arr, row, S):
    return np.unpackbits(arr[row].reshape(S, -1), axis=-1)[:, :S]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=768)
    a = ap.parse_args()
    S = a.size

    idx = json.loads((CACHE / f"index_{S}.json").read_text())
    entry_ids = idx["entry_ids"]
    row_of = {e: i for i, e in enumerate(entry_ids)}
    msks = np.load(CACHE / f"masks_{S}.npy", mmap_mode="r")

    d = json.load(open(ANN, encoding="utf-8"))
    by_entry = defaultdict(list)
    for x in d["annotations"]:
        by_entry[x["image_id"]].append(x)

    out = np.lib.format.open_memmap(CACHE / f"tubes_{S}.npy", mode="w+",
                                    dtype=np.uint8, shape=(len(entry_ids), S, S))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * DILATE_R + 1, 2 * DILATE_R + 1))
    t0 = time.time()
    raw_px = kept_px = 0
    for entry, anns in by_entry.items():
        r = row_of.get(entry)
        if r is None:
            continue
        # Rasterise and dilate at NATIVE resolution, then downsample. The kernel
        # has to be sized against the structure, not the canvas: a filament is
        # about 12 px wide at 2048 but only 4.5 px at 768, so dilating by radius
        # 2 after downsampling makes the tube as wide as the whole filament and
        # two thirds of it lands outside the label. Measured at 768: the raw
        # polyline is 94.4% inside the mask, radius-1 85.6%, radius-2 68.9% --
        # the mismatch was the kernel, not the annotation.
        big = np.zeros((FULL, FULL), np.uint8)
        for x in anns:
            p = np.asarray(x.get("spine", []), np.float32).reshape(-1, 2)
            if len(p) >= 2:
                cv2.polylines(big, [np.round(p).astype(np.int32)], False, 1,
                              thickness=1)
        if not big.any():
            continue
        big = cv2.dilate(big, kernel)
        tube = (cv2.resize(big, (S, S), interpolation=cv2.INTER_AREA) > 0
                ).astype(np.uint8)
        mask = unpack(msks, row_of[entry], S) > 0
        raw_px += int(tube.sum())
        tube = (tube > 0) & mask          # keep the tube inside the label
        kept_px += int(tube.sum())
        out[r] = tube.astype(np.uint8)
    out.flush()

    frac = kept_px / max(raw_px, 1)
    log(f"wrote tubes_{S}.npy in {time.time()-t0:.0f}s")
    log(f"  tube pixels inside the GT mask: {100*frac:.1f}% "
        f"({kept_px} of {raw_px})")
    log("  a high fraction means the human centreline sits within the label, so "
        "the recall term pulls the prediction toward the label rather than out "
        "of it; a low one would have made the substitution unsafe.")
    cover = float((np.asarray(out[:200]) > 0).mean())
    log(f"  tube covers {100*cover:.2f}% of pixels (first 200 entries)")


if __name__ == "__main__":
    main()
