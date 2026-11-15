"""Rasterise the manually-annotated spines into a per-entry cache.

Organizer-sanctioned supervision (discussion 736543: spine, bbox, area and
category_id may all be used; only external ground truth is forbidden), and the
precondition measured: annotators agree on the centreline substantially better
than on the boundary (ribbon IoU 0.708 vs mask IoU 0.629, same unit).

Target design. A hard 1-px polyline is a degenerate target for a convolutional
head -- almost all pixels are background and a one-pixel miss costs everything.
Instead each spine is drawn at SPINE_W px and then distance-decayed, so the
target is a soft ridge: 1.0 on the centreline, falling smoothly to 0 by
DECAY_R px. The head therefore learns "how central am I", which is exactly the
width-invariant quantity annotators agree on, rather than a brittle skeleton.

Layout mirrors the semantic cache so the training loader can address it with
the same entry index: spines_{S}.npy, uint8, one HxW plane per annotator entry
in the same order as index_{S}.json's entry_ids.

Usage: python src/data/build_spine_cache.py --size 768
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
SPINE_W = 3          # px at native resolution before decay
DECAY_R = 12.0       # px at native resolution; target reaches 0 here


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=768)
    a = ap.parse_args()
    S = a.size

    idx = json.loads((CACHE / f"index_{S}.json").read_text())
    entry_ids = idx["entry_ids"]
    row_of = {e: i for i, e in enumerate(entry_ids)}
    log(f"{len(entry_ids)} entries in the size-{S} cache")

    d = json.load(open(ANN, encoding="utf-8"))
    by_entry = defaultdict(list)
    for x in d["annotations"]:
        by_entry[x["image_id"]].append(x)

    out = np.lib.format.open_memmap(CACHE / f"spines_{S}.npy", mode="w+",
                                    dtype=np.uint8, shape=(len(entry_ids), S, S))
    t0 = time.time()
    scale = S / FULL
    missing = 0
    for entry, anns in by_entry.items():
        # the cache keys entries by the annotator-image id; every entry in the
        # annotation file must exist there or the ordering assumption is broken
        r = row_of.get(entry)
        if r is None:
            missing += 1
            continue
        canvas = np.zeros((S, S), np.uint8)
        for x in anns:
            p = np.asarray(x.get("spine", []), np.float32).reshape(-1, 2)
            if len(p) < 2:
                continue
            cv2.polylines(canvas, [np.round(p * scale).astype(np.int32)],
                          False, 255, thickness=max(int(round(SPINE_W * scale)), 1))
        if canvas.any():
            # distance decay: 1 on the line, 0 at DECAY_R (scaled)
            dist = cv2.distanceTransform(255 - canvas, cv2.DIST_L2, 3)
            soft = np.clip(1.0 - dist / max(DECAY_R * scale, 1e-3), 0, 1)
            out[r] = (soft * 255).astype(np.uint8)
    out.flush()
    if missing:
        log(f"WARNING: {missing} annotation entries absent from the cache index")
    log(f"wrote spines_{S}.npy ({out.shape}) in {time.time()-t0:.0f}s")
    nz = float((np.asarray(out[:64]) > 0).mean())
    log(f"  sanity: {100*nz:.2f}% of pixels carry spine signal in the first 64 entries")


if __name__ == "__main__":
    main()
