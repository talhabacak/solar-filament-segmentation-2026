"""E162 -- the magnetogram as a training input: a cache aligned with the semantic cache.

E161 measured what a network could hope to get from the field at the pixel level:
magnetic inputs alone predict "some annotator marked this pixel" at AUC 0.673, and
added to the production U-Net's own field they lift it 0.884 -> 0.911. That is
the funded reading for a U-Net input channel; the two cheaper uses (selector
summaries, E160; a generation prior, E159) did not pay.

This writes, for the cache size S, a uint8 memmap `mag_{S}.npy` of shape
(n_files, 2, S, S) in the SAME file order as `images_{S}.npy` (index_{S}.json), so
the cached dataset can read it by the same row:

    channel 0   the smoothed line-of-sight field, clipped to +/-50 G and mapped to
                0..255 with 128 at zero -- the sign, i.e. the polarity, is the
                information; magnitude beyond 50 G is saturated on purpose because
                filaments live in weak-field corridors and sunspots would dominate
                the scale
    channel 1   log(1 + distance to the nearest PIL) mapped 0..255 over 0..log(201)

and `mag_{S}_has.npy`, a uint8 vector with 1 where a registered magnetogram
existed. Files without one get the neutral values (128, 255) and flag 0, which is
exactly what the trainer's channel dropout produces, so a missing frame at test
time is in-distribution rather than an outlier.

Usage: python src/data/build_magnetic_cache.py --size 1536
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
from magnetogram_features import load_registered, pil_maps  # noqa: E402

CACHE = ROOT / "data" / "semantic_cache"
B_CLIP = 50.0
D_MAX = np.log1p(200.0)
NEUTRAL = (128, 255)


def encode(B):
    m = pil_maps(B)
    bs = np.clip(m["Bs"] / B_CLIP, -1, 1)
    c0 = np.rint(128 + 127 * bs).astype(np.uint8)
    c1 = np.rint(255 * np.clip(np.log1p(m["dist"]) / D_MAX, 0, 1)).astype(np.uint8)
    c0[~m["on"]] = NEUTRAL[0]
    c1[~m["on"]] = NEUTRAL[1]
    return c0, c1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=1536)
    a = ap.parse_args()
    S = a.size
    idx = json.loads((CACHE / f"index_{S}.json").read_text())
    files = idx["files"]
    out = np.lib.format.open_memmap(CACHE / f"mag_{S}.npy", mode="w+", dtype=np.uint8,
                                    shape=(len(files), 2, S, S))
    has = np.zeros(len(files), np.uint8)
    t0 = time.time()
    for i, fn in enumerate(files):
        stem = fn.replace(".jpeg", "")
        B = load_registered(stem)
        if B is None:
            out[i, 0] = NEUTRAL[0]
            out[i, 1] = NEUTRAL[1]
            continue
        c0, c1 = encode(B)
        out[i, 0] = cv2.resize(c0, (S, S), interpolation=cv2.INTER_LINEAR)
        out[i, 1] = cv2.resize(c1, (S, S), interpolation=cv2.INTER_LINEAR)
        has[i] = 1
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(files)} ({time.time() - t0:.0f}s)", flush=True)
    out.flush()
    np.save(CACHE / f"mag_{S}_has.npy", has)
    print(f"mag_{S}.npy: {len(files)} files, {int(has.sum())} with a magnetogram "
          f"({out.nbytes / 1e9:.2f} GB, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
