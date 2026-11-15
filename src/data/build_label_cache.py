"""Per-entry INSTANCE label maps, the target an instance-balanced loss needs.

Our semantic cache stores a binary union per annotator entry, which is all an
image-level Dice ever needed. An instance-balanced loss needs to know which
pixels belong to which filament, so this writes a parallel label plane: 0 for
background, 1..N for the entry's filaments in annotation order.

Rasterisation follows `build_semantic_cache.py` exactly -- render at native 2048,
downsample with INTER_AREA, keep pixels above 0.5 -- so a label map and the mask
it accompanies agree pixel for pixel. Doing it any other way (nearest-downsample
of a label image, say) would put the two planes out of register and silently
teach the loss to chase pixels the mask calls background.

uint8 is enough: the busiest entry in the training set carries far fewer than 255
filaments, and the script asserts that rather than trusting it.

Overlap is possible in principle after downsampling even though one annotator's
instances are disjoint at 2048, because two thin neighbours can round into the
same 768 pixel. The later instance wins and the count is reported, because a
large number would mean the resolution is destroying the very instances the loss
is meant to protect.

Usage: python src/data/build_label_cache.py --size 768
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament.rle import rles_to_layers

CACHE = ROOT / "data" / "semantic_cache"


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

    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    by_entry = defaultdict(list)
    for r in gt.itertuples():
        by_entry[r.entry_id].append(r.segmentation_rle)
    n_max = max(len(v) for v in by_entry.values())
    assert n_max < 255, f"{n_max} instances in one entry, uint8 is not enough"
    log(f"{len(entry_ids)} entries, busiest carries {n_max} filaments")

    out = np.lib.format.open_memmap(CACHE / f"labels_{S}.npy", mode="w+",
                                    dtype=np.uint8, shape=(len(entry_ids), S, S))
    t0, overlaps, empties, kept = time.time(), 0, 0, 0
    for e, rles in by_entry.items():
        r = row_of.get(e)
        if r is None:
            continue
        lab = np.zeros((S, S), np.uint8)
        for k, rle in enumerate(rles, start=1):
            m = rles_to_layers([rle])[0].astype(np.float32)
            small = cv2.resize(m, (S, S), interpolation=cv2.INTER_AREA) > 0.5
            if not small.any():
                empties += 1
                continue
            overlaps += int(np.logical_and(small, lab > 0).sum())
            lab[small] = k
            kept += 1
        out[r] = lab
    out.flush()

    n_inst = np.array([len(v) for v in by_entry.values()])
    log(f"wrote labels_{S}.npy in {time.time()-t0:.0f}s")
    log(f"  filaments rasterised   : {kept}")
    log(f"  vanished at {S}px       : {empties} "
        f"({100*empties/max(kept+empties,1):.2f}%) -- these become background and "
        f"the loss cannot see them")
    log(f"  overlapping pixels     : {overlaps} (later instance wins)")
    log(f"  instances per entry    : mean {n_inst.mean():.1f}, max {n_inst.max()}")
    chk = np.asarray(out[:200])
    log(f"  sanity: {100*(chk > 0).mean():.2f}% of pixels labelled over the "
        f"first 200 entries")


if __name__ == "__main__":
    main()
