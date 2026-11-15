"""Soft pseudo-labels for the 180 test images, from our own ensemble -- the
transductive lane.

Legality, settled earlier and recorded in docs/01: the host prohibits external
GROUND TRUTH; the test IMAGES are competition data, and training on them with
our own model's soft outputs uses no label information we do not already have.
Declared in the final writeup as transductive self-training.

Why soft targets: same argument as the GONG distillation -- a threshold would
discard exactly the information that makes a noisy teacher usable (where it is
unsure). The teacher is the recipe3 ensemble's cached mean probability map over
flip TTA (test_probs_recipe3.npz), our best semantic estimate of the test set.

Output matches the gong_pseudo cache format (img_768_test.npy, tgt_768_test.npy,
meta_768_test.parquet with tag "_test"), so pretrain_gong.py's dataset machinery
can consume it unchanged.

Usage: python src/data/build_test_pseudo.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

SEM = ROOT / "outputs" / "semantic" / "test_probs_recipe3.npz"
TEST = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "test" / "test_images"
OUT = ROOT / "data" / "gong_pseudo"
S = 768


def main():
    z = np.load(SEM)
    stems = sorted(z.files)
    print(f"{len(stems)} test probability maps from the recipe3 ensemble")
    imgs = np.lib.format.open_memmap(OUT / f"img_{S}_test.npy", mode="w+",
                                     dtype=np.uint8, shape=(len(stems), S, S))
    tgts = np.lib.format.open_memmap(OUT / f"tgt_{S}_test.npy", mode="w+",
                                     dtype=np.uint8, shape=(len(stems), S, S))
    for i, stem in enumerate(stems):
        im = cv2.imread(str(TEST / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        imgs[i] = cv2.resize(im, (S, S), interpolation=cv2.INTER_AREA)
        tgts[i] = cv2.resize(z[stem], (S, S), interpolation=cv2.INTER_AREA)
    imgs.flush(); tgts.flush()
    pd.DataFrame(dict(idx=range(len(stems)), stem=stems,
                      is_test_frame=1)).to_parquet(OUT / f"meta_{S}_test.parquet")
    print(f"wrote img/tgt/meta _test caches under {OUT}")


if __name__ == "__main__":
    main()
