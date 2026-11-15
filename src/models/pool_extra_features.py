"""Compute the photometry and orientation features of a labelled candidate pool.

The selector reads 42 features per candidate: 32 written by build_fused_pool.py and ten computed
here by `extra_features.extra_features` (contrast against a ring, dark core, structure-tensor
coherence and texture alignment; see that module). They are computed by the same function the
submission path calls (`submit_v2.featurise_one`), on the same object: the SHRUNK mask, which is
what a labelled pool's `rle` column holds (label_pool.py). Using one function on both sides is
deliberate; an earlier float/uint8 difference between training and deployment features flipped
5.9 % of keep decisions.

This is the step that produced outputs/eda/extra_features_t020.parquet. It reads the training
images from data/MAGFiLO_1.0_Kaggle_2026/train/train_images.

Usage: python src/models/pool_extra_features.py fused_pool_all_blob5_t020
Output: outputs/eda/extra_features_<last part of the name>.parquet (e.g. extra_features_t020.parquet)
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/models"):
    sys.path.insert(0, str(ROOT / p))
from filament.rle import decode_rle  # noqa: E402
from extra_features import EXTRA, extra_features, structure_tensor  # noqa: E402

IMG = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"


def main():
    name = sys.argv[1]
    pool = pd.read_parquet(ROOT / "outputs" / "eda" / f"{name}_labelled.parquet")
    rows, t0 = [], time.time()
    for k, (stem, g) in enumerate(pool.groupby("stem", sort=False)):
        img = cv2.imread(str(IMG / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        T = structure_tensor(img)
        for cid, rle in zip(g["cand_id"], g["rle"]):
            d = extra_features(decode_rle(rle).astype(bool), img, T)
            d["cand_id"] = cid
            rows.append(d)
        if k % 100 == 0:
            print(f"  {k} images, {len(rows)} candidates ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)[["cand_id"] + list(EXTRA)]
    out = ROOT / "outputs" / "eda" / f"extra_features_{name.split('_')[-1]}.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out}: {len(df)} candidates, {len(EXTRA)} features ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
