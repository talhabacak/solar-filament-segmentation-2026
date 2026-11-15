"""Pre-flight checks on a submission CSV, before a slot is spent on it.

Each check corresponds to a way this competition can silently reject or
under-score a file:

  round-trip      every RLE must decode to a 2048x2048 mask and re-encode to the
                  same string. A mismatch means the scorer sees something other
                  than what we drew.
  >= 1024 masks   pycocotools overflows 32-bit at h*w*n >= 2^32, which is exactly
                  1024 masks at 2048^2. A frame at or above that limit fails to
                  decode in bulk with "Invalid RLE mask representation" -- a hard
                  submission constraint, not a warning.
  overlap         Panoptic Quality requires pixel-disjoint instances. Two
                  overlapping masks cannot both exceed IoU 0.5 against the same
                  ground truth, so an overlap is at best wasted and at worst
                  illegal.
  empty images    an image with no prediction can only contribute FN. Worth
                  knowing the count, since the value rule says a candidate worth
                  more than PQ/2 should have been kept.
  area sanity     a median far from the training distribution's is the signature
                  of a threshold or shrink applied twice, or not at all.

Usage:
  python src/models/validate_submission.py outputs/submissions/sub008_*.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament import IMG_H, IMG_W
from filament.rle import decode_rle, encode_mask, overlap_pixel_count

TEST = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "test" / "test_images"
RLE_LIMIT = (2 ** 32) // (IMG_H * IMG_W)      # 1024 at 2048^2


def log(*a):
    print(*a, flush=True)


def main():
    path = Path(sys.argv[1])
    df = pd.read_csv(path)
    log(f"{path.name}")
    log(f"rows {len(df)} cols {list(df.columns)}")

    stems = sorted(p.stem for p in TEST.glob("*.jpeg"))
    df["stem"] = df["filament_id"].astype(str).str.rsplit("_", n=1).str[0]
    present = set(df["stem"])
    missing = [s for s in stems if s not in present]
    log(f"test images {len(stems)} | present in submission {len(present)}")
    log(f"images with NO prediction: {missing}")

    per = df.groupby("stem").size()
    log(f"instances per image: min {per.min()} median {int(per.median())} "
        f"max {per.max()}, and >= {RLE_LIMIT}? {bool((per >= RLE_LIMIT).any())}")

    bad_rt = bad_shape = 0
    areas, overlaps, ov_imgs = [], 0, 0
    for stem, g in df.groupby("stem"):
        masks = []
        for r in g["segmentation_rle"]:
            m = decode_rle(r)
            if m.shape != (IMG_H, IMG_W):
                bad_shape += 1
                continue
            if encode_mask(m.astype(np.uint8)) != r:
                bad_rt += 1
            masks.append(m.astype(bool))
            areas.append(int(m.sum()))
        ov = overlap_pixel_count(np.array(masks)) if len(masks) > 1 else 0
        if ov:
            overlaps += ov
            ov_imgs += 1
    log(f"round-trip failures {bad_rt}, wrong shape {bad_shape}")
    log(f"images with overlapping pixels: {ov_imgs} (total {overlaps} px)"
        f"  -> {'LEGAL' if overlaps == 0 else 'ILLEGAL'}")
    a = np.array(areas)
    log(f"instance area: median {int(np.median(a))}px, min {a.min()}, max {a.max()}")

    ok = (bad_rt == 0 and bad_shape == 0 and overlaps == 0
          and not bool((per >= RLE_LIMIT).any()))
    log(f"\nVERDICT: {'ready to submit' if ok else 'DO NOT SUBMIT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
