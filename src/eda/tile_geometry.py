"""Choose the tiling geometry from OUR data, rather than copying a public one.

Everything downstream depends on three numbers and each is measurable:

  disk radius        where the frame stops being sky. Nothing outside it can hold
                     a filament, so predicting there is pure FP risk.
  GT reach           the largest distance from centre at which any annotator ever
                     drew a filament. The centred crop must contain it with margin;
                     anything beyond is wasted compute.
  foreground rate    the share of pixels that are filament. This sets whether
                     uniform random crops are viable or whether sampling has to be
                     anchored on foreground -- at a fraction of a percent, a random
                     1056px window is mostly empty and the model spends its epochs
                     learning that sky is not a filament.

Then the crop and tile sizes are chosen so that:
  * the crop contains the GT reach with margin,
  * the tiles tile it exactly with a usable overlap,
  * every tile lies WHOLLY inside the crop, so no pixel a tile sees is padded or
    reflected -- reflection at a tile edge invents chromosphere that is not there,
  * the tile side is a multiple of 32, the total stride of the encoder, so no
    internal padding shifts the feature grid.

Usage: python src/eda/tile_geometry.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament import IMG_H, IMG_W
from filament.rle import decode_rle

OUT = ROOT / "outputs" / "eda"
FULL = IMG_H


def log(*a):
    print(*a, flush=True)


def main():
    import cv2
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    log(f"{len(gt)} GT instances over {gt['file_name'].nunique()} images\n")

    # ---- 1. disk radius, from the images themselves ----------------------
    imdir = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
    paths = sorted(imdir.rglob("*.jpeg"))[:40]
    radii, centres = [], []
    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        on = im > 25                       # well above the black sky, below the disk
        ys, xs = np.nonzero(on)
        cy, cx = ys.mean(), xs.mean()
        r = np.sqrt(on.sum() / np.pi)      # radius of the equivalent-area circle
        radii.append(r)
        centres.append((cy, cx))
    radii = np.array(radii)
    cy, cx = np.mean([c[0] for c in centres]), np.mean([c[1] for c in centres])
    log(f"1. DISK ({len(paths)} frames)")
    log(f"   centre  ({cy:.1f}, {cx:.1f})   nominal ({(FULL-1)/2:.1f}, {(FULL-1)/2:.1f})")
    log(f"   radius  median {np.median(radii):.1f} px  "
        f"min {radii.min():.1f}  max {radii.max():.1f}")

    # ---- 2. how far out does the ground truth actually reach? ------------
    c = (FULL - 1) / 2.0
    yy, xx = np.mgrid[0:FULL, 0:FULL].astype(np.float32)
    dist = np.sqrt((yy - c) ** 2 + (xx - c) ** 2)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(gt), min(1500, len(gt)), replace=False)
    reach, fg_px, n_img_px = [], 0, 0
    for r in gt["segmentation_rle"].iloc[idx]:
        m = decode_rle(r).astype(bool)
        if not m.any():
            continue
        reach.append(float(dist[m].max()))
        fg_px += int(m.sum())
    reach = np.array(reach)
    log(f"\n2. GT REACH ({len(reach)} instances sampled)")
    for q in (50, 90, 99, 100):
        log(f"   p{q:<3d} {np.percentile(reach, q):7.1f} px  "
            f"({np.percentile(reach, q)/np.median(radii):.3f} R)")

    # Foreground rate over the UNION of annotators. Summing `area` across rows
    # would count a filament once per annotator that drew it (1.63 entries per
    # image on average), inflating the rate by that factor -- the union is what a
    # crop actually sees.
    union_px = []
    for fn, g in list(gt.groupby("file_name"))[:60]:
        m = None
        for r in g["segmentation_rle"]:
            d = decode_rle(r).astype(bool)
            m = d if m is None else (m | d)
        union_px.append(int(m.sum()))
    fg_rate = float(np.mean(union_px) / (FULL * FULL))
    naive = float(gt.groupby("file_name")["area"].sum().mean() / (FULL * FULL))
    log(f"\n3. FOREGROUND RATE ({len(union_px)} images)")
    log(f"   union of annotators {np.mean(union_px):.0f} px "
        f"of {FULL*FULL} = {100*fg_rate:.3f}%")
    log(f"   (summing rows instead would say {100*naive:.3f}% -- it counts a "
        f"filament once per annotator)")

    # ---- 4. pick the geometry -------------------------------------------
    # The objective is the CHEAPEST configuration that satisfies the constraints,
    # not the one with the most overlap. Maximising overlap picks 3x3 tiles of
    # 1408 at step 231 -- nine forward passes and 84% redundancy, which buys
    # nothing: the overlap only has to be wide enough for the blend ramp to
    # cross-fade and for both tiles to see a straddling filament with context.
    # A filament is 10-30 px wide, so ~192 px is already generous.
    #
    # Margin: the reach is measured on TRAIN. Test filaments could sit slightly
    # further out, so the crop keeps more than the bare minimum.
    MIN_OVERLAP, MIN_MARGIN = 192, 48
    need = float(np.max(reach))
    log(f"\n4. GEOMETRY -- crop must contain {need:.0f} px of reach "
        f"(+{MIN_MARGIN} margin), overlap >= {MIN_OVERLAP}, tile %32 == 0")
    log(f"   {'crop':>6s} {'tiles':>5s} {'tile':>6s} {'step':>6s} {'overlap':>7s} "
        f"{'margin':>7s} {'passes':>7s} {'px/image':>10s}")
    best = None
    for tiles in (2, 3):
        for tile in range(768, 1409, 32):
            for crop in range(int(2 * (need + MIN_MARGIN)), FULL + 1, 16):
                step = (crop - tile) // (tiles - 1)
                overlap = tile - step
                if step <= 0 or overlap < MIN_OVERLAP:
                    continue
                if tile + step * (tiles - 1) != crop:
                    continue           # tiles must cover the crop exactly
                margin = crop / 2 - need
                if margin < MIN_MARGIN:
                    continue
                cost = tiles * tiles * tile * tile          # pixels forwarded
                row = (crop, tiles, tile, step, overlap, margin, cost)
                if best is None or cost < best[6]:
                    best = row
                log(f"   {crop:6d} {tiles:5d} {tile:6d} {step:6d} {overlap:7d} "
                    f"{margin:7.1f} {tiles*tiles:7d} {cost/1e6:9.1f}M")
    if best:
        crop, tiles, tile, step, overlap, margin, cost = best
        log(f"\n   CHOSEN (cheapest that satisfies the constraints): crop {crop}, "
            f"{tiles}x{tiles} tiles of {tile}, step {step}, overlap {overlap} px, "
            f"margin {margin:.0f} px, {cost/1e6:.1f}M px/image")
        log(f"   offset into the 2048 frame: {(FULL - crop) // 2}")
        log(f"   forward passes per image: {tiles*tiles}")
        OUT.mkdir(parents=True, exist_ok=True)
        json.dump(dict(full=FULL, crop=int(crop), tiles=int(tiles),
                       tile=int(tile), step=int(step), overlap=int(overlap),
                       offset=int((FULL - crop) // 2),
                       disk_radius=float(np.median(radii)),
                       gt_reach_max=float(need), fg_rate=fg_rate),
                  open(OUT / "tile_geometry.json", "w"), indent=1)
        log(f"\nsaved -> {OUT/'tile_geometry.json'}")


if __name__ == "__main__":
    main()
