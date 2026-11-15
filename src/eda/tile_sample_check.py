"""Do the tiled training samples actually look right, and is the mask still on the filament?

Before a multi-hour run, two things have to be true and neither raises an
exception when it is not:

  alignment   the single affine warps image and mask with different interpolations
              (linear vs nearest) and a shared matrix. If the matrix were applied
              inconsistently, training would proceed and learn nothing. The
              numeric test is the one that worked on the whole-image lane:
              filaments are DARK against their LOCAL surroundings -- about -15 DN
              against a dilated ring. Against a whole-frame mean they come out
              brighter, because limb darkening drags the mean down while
              filaments sit at mid-latitudes, so the ring is the reference.

  sampling    FG_PROB of the windows are supposed to be anchored on a filament.
              If the anchoring silently failed, most tiles would be empty sky and
              the loss would still fall -- to the all-background solution.

Also renders a sheet, because the numbers cannot show a tile that is half black
sky from falling off the frame, or a rotation that clipped a filament in two.

Usage: python src/eda/tile_sample_check.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

import json
import pandas as pd
from train_tiled import TiledEntries
from tiling import FULL, TILE, OFFSET, CROP

CACHE = ROOT / "data" / "tile_cache"
OUT = ROOT / "outputs" / "gallery"
KER = np.ones((15, 15), np.uint8)
N = 240


def log(*a):
    print(*a, flush=True)


def main():
    idx = json.loads((CACHE / f"index_{FULL}.json").read_text())
    fgz = np.load(CACHE / f"fg_{FULL}.npz")
    fg = {k: fgz[k] for k in fgz.files}
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"], folds["fold"]))
    rows = [(j, ifile, eid)
            for j, (eid, ifile) in enumerate(zip(idx["entry_ids"], idx["entry_file"]))
            if fold_of[idx["files"][ifile]] != 0]
    ds = TiledEntries(rows, fg, train=True, crops_per=1)

    fg_frac, contr, empty, black = [], [], 0, []
    for i in range(N):
        x, y = ds[i % len(ds)]
        im = (x[0].numpy() * 255).astype(np.uint8)
        mk = y[0].numpy() > 0.5
        fg_frac.append(float(mk.mean()))
        black.append(float((im < 5).mean()))
        if mk.sum() < 100:
            empty += 1
            continue
        ring = (cv2.dilate(mk.astype(np.uint8), KER, iterations=2) > 0) & ~mk & (im > 8)
        if ring.sum() < 100:
            continue
        contr.append(float(im[mk].mean()) - float(im[ring].mean()))

    log(f"{N} training samples drawn\n")
    log(f"ALIGNMENT  filament minus local ring: {np.mean(contr):+.2f} DN "
        f"(sd {np.std(contr):.2f}, n={len(contr)})")
    log(f"           whole-image lane measured -15.7; a value near 0 would mean "
        f"the mask is no longer on the filament")
    log(f"\nSAMPLING   tiles with < 100 filament px: {empty}/{N} "
        f"({100*empty/N:.0f}%)")
    log(f"           mean foreground fraction per tile: {100*np.mean(fg_frac):.3f}%")
    log(f"           frame-wide foreground rate is 0.302%, so anchoring should "
        f"lift this well above it")
    log(f"\nFRAMING    mean share of near-black pixels per tile: "
        f"{100*np.mean(black):.1f}%  (max {100*np.max(black):.1f}%)")
    log(f"           high values mean windows are falling off the disk")

    tiles = []
    for r in range(3):
        row = []
        for c in range(4):
            x, y = ds[(r * 4 + c) * 7 % len(ds)]
            im = (x[0].numpy() * 255).astype(np.uint8)
            mk = (y[0].numpy() > 0.5).astype(np.uint8)
            small = cv2.resize(im, (320, 320), interpolation=cv2.INTER_AREA)
            small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
            ms = cv2.resize(mk, (320, 320), interpolation=cv2.INTER_NEAREST)
            cnt, _ = cv2.findContours(ms, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(small, cnt, -1, (0, 200, 255), 1)
            cv2.putText(small, f"fg {100*mk.mean():.2f}%", (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
            row.append(small)
        tiles.append(np.hstack(row))
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "tile_samples.png"
    cv2.imwrite(str(p), np.vstack(tiles))
    log(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
