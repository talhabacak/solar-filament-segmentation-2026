"""E159b -- why are filaments 50 px from the nearest PIL? Two suspects, both cheap to test.

E159 (2 G floor) put ground-truth filaments a median 49.8 half-scale px from the
nearest polarity inversion line, against 66.1 for random same-size patches. That is
the right sign but the wrong magnitude: 50 px here is ~100 full-frame px, about
75 Mm, and a filament sits on its inversion line to within a few Mm.

  suspect 1  the PIL map is still too sparse or too smoothed: an inversion line
             found on a sigma-4 field can wander far from the true one where the
             field is weak, and a 2 G floor may still cut most quiet-Sun crossings
  suspect 2  a TRANSLATION error in the registration. E154c verified the rotation
             (a sharp peak at 0 deg) but never swept a shift, and the H-alpha disk
             centre is only known to ~2 px on the frames that fitted cleanly

Both are read from one experiment: the fraction of GROUND-TRUTH mask pixels that lie
within k px of a PIL, as a function of (a) an applied shift of the magnetogram in x
and y, and (b) the smoothing sigma and floor. If the fraction peaks off (0,0) the
registration is offset; if it stays flat and low at every shift the PIL map itself
is the problem.

Usage: python src/eda/pil_diagnostics.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/eda", "src/models"):
    sys.path.insert(0, str(ROOT / p))
from filament.rle import decode_rle  # noqa: E402
import magnetogram_features as mf  # noqa: E402

OUT = ROOT / "outputs" / "eda"
MAG = ROOT / "outputs" / "magnetograms"
K = 6


def log(*a):
    print(*a, flush=True)


def gt_half_masks(stem, gt):
    import cv2
    ms = []
    for r in gt.loc[gt["stem"] == stem, "segmentation_rle"]:
        m = decode_rle(r).astype(np.uint8)
        ms.append(cv2.resize(m, (1024, 1024), interpolation=cv2.INTER_AREA) > 0.25)
    return ms


def frac_near(masks, dist, k=K):
    tot = sum(int(m.sum()) for m in masks)
    near = sum(int((dist[m] <= k).sum()) for m in masks)
    return near / max(tot, 1)


def main():
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    have = sorted({p.stem.replace("reg_", "") for p in MAG.glob("reg_*.npy")})
    stems = have[:: max(len(have) // 30, 1)][:30]
    log(f"{len(stems)} images")

    # (a) PIL map density and the shift sweep at the current settings
    shifts = [-40, -20, -10, -5, 0, 5, 10, 20, 40]
    acc = np.zeros((len(shifts), len(shifts)))
    dens = []
    for stem in stems:
        B = mf.load_registered(stem)
        maps = mf.pil_maps(B)
        dens.append(maps["pil"][maps["on"]].mean())
        masks = gt_half_masks(stem, gt)
        if not masks:
            continue
        for i, dy in enumerate(shifts):
            for j, dx in enumerate(shifts):
                d = np.roll(np.roll(maps["dist"], dy, axis=0), dx, axis=1)
                acc[i, j] += frac_near(masks, d)
    acc /= len(stems)
    log(f"\nPIL pixels as a fraction of the disk, current settings (sigma {mf.SIGMA}, floor {mf.B_FLOOR} G): "
        f"{np.mean(dens):.4f}")
    log(f"fraction of ground-truth pixels within {K} px of a PIL, magnetogram shifted by (dy, dx):")
    log("      dx " + " ".join(f"{dx:+5d}" for dx in shifts))
    for i, dy in enumerate(shifts):
        log(f"dy {dy:+4d}   " + " ".join(f"{acc[i, j]:5.3f}" for j in range(len(shifts))))
    i0, j0 = np.unravel_index(np.argmax(acc), acc.shape)
    log(f"peak at dy {shifts[i0]:+d}, dx {shifts[j0]:+d} ({acc[i0, j0]:.3f}); at (0,0) {acc[len(shifts)//2, len(shifts)//2]:.3f}")

    # (b) the PIL definition itself, at zero shift
    log(f"\nfraction of ground-truth pixels within {K} px of a PIL vs the PIL definition (no shift):")
    log(f"  {'sigma':>6s} {'floor G':>8s} {'PIL density':>12s} {'GT near':>8s} {'random near':>12s}")
    rng = np.random.default_rng(0)
    for sigma in (2.0, 4.0, 8.0):
        for floor in (0.5, 1.0, 2.0, 5.0):
            mf.SIGMA, mf.B_FLOOR = sigma, floor
            g_near, r_near, d_all = [], [], []
            for stem in stems[:12]:
                B = mf.load_registered(stem)
                maps = mf.pil_maps(B)
                d_all.append(maps["pil"][maps["on"]].mean())
                masks = gt_half_masks(stem, gt)
                if not masks:
                    continue
                g_near.append(frac_near(masks, maps["dist"]))
                # random control: the same masks rolled to a random on-disk offset
                rm = [np.roll(np.roll(m, int(rng.integers(-250, 250)), 0), int(rng.integers(-250, 250)), 1) & maps["on"]
                      for m in masks]
                r_near.append(frac_near(rm, maps["dist"]))
            log(f"  {sigma:6.1f} {floor:8.1f} {np.mean(d_all):12.4f} {np.mean(g_near):8.3f} {np.mean(r_near):12.3f}")
    mf.SIGMA, mf.B_FLOOR = 4.0, 2.0


if __name__ == "__main__":
    main()
