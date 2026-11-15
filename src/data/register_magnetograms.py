"""E154c -- put the magnetogram into the H-alpha frame, and prove it landed there.

The fetch stores HMI frames in HMI's own geometry. To be usable they have to sit
in our 2048-pixel H-alpha frame, pixel for pixel. Analytically the map is fully
determined: the HMI header gives the disk centre (CRPIX), the plate scale (CDELT),
the solar radius (RSUN_OBS) and the roll (CROTA2 ~ 180 degrees), and E155 measured
our own disk at radius 904.1 +/- 2.8 pixels centred on the frame centre.

One thing the headers cannot settle is the orientation of the GONG H-alpha frames.
Ground-based full-disk images may be stored solar-north-up (the P angle removed) or
geocentric-north-up (the P angle left in), and P swings +/- 26 degrees over a year.
Getting that wrong would rotate the magnetogram against the H-alpha by up to 26
degrees and turn a physical prior into noise.

So the registration is verified rather than assumed, using a physical anchor that
needs no ephemeris: sunspots are dark in H-alpha and carry the strongest fields, so
the correlation between H-alpha darkness and log|B| peaks at the correct rotation.
The script sweeps the rotation per image, reports where the peak sits, and then
asks whether the fitted angle tracks the date -- which is the signature of an
uncorrected P angle -- or stays at zero, which means GONG has already removed it.

Outputs (only after the check passes):
  outputs/magnetograms/reg_<stem>.npy   float16 1024x1024, B_LOS in Gauss, on our
                                        disk geometry at half our frame scale

Usage:
  python src/data/register_magnetograms.py --check 40     # fit the angle, no writes
  python src/data/register_magnetograms.py --angle 0      # register everything
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/eda", "src/models"):
    sys.path.insert(0, str(ROOT / p))

MAG = ROOT / "outputs" / "magnetograms"
OUT = ROOT / "outputs" / "eda"
IMG = {"train": ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images",
       "test": ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "test" / "test_images"}
SIZE = 1024                  # half our 2048 frame; the fields are smooth at this scale
DISK_R_HA = 904.1 / 2.0      # E155, in the half-scale frame
DISK_C_HA = SIZE / 2.0
REBIN = 4
FLIP, PREFIX, INDEX = -1.0, "mag", "index.parquet"


def log(*a):
    print(*a, flush=True)


def halpha_half(stem):
    for d in IMG.values():
        f = d / f"{stem}.jpeg"
        if f.exists():
            g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            return cv2.resize(g, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
    return None


def sample_grid(row, angle_deg):
    """Where each pixel of our half-scale frame falls in the rebinned HMI array.

    Solar coordinates (X west, Y north) in units of the solar radius are read off
    our frame, rotated by the trial angle, then written into HMI pixels. CROTA2 is
    about 180 degrees, so HMI's +x is solar east and +y is solar south; that is the
    origin of both minus signs below.
    """
    u, v = np.meshgrid(np.arange(SIZE, dtype=np.float32), np.arange(SIZE, dtype=np.float32))
    X = (u - DISK_C_HA) / DISK_R_HA
    Y = (DISK_C_HA - v) / DISK_R_HA
    t = np.deg2rad(angle_deg)
    Xr = X * np.cos(t) - Y * np.sin(t)
    Yr = X * np.sin(t) + Y * np.cos(t)
    r_hmi = float(row["rsun_obs"]) / float(row["cdelt1"]) / REBIN
    cx = (float(row["crpix1"]) - 2.5) / REBIN          # FITS 1-based -> rebinned 0-based
    cy = (float(row["crpix2"]) - 2.5) / REBIN
    col = cx + FLIP * Xr * r_hmi
    rowi = cy + FLIP * Yr * r_hmi
    return col.astype(np.float32), rowi.astype(np.float32), np.hypot(X, Y)


def register(stem, row, angle_deg):
    p = MAG / f"{PREFIX}_{stem}.npy"
    if not p.exists():
        return None, None
    m = np.load(p).astype(np.float32)
    col, rowi, rr = sample_grid(row, angle_deg)
    # HMI stores the bottom FITS row first, which is how numpy reads it, so the row
    # index computed above indexes the array directly
    out = cv2.remap(m, col, rowi, interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    out[rr > 1.0] = np.nan
    return out, rr


def score_alignment(ha, mag, rr):
    """Sunspots are dark in H-alpha and strong in |B|; their agreement peaks at the
    right rotation. Measured on the inner disk, where foreshortening is mild."""
    m = (rr < 0.85) & np.isfinite(mag) & (ha > 0)
    if m.sum() < 5000:
        return np.nan
    a = ha[m]
    b = np.log1p(np.abs(mag[m]))
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    return float(-(a * b).mean())          # darkness against field: expect positive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=int, default=0, help="fit the angle on N images and stop")
    ap.add_argument("--angle", type=float, default=None, help="apply this rotation and write")
    ap.add_argument("--span", type=float, default=30.0)
    ap.add_argument("--step", type=float, default=2.0)
    # AIA frames are stored north-up (CROTA2 ~ 0): no 180-degree flip, so the two minus
    # signs in sample_grid become plus. Everything else (centre, scale) is identical.
    ap.add_argument("--series", default="hmi", choices=["hmi", "aia304", "aia193"])
    a_ = ap.parse_args()
    global FLIP, PREFIX, INDEX
    FLIP = -1.0 if a_.series == "hmi" else 1.0
    PREFIX = {"hmi": "mag", "aia304": "a304", "aia193": "a193"}[a_.series]
    INDEX = "index.parquet" if a_.series == "hmi" else f"index_{a_.series}.parquet"

    idx = pd.read_parquet(MAG / INDEX)
    idx = idx[idx["status"] == "ok"].set_index("stem")
    have = [p.stem[len(PREFIX) + 1:] for p in MAG.glob(f"{PREFIX}_*.npy")]
    stems = [s for s in have if s in idx.index]
    log(f"{len(stems)} magnetograms with geometry available")

    if a_.check:
        angles = np.arange(-a_.span, a_.span + 1e-6, a_.step)
        rows = []
        for stem in stems[:: max(len(stems) // a_.check, 1)][: a_.check]:
            ha = halpha_half(stem)
            if ha is None:
                continue
            sc = []
            for ang in angles:
                mag, rr = register(stem, idx.loc[stem], float(ang))
                sc.append(np.nan if mag is None else score_alignment(ha, mag, rr))
            sc = np.array(sc, dtype=np.float64)
            if not np.isfinite(sc).any():
                continue
            best = float(angles[int(np.nanargmax(sc))])
            rows.append(dict(stem=stem, best_angle=best, peak=float(np.nanmax(sc)),
                             at_zero=float(sc[int(np.argmin(np.abs(angles)))]),
                             doy=int(pd.Timestamp(f"{stem[:4]}-{stem[4:6]}-{stem[6:8]}").dayofyear),
                             site=stem[-2]))
            log(f"  {stem}  best {best:+5.1f} deg  peak {rows[-1]['peak']:+.3f}  "
                f"at 0 deg {rows[-1]['at_zero']:+.3f}")
        d = pd.DataFrame(rows)
        if not len(d):
            log("no images could be scored"); return
        log(f"\nfitted rotation over {len(d)} images: median {d['best_angle'].median():+.1f} deg, "
            f"sd {d['best_angle'].std():.1f}, range {d['best_angle'].min():+.0f} .. "
            f"{d['best_angle'].max():+.0f}")
        log(f"  correlation of the H-alpha darkness with log|B| at the fitted angle: "
            f"{d['peak'].mean():+.3f}; at zero rotation: {d['at_zero'].mean():+.3f}")
        # an uncorrected P angle would swing about +/-26 degrees with a one-year period
        ph = 2 * np.pi * d["doy"] / 365.25
        des = np.stack([np.ones(len(d)), np.sin(ph), np.cos(ph)], axis=1)
        coef, *_ = np.linalg.lstsq(des, d["best_angle"].to_numpy(), rcond=None)
        amp = float(np.hypot(coef[1], coef[2]))
        log(f"  annual component of the fitted angle: amplitude {amp:.1f} deg "
            f"(an uncorrected P angle would give about 26)")
        log("  -> " + ("GONG frames appear P-corrected; a single angle registers them"
                       if amp < 6 else
                       "the P angle is NOT removed; it has to be computed per date"))
        d.to_parquet(OUT / "magnetogram_alignment.parquet", index=False)
        json.dump(dict(n=len(d), median_angle=float(d["best_angle"].median()),
                       sd=float(d["best_angle"].std()), annual_amplitude=amp,
                       corr_at_best=float(d["peak"].mean()), corr_at_zero=float(d["at_zero"].mean())),
                  open(OUT / "magnetogram_alignment.json", "w"), indent=1)
        log(f"\nwrote {OUT / 'magnetogram_alignment.json'}")
        return

    if a_.angle is None:
        log("pass --check N first, then --angle with what it found"); return
    n = 0
    for stem in stems:
        dest = MAG / (f"reg_{stem}.npy" if PREFIX == "mag" else f"reg{PREFIX}_{stem}.npy")
        if dest.exists():
            continue
        mag, _ = register(stem, idx.loc[stem], a_.angle)
        if mag is None:
            continue
        np.save(dest, mag.astype(np.float16))
        n += 1
        if n % 100 == 0:
            log(f"  registered {n}")
    log(f"registered {n} frames at {a_.angle:+.1f} deg into {MAG}")


if __name__ == "__main__":
    main()
