"""E155 -- the solar disk is not where the pipeline thinks it is.

Every radial quantity in this project is computed against two hard-coded numbers:

    DISK_C = 1023.5     the assumed disk centre, the frame centre
    DISK_R = 900.0      the assumed disk radius in pixels

They appear in selector.py, yolo_infer.py, cross_cache.py, sweep_oof.py,
cache_*_preds.py and a dozen analysis scripts, and they feed r_norm and abs_lat,
which are selector features, and R_MAX = 0.97, which is a hard cut applied to
every detector candidate before it can enter the pool.

A five-image spot check says both numbers are wrong. Measured radii ran 983 to
1033 pixels and centres moved by up to 30 pixels between images. If the true
radius is about 1010, then r_norm <= 0.97 keeps only distance <= 873 px, which is
0.86 of the real disk -- so the pipeline has been discarding a ring between 0.86 R
and the limb, in exactly the region where the miss rate is already 0.50 beyond
r 0.8 and 0.93 beyond r 0.9 (ledger §16).

This audit measures the disk properly on all 707 training and 180 test images and
quantifies three things:

  1. how far the assumed geometry is from the real one, per image and per
     observatory (six GONG sites, whose plate scales need not agree);
  2. what fraction of the true disk R_MAX = 0.97 actually admits, and how many
     ground-truth rows sit in the ring it cuts away;
  3. how much of the spread is between observatories (a fixable constant per site)
     versus within one (a genuine per-image measurement).

The measurement is a threshold plus a centroid-and-area circle estimate, then a
refinement that fits the limb by the median radius of the boundary pixels, which
is robust to the dark limb darkening and to filaments touching the edge.

Usage: python src/eda/disk_geometry_audit.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/eda", "src/models"):
    sys.path.insert(0, str(ROOT / p))
from filament.rle import decode_rle  # noqa: E402

OUT = ROOT / "outputs" / "eda"
TRAIN = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
TEST = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "test" / "test_images"
ASSUMED_C, ASSUMED_R, R_MAX = 1023.5, 900.0, 0.97


def log(*a):
    print(*a, flush=True)


def measure_disk(path):
    """Centre and radius of the solar disk, by the limb rather than by a threshold.

    The first attempt thresholded the image and took the largest component. It
    failed on a good fraction of the 887 frames (radii from 700 to 1427 px), because
    the six GONG sites differ in exposure and vignetting and some frames carry a
    bright surround. The limb itself is the one feature every frame shares: a strong,
    circular intensity drop. So the disk is found by casting rays from an initial
    centre, taking the steepest fall along each ray, and fitting a circle to those
    points by least squares, with the worst tenth of rays rejected -- which is enough
    to absorb a clipped edge or a bright artefact.
    """
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    img = cv2.GaussianBlur(g, (0, 0), 2.0).astype(np.float32)
    h, w = img.shape
    cx, cy = w / 2.0, h / 2.0
    n_rays = 720
    ang = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    rr = np.arange(0.55 * min(h, w) / 2, 0.55 * min(h, w), 1.0)
    for _ in range(4):
        xs = cx + np.cos(ang)[:, None] * rr[None, :]
        ys = cy + np.sin(ang)[:, None] * rr[None, :]
        ok = (xs >= 0) & (xs < w - 1) & (ys >= 0) & (ys < h - 1)
        prof = np.where(ok, img[np.clip(ys, 0, h - 1).astype(int),
                                np.clip(xs, 0, w - 1).astype(int)], np.nan)
        d = np.diff(prof, axis=1)                       # the limb is the steepest fall
        d = np.where(np.isnan(d), 0.0, d)
        j = np.argmin(d, axis=1)
        strength = -d[np.arange(n_rays), j]
        r_hit = rr[j]
        good = (strength > np.percentile(strength, 10)) & np.isfinite(r_hit)
        if good.sum() < 100:
            return None
        px = cx + np.cos(ang[good]) * r_hit[good]
        py = cy + np.sin(ang[good]) * r_hit[good]
        # algebraic circle fit: x^2+y^2 = a*x + b*y + c
        A = np.stack([px, py, np.ones_like(px)], axis=1)
        b = px ** 2 + py ** 2
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx_n, cy_n = sol[0] / 2, sol[1] / 2
        rad = float(np.sqrt(max(sol[2] + cx_n ** 2 + cy_n ** 2, 1.0)))
        if not np.isfinite(rad) or rad < 0.3 * min(h, w) / 2 or rad > 0.75 * min(h, w):
            return None
        cx, cy = float(cx_n), float(cy_n)
    resid = np.hypot(px - cx, py - cy) - rad
    clipped = float(1.0 - ((px > 2) & (px < w - 3) & (py > 2) & (py < h - 3)).mean())
    return dict(cx=cx, cy=cy, radius=rad, r_iqr=float(np.percentile(np.abs(resid), 90)),
                clipped=clipped, area_px=int(np.pi * rad ** 2))


def main():
    rows = []
    for split, d in (("train", TRAIN), ("test", TEST)):
        files = sorted(d.glob("*.jpeg"))
        for i, f in enumerate(files):
            m = measure_disk(f)
            if m is None:
                continue
            m.update(stem=f.stem, split=split, site=f.stem[-2])
            rows.append(m)
            if i % 200 == 0:
                log(f"  {split}: {i}/{len(files)}")
    g = pd.DataFrame(rows)
    g.to_parquet(OUT / "disk_geometry.parquet", index=False)
    log(f"\nmeasured {len(g)} images ({(g['split'] == 'train').sum()} train, "
        f"{(g['split'] == 'test').sum()} test)")

    log("\nassumed centre 1023.5, radius 900.0. Measured:")
    log(f"  centre x  {g['cx'].min():7.1f} .. {g['cx'].max():7.1f}   median {g['cx'].median():7.1f}")
    log(f"  centre y  {g['cy'].min():7.1f} .. {g['cy'].max():7.1f}   median {g['cy'].median():7.1f}")
    log(f"  radius    {g['radius'].min():7.1f} .. {g['radius'].max():7.1f}   median {g['radius'].median():7.1f}")
    log(f"  the assumed radius is {100 * (ASSUMED_R / g['radius'].median() - 1):+.1f}% of the real one")
    log(f"  centre offset from the frame centre: median "
        f"{np.hypot(g['cx'] - ASSUMED_C, g['cy'] - ASSUMED_C).median():.1f} px, "
        f"max {np.hypot(g['cx'] - ASSUMED_C, g['cy'] - ASSUMED_C).max():.1f} px")

    log("\nby observatory (the six GONG sites):")
    log(f"  {'site':4s} {'n':>4s} {'radius median':>14s} {'radius sd':>10s} "
        f"{'centre sd x':>12s} {'centre sd y':>12s}")
    for s, d in g.groupby("site"):
        log(f"  {s:4s} {len(d):4d} {d['radius'].median():14.1f} {d['radius'].std():10.2f} "
            f"{d['cx'].std():12.2f} {d['cy'].std():12.2f}")
    within = g.groupby("site")["radius"].std().mean()
    between = g.groupby("site")["radius"].median().std()
    log(f"\n  radius spread within a site {within:.2f} px, between sites {between:.2f} px")
    log("  (if between dominates, a constant per site fixes most of it; if within "
        "dominates, the disk has to be measured per image)")

    # what the R_MAX cut actually does in true units
    g["rmax_true"] = R_MAX * ASSUMED_R / g["radius"]
    log(f"\nR_MAX = 0.97 in assumed units keeps distance <= {R_MAX * ASSUMED_R:.0f} px, which is")
    log(f"  {g['rmax_true'].median():.3f} of the TRUE radius (range "
        f"{g['rmax_true'].min():.3f} .. {g['rmax_true'].max():.3f})")

    # how many ground-truth rows live in the ring the cut removes
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    geo = g.set_index("stem")[["cx", "cy", "radius"]].to_dict("index")
    n_tot, n_cut, n_cut_assumed = 0, 0, 0
    for stem, gg in gt.groupby("stem"):
        if stem not in geo:
            continue
        e = geo[stem]
        for rle in gg["segmentation_rle"]:
            m = decode_rle(rle).astype(bool)
            ys, xs = np.nonzero(m)
            if len(ys) == 0:
                continue
            cy, cx = ys.mean(), xs.mean()
            r_true = np.hypot(cx - e["cx"], cy - e["cy"]) / e["radius"]
            r_assumed = np.hypot(cx - ASSUMED_C, cy - ASSUMED_C) / ASSUMED_R
            n_tot += 1
            n_cut += r_true > R_MAX
            n_cut_assumed += r_assumed > R_MAX
    log(f"\nground-truth rows: {n_tot}")
    log(f"  beyond 0.97 of the TRUE radius        {n_cut:5d} ({n_cut / n_tot:.2%})")
    log(f"  beyond 0.97 in the ASSUMED units      {n_cut_assumed:5d} ({n_cut_assumed / n_tot:.2%})")
    log(f"  rows the current cut discards but should not: {n_cut_assumed - n_cut} "
        f"({(n_cut_assumed - n_cut) / n_tot:.2%} of all ground truth)")

    json.dump(dict(n=len(g), radius_median=float(g["radius"].median()),
                   radius_min=float(g["radius"].min()), radius_max=float(g["radius"].max()),
                   assumed_radius=ASSUMED_R, rmax_true_median=float(g["rmax_true"].median()),
                   within_site_sd=float(within), between_site_sd=float(between),
                   gt_rows=n_tot, cut_true=int(n_cut), cut_assumed=int(n_cut_assumed)),
              open(OUT / "disk_geometry_audit.json", "w"), indent=1, default=float)
    log(f"\nwrote {OUT / 'disk_geometry.parquet'} and the summary json")


if __name__ == "__main__":
    main()
