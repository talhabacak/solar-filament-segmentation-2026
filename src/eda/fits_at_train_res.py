"""Does the FITS advantage survive the resize the model actually applies?

Two results stand so far. The original FITS separates filaments from their local
background a factor 1.13 better than the competition JPEG, and a bilateral filter
on the JPEG closes 99% of that gap -- so the deficit is lossy-compression noise,
not tonal depth. Both were measured at the 2048 label resolution.

The model never sees 2048. `train_semantic` resizes to 1536 (or 768) with
INTER_AREA, which averages every output pixel over a 1.8x1.8 (or 7x7) box. That
is itself a low-pass filter, applied to every input the network has ever been
shown. If it already removes the compression noise, then the FITS advantage, the
bilateral filter, and the whole denoising family are measuring something that was
never in the model's input -- and the axis is closed before any GPU time is spent.

So the gap is re-read at each training resolution, with the ground-truth masks
carried down by the same interpolation the pipeline uses, so filament and halo
are the same populations the loss sees.

Usage: python src/eda/fits_at_train_res.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament.rle import decode_rle

OUT = ROOT / "outputs" / "eda"
FITS_DIR = ROOT / "data" / "external" / "gong_fits"
IMG = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
SIZES = [2048, 1536, 1024, 768]


def log(*a):
    print(*a, flush=True)


def load_fits(path, union):
    from astropy.io import fits as afits
    with afits.open(path) as h:
        arr = next(x.data for x in h if x.data is not None).astype(np.float32)
    if arr.shape != union.shape:
        return None
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    halo = (cv2.dilate(union.astype(np.uint8), k) > 0) & ~union
    best = None
    for cand in (arr, arr[::-1], arr[:, ::-1], arr[::-1, ::-1]):
        c = (cand[halo].mean() - cand[union].mean()) / max(cand[halo].mean(), 1e-6)
        if best is None or c > best[0]:
            best = (c, cand)
    arr = best[1]
    for dy in range(-6, 7, 2):
        for dx in range(-6, 7, 2):
            s = np.roll(np.roll(arr, dy, 0), dx, 1)
            c = (s[halo].mean() - s[union].mean()) / max(s[halo].mean(), 1e-6)
            if c > best[0]:
                best = (c, s)
    return np.ascontiguousarray(best[1])


def main():
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    un = pd.read_parquet(OUT / "true_unreachable.parquet")
    un["unreach"] = (un["best"] <= 0.5).astype(int)
    gt = gt.merge(un[["filament_id", "unreach"]], on="filament_id", how="inner")

    stems = sorted(p.name.replace(".fits.fz", "") for p in FITS_DIR.glob("*.fits.fz"))
    log(f"{len(stems)} images with FITS on disk")

    rows = []
    for n, stem in enumerate(stems):
        g = gt[gt["stem"] == stem]
        if g.empty:
            continue
        masks = [decode_rle(r.segmentation_rle).astype(bool) for r in g.itertuples()]
        union = np.zeros_like(masks[0])
        for m in masks:
            union |= m
        F = load_fits(FITS_DIR / f"{stem}.fits.fz", union)
        if F is None:
            continue
        J = cv2.imread(str(IMG / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        for S in SIZES:
            # exactly the pipeline's own resize, applied to image and label alike
            js = J if S == 2048 else cv2.resize(J, (S, S), interpolation=cv2.INTER_AREA)
            fs = F if S == 2048 else cv2.resize(F, (S, S), interpolation=cv2.INTER_AREA)
            us = union if S == 2048 else cv2.resize(
                union.astype(np.float32), (S, S), interpolation=cv2.INTER_AREA) > 0.5
            r_ker = max(3, int(round(41 * S / 2048)) | 1)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_ker, r_ker))
            for r, m in zip(g.itertuples(), masks):
                ms = m if S == 2048 else cv2.resize(
                    m.astype(np.float32), (S, S), interpolation=cv2.INTER_AREA) > 0.5
                if ms.sum() < 20:
                    continue
                halo = (cv2.dilate(ms.astype(np.uint8), k) > 0) & ~us
                if halo.sum() < 20:
                    continue
                d = {}
                for nm, A in (("j", js), ("f", fs)):
                    a, b = A[ms], A[halo]
                    d[nm] = (b.mean() - a.mean()) / np.sqrt((a.var() + b.var()) / 2 + 1e-9)
                rows.append(dict(stem=stem, fid=r.filament_id, unreach=r.unreach,
                                 size=S, d_jpeg=d["j"], d_fits=d["f"]))
        log(f"  [{n+1}/{len(stems)}] {stem}")

    d = pd.DataFrame(rows)
    d.to_parquet(OUT / "fits_at_train_res.parquet", index=False)

    log(f"\nFITS advantage by the resolution the network is actually fed")
    log(f"  {'size':>6}{'n':>6}{'JPEG':>9}{'FITS':>9}{'gap':>9}{'vs 2048':>10}"
        f"{'FITS wins':>11}")
    ref, res = None, {}
    for S in SIZES:
        t = d[d["size"] == S]
        gap = t.d_fits.median() - t.d_jpeg.median()
        ref = gap if ref is None else ref
        res[S] = float(gap)
        log(f"  {S:>6}{len(t):>6}{t.d_jpeg.median():>9.3f}{t.d_fits.median():>9.3f}"
            f"{gap:>9.3f}{100*gap/ref:>9.0f}%{100*(t.d_fits>t.d_jpeg).mean():>10.0f}%")

    log(f"\nsame, restricted to the unreachable rows -- the population that costs us")
    log(f"  {'size':>6}{'n':>6}{'JPEG':>9}{'FITS':>9}{'gap':>9}")
    u = d[d.unreach == 1]
    for S in SIZES:
        t = u[u["size"] == S]
        log(f"  {S:>6}{len(t):>6}{t.d_jpeg.median():>9.3f}{t.d_fits.median():>9.3f}"
            f"{t.d_fits.median()-t.d_jpeg.median():>9.3f}")

    prod = 1536
    log(f"\nthe production semantic lane trains at {prod}.")
    log(f"  gap there {res[prod]:+.3f} d', which is {100*res[prod]/res[2048]:.0f}% "
        f"of the gap visible at 2048.")

    json.dump(dict(sizes=res, n=len(d)), open(OUT / "fits_at_train_res.json", "w"),
              indent=1)
    log(f"\nwrote {OUT/'fits_at_train_res.json'}")


if __name__ == "__main__":
    main()
