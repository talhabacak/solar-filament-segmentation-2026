"""Can a denoiser on the existing JPEG buy what the FITS buys?

The FITS is worth a factor 1.13 in filament separability, and the mechanism is
not 8-bit quantisation -- that step accounts for 0.1% of the local variance,
worth 1.0005 -- so what the JPEG is losing is lossy-compression noise. Compression
noise is high-frequency and structured on the 8x8 DCT grid, which is exactly what
an ordinary denoiser removes. If a denoiser on the JPEG we already have recovers
most of the gap, the FITS axis costs 661 downloads and a full five-fold retrain
to buy something a CPU filter delivers in minutes.

The comparison has to be fair, and the obvious version is not. d' is a mean
difference over a pooled standard deviation, so ANY smoothing raises it
mechanically: the numerator survives, the denominator falls, and no information
has been added. Reporting "bilateral filtering gives +0.2 d'" would be measuring
the filter's blur, not its cleaning.

So every treatment is applied to BOTH images and the FITS-minus-JPEG gap is read
at matched resolution. If smoothing both closes the gap, the JPEG's deficit was
high-frequency noise and a denoiser is a sufficient substitute. If the gap
survives identical treatment, the FITS carries information the JPEG does not
have and no filter can invent.

Usage: python src/eda/fits_vs_denoise.py
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
KER = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))


def log(*a):
    print(*a, flush=True)


def treatments():
    """Each maps an image to a filtered image. Applied identically to both arms."""
    def nlm(x):
        # non-local means wants 8-bit; rescale per image, filter, rescale back so
        # the FITS is not silently quantised harder than the JPEG
        lo, hi = np.percentile(x, [0.5, 99.5])
        u = np.clip((x - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        f = cv2.fastNlMeansDenoising(u, None, 7, 7, 21).astype(np.float32)
        return f / 255.0 * (hi - lo) + lo
    return {
        "none": lambda x: x,
        "gauss s=1": lambda x: cv2.GaussianBlur(x, (0, 0), 1.0),
        "gauss s=2": lambda x: cv2.GaussianBlur(x, (0, 0), 2.0),
        "median 5": lambda x: cv2.medianBlur(x, 5),
        "bilateral": lambda x: cv2.bilateralFilter(x, 9, 40.0, 9.0),
        "nl-means": nlm,
    }


def load_fits(path, union):
    from astropy.io import fits as afits
    with afits.open(path) as h:
        arr = next(x.data for x in h if x.data is not None).astype(np.float32)
    if arr.shape != union.shape:
        return None
    halo = (cv2.dilate(union.astype(np.uint8), KER) > 0) & ~union
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

    stems = sorted(p.stem.replace(".fits", "") for p in FITS_DIR.glob("*.fits.fz"))
    log(f"{len(stems)} images with FITS on disk")
    TR = treatments()
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
        # filter once per treatment per image, not once per filament
        filt = {k: (f(J), f(F)) for k, f in TR.items()}
        for r, m in zip(g.itertuples(), masks):
            if m.sum() < 100:
                continue
            halo = (cv2.dilate(m.astype(np.uint8), KER) > 0) & ~union
            if halo.sum() < 100:
                continue
            rec = dict(stem=stem, fid=r.filament_id, unreach=r.unreach)
            for k, (jj, ff) in filt.items():
                for nm, A in (("j", jj), ("f", ff)):
                    a, b = A[m], A[halo]
                    rec[f"{nm}|{k}"] = ((b.mean() - a.mean())
                                        / np.sqrt((a.var() + b.var()) / 2 + 1e-9))
            rows.append(rec)
        log(f"  [{n+1}/{len(stems)}] {stem}")

    d = pd.DataFrame(rows)
    d.to_parquet(OUT / "fits_vs_denoise.parquet", index=False)
    log(f"\n{len(d)} filaments, {int(d.unreach.sum())} unreachable\n")

    base_gap = d["f|none"].median() - d["j|none"].median()
    log("d' at matched resolution -- the same filter is applied to both arms")
    log(f"  {'treatment':<12}{'JPEG':>8}{'FITS':>8}{'gap':>8}{'gap left':>10}"
        f"{'JPEG gain':>11}")
    res = {}
    for k in TR:
        j, f = d[f"j|{k}"].median(), d[f"f|{k}"].median()
        gap = f - j
        res[k] = dict(jpeg=float(j), fits=float(f), gap=float(gap))
        log(f"  {k:<12}{j:>8.3f}{f:>8.3f}{gap:>8.3f}{100*gap/base_gap:>9.0f}%"
            f"{j - d['j|none'].median():>+11.3f}")

    log(f"\n  baseline FITS advantage {base_gap:+.3f} d'")
    best = min((k for k in TR if k != "none"), key=lambda k: res[k]["gap"])
    log(f"  best denoiser at closing it: {best} -- leaves "
        f"{100*res[best]['gap']/base_gap:.0f}% of the gap")
    log(f"  and it raises the JPEG itself by "
        f"{res[best]['jpeg'] - res['none']['jpeg']:+.3f} d', but that number is"
        f" NOT a gain:")
    log("  smoothing shrinks the denominator of d' whether or not it removes")
    log("  anything real, which is why the gap is read after treating both arms.")

    u = d[d.unreach == 1]
    log(f"\nsame, restricted to the {len(u)} unreachable rows:")
    for k in TR:
        j, f = u[f"j|{k}"].median(), u[f"f|{k}"].median()
        log(f"  {k:<12}{j:>8.3f}{f:>8.3f}{f-j:>8.3f}")

    json.dump(dict(n=len(d), base_gap=float(base_gap), per_treatment=res,
                   best=best, gap_left=float(res[best]["gap"] / base_gap)),
              open(OUT / "fits_vs_denoise.json", "w"), indent=1)
    log(f"\nwrote {OUT/'fits_vs_denoise.json'}")


if __name__ == "__main__":
    main()
