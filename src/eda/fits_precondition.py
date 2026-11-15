"""Does the original FITS recover the faint filaments the JPEG loses?

The organiser ruled (discussion 738951) that the original GONG FITS files may be
used. The measured binding constraint on this project is a faint population the
model is blind to: the 414 truly-unreachable ground-truth rows sit at median
local contrast 0.082 against 0.112 for reachable rows, and the semantic map's
median in-mask maximum probability over them is 0.004. Low-contrast structure is
exactly what 8-bit quantisation and JPEG compression destroy -- so the FITS is
worth a precondition test before any pipeline work.

The test must avoid the trap the first pass fell into. Splitting filaments into
"faintest third" BY THEIR JPEG SEPARABILITY and then showing the FITS looks
better on that third measures regression to the mean, not the FITS. Here the
split is reachable vs unreachable, which is decided by the candidate pool's
geometry and is independent of either image's intensities.

What is actually being asked: is the JPEG's 8-bit quantisation the limiting
noise? If the filament-background signal is ~10 DN and the total noise ~7 DN,
the 1 DN quantisation step contributes under 1% of the variance and the FITS
cannot help by adding tonal resolution -- any gain it shows comes from JPEG
compression artefacts instead. The script reports both terms so the mechanism is
readable, not just the delta.

Usage: python src/eda/fits_precondition.py --n-images 12
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
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
BASE = "https://gong2.nso.edu/HA/haf"
KER = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))


def log(*a):
    print(*a, flush=True)


def fetch(stem):
    """Download one FITS. Returns the path, or None if the archive lacks it."""
    dst = FITS_DIR / f"{stem}.fits.fz"
    if dst.exists():
        return dst
    url = f"{BASE}/{stem[:6]}/{stem[:8]}/{stem}.fits.fz"
    try:
        with urllib.request.urlopen(url, timeout=90) as r:
            data = r.read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        log(f"    {stem}: {type(e).__name__} {getattr(e, 'code', '')}")
        return None
    dst.write_bytes(data)
    return dst


def load_fits(path, gt_union):
    """Read the FITS and put it in the JPEG's frame.

    The archive stores the disk vertically flipped relative to the competition
    JPEG; the residual shift is a couple of pixels. Both are resolved by
    maximising filament darkness under the ground-truth union, which is the only
    signal that distinguishes orientations on a near-radially-symmetric disk --
    plain image correlation gives ~0.96 for every flip and decides nothing.
    """
    from astropy.io import fits as afits
    with afits.open(path) as h:
        arr = next(x.data for x in h if x.data is not None).astype(np.float32)
    if arr.shape != gt_union.shape:
        return None, None
    halo = (cv2.dilate(gt_union.astype(np.uint8), KER) > 0) & ~gt_union
    best = None
    for name, cand in (("as-is", arr), ("flipud", arr[::-1]),
                       ("fliplr", arr[:, ::-1]), ("rot180", arr[::-1, ::-1])):
        c = (cand[halo].mean() - cand[gt_union].mean()) / max(cand[halo].mean(), 1e-6)
        if best is None or c > best[0]:
            best = (c, name, cand)
    _, name, arr = best
    for dy in range(-6, 7, 2):
        for dx in range(-6, 7, 2):
            s = np.roll(np.roll(arr, dy, 0), dx, 1)
            c = (s[halo].mean() - s[gt_union].mean()) / max(s[halo].mean(), 1e-6)
            if c > best[0]:
                best = (c, f"{name}{dy:+d}{dx:+d}", s)
    return best[2], best[1]


def dprime(a, img, halo):
    """Separability of filament from its own local halo, in noise units."""
    x, y = img[a], img[halo]
    return ((y.mean() - x.mean()) / np.sqrt((x.var() + y.var()) / 2 + 1e-9),
            y.mean() - x.mean(), np.sqrt((x.var() + y.var()) / 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=12)
    ap.add_argument("--min-area", type=int, default=100)
    a = ap.parse_args()

    FITS_DIR.mkdir(parents=True, exist_ok=True)
    un = pd.read_parquet(OUT / "true_unreachable.parquet")
    un["unreach"] = (un["best"] <= 0.5).astype(int)
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt = gt.merge(un[["filament_id", "unreach", "best"]], on="filament_id", how="inner")

    # rank images by how much unreachable mass they carry, so the comparison has
    # enough of the population the FITS is supposed to rescue
    per_img = gt.groupby("stem")["unreach"].agg(n_unreach="sum", n_gt="size")
    order = per_img[per_img["n_unreach"] >= 3].sort_values("n_unreach", ascending=False)
    log(f"{len(gt)} labelled ground-truth rows, {int(gt['unreach'].sum())} unreachable, "
        f"across {gt['stem'].nunique()} images")

    rows, got = [], 0
    for stem in order.index:
        if got >= a.n_images:
            break
        g = gt[gt["stem"] == stem]
        masks = {r.filament_id: decode_rle(r.segmentation_rle).astype(bool)
                 for r in g.itertuples()}
        union = np.zeros((2048, 2048), bool)
        for m in masks.values():
            union |= m
        t0 = time.time()
        p = fetch(stem)
        if p is None:
            continue
        F, orient = load_fits(p, union)
        if F is None:
            log(f"    {stem}: shape mismatch")
            continue
        J = cv2.imread(str(IMG / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        got += 1
        log(f"  [{got:2d}/{a.n_images}] {stem} {orient:12s} "
            f"{int(g['unreach'].sum()):2d}/{len(g):2d} unreachable ({time.time()-t0:.0f}s)")
        for r in g.itertuples():
            m = masks[r.filament_id]
            if m.sum() < a.min_area:
                continue
            halo = (cv2.dilate(m.astype(np.uint8), KER) > 0) & ~union
            if halo.sum() < a.min_area:
                continue
            dj, sj, nj = dprime(m, J, halo)
            df_, sf, nf = dprime(m, F, halo)
            rows.append(dict(stem=stem, fid=r.filament_id, unreach=r.unreach,
                             best=r.best, area=int(m.sum()),
                             d_jpeg=dj, d_fits=df_, sig_jpeg=sj, sig_fits=sf,
                             noise_jpeg=nj, noise_fits=nf,
                             fits_scale=float(F[union].mean())))

    d = pd.DataFrame(rows)
    d.to_parquet(OUT / "fits_precondition.parquet", index=False)
    log(f"\n{len(d)} filaments over {d['stem'].nunique()} images "
        f"({int(d['unreach'].sum())} unreachable)")

    log("\nseparability d' = (halo_mu - filament_mu) / pooled_sd")
    log(f"  {'population':<14}{'n':>5}{'JPEG':>9}{'FITS':>9}{'ratio':>8}{'FITS wins':>11}")
    for nm, sub in (("reachable", d[d.unreach == 0]), ("unreachable", d[d.unreach == 1]),
                    ("all", d)):
        if sub.empty:
            continue
        w = (sub.d_fits > sub.d_jpeg).mean()
        log(f"  {nm:<14}{len(sub):>5}{sub.d_jpeg.median():>9.3f}"
            f"{sub.d_fits.median():>9.3f}{sub.d_fits.median()/sub.d_jpeg.median():>8.3f}"
            f"{100*w:>10.0f}%")

    # the mechanism question: is 8-bit quantisation actually the limiting noise?
    # uniform quantisation with step 1 DN contributes variance 1/12
    qv = 1 / 12
    frac = qv / (d.noise_jpeg ** 2).median()
    log(f"\nis the JPEG's 8-bit step the binding noise?")
    log(f"  median filament-halo signal   JPEG {d.sig_jpeg.median():6.2f} DN of 256 levels")
    log(f"  median local noise sd         JPEG {d.noise_jpeg.median():6.2f} DN")
    log(f"  quantisation share of variance     {100*frac:6.3f}%  "
        f"(1/12 over {d.noise_jpeg.median()**2:.1f})")
    log(f"  -> a perfect-precision 8-bit image would gain d' by a factor "
        f"{1/np.sqrt(1-frac):.4f}")
    log(f"  measured FITS gain factor          {d.d_fits.median()/d.d_jpeg.median():.4f}")

    # paired test over filaments, clustered by image
    per = d.groupby("stem").apply(
        lambda t: pd.Series(dict(dj=t.d_jpeg.median(), df=t.d_fits.median())),
        include_groups=False)
    diff = (per.df - per.dj).to_numpy()
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    log(f"\npaired over {len(diff)} images (image-level medians): "
        f"delta d' {diff.mean():+.4f} +/- {se:.4f}, "
        f"{int((diff > 0).sum())}/{len(diff)} images positive")

    json.dump(dict(n=len(d), n_images=int(d["stem"].nunique()),
                   n_unreach=int(d["unreach"].sum()),
                   d_jpeg=float(d.d_jpeg.median()), d_fits=float(d.d_fits.median()),
                   quant_share=float(frac), delta_img=float(diff.mean()),
                   se_img=float(se)),
              open(OUT / "fits_precondition.json", "w"), indent=1)
    log(f"\nwrote {OUT/'fits_precondition.json'}")


if __name__ == "__main__":
    main()
