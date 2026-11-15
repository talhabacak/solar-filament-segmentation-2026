"""Are the augmentation ranges calibrated to the data's own variation?

Picking sigma 0.5-1.5 for blur and 0.9-1.1 for contrast is a guess unless it is
checked against how much those quantities already vary in the 707 competition
images. That check is what separates a study from a list of albumentations
defaults, and it cuts both ways:

  range too narrow   the augmentation is a no-op — the model already sees more
                     variation than we are synthesising, so it costs CPU and
                     teaches nothing
  range too wide     we are synthesising images the test set cannot contain, and
                     spending capacity on a distribution that will never be
                     scored

The dataset has a natural source of exactly this variation: six GONG stations at
different sites and altitudes, each with its own seeing, transparency and
exposure. So per-station spread is the yardstick — it is real variation the test
set genuinely contains.

Measured per image, on-disk only (r <= 0.97, matching the pipeline's mask):

  sharpness    variance of the Laplacian, the standard seeing proxy. Converted
               to an equivalent Gaussian sigma by blurring the sharpest decile
               and finding which sigma reproduces the blurriest decile.
  level        mean intensity — exposure and transparency
  contrast     robust spread (p84 - p16) / 2, an outlier-resistant sigma
  encoding     bytes per on-disk pixel, the JPEG-quality proxy

Usage: python src/eda/aug_calibration.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "outputs" / "eda"
S = 768


def log(*a):
    print(*a, flush=True)


def disk_mask(S, frac=0.97):
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    c = (S - 1) / 2.0
    return ((xx - c) ** 2 + (yy - c) ** 2) <= (frac * c) ** 2


def main():
    idx = json.loads((ROOT / "data" / "semantic_cache" / f"index_{S}.json").read_text())
    imgs = np.load(ROOT / "data" / "semantic_cache" / f"images_{S}.npy", mmap_mode="r")
    files = idx["files"]
    dk = disk_mask(S)

    raw_dir = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"
    sizes = {p.name: p.stat().st_size for p in raw_dir.rglob("*.jpeg")}

    rows = []
    for i, fn in enumerate(files):
        im = np.asarray(imgs[i])
        v = im[dk].astype(np.float32)
        lap = cv2.Laplacian(im, cv2.CV_32F)
        rows.append(dict(
            file=fn,
            station=fn[14] if len(fn) > 14 else "?",   # GONG site letter
            sharp=float(lap[dk].var()),
            level=float(v.mean()),
            contrast=float((np.percentile(v, 84) - np.percentile(v, 16)) / 2),
            bytes_px=sizes.get(fn, np.nan) / float(dk.sum())))
    df = pd.DataFrame(rows)
    log(f"{len(df)} images, {df['station'].nunique()} stations\n")

    log("per-station (the natural variation the test set really contains):")
    g = df.groupby("station").agg(n=("file", "size"), sharp=("sharp", "median"),
                                  level=("level", "mean"),
                                  contrast=("contrast", "mean"),
                                  bytes_px=("bytes_px", "mean"))
    log(g.to_string(float_format=lambda v: f"{v:.3f}"))

    log("\nwhole-dataset spread (p10 / p50 / p90) and what it implies:")
    imp = {}
    # bytes_px divides a 2048^2 file size by the 768^2 disk area, so it is a
    # RELATIVE encoding-cost index across images, not an absolute rate; the
    # absolute quality question is answered natively further down.
    for col, aug, lo_hi in (("level", "brightness shift b", (-5, 5)),
                            ("contrast", "contrast scale a", (0.78, 1.22)),
                            ("bytes_px", "relative encoding cost index", None),
                            ("sharp", "GaussianBlur sigma 0.2-0.7", None)):
        p10, p50, p90 = np.nanpercentile(df[col], [10, 50, 90])
        log(f"  {col:9s} {p10:8.3f} / {p50:8.3f} / {p90:8.3f}"
            f"   <- {aug}")
        imp[col] = dict(p10=float(p10), p50=float(p50), p90=float(p90))

    # level: the first guess was +-12 DN, which this showed to be 3.3x too wide.
    p10, p50, p90 = imp["level"]["p10"], imp["level"]["p50"], imp["level"]["p90"]
    log(f"\nbrightness: natural p10-p90 span is {p90-p10:.1f} DN "
        f"(+-{(p90-p10)/2:.1f}); augmentation uses +-5 -> "
        f"{'calibrated' if 0.5 < 5/((p90-p10)/2) < 2.0 else 'MISCALIBRATED'}")

    # contrast: the first guess was +-0.10, which this showed to be 2.4x too
    # narrow — the opposite error to brightness, on the axis that matters more.
    c10, c50, c90 = imp["contrast"]["p10"], imp["contrast"]["p50"], imp["contrast"]["p90"]
    nat = (c90 - c10) / 2 / c50
    log(f"contrast: natural relative spread +-{nat:.3f}; augmentation uses "
        f"+-0.220 -> {'calibrated' if 0.5 < 0.22/nat < 2.0 else 'MISCALIBRATED'}")

    # sharpness: convert the natural spread into an equivalent blur sigma
    sharp = df["sharp"].to_numpy()
    top = np.argsort(sharp)[-max(8, len(df)//10):]          # sharpest decile
    tgt = float(np.percentile(sharp, 10))                   # blurriest decile
    log(f"\nseeing: sharpest decile Laplacian var {np.median(sharp[top]):.1f}, "
        f"blurriest decile {tgt:.1f}")
    best = None
    for sig in np.arange(0.1, 3.01, 0.05):
        vals = []
        for i in top[:12]:
            b = cv2.GaussianBlur(np.asarray(imgs[int(i)]), (0, 0), float(sig))
            vals.append(cv2.Laplacian(b, cv2.CV_32F)[dk].var())
        m = float(np.median(vals))
        if best is None or abs(m - tgt) < abs(best[1] - tgt):
            best = (float(sig), m)
    log(f"  sigma that turns the sharpest decile into the blurriest: "
        f"{best[0]:.1f} (Laplacian var {best[1]:.1f} vs target {tgt:.1f})")
    log(f"  augmentation uses sigma 0.2-0.7 -> "
        f"{'calibrated' if 0.7 >= best[0] >= 0.2 else 'MISCALIBRATED'} "
        f"(the natural spread must sit INSIDE the range, not at its floor)")

    # encoding: what quality do the competition frames correspond to?
    #
    # This has to be done on the NATIVE 2048^2 file, not the 768 cache. Comparing
    # a 2048^2 file size against a 768^2 pixel count is off by (2048/768)^2 = 7.1
    # and downsampling strips the high frequencies that dominate JPEG cost, so
    # the cached image re-encodes far smaller at every quality.
    log("\nencoding: matching the competition JPEGs' own quality "
        "(native 2048^2, same file re-encoded)")
    probe = [f for f in files[:40] if f in sizes][:6]
    est = []
    for fn in probe:
        p = next(raw_dir.rglob(fn))
        raw = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        actual = sizes[fn]
        curve = []
        for q in range(40, 101, 5):
            ok, enc = cv2.imencode(".jpg", raw, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            curve.append((q, len(enc)))
        qbest = min(curve, key=lambda t: abs(t[1] - actual))[0]
        est.append(qbest)
        log(f"    {fn[:20]}  actual {actual/1024:6.0f} KB  -> q~{qbest}")
    log(f"  competition frames sit near JPEG q~{int(np.median(est))} "
        f"(median of {len(est)} probes)")
    qm = int(np.median(est))
    log(f"  augmentation jitters q in 78-95 -> "
        f"{'centred on it' if 78 <= qm <= 95 else 'RECENTRE'}")

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "aug_calibration.parquet", index=False)
    with open(OUT / "aug_calibration.json", "w") as f:
        json.dump(dict(per_station=g.to_dict(), spread=imp,
                       equiv_blur_sigma=best[0]), f, indent=1, default=float)
    log(f"\nsaved -> {OUT/'aug_calibration.json'}")


if __name__ == "__main__":
    main()
