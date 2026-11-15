"""Trusted per-image solar disk geometry via full-resolution limb fitting.

GONG display JPEGs: disk interior ~85, a thin BRIGHT rim at the limb (~200),
then a dim halo (~40) fading to black. The photospheric limb is located per ray
as the steepest intensity DROP just outside the rim peak; a robust (iterative
outlier-rejecting) Kasa circle fit over ~720 rays gives (cx, cy, R).

Resolves the manifest (R~1036, impossible) vs morphology (R~938) contradiction.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"
OUT = ROOT / "outputs" / "eda"
CHECK = OUT / "disk_refit_check"
CHECK.mkdir(parents=True, exist_ok=True)

N_RAYS = 720


def coarse_disk(img):
    small = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    m = (small > 30).astype(np.uint8)
    ys, xs = np.nonzero(m)
    if len(xs) < 100:
        return 1024.0, 1024.0, 940.0
    cx, cy = xs.mean() * 8, ys.mean() * 8
    r = np.sqrt(m.sum() / np.pi) * 8
    return cx, cy, r


def limb_points(img, cx, cy, r0):
    """Limb point per ray = the bright rim peak.

    GONG display JPEGs always carry a thin bright ring (values ~160-220, brighter
    than both the ~110 interior and any outer gray annulus) exactly at the
    photospheric limb; peak detection is robust where gradient-drop logic fails
    (thick bright annuli in some site/era processings). Absolute search window
    820-1040 px covers the physical range (plate scale ~1.05-1.08 arcsec/px,
    solar radius 944-976 arcsec) with margin.
    """
    thetas = np.linspace(0, 2 * np.pi, N_RAYS, endpoint=False)
    radii = np.arange(820.0, 1040.0, 1.0)
    pts = []
    for th in thetas:
        xs = cx + radii * np.cos(th)
        ys = cy + radii * np.sin(th)
        ok = (xs >= 0) & (xs < 2048) & (ys >= 0) & (ys < 2048)
        if ok.sum() < 50:
            continue
        prof = img[ys[ok].astype(int), xs[ok].astype(int)].astype(np.float32)
        sm = np.convolve(prof, np.ones(3) / 3, mode="same")
        peak = int(np.argmax(sm))
        interior = np.median(prof[:30]) if len(prof) > 40 else 100.0
        if sm[peak] < max(interior * 1.25, 130):     # no clear rim (cloud/prominence gap)
            continue
        rr = radii[ok][peak]
        pts.append((cx + rr * np.cos(th), cy + rr * np.sin(th)))
    return np.array(pts)


def kasa_fit(pts):
    x, y = pts[:, 0], pts[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0], sol[1]
    r = np.sqrt(sol[2] + cx ** 2 + cy ** 2)
    resid = np.abs(np.hypot(x - cx, y - cy) - r)
    return cx, cy, r, resid


def fit_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    cx, cy, r0 = coarse_disk(img)
    pts = limb_points(img, cx, cy, r0)
    if len(pts) < 100:
        return dict(cx=np.nan, cy=np.nan, R=np.nan, rms=np.nan, n_pts=len(pts), suspect=True)
    for _ in range(3):                                   # iterative outlier rejection
        cx, cy, r, resid = kasa_fit(pts)
        mad = np.median(np.abs(resid - np.median(resid))) + 1e-6
        keep = resid < np.median(resid) + 3 * 1.4826 * mad
        if keep.sum() < 100:
            break
        pts = pts[keep]
    cx, cy, r, resid = kasa_fit(pts)
    rms = float(np.sqrt((resid ** 2).mean()))
    return dict(cx=float(cx), cy=float(cy), R=float(r), rms=rms,
                n_pts=int(len(pts)), suspect=bool(rms > 6.0 or r > 1010 or r < 850))


def overlay(path, fit, dest):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if not np.isnan(fit["R"]):
        cv2.circle(vis, (int(fit["cx"]), int(fit["cy"])), int(fit["R"]), (0, 0, 255), 3)
        cv2.circle(vis, (int(fit["cx"]), int(fit["cy"])), 8, (0, 255, 0), -1)
    cv2.imwrite(str(dest), cv2.resize(vis, (1024, 1024)))


if __name__ == "__main__":
    rows = []
    for split, d in [("train", DATA / "train" / "train_images"),
                     ("test", DATA / "test" / "test_images")]:
        for p in sorted(d.glob("*.jpeg")):
            fit = fit_image(p)
            fit.update(file_name=p.name, split=split)
            rows.append(fit)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "disk_geometry_v2.parquet", index=False)

    print(df.groupby("split")["R"].describe().round(1))
    print("\nR percentiles:", df["R"].quantile([0.01, 0.05, 0.5, 0.95, 0.99]).round(1).to_dict())
    df["site"] = df["file_name"].str[14]
    df["offset"] = np.hypot(df["cx"] - 1023.5, df["cy"] - 1023.5)
    print("\nper-site R median:", df.groupby("site")["R"].median().round(1).to_dict())
    print("per-site center offset median:", df.groupby("site")["offset"].median().round(1).to_dict())
    print("suspect fits:", int(df["suspect"].sum()))
    print(df[df["suspect"]][["file_name", "R", "rms", "n_pts"]].head(12).to_string())

    # visual checks: normal, claimed outliers from both old caches, C-site offset
    checks = ["20110120105534Ch.jpeg", "20200619123010Th.jpeg", "20190926165810Mh.jpeg",
              "20190704123830Ch.jpeg", "20140609195854Bh.jpeg", "20120703070814Th.jpeg"]
    for fn in checks:
        row = df[df["file_name"] == fn]
        if len(row):
            split = row["split"].iloc[0]
            src = DATA / split / f"{split}_images" / fn
            overlay(src, row.iloc[0].to_dict(), CHECK / f"check_{fn}.png")
    print("overlays written to", CHECK)
