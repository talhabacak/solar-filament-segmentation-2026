"""Does our score depend on the Earth-Sun distance, or on the solar B0 angle?

GONG remaps every frame to a disc of radius ~900 px (disk_geometry_v2: 899.95 +- 0.73 px, no seasonal
trend), so the physical apparent size, which swings by 3.3 % between perihelion (early January) and
aphelion (early July), becomes a change of effective resolution: January frames are shrunk by ~1.7 %,
July frames enlarged by ~1.7 %. B0, the tilt of the solar rotation axis toward us, runs over the same
year (0 in early June and December, -7.25 deg in early March, +7.25 deg in early September) and is
ignored by our latitude feature and the +-70 deg longitude check.

Both are computed from the date in the file name with low-precision solar ephemeris formulas
(accurate to ~0.001 AU and ~0.1 deg, far below what matters here), then set against hf, GT count and
sub036's out-of-fold score: Spearman over images, and pooled PQ with image-bootstrap CI by quartile.
Needs outputs/eda/sub036_oof_per_image.parquet (written by _hf_mix_by_quartile.py).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/eda"
C4 = ["tp", "fp", "fn", "iou"]


def pq(tp, fp, fn, iou):
    return iou / (tp + fp / 2 + fn / 2) if tp + fp + fn else 0.0


def ephemeris(ts):
    """Earth-Sun distance (AU) and solar B0 (deg) from a UTC timestamp."""
    jd = ts.to_julian_date()
    n = jd - 2451545.0
    g = np.radians((357.529 + 0.98560028 * n) % 360)                  # mean anomaly
    dist = 1.00014 - 0.01671 * np.cos(g) - 0.00014 * np.cos(2 * g)
    # ecliptic longitude of the Sun: mean longitude + equation of centre (degrees -> radians)
    lam = np.radians((280.459 + 0.98564736 * n) % 360 + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g))
    K = np.radians(73.6667 + 1.3958333 * (jd - 2396758.0) / 36525)    # ascending node of the solar equator
    I = np.radians(7.25)                                              # inclination of the solar equator
    b0 = np.degrees(np.arcsin(np.sin(lam - K) * np.sin(I)))
    return dist, b0


q = pd.read_parquet(OUT / "image_quality.parquet")
P = pd.read_parquet(OUT / "sub036_oof_per_image.parquet")
ts = pd.to_datetime(q["stem"].str[:14], format="%Y%m%d%H%M%S")
q["dist_au"], q["b0"] = ephemeris(pd.DatetimeIndex(ts))
q["abs_b0"] = q["b0"].abs()
q["scale"] = 1 / q["dist_au"]                                         # apparent size relative to 1 AU

chk = q.assign(m=ts.dt.month).groupby("m")[["dist_au", "b0"]].mean().round(4)
print("sanity, monthly means (expect distance min in Jan, max in Jul; B0 ~ -7 in Mar, ~ +7 in Sep):")
print("  " + "  ".join(f"{m}: {r.dist_au:.4f}/{r.b0:+.1f}" for m, r in chk.iterrows()))

tr = q[q.split == "train"].merge(P, on="stem")
te = q[q.split == "test"]
tr["den"] = tr.tp + tr.fp / 2 + tr.fn / 2
tr["pq_img"] = np.where(tr.den > 0, tr.iou / tr.den.replace(0, np.nan), np.nan)
tr["recall_img"] = tr.tp / (tr.tp + tr.fn).replace(0, np.nan)
tr["gt"] = tr.tp + tr.fn

print("\nSpearman over training images (n = %d)" % len(tr))
for x in ("dist_au", "abs_b0"):
    cells = []
    for y in ("hf", "gt", "pq_img", "recall_img"):
        d = tr[[x, y]].dropna()
        r, p = spearmanr(d[x], d[y])
        cells.append(f"{y} {r:+.3f} (p {p:.2f})")
    print(f"  {x:8s} " + "   ".join(cells))

rng = np.random.default_rng(0)
tot = pq(*tr[C4].sum().to_numpy())
for x, lab in (("dist_au", "Earth-Sun distance (AU)"), ("abs_b0", "|B0| (deg)")):
    edges = tr[x].quantile([.25, .5, .75]).to_numpy()
    tr["b"] = pd.cut(tr[x], [-np.inf, *edges, np.inf], labels=["Q1", "Q2", "Q3", "Q4"])
    te_b = pd.cut(te[x], [-np.inf, *edges, np.inf], labels=["Q1", "Q2", "Q3", "Q4"])
    print(f"\n{lab} quartiles (training edges), pooled out-of-fold PQ")
    shares = {}
    for b, g in tr.groupby("b", observed=True):
        X = g[C4].to_numpy()
        boots = [pq(*X[rng.integers(0, len(X), len(X))].sum(0)) for _ in range(2000)]
        tp, fp, fn, iou = X.sum(0)
        shares[b] = (te_b == b).mean() / (len(g) / len(tr))
        print(f"  {b} {g[x].min():.4f}-{g[x].max():.4f}  train {len(g)}  test {int((te_b == b).sum())}  "
              f"PQ {pq(tp, fp, fn, iou):.4f} [{np.quantile(boots, .025):.4f}, {np.quantile(boots, .975):.4f}]  "
              f"recall {tp / (tp + fn):.4f}  hf median {g.hf.median():.0f}")
    w = tr["b"].map(shares).astype(float).to_numpy()
    print(f"  PQ shift under the test mix {pq(*(tr[C4].to_numpy() * w[:, None]).sum(0)) - tot:+.4f}")

# the distance quartiles split into a nearer and a farther pair: is the gap real, and does it survive
# holding the station fixed (seeing has its own season, opposite in the two hemispheres)?
far = (tr.dist_au > tr.dist_au.median()).to_numpy()
X = tr[C4].to_numpy()
gaps = []
for _ in range(4000):
    i = rng.integers(0, len(tr), len(tr))
    gaps.append(pq(*X[i][far[i]].sum(0)) - pq(*X[i][~far[i]].sum(0)))
print(f"\nfarther half - nearer half: {pq(*X[far].sum(0)) - pq(*X[~far].sum(0)):+.4f}  "
      f"95% CI [{np.quantile(gaps, .025):+.4f}, {np.quantile(gaps, .975):+.4f}]")
within = []
for s, g in tr.groupby("site"):
    f = (g.dist_au > tr.dist_au.median()).to_numpy()
    d = pq(*g[f][C4].sum().to_numpy()) - pq(*g[~f][C4].sum().to_numpy())
    within.append(d)
    print(f"  {s}: farther {pq(*g[f][C4].sum().to_numpy()):.4f} (n {f.sum()})  "
          f"nearer {pq(*g[~f][C4].sum().to_numpy()):.4f} (n {(~f).sum()})  gap {d:+.4f}")
print(f"  stations with a negative gap: {sum(d < 0 for d in within)} of {len(within)}")
