"""Is the model's blindness to faint filaments actually SNR-limited?

The FITS precondition measured a real separability gain over the competition
JPEG: d' rises by a factor 1.13, 14/14 images positive, and the mechanism is JPEG
compression noise rather than 8-bit quantisation (the quantisation step accounts
for 0.1% of the local variance, worth a factor 1.0005). But the gain is UNIFORM:
reachable rows gain 1.118x and unreachable rows 1.111x, so the population that
actually costs us keeps its relative position and the gap between the two widens
slightly. That is the shape of every ceiling-raise this project has already
watched die in the selector.

Before spending twenty GPU hours rebuilding the pipeline on FITS inputs, this
asks the question that decides the axis: does the model's response to a ground
truth row depend on that row's input separability at all, in the faint regime?

If p_max rises steeply with d' where the unreachable rows live, a 0.15 shift in
d' buys a predictable amount of probability and the axis is worth pursuing. If
p_max is flat there -- the model is not failing for want of contrast but because
the structure is outside what it has learned to call a filament -- then a
cleaner input cannot help, and the measured 13% is a number about photographs
rather than about the metric.

The conversion is done on the model's own measured sensitivity, evaluated at the
faint population's own operating point, which is the gating rule this project
adopted after thirteen gains failed to survive: gate on the targeted population,
never on a proxy for it.

Usage: python src/eda/snr_binding.py
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
IMG = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
KER = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
CACHE = OUT / "_snr_binding_cache.parquet"


def log(*a):
    print(*a, flush=True)


def prob_maps(tag="_cldice"):
    """stem -> out-of-fold semantic probability map (the production member)."""
    idx = {}
    for k in range(5):
        p = ROOT / "outputs" / "semantic" / f"fold{k}_probs{tag}.npz"
        if not p.exists():
            continue
        with np.load(p) as z:
            for s in z.files:
                idx[s] = (p, s)
    return idx


def build():
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    un = pd.read_parquet(OUT / "true_unreachable.parquet")
    un["unreach"] = (un["best"] <= 0.5).astype(int)
    gt = gt.merge(un[["filament_id", "unreach", "best"]], on="filament_id", how="inner")

    idx = prob_maps()
    log(f"{len(gt)} labelled rows over {gt['stem'].nunique()} images; "
        f"{len(idx)} probability maps cached")
    stems = [s for s in sorted(gt["stem"].unique()) if s in idx]
    log(f"{len(stems)} images have both")

    rows = []
    open_files = {}
    for i, stem in enumerate(stems):
        g = gt[gt["stem"] == stem]
        J = cv2.imread(str(IMG / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        if J is None:
            continue
        J = J.astype(np.float32)
        path, key = idx[stem]
        if path not in open_files:
            open_files.clear()
            open_files[path] = np.load(path)
        P = open_files[path][key].astype(np.float32)
        if P.max() > 1.5:                       # stored as uint8 quantised probs
            P = P / 255.0
        masks = [decode_rle(r.segmentation_rle).astype(bool) for r in g.itertuples()]
        union = np.zeros_like(masks[0])
        for m in masks:
            union |= m
        # the probability map is smaller than the 2048 label frame; compare at the
        # map's own resolution so no interpolation is invented for thin structures
        sc = P.shape[0] / 2048.0
        for r, m in zip(g.itertuples(), masks):
            if m.sum() < 100:
                continue
            halo = (cv2.dilate(m.astype(np.uint8), KER) > 0) & ~union
            if halo.sum() < 100:
                continue
            a, b = J[m], J[halo]
            d = (b.mean() - a.mean()) / np.sqrt((a.var() + b.var()) / 2 + 1e-9)
            ms = (cv2.resize(m.astype(np.uint8), P.shape[::-1],
                             interpolation=cv2.INTER_AREA) > 0.5) if sc != 1 else m
            if ms.sum() == 0:
                continue
            rows.append(dict(stem=stem, fid=r.filament_id, unreach=r.unreach,
                             best=r.best, area=int(m.sum()), dprime=d,
                             p_max=float(P[ms].max()), p_mean=float(P[ms].mean())))
        if (i + 1) % 40 == 0:
            log(f"  {i+1}/{len(stems)} images")
    d = pd.DataFrame(rows)
    d.to_parquet(CACHE, index=False)
    return d


def main():
    d = pd.read_parquet(CACHE) if CACHE.exists() else build()
    log(f"\n{len(d)} rows, {int(d['unreach'].sum())} unreachable, "
        f"{d['stem'].nunique()} images")

    log("\nmodel response against input separability (JPEG d'):")
    d["bin"] = pd.qcut(d["dprime"], 8, duplicates="drop")
    t = d.groupby("bin", observed=True).agg(
        n=("p_max", "size"), d=("dprime", "median"),
        p_max=("p_max", "median"), p_mean=("p_mean", "median"),
        unreach=("unreach", "mean"))
    log(f"  {'d-prime':>9}{'n':>6}{'p_max':>9}{'p_mean':>9}{'% unreach':>11}")
    for _, r in t.iterrows():
        log(f"  {r.d:>9.2f}{int(r.n):>6}{r.p_max:>9.3f}{r.p_mean:>9.3f}"
            f"{100*r.unreach:>10.0f}%")

    from scipy.stats import spearmanr
    log(f"\n  Spearman(d', p_max) over all rows      {spearmanr(d.dprime, d.p_max).statistic:+.3f}")
    u = d[d.unreach == 1]
    log(f"  Spearman(d', p_max) over unreachable   {spearmanr(u.dprime, u.p_max).statistic:+.3f}"
        f"   (n={len(u)})")

    # p_max saturates at 1.0 in every bin -- ground-truth masks are large enough
    # that one pixel is always confident -- so the model's response is read off
    # p_mean, the fraction of the annotated body the map actually lights up
    log(f"\n  Spearman(d', p_mean) over all rows     "
        f"{spearmanr(d.dprime, d.p_mean).statistic:+.3f}")
    log(f"  Spearman(d', p_mean) over unreachable  "
        f"{spearmanr(u.dprime, u.p_mean).statistic:+.3f}")

    # Convert the measured separability gain into reachability. A logistic fit of
    # unreachability on log d' is evaluated twice per row: at its own d', and at
    # the d' the FITS was measured to deliver. The difference is the number of
    # rows the axis could rescue IF the relationship is causal.
    pre = json.load(open(OUT / "fits_precondition.json"))
    g = pre["d_fits"] / pre["d_jpeg"]
    from sklearn.linear_model import LogisticRegression
    X = np.log(np.maximum(d.dprime.to_numpy(), 0.05)).reshape(-1, 1)
    m = LogisticRegression().fit(X, d.unreach.to_numpy())
    p_now = m.predict_proba(X)[:, 1]
    p_aft = m.predict_proba(X + np.log(g))[:, 1]
    log(f"\nconverting the FITS separability gain into reachability")
    log(f"  gain factor {g:.4f}, i.e. every row's d' scaled by {g:.3f}")
    log(f"  expected unreachable rows now            {p_now.sum():7.1f} "
        f"(actual {int(d.unreach.sum())})")
    log(f"  expected unreachable rows after FITS     {p_aft.sum():7.1f}")
    log(f"  rows rescued                             {p_now.sum()-p_aft.sum():7.1f}"
        f"  = {100*(p_now.sum()-p_aft.sum())/p_now.sum():.1f}% of the block")

    # what that is worth if every rescued row converted into a matched instance,
    # which it will not -- the pool still has to propose it and the selector keep it
    tp, fn = int((d.unreach == 0).sum()), int(d.unreach.sum())
    resc = float(p_now.sum() - p_aft.sum())
    pq_now = 0.6 * tp / (tp + 0.5 * (tp * 1.9) + 0.5 * fn)
    pq_aft = 0.6 * (tp + resc) / ((tp + resc) + 0.5 * (tp * 1.9) + 0.5 * (fn - resc))
    log(f"\n  crude PQ ceiling effect (every rescued row matches, SQ 0.6, FP held)")
    log(f"    {pq_now:.4f} -> {pq_aft:.4f}   delta {pq_aft-pq_now:+.4f}")
    log(f"  this is an UPPER bound twice over: the d'-reachability relation is")
    log(f"  correlational -- brighter, better-defined filaments are easier for the")
    log(f"  annotator and the model alike -- and nothing here makes the selector")
    log(f"  keep the extra candidates. Thirteen ceiling gains have died at that step.")

    json.dump(dict(n=len(d), gain=float(g), unreach_now=float(p_now.sum()),
                   unreach_after=float(p_aft.sum()), rescued=resc,
                   pq_ceiling_delta=float(pq_aft - pq_now),
                   rho_dp_pmean=float(spearmanr(d.dprime, d.p_mean).statistic)),
              open(OUT / "snr_binding.json", "w"), indent=1)
    log(f"\nwrote {OUT/'snr_binding.json'}")


if __name__ == "__main__":
    main()
