"""E120 -- limb-relative selector features (ledger §16a's conclusion).

E119 showed the selectable limb rows share their p_sel band with limb false
positives, so no threshold separates them. The selector's 32 features judge
every candidate against absolute statistics; at the limb the background is
darker and less contrasted, so an honest limb filament looks like a disc FP.
This adds four features that judge a candidate against ITS OWN RADIUS BAND
and against the projection geometry, computed from the raw JPEG at native
resolution and the candidate mask, no labels involved:

  img_dprime      mask-vs-halo separability d' on the raw image (docs/23's
                  instrument; the halo is a 12 px dilation minus the mask)
  band_contrast   (halo mean - mask mean) / std of the image inside the
                  candidate's radius band (r +/- 0.05): contrast relative to
                  what the limb band itself offers
  flat_dprime     the same d' after dividing the image by its own fitted
                  limb-darkening profile (median intensity in 40 radial bins,
                  smoothed) -- Talha's flattening question, as a feature
  elong_radial    extent along the radial direction / extent along the
                  tangential direction of the mask (foreshortening signature;
                  limb filaments are squeezed radially)

Then the E115b selector recipe is refit out of fold with 32 + 4 features and
scored at the production policy, against the 32-feature control on the same
folds (same seeds, same labelling, same GT scope = all folds_v1 stems).

PRE-REGISTERED: gate = mean Δ over folds >= +0.002 (Talha's rule) AND >= 4/5
folds positive; the paired image bootstrap of E115c is run for the interval.
Diagnostic, not verdict: reach by radius band under both selectors (the
mechanism must show at r > 0.8, else the gain is somewhere else and the
feature story is wrong).

Usage: python src/eda/limb_relative_features.py [--build-only]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "eda"))

from filament.pq import score
from filament.rle import decode_rle
from e115c_threshold_ci import oof_scores, per_image, pq

OUT = ROOT / "outputs" / "eda"
IMG = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
BUNDLE = ROOT / "outputs" / "selector" / "selector_v5_shrunkblob5prodts.joblib"
DISK_C, DISK_R = 1023.5, 900.0
NEW = ["img_dprime", "band_contrast", "flat_dprime", "elong_radial"]
KER = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
FLOOR, EQUIV_BAND, REPS = 0.002, 0.005, 4000


def log(*a):
    print(*a, flush=True)


def radius_map():
    yy, xx = np.mgrid[0:2048, 0:2048].astype(np.float32)
    return np.hypot(xx - DISK_C, yy - DISK_C) / DISK_R


def flatten(img, r, nbins=40):
    """Divide by the image's own radial median profile (limb darkening)."""
    rb = np.clip((r * nbins).astype(int), 0, nbins - 1)
    prof = np.array([np.median(img[rb == b]) if (rb == b).any() else np.nan for b in range(nbins)])
    prof = pd.Series(prof).interpolate(limit_direction="both").rolling(3, center=True, min_periods=1).mean().to_numpy()
    prof = np.maximum(prof, 1.0)
    return img / prof[rb] * float(np.median(img[r < 0.5]))


def dprime(A, m, halo):
    a, b = A[m], A[halo]
    return float((b.mean() - a.mean()) / np.sqrt((a.var() + b.var()) / 2 + 1e-9))


def build_features(cand):
    R = radius_map()
    rows = []
    for k, (stem, g) in enumerate(cand.groupby("stem")):
        J = cv2.imread(str(IMG / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        if J is None:
            continue
        J = J.astype(np.float32)
        F = flatten(J, R)
        for r in g.itertuples():
            m = decode_rle(r.rle).astype(bool)
            if m.sum() < 16:
                rows.append(dict(cand_id=r.cand_id, **{n: np.nan for n in NEW}))
                continue
            halo = (cv2.dilate(m.astype(np.uint8), KER) > 0) & ~m
            ys, xs = np.nonzero(m)
            cy, cx = ys.mean(), xs.mean()
            rc = float(np.hypot(cx - DISK_C, cy - DISK_C) / DISK_R)
            band = (R > rc - 0.05) & (R < rc + 0.05)
            bstd = float(J[band].std()) + 1e-6
            # radial / tangential extent of the mask
            ux, uy = (cx - DISK_C), (cy - DISK_C)
            n = np.hypot(ux, uy) + 1e-6
            ux, uy = ux / n, uy / n
            proj_r = (xs - cx) * ux + (ys - cy) * uy
            proj_t = -(xs - cx) * uy + (ys - cy) * ux
            er = (np.percentile(proj_r, 97) - np.percentile(proj_r, 3) + 1.0) / \
                 (np.percentile(proj_t, 97) - np.percentile(proj_t, 3) + 1.0)
            rows.append(dict(cand_id=r.cand_id,
                             img_dprime=dprime(J, m, halo) if halo.any() else np.nan,
                             band_contrast=float((J[halo].mean() - J[m].mean()) / bstd) if halo.any() else np.nan,
                             flat_dprime=dprime(F, m, halo) if halo.any() else np.nan,
                             elong_radial=float(er)))
        if (k + 1) % 50 == 0:
            log(f"  features {k+1} images")
    return pd.DataFrame(rows)


def exact_at(cand, gt, pcol, fold=None):
    sub = cand[(cand["conf"] >= 0.40) & (cand[pcol] >= 0.60)]
    g = gt
    if fold is not None:
        sub, g = sub[sub["fold"] == fold], gt[gt["fold"] == fold]
    pred = sub[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
    return score(g[["filament_id", "segmentation_rle"]], pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-only", action="store_true")
    a = ap.parse_args()
    import joblib
    feats = list(joblib.load(BUNDLE)["features"])
    cand = pd.read_parquet(OUT / "e115b_labelled_pool.parquet")
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    cand["fold"] = cand["stem"].map(fold_of).astype(int)
    gt["fold"] = gt["stem"].map(fold_of)
    gt = gt[gt["fold"].notna()].copy()
    gt["fold"] = gt["fold"].astype(int)

    fcache = OUT / "limb_relative_features.parquet"
    if fcache.exists():
        nf = pd.read_parquet(fcache)
        log(f"features from cache ({len(nf)} rows)")
    else:
        log(f"building limb-relative features for {len(cand)} candidates on {cand['stem'].nunique()} images")
        nf = build_features(cand)
        nf.to_parquet(fcache, index=False)
    cand = cand.merge(nf, on="cand_id", how="left")
    for n in NEW:
        cand[n] = cand[n].fillna(cand[n].median())
    log("  feature medians by band (r<=0.8 vs >0.8): " + "; ".join(
        f"{n} {cand.loc[cand['r_norm'] <= 0.8, n].median():.3f}/{cand.loc[cand['r_norm'] > 0.8, n].median():.3f}" for n in NEW))
    if a.build_only:
        return

    log("\nOOF selector, control (32 features):")
    cand["p_ctrl"] = oof_scores(cand, feats, cand["fold"].to_numpy())
    log("OOF selector, treatment (32 + 4 limb-relative):")
    cand["p_limb"] = oof_scores(cand, feats + NEW, cand["fold"].to_numpy())

    res = {}
    log(f"\n  {'fold':<5}{'ctrl PQ':>9}{'limb PQ':>9}{'Δ':>9}{'TP':>11}{'FP':>11}")
    d = []
    for f in range(5):
        c = exact_at(cand, gt, "p_ctrl", f)
        t = exact_at(cand, gt, "p_limb", f)
        d.append(t["pq"] - c["pq"])
        res[f"fold{f}"] = dict(ctrl=c["pq"], limb=t["pq"], delta=t["pq"] - c["pq"], tp=(c["tp"], t["tp"]), fp=(c["fp"], t["fp"]))
        log(f"  {f:<5}{c['pq']:>9.4f}{t['pq']:>9.4f}{t['pq']-c['pq']:>+9.4f}{c['tp']:>5}>{t['tp']:<5}{c['fp']:>5}>{t['fp']:<5}")
    dm, pos = float(np.mean(d)), sum(x > 0 for x in d)
    verdict = "GO (>= +0.002, >= 4/5): submission candidate, ask Talha" if (dm >= FLOOR and pos >= 4) else "NO GO"
    log(f"  mean Δ {dm:+.4f}, folds positive {pos}/5 -> {verdict}")

    # reach by radius band (mechanism check): matched GT rows per band under each selector
    Rg = []
    for rle in gt["segmentation_rle"]:
        m = decode_rle(rle).astype(np.uint8); ys, xs = np.nonzero(m)
        Rg.append(float(np.hypot(xs.mean() - DISK_C, ys.mean() - DISK_C) / DISK_R) if len(xs) else np.nan)
    gt["r"] = Rg
    log("\n  reach by radius band (rows matched / rows):")
    for pcol, name in (("p_ctrl", "ctrl"), ("p_limb", "limb")):
        sub = cand[(cand["conf"] >= 0.40) & (cand[pcol] >= 0.60)]
        pred = sub[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
        r = score(gt[["filament_id", "segmentation_rle"]], pred, return_details=True)
        hit = set(r["matches"]["gt_filament_id"])
        parts = []
        for lo, hi in [(0, 0.5), (0.5, 0.8), (0.8, 0.9), (0.9, 1.01)]:
            g = gt[(gt["r"] > lo) & (gt["r"] <= hi)]
            parts.append(f"({lo},{hi}] {100*g['filament_id'].isin(hit).mean():.1f}%")
        log(f"    {name}: " + "  ".join(parts) + f"   pooled PQ {r['pq']:.4f}")
        res[f"reach_{name}"] = parts

    # paired interval
    stems = np.array(sorted(cand["stem"].unique()))
    ncopy = gt.groupby("stem")["entry_id"].nunique().reindex(stems).fillna(1).to_numpy()
    parts = {}
    for pcol in ("p_ctrl", "p_limb"):
        sub = cand[(cand["conf"] >= 0.40) & (cand[pcol] >= 0.60)]
        st, r = per_image(sub, gt, stems, ncopy)
        tot = pq(st["tp"].sum(), st["fp"].sum(), st["fn"].sum(), st["iou"].sum())
        if abs(tot - r["pq"]) > 5e-4:
            log("  guard failed; no interval"); parts = None; break
        parts[pcol] = st
    if parts:
        rng = np.random.default_rng(0); n = len(stems); boot = np.empty(REPS)
        for i in range(REPS):
            idx = rng.integers(0, n, n)
            v = [pq(s["tp"][idx].sum(), s["fp"][idx].sum(), s["fn"][idx].sum(), s["iou"][idx].sum()) for s in (parts["p_ctrl"], parts["p_limb"])]
            boot[i] = v[1] - v[0]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        log(f"  paired image bootstrap: 95% CI [{lo:+.4f}, {hi:+.4f}], sd {boot.std():.4f}, P(Δ>0) {(boot > 0).mean():.3f}")
        res["interval"] = dict(ci=[float(lo), float(hi)], sd=float(boot.std()), p_gt0=float((boot > 0).mean()))
    res.update(mean_delta=dm, folds_positive=pos, verdict=verdict)
    json.dump(res, open(OUT / "limb_relative_features.json", "w"), indent=1, default=float)
    log(f"\nwrote {OUT/'limb_relative_features.json'}")


if __name__ == "__main__":
    main()
