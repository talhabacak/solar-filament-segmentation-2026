"""Image-quality features in the selector (S007 step 2; E120 protocol).

The precondition (image_quality_residual.py) found weak per-image signal
(max partial Spearman 0.10; grouped-OOF R^2 gain +0.04/+0.05 on 3-4 of 5
folds against a baseline that is itself below zero). The pre-registered gate
was formally met, so the single-variable test follows: the six image-level
quality features (Laplacian variance, high-frequency energy, JPEG block
excess, non-radial low-frequency residual, off-limb noise, mean intensity)
are appended to the 32 production features and the E115b honest OOF selector
is refit on the same pool. Exact PQ at (0.40, 0.60), per fold and pooled,
paired image bootstrap; bands as E120: pays >= +0.003, hurts <= -0.003, TIED
if the 95% interval is inside +-0.005.

Usage (on the box, after image_quality_residual.py): python src/eda/quality_selector_features.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "eda"))
from filament.pq import score  # noqa: E402
from e115c_threshold_ci import per_image, pq, oof_scores  # noqa: E402

OUT = ROOT / "outputs" / "eda"
BUNDLE = ROOT / "outputs" / "selector" / "selector_v5_shrunkblob5prodts.joblib"
A_POOL = OUT / "e098d_fused_pool_all_blob5_rebuild_labelled.parquet"
QCOLS = ["lap_var", "hf_energy", "jpeg8", "lowfreq", "limb_noise", "mean_int"]
POLICY, REPS, FLOOR, EQUIV = (0.40, 0.60), 4000, 0.003, 0.005


def log(*a):
    print(*a, flush=True)


def exact(cand, gt, col, fold=None):
    sub = cand[(cand["conf"] >= POLICY[0]) & (cand[col] >= POLICY[1])]
    g = gt
    if fold is not None:
        sub, g = sub[sub["fold"] == fold], gt[gt["fold"] == fold]
    pred = sub[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
    return score(g[["filament_id", "segmentation_rle"]], pred)


def main():
    feats = list(joblib.load(BUNDLE)["features"])
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    gt["fold"] = gt["stem"].map(fold_of)
    gt = gt[gt["fold"].notna()].copy()
    gt["fold"] = gt["fold"].astype(int)
    a = pd.read_parquet(A_POOL)
    a["fold"] = a["stem"].map(fold_of).astype(int)
    q = pd.read_parquet(OUT / "image_quality_residual.parquet")[["stem"] + QCOLS]
    a = a.merge(q, on="stem", how="left")
    log(f"pool {len(a)} rows; quality features joined, missing rows {int(a[QCOLS].isna().any(axis=1).sum())}")
    stems = np.array(sorted(set(a["stem"])))
    gts = gt[gt["stem"].isin(stems)]
    ncopy = gts.groupby("stem")["entry_id"].nunique().reindex(stems).fillna(1).to_numpy()

    a["p_base"] = oof_scores(a, feats, a["fold"].to_numpy())
    a["p_qual"] = oof_scores(a, feats + QCOLS, a["fold"].to_numpy())
    res = {}
    rows = {}
    for name, col in (("base", "p_base"), ("quality", "p_qual")):
        per = [exact(a, gts, col, fold=f)["pq"] for f in range(5)]
        r = exact(a, gts, col)
        rows[name] = dict(pq=r["pq"], tp=r["tp"], fp=r["fp"], fn=r["fn"], per_fold=per)
        log(f"{name:8s} pooled PQ {r['pq']:.4f} TP {r['tp']} FP {r['fp']} FN {r['fn']}  per fold " + " ".join(f"{x:.4f}" for x in per))
    d = [rows["quality"]["per_fold"][i] - rows["base"]["per_fold"][i] for i in range(5)]
    log(f"delta quality - base per fold " + " ".join(f"{x:+.4f}" for x in d) + f"  mean {np.mean(d):+.4f}, folds+ {sum(x > 0 for x in d)}/5")
    parts = {}
    for name, col in (("base", "p_base"), ("quality", "p_qual")):
        sub = a[(a["conf"] >= POLICY[0]) & (a[col] >= POLICY[1])]
        st, r = per_image(sub, gts, stems, ncopy)
        tot = pq(st["tp"].sum(), st["fp"].sum(), st["fn"].sum(), st["iou"].sum())
        log(f"  guard {name}: decomposed {tot:.4f} vs exact {r['pq']:.4f}")
        parts[name] = st
    rng = np.random.default_rng(0)
    n = len(stems)
    boot = np.empty(REPS)
    for i in range(REPS):
        idx = rng.integers(0, n, n)
        v = [pq(s["tp"][idx].sum(), s["fp"][idx].sum(), s["fn"][idx].sum(), s["iou"][idx].sum()) for s in (parts["base"], parts["quality"])]
        boot[i] = v[1] - v[0]
    obs = rows["quality"]["pq"] - rows["base"]["pq"]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    verdict = ("PAYS" if obs >= FLOOR else "HURTS" if obs <= -FLOOR else ("TIED" if lo > -EQUIV and hi < EQUIV else "UNDERPOWERED"))
    log(f"pooled delta {obs:+.4f}; bootstrap {REPS} over {n} images: 95% CI [{lo:+.4f}, {hi:+.4f}], sd {boot.std():.4f}, P(delta>0) {(boot > 0).mean():.3f}")
    log(f"VERDICT: {verdict}")
    res = dict(base=rows["base"], quality=rows["quality"], delta_folds=d, pooled=float(obs), ci=[float(lo), float(hi)], verdict=verdict)
    json.dump(res, open(OUT / "quality_selector_features.json", "w"), indent=1, default=float)
    log(f"wrote {OUT / 'quality_selector_features.json'}")


if __name__ == "__main__":
    main()
