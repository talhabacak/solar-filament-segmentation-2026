"""E125 -- the two feature families that pointed the right way, together.

Seven selector feature families were measured under the E120 protocol and all
were TIED, but two moved in the same direction on 4 of 5 folds:

    photometry (E124)   +0.0017  CI [-0.0007, +0.0040]  P(>0) 0.92
    orientation (E123)  +0.0010  CI [-0.0014, +0.0034]  P(>0) 0.77

This is the union of those two, and it is declared for what it is: an eighth
test on the same pool, chosen after seeing which families looked best, so it
carries a selection risk that a first-look test does not. It is therefore held
to the stricter bar, stated before the run:

    ADOPT only if delta >= +0.003 AND the 95% interval excludes zero AND at
    least 4 of 5 folds are positive. Anything else is recorded as TIED and the
    selector feature axis is closed for good.

Features are read from the caches the two experiments already wrote, so nothing
is recomputed.

Usage: python src/eda/combined_feature_families.py
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
POLICY, REPS, FLOOR = (0.40, 0.60), 4000, 0.003
PHOTO = ["img_contrast", "dark_core", "fg_p05", "ring_med"]
ORI = ["coh_in", "coh_ring", "tex_align", "ring_align", "grad_norm", "aniso_ratio"]


def log(*a):
    print(*a, flush=True)


def main():
    feats = list(joblib.load(BUNDLE)["features"])
    a = pd.read_parquet(A_POOL)
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    a["fold"] = a["stem"].map(fold_of).astype(int)
    a = a.merge(pd.read_parquet(OUT / "shape_photometry_features.parquet")[["cand_id"] + PHOTO], on="cand_id", how="left")
    a = a.merge(pd.read_parquet(OUT / "orientation_context_features.parquet")[["cand_id"] + ORI], on="cand_id", how="left")
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt["fold"] = gt["stem"].map(fold_of)
    gt = gt[gt["fold"].notna()].copy()
    gt["fold"] = gt["fold"].astype(int)
    stems = np.array(sorted(set(a["stem"])))
    gts = gt[gt["stem"].isin(stems)]
    ncopy = gts.groupby("stem")["entry_id"].nunique().reindex(stems).fillna(1).to_numpy()
    log(f"pool {len(a)} rows; adding {len(PHOTO) + len(ORI)} features to {len(feats)}")

    def exact(col, fold=None):
        sub = a[(a["conf"] >= POLICY[0]) & (a[col] >= POLICY[1])]
        g = gts
        if fold is not None:
            sub, g = sub[sub["fold"] == fold], gts[gts["fold"] == fold]
        pred = sub[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
        return score(g[["filament_id", "segmentation_rle"]], pred)

    res, parts = {}, {}
    for name, fl in (("base", feats), ("photo+orient", feats + PHOTO + ORI)):
        col = f"p_{name}"
        a[col] = oof_scores(a, fl, a["fold"].to_numpy())
        per = [exact(col, f)["pq"] for f in range(5)]
        r = exact(col)
        sub = a[(a["conf"] >= POLICY[0]) & (a[col] >= POLICY[1])]
        st, _ = per_image(sub, gts, stems, ncopy)
        parts[name] = st
        res[name] = dict(pq=r["pq"], tp=r["tp"], fp=r["fp"], fn=r["fn"], kept=int(len(sub)), per_fold=per)
        log(f"{name:13s} PQ {r['pq']:.4f} kept {len(sub)} TP {r['tp']} FP {r['fp']} FN {r['fn']} | per fold "
            + " ".join(f"{x:.4f}" for x in per))
    rng = np.random.default_rng(0)
    n = len(stems)
    boot = np.empty(REPS)
    for i in range(REPS):
        idx = rng.integers(0, n, n)
        v = [pq(s["tp"][idx].sum(), s["fp"][idx].sum(), s["fn"][idx].sum(), s["iou"][idx].sum())
             for s in (parts["base"], parts["photo+orient"])]
        boot[i] = v[1] - v[0]
    d = res["photo+orient"]["pq"] - res["base"]["pq"]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    dpf = [res["photo+orient"]["per_fold"][f] - res["base"]["per_fold"][f] for f in range(5)]
    ok = d >= FLOOR and lo > 0 and sum(x > 0 for x in dpf) >= 4
    log(f"\ndelta {d:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  P(>0) {(boot > 0).mean():.3f}  folds+ {sum(x > 0 for x in dpf)}/5")
    log("VERDICT: " + ("ADOPT (clears the stricter post-hoc bar)" if ok else
                       "TIED -> selector feature axis closed"))
    res["delta"] = dict(value=float(d), ci=[float(lo), float(hi)], p_gt0=float((boot > 0).mean()),
                        folds=dpf, adopt=bool(ok))
    json.dump(res, open(OUT / "combined_feature_families.json", "w"), indent=1, default=float)
    log(f"wrote {OUT / 'combined_feature_families.json'}")


if __name__ == "__main__":
    main()
