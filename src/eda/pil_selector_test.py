"""E160 -- the cheap magnetogram test: nine PIL features into the selector, single variable.

The magnetogram is information the H-alpha frame does not carry, which makes it the
one exception the annotator-ceiling argument (E152) allows: that result closed
every axis that REORDERS existing information, and this adds some.

E159b established that the signal is real but modest. With the field smoothed to
sigma 8 and a 1 G floor, 6.5% of ground-truth filament pixels lie within 6 px of a
polarity inversion line against 1.2% of random same-size patches, a 5.4x
enrichment; registration is verified in rotation (E154c) and translation (E159b).
The hypothesis that the never-proposed filaments sit on stronger PILs than the
found ones got no support (53.0 vs 51.4 px), so this is not expected to open
candidate generation. It CAN still sharpen selection, and that is what this asks.

Protocol is E151b's, unchanged: the same pool, masks, gate and two-stage GBDT
recipe, fitted on 42 features and on 42 + 9 (distance to the nearest PIL, minimum
distance, fraction of the mask in the PIL band, alignment of the mask's axis with
the PIL tangent as a director, |B| in the mask, field gradient, field on either
side, PIL strength). Out of fold, exact scorer, paired image bootstrap.

Coverage: features exist only for images with a registered magnetogram; elsewhere
they are NaN, which HistGradientBoosting treats as its own branch. So a run on
partial coverage is a valid but diluted read, and the result is reported with the
coverage next to it. The feature cache is keyed by candidate and grows with the
fetch, so re-running after more frames arrive only computes the new ones.

Usage: python src/eda/pil_selector_test.py
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/eda", "src/models"):
    sys.path.insert(0, str(ROOT / p))
from filament.rle import decode_rle  # noqa: E402
from e115c_threshold_ci import pq  # noqa: E402
from extra_features import EXTRA  # noqa: E402
from magnetogram_features import load_registered, pil_maps, mask_features, PIL_FEATS  # noqa: E402
from patch_selector_eval import run_selector  # noqa: E402

OUT = ROOT / "outputs" / "eda"
GATE = (0.40, 0.60)
REPS, FLOOR, EQUIV = 4000, 0.003, 0.005
CACHE = OUT / "pil_features_cache.parquet"


def log(*a):
    print(*a, flush=True)


def build_features(a):
    have = pd.read_parquet(CACHE) if CACHE.exists() else pd.DataFrame(columns=["cand_id"])
    done = set(have["cand_id"])
    reg = {p.stem.replace("reg_", "") for p in (ROOT / "outputs" / "magnetograms").glob("reg_*.npy")}
    todo = a[a["stem"].isin(reg) & ~a["cand_id"].isin(done)]
    log(f"{len(reg)} registered images; {len(done)} candidates cached, {len(todo)} to compute")
    rows, t0 = [], time.time()
    for i, (stem, gg) in enumerate(todo.groupby("stem")):
        maps = pil_maps(load_registered(stem))
        for cid, rle in zip(gg["cand_id"], gg["rle"]):
            f = mask_features(decode_rle(rle).astype(bool), maps)
            f["cand_id"] = cid
            rows.append(f)
        if (i + 1) % 25 == 0:
            log(f"  {i + 1} images, {len(rows)} candidates ({time.time() - t0:.0f}s)")
    if rows:
        new = pd.DataFrame(rows)
        have = pd.concat([have, new], ignore_index=True) if len(have) else new
        have.to_parquet(CACHE, index=False)
    return have


def main():
    base_feats = list(joblib.load(ROOT / "outputs/selector/selector_v5_shrunkblob5prodts.joblib")["features"]) + list(EXTRA)
    a = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    a = a.merge(pd.read_parquet(OUT / "extra_features_t020.parquet")[["cand_id"] + EXTRA], on="cand_id", how="left")
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    a["fold"] = a["stem"].map(fold_of).astype(int)

    feats = build_features(a)
    a = a.merge(feats[["cand_id"] + PIL_FEATS], on="cand_id", how="left")
    cov = a["pil_dist_mean"].notna().mean()
    log(f"coverage: {cov:.1%} of pool candidates carry magnetic features")

    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt = gt[gt["stem"].isin(set(a["stem"]))]
    stems_by_fold, ncopy_by_fold = {}, {}
    for F in range(5):
        s = np.array(sorted(set(a.loc[a["fold"] == F, "stem"])))
        stems_by_fold[F] = s
        gF = gt[gt["stem"].map(fold_of) == F]
        ncopy_by_fold[F] = gF.groupby("stem")["entry_id"].nunique().reindex(s).fillna(1).to_numpy()

    # a quick read of the features' own discriminative power on covered candidates
    from sklearn.metrics import roc_auc_score
    c = a[a["pil_dist_mean"].notna() & (a["conf"] >= GATE[0])]
    log(f"on {len(c)} covered gated candidates ({c['y'].mean():.1%} true), single-feature AUC:")
    for f in PIL_FEATS:
        v = c[f].fillna(c[f].median())
        auc = roc_auc_score(c["y"], v)
        log(f"  {f:14s} {max(auc, 1 - auc):.3f}")

    parts, res = {}, {}
    for name, fl in (("42 features", base_feats), ("51 features (+PIL)", base_feats + PIL_FEATS)):
        p, kept = run_selector(a, fl, gt, fold_of, stems_by_fold, ncopy_by_fold)
        parts[name] = p
        cat = lambda k: np.concatenate([p[F][k] for F in range(5)])
        val = pq(cat("tp").sum(), cat("fp").sum(), cat("fn").sum(), cat("iou").sum())
        res[name] = dict(pq=float(val), kept=kept)
        log(f"{name:20s} kept {kept:5d}  PQ {val:.4f}  TP {int(cat('tp').sum())} FP {int(cat('fp').sum())}")
    A, B = parts["42 features"], parts["51 features (+PIL)"]
    cat = lambda P, k: np.concatenate([P[F][k] for F in range(5)])
    t0, f0, n0, i0 = cat(A, "tp"), cat(A, "fp"), cat(A, "fn"), cat(A, "iou")
    t1, f1, n1, i1 = cat(B, "tp"), cat(B, "fp"), cat(B, "fn"), cat(B, "iou")
    d = pq(t1.sum(), f1.sum(), n1.sum(), i1.sum()) - pq(t0.sum(), f0.sum(), n0.sum(), i0.sum())
    rng = np.random.default_rng(0)
    n = len(t0)
    boot = np.empty(REPS)
    for i in range(REPS):
        idx = rng.integers(0, n, n)
        boot[i] = (pq(t1[idx].sum(), f1[idx].sum(), n1[idx].sum(), i1[idx].sum())
                   - pq(t0[idx].sum(), f0[idx].sum(), n0[idx].sum(), i0[idx].sum()))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    verdict = ("PAYS" if d >= FLOOR else "HURTS" if d <= -FLOOR
               else ("TIED" if lo > -EQUIV and hi < EQUIV else "UNDERPOWERED"))
    log(f"\ndelta {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  P(delta>0) {(boot > 0).mean():.3f} "
        f"-> {verdict}   (coverage {cov:.0%})")
    res.update(delta=float(d), ci=[float(lo), float(hi)], verdict=verdict, coverage=float(cov))
    json.dump(res, open(OUT / "pil_selector_test.json", "w"), indent=1, default=float)
    log(f"wrote {OUT / 'pil_selector_test.json'}")


if __name__ == "__main__":
    main()
