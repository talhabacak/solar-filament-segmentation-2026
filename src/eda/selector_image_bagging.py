"""SI-S2 (docs/32 §5.2) -- real bagging of the selector, by image bootstrap.

E122's "bag5" arm refitted the same HistGradientBoosting pair on the SAME rows with
five seeds and got byte-identical scores (AUC 0.90668, PQ 0.43613, TP/FP identical):
the learner is deterministic given the data, so seeds create no diversity and that
experiment said nothing about ensembling the selector. This is the experiment it
was taken for: five models, each fitted on a bootstrap sample of IMAGES (all of an
image's candidates and annotator copies stay together), scores averaged, everything
else the deployed recipe.

Usage: python src/eda/selector_image_bagging.py
"""
import sys, json, pathlib, numpy as np, pandas as pd, joblib
ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in ("src", "src/eda", "src/models"): sys.path.insert(0, str(ROOT / p))
from e115c_threshold_ci import make_cls, make_reg, per_image, pq
from extra_features import EXTRA
OUT = ROOT / "outputs" / "eda"
GATE = (0.40, 0.60); REPS, FLOOR, EQUIV = 4000, 0.003, 0.005; NBAG = 5

def fit_pair(trn, feats, seed=None):
    X, y, v = trn[feats].to_numpy(np.float64), trn["y"].to_numpy(), trn["iou_sum"].to_numpy(np.float64)
    return make_cls().fit(X, y), make_reg().fit(X[y == 1], v[y == 1])

def raw_score(models, Z):
    return np.mean([c.predict_proba(Z)[:, 1] * np.clip(r.predict(Z), 0, None) for c, r in models], axis=0)

def run(a, feats, gt, fold_of, bag):
    parts, kept_n = {}, 0
    rng = np.random.default_rng(0)
    for F in range(5):
        trn = a[a["fold"] != F]
        def models_for(sub):
            if not bag: return [fit_pair(sub, feats)]
            stems = sub["stem"].unique(); out = []
            for b in range(NBAG):
                pick = rng.choice(stems, len(stems), replace=True)
                boot = pd.concat([sub[sub["stem"] == s] for s in pick], ignore_index=True)
                out.append(fit_pair(boot, feats))
            return out
        inner = np.zeros(len(trn))
        for g in [k for k in range(5) if k != F]:
            itr, msk = (trn["fold"] != g).to_numpy(), (trn["fold"] == g).to_numpy()
            inner[msk] = raw_score(models_for(trn[itr]), trn[feats].to_numpy(np.float64)[msk])
        ref = np.sort(inner); models = models_for(trn)
        te = a[a["fold"] == F].copy()
        te["p"] = np.searchsorted(ref, raw_score(models, te[feats].to_numpy(np.float64)), side="right") / len(ref)
        sub = te[(te["conf"] >= GATE[0]) & (te["p"] >= GATE[1])]; kept_n += len(sub)
        stems = np.array(sorted(s for s, f_ in fold_of.items() if f_ == F))
        gF = gt[gt["stem"].map(fold_of) == F]
        ncopy = gF.groupby("stem")["entry_id"].nunique().reindex(stems).fillna(1).to_numpy()
        parts[F] = per_image(sub[["cand_id", "rle", "stem"]], gF, stems, ncopy)[0]
        print(f"  fold {F} {'bagged' if bag else 'single'}: kept {len(sub)}", flush=True)
    return parts, kept_n

def main():
    feats = list(joblib.load(ROOT / "outputs/selector/selector_v5_shrunkblob5prodts.joblib")["features"]) + list(EXTRA)
    a = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    a = a.merge(pd.read_parquet(OUT / "extra_features_t020.parquet")[["cand_id"] + EXTRA], on="cand_id", how="left")
    folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    a["fold"] = a["stem"].map(fold_of).astype(int)
    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet"); gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    res, parts = {}, {}
    for name, bag in (("single (deployed)", False), (f"image-bagged x{NBAG}", True)):
        p, kept = run(a, feats, gt, fold_of, bag); parts[name] = p
        cat = lambda k: np.concatenate([p[F][k] for F in range(5)])
        val = pq(cat("tp").sum(), cat("fp").sum(), cat("fn").sum(), cat("iou").sum())
        res[name] = dict(pq=float(val), kept=kept); print(f"{name:22s} kept {kept}  PQ {val:.4f}", flush=True)
    A, B = parts["single (deployed)"], parts[f"image-bagged x{NBAG}"]
    cat = lambda P, k: np.concatenate([P[F][k] for F in range(5)])
    t0, f0, n0, i0 = cat(A, "tp"), cat(A, "fp"), cat(A, "fn"), cat(A, "iou"); t1, f1, n1, i1 = cat(B, "tp"), cat(B, "fp"), cat(B, "fn"), cat(B, "iou")
    d = pq(t1.sum(), f1.sum(), n1.sum(), i1.sum()) - pq(t0.sum(), f0.sum(), n0.sum(), i0.sum())
    rng = np.random.default_rng(0); n = len(t0); boot = np.empty(REPS)
    for i in range(REPS):
        idx = rng.integers(0, n, n)
        boot[i] = pq(t1[idx].sum(), f1[idx].sum(), n1[idx].sum(), i1[idx].sum()) - pq(t0[idx].sum(), f0[idx].sum(), n0[idx].sum(), i0[idx].sum())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    verdict = "PAYS" if d >= FLOOR else "HURTS" if d <= -FLOOR else ("TIED" if lo > -EQUIV and hi < EQUIV else "UNDERPOWERED")
    print(f"delta {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  P(delta>0) {(boot > 0).mean():.3f} -> {verdict}")
    res.update(delta=float(d), ci=[float(lo), float(hi)], verdict=verdict)
    json.dump(res, open(OUT / "selector_image_bagging.json", "w"), indent=1)
if __name__ == "__main__":
    main()
