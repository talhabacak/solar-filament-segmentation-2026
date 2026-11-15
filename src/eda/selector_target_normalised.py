"""SI-S1 (docs/32 §5.1) -- normalise the regressor target by the annotator copy count.

The deployed selector scores s(x) = P(match | x) * E[iou_sum | match, x]. iou_sum is
summed over annotator copies, so on a three-copy image the same mask earns up to
three times the target of a one-copy image, while its break-even cost in the PQ
denominator is 0.5 * n_copies. The value the decision actually needs is per copy:
iou_sum / n_copies. In the current pool the positive mean is 0.674 / 1.084 / 1.416
for 1 / 2 / 3 copies, i.e. 0.674 / 0.542 / 0.472 per copy -- the learned magnitude
grows with copies while the per-copy value falls.

Single variable: the same classifier, the same positive rows, the same 42 features,
recipe, inner-OOF CDF reference, folds and gate; only the regressor target changes.
The copy count is never an input, so nothing is needed at test time.

Usage: python src/eda/selector_target_normalised.py
"""
import sys, json, pathlib, numpy as np, pandas as pd, joblib
ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in ("src", "src/eda", "src/models"): sys.path.insert(0, str(ROOT / p))
from e115c_threshold_ci import make_cls, make_reg, per_image, pq
from extra_features import EXTRA
OUT = ROOT / "outputs" / "eda"
GATE = (0.40, 0.60); REPS, FLOOR, EQUIV = 4000, 0.003, 0.005

def run(a, feats, gt, fold_of, target):
    parts, kept_n = {}, 0
    for F in range(5):
        trn = a[a["fold"] != F]
        X, y, v = trn[feats].to_numpy(np.float64), trn["y"].to_numpy(), trn[target].to_numpy(np.float64)
        inner = np.zeros(len(trn))
        for g in [k for k in range(5) if k != F]:
            itr, msk = (trn["fold"] != g).to_numpy(), (trn["fold"] == g).to_numpy()
            c = make_cls().fit(X[itr], y[itr]); r = make_reg().fit(X[itr][y[itr] == 1], v[itr][y[itr] == 1])
            inner[msk] = c.predict_proba(X[msk])[:, 1] * np.clip(r.predict(X[msk]), 0, None)
        ref = np.sort(inner); cls, reg = make_cls().fit(X, y), make_reg().fit(X[y == 1], v[y == 1])
        te = a[a["fold"] == F].copy(); Z = te[feats].to_numpy(np.float64)
        te["p"] = np.searchsorted(ref, cls.predict_proba(Z)[:, 1] * np.clip(reg.predict(Z), 0, None), side="right") / len(ref)
        sub = te[(te["conf"] >= GATE[0]) & (te["p"] >= GATE[1])]; kept_n += len(sub)
        stems = np.array(sorted(s for s, f_ in fold_of.items() if f_ == F))     # manifest universe
        gF = gt[gt["stem"].map(fold_of) == F]
        ncopy = gF.groupby("stem")["entry_id"].nunique().reindex(stems).fillna(1).to_numpy()
        parts[F] = per_image(sub[["cand_id", "rle", "stem"]], gF, stems, ncopy)[0]
    return parts, kept_n

def main():
    feats = list(joblib.load(ROOT / "outputs/selector/selector_v5_shrunkblob5prodts.joblib")["features"]) + list(EXTRA)
    a = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    a = a.merge(pd.read_parquet(OUT / "extra_features_t020.parquet")[["cand_id"] + EXTRA], on="cand_id", how="left")
    folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    a["fold"] = a["stem"].map(fold_of).astype(int)
    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet"); gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    nc = gt.groupby("stem")["entry_id"].nunique()
    a["n_copies"] = a["stem"].map(nc).fillna(1).astype(float)
    a["iou_per_copy"] = a["iou_sum"] / a["n_copies"]
    pos = a[a["y"] == 1]
    print("positives, mean target by copy count:")
    print(pos.groupby("n_copies")[["iou_sum", "iou_per_copy"]].mean().round(4).to_string())
    res, parts = {}, {}
    for name, tgt in (("iou_sum (deployed)", "iou_sum"), ("iou_sum / n_copies (SI-S1)", "iou_per_copy")):
        p, kept = run(a, feats, gt, fold_of, tgt); parts[name] = p
        cat = lambda k: np.concatenate([p[F][k] for F in range(5)])
        val = pq(cat("tp").sum(), cat("fp").sum(), cat("fn").sum(), cat("iou").sum())
        res[name] = dict(pq=float(val), kept=kept, tp=int(cat("tp").sum()), fp=int(cat("fp").sum()), fn=int(cat("fn").sum()))
        print(f"{name:28s} kept {kept:5d}  PQ {val:.4f}  TP {res[name]['tp']} FP {res[name]['fp']} FN {res[name]['fn']}", flush=True)
    A, B = parts["iou_sum (deployed)"], parts["iou_sum / n_copies (SI-S1)"]
    cat = lambda P, k: np.concatenate([P[F][k] for F in range(5)])
    t0, f0, n0, i0 = cat(A, "tp"), cat(A, "fp"), cat(A, "fn"), cat(A, "iou"); t1, f1, n1, i1 = cat(B, "tp"), cat(B, "fp"), cat(B, "fn"), cat(B, "iou")
    d = pq(t1.sum(), f1.sum(), n1.sum(), i1.sum()) - pq(t0.sum(), f0.sum(), n0.sum(), i0.sum())
    rng = np.random.default_rng(0); n = len(t0); boot = np.empty(REPS)
    for i in range(REPS):
        idx = rng.integers(0, n, n)
        boot[i] = pq(t1[idx].sum(), f1[idx].sum(), n1[idx].sum(), i1[idx].sum()) - pq(t0[idx].sum(), f0[idx].sum(), n0[idx].sum(), i0[idx].sum())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    verdict = "PAYS" if d >= FLOOR else "HURTS" if d <= -FLOOR else ("TIED" if lo > -EQUIV and hi < EQUIV else "UNDERPOWERED")
    per_fold = [float(pq(A[F]["tp"].sum(), A[F]["fp"].sum(), A[F]["fn"].sum(), A[F]["iou"].sum())) for F in range(5)]
    per_fold_b = [float(pq(B[F]["tp"].sum(), B[F]["fp"].sum(), B[F]["fn"].sum(), B[F]["iou"].sum())) for F in range(5)]
    print(f"\nper fold deployed {np.round(per_fold, 4).tolist()}\nper fold SI-S1    {np.round(per_fold_b, 4).tolist()}")
    print(f"delta {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  P(delta>0) {(boot > 0).mean():.3f}  folds positive {sum(b > a_ for a_, b in zip(per_fold, per_fold_b))}/5 -> {verdict}")
    res.update(delta=float(d), ci=[float(lo), float(hi)], verdict=verdict, per_fold=per_fold, per_fold_b=per_fold_b)
    json.dump(res, open(OUT / "selector_target_normalised.json", "w"), indent=1)
if __name__ == "__main__":
    main()
