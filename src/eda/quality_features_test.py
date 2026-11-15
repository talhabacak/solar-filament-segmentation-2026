"""E206 -- the image-quality block in the selector.

E205 measured four per-frame quantities that need only the image: the limb 10-90% transition
width (seeing, filament-independent because the limb is a step in truth), high-frequency energy
on the disk, disk mean and disk sd. Only `hf` predicts our PQ, and it predicts it well: pooled PQ
falls 0.464 -> 0.416 across its quartiles, Spearman -0.140.

E205b located the loss. Across hf quartiles SQ is flat (0.683 -> 0.676) and FP/GT is flat
(0.210 -> 0.212); the whole thing is recall (0.623 -> 0.540). By outcome, the rise is selector
+0.030, shape +0.021, never +0.012 -- so about half the loss is the gate discarding real
candidates in busy frames, not the model failing to propose them.

And it is new information: partialling out candidate count, GT count and the selector's own
density features (n_cands_img, img_cand_density, img_sem_area_frac) leaves rho -0.098 (p 0.010),
and controlling for the other quality axes as well strengthens it to -0.150.

So the selector is missing a covariate it can act on. This adds the block and reads it under the
deployed protocol: bagged x5, inner out-of-fold quantile map, gate (0.40, 0.60), exact scorer,
full folds_v1 scope -- overall AND on the Q4 slice the block is supposed to help.

Usage: python src/eda/quality_features_test.py [--bag 5] [--feats hf|all]
"""
import argparse, json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/eda", "src/models"):
    sys.path.insert(0, str(ROOT / p))
import selector_image_bagging as B  # noqa: E402
from e115c_threshold_ci import pq  # noqa: E402
from extra_features import EXTRA  # noqa: E402

OUT = ROOT / "outputs" / "eda"
QUAL = {"hf": ["hf"], "all": ["hf", "limb_width", "limb_ok", "disk_sd", "disk_mean"],
        "regime": None,      # the 20 frame-relative features of E207, loaded from their parquet
        "both": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", type=int, default=5)
    ap.add_argument("--feats", default="hf", choices=list(QUAL))
    a = ap.parse_args()
    base = list(joblib.load(ROOT / "outputs/selector/selector_v5_shrunk_t020pox.joblib")["features"])
    add = QUAL[a.feats]
    pool = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    pool = pool.merge(pd.read_parquet(OUT / "extra_features_t020.parquet")[["cand_id"] + EXTRA], on="cand_id")
    q = pd.read_parquet(OUT / "image_quality.parquet")
    q = q[q.split == "train"][["stem"] + QUAL["all"]]
    pool = pool.merge(q, on="stem", how="left", validate="many_to_one")
    if a.feats in ("regime", "both"):
        from regime_relative_features import REG
        pool = pool.merge(pd.read_parquet(OUT / "regime_features_t020.parquet"), on="cand_id",
                          validate="one_to_one")
        add = list(REG) + (QUAL["all"] if a.feats == "both" else [])
    assert not pool[base].isna().any().any(), "taban ozellikleri eksik"
    # limb_width is NaN on the 15 frames whose limb step is too weak to fit; HistGradientBoosting
    # takes NaN natively and limb_ok carries the fact, so these are left as they are.
    for c in add:
        n = int(pool[c].isna().sum())
        if n:
            print(f"  {c}: {n} adayda NaN (HGB dogal olarak isliyor)")
    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    pool["fold"] = pool["stem"].map(fold_of).astype(int)

    res, parts = {}, {}
    for name, feats in (("taban 42", base), (f"+{a.feats} ({len(add)})", base + add)):
        P, kept = B.run(pool, feats, gt, fold_of, bool(a.bag))
        parts[name] = P
        cat = lambda k: np.concatenate([P[F][k] for F in range(5)])
        per = [float(pq(*[P[F][k].sum() for k in ("tp", "fp", "fn", "iou")])) for F in range(5)]
        val = float(pq(cat("tp").sum(), cat("fp").sum(), cat("fn").sum(), cat("iou").sum()))
        res[name] = dict(pq=val, per_fold=per, kept=int(kept))
        print(f"{name:16s} tutulan {kept:5d}  PQ {val:.4f}  fold {[round(x,4) for x in per]}", flush=True)

    A, Bb = parts["taban 42"], parts[f"+{a.feats} ({len(add)})"]
    cat = lambda P, k: np.concatenate([P[F][k] for F in range(5)])
    dpf = [res[f"+{a.feats} ({len(add)})"]["per_fold"][i] - res["taban 42"]["per_fold"][i] for i in range(5)]
    d = res[f"+{a.feats} ({len(add)})"]["pq"] - res["taban 42"]["pq"]
    rng = np.random.default_rng(0)
    ta, tb = {k: cat(A, k) for k in ("tp","fp","fn","iou")}, {k: cat(Bb, k) for k in ("tp","fp","fn","iou")}
    n = len(ta["tp"]); boot = np.empty(4000)
    for i in range(4000):
        idx = rng.integers(0, n, n)
        boot[i] = (pq(tb["tp"][idx].sum(), tb["fp"][idx].sum(), tb["fn"][idx].sum(), tb["iou"][idx].sum())
                   - pq(ta["tp"][idx].sum(), ta["fp"][idx].sum(), ta["fn"][idx].sum(), ta["iou"][idx].sum()))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\nfark {d:+.4f}  fold ort {np.mean(dpf):+.4f}  fold basina {[round(x,4) for x in dpf]}")
    print(f"  CI [{lo:+.4f}, {hi:+.4f}]  P(d>0) {(boot>0).mean():.3f}  +fold {sum(1 for x in dpf if x>0)}/5")

    # The slice the block is aimed at. selector_image_bagging.run scores fold by fold with
    # `sorted(stems of fold F)`, over ALL folds_v1 stems (707), and concatenates in fold order --
    # not the pool's 703 in global order. Rebuild the covariate in exactly that order.
    order = np.concatenate([np.array(sorted(s_ for s_, f_ in fold_of.items() if f_ == F))
                            for F in range(5)])
    hf = q.set_index("stem")["hf"].reindex(order)
    assert len(hf) == len(ta["tp"]), f"{len(hf)} vs {len(ta['tp'])}"
    qu = pd.qcut(hf, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    print(f"\nhf ceyregine gore (hedef dilim Q4; NaN {int(hf.isna().sum())}):")
    for lab in ["Q1", "Q2", "Q3", "Q4"]:
        m = (qu == lab).to_numpy()
        va = pq(ta["tp"][m].sum(), ta["fp"][m].sum(), ta["fn"][m].sum(), ta["iou"][m].sum())
        vb = pq(tb["tp"][m].sum(), tb["fp"][m].sum(), tb["fn"][m].sum(), tb["iou"][m].sum())
        rec_a = ta["tp"][m].sum() / max(ta["tp"][m].sum() + ta["fn"][m].sum(), 1)
        rec_b = tb["tp"][m].sum() / max(tb["tp"][m].sum() + tb["fn"][m].sum(), 1)
        print(f"  {lab}  n={int(m.sum()):3d}  taban {va:.4f}  +{a.feats} {vb:.4f}   {vb-va:+.4f}"
              f"   recall {rec_a:.4f} -> {rec_b:.4f} ({rec_b-rec_a:+.4f})")
    res["delta"] = dict(pooled=d, mean_per_fold=float(np.mean(dpf)), ci=[float(lo), float(hi)],
                        p=float((boot > 0).mean()), per_fold=dpf)
    json.dump(res, open(OUT / f"quality_features_{a.feats}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
