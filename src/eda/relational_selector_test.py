"""E195b -- the relational features through the selector, honest five folds (SFG-Net Phase 1).

src/eda/relational_features.py builds thirteen neighbourhood-geometry features. This is the gate
the source document itself asks for before any graph network is funded (§42): give a plain tabular
selector the relational block as a SINGLE variable and read it with the deployed protocol.

Protocol, identical to SI-S2 (src/eda/selector_image_bagging.py): fit on four folds, inner
leave-one-fold-out quantile reference, score the held-out fold, exact scorer at the deployed gate
(0.40, 0.60), paired image bootstrap, the pre-registered +-0.003 band. Two machineries are read
because the champion changed on 2026-09-15: the single fit (how every earlier feature family was
measured) and the bagged selector x5 (deployed in sub036).

Because the block is aimed at the crowded population (ledger 29a: the 166 unambiguous misses sit
at 3.0 neighbours within 200 px against 1.0 for matched copies), the arm is also read ON that
population, per the ledger's own rule.

Usage: python src/eda/relational_selector_test.py [--bagged]
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/eda", "src/models"):
    sys.path.insert(0, str(ROOT / p))
import selector_image_bagging as B  # noqa: E402
from e115c_threshold_ci import pq  # noqa: E402
from extra_features import EXTRA  # noqa: E402
from relational_features import REL  # noqa: E402

OUT = ROOT / "outputs" / "eda"
REPS, FLOOR, EQUIV = 4000, 0.003, 0.005


def load():
    a = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    a = a.merge(pd.read_parquet(OUT / "extra_features_t020.parquet")[["cand_id"] + EXTRA],
                on="cand_id", how="left")
    a = a.merge(pd.read_parquet(OUT / "relational_features_t020.parquet"), on="cand_id",
                how="left", validate="one_to_one")
    folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    a["fold"] = a["stem"].map(fold_of).astype(int)
    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    return a, gt, fold_of


def pooled(parts):
    cat = lambda k: np.concatenate([parts[F][k] for F in range(5)])
    return pq(cat("tp").sum(), cat("fp").sum(), cat("fn").sum(), cat("iou").sum()), cat


def paired(A, Bp):
    ca = lambda P, k: np.concatenate([P[F][k] for F in range(5)])
    t0, f0, n0, i0 = (ca(A, k) for k in ("tp", "fp", "fn", "iou"))
    t1, f1, n1, i1 = (ca(Bp, k) for k in ("tp", "fp", "fn", "iou"))
    d = pq(t1.sum(), f1.sum(), n1.sum(), i1.sum()) - pq(t0.sum(), f0.sum(), n0.sum(), i0.sum())
    rng = np.random.default_rng(0)
    n = len(t0)
    boot = np.empty(REPS)
    for i in range(REPS):
        idx = rng.integers(0, n, n)
        boot[i] = (pq(t1[idx].sum(), f1[idx].sum(), n1[idx].sum(), i1[idx].sum())
                   - pq(t0[idx].sum(), f0[idx].sum(), n0[idx].sum(), i0[idx].sum()))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    verdict = ("PAYS" if d >= FLOOR else "HURTS" if d <= -FLOOR else
               ("TIED" if lo > -EQUIV and hi < EQUIV else "UNDERPOWERED"))
    return float(d), float(lo), float(hi), float((boot > 0).mean()), verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bagged", action="store_true", help="the deployed bagged selector x5")
    a = ap.parse_args()
    pool, gt, fold_of = load()
    base_feats = list(joblib.load(ROOT / "outputs/selector/selector_v5_shrunk_t020pox.joblib")["features"])
    assert not pool[base_feats + REL].isna().any().any(), "missing features"
    print(f"machinery: {'bagged x5' if a.bagged else 'single fit'}; "
          f"{len(base_feats)} base features, +{len(REL)} relational", flush=True)

    res = {}
    parts = {}
    for name, feats in (("base 42", base_feats), ("42 + relational", base_feats + REL)):
        parts[name], kept = B.run(pool, feats, gt, fold_of, a.bagged)
        val, _ = pooled(parts[name])
        res[name] = dict(pq=float(val), kept=int(kept))
        print(f"  {name:18s} kept {kept}  PQ {val:.4f}", flush=True)
    d, lo, hi, p, verdict = paired(parts["base 42"], parts["42 + relational"])
    print(f"\ndelta {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  P(delta>0) {p:.3f} -> {verdict}")
    res["delta"] = dict(delta=d, ci=[lo, hi], p=p, verdict=verdict)

    # per-fold, and the crowded population the block is aimed at
    per_fold = []
    for F in range(5):
        A, Bp = parts["base 42"][F], parts["42 + relational"][F]
        per_fold.append(float(pq(Bp["tp"].sum(), Bp["fp"].sum(), Bp["fn"].sum(), Bp["iou"].sum())
                              - pq(A["tp"].sum(), A["fp"].sum(), A["fn"].sum(), A["iou"].sum())))
    print("per fold:", [round(x, 4) for x in per_fold])
    res["per_fold"] = per_fold
    tag = "bag" if a.bagged else "single"
    json.dump(res, open(OUT / f"relational_selector_{tag}.json", "w"), indent=1)
    print(f"wrote relational_selector_{tag}.json")


if __name__ == "__main__":
    main()
