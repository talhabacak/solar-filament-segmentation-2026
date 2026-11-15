"""E199 -- extraction threshold below 0.20, read the way a distribution change must be read.

The sweep in extraction_schemes.json stops at 0.20 and 0.3 -> 0.2 was worth +0.0050 (ledger 19).
Three things now say the optimum is lower still: the metric's own arithmetic (a pixel drawn by k
of N annotators pays whenever k/N > IoU/(1+IoU) = 0.40 at IoU 0.67, 0.29 in the near-miss band),
the field reading 0.15 in the gap where annotator agreement is 0.57-1.0 (so the model sits under
its own Bayes target there), and the fold-0 reach table: 0.20 -> 0.10 gains 19 reachable copies
for 9 extra candidates.

Ledger 19's lesson is the protocol here: a threshold change needs its OWN selector. The same
sweep read TIED (+0.0011) with the selector frozen and PAYS (+0.0050) with it refitted, because
the frozen selector is being asked to score a distribution it never saw. So each pool gets the
full deployed recipe fitted on its own folds: 42 features, image-bootstrap bagging x5, inner
out-of-fold quantile map, gate (0.40, 0.60), exact scorer, full folds_v1 scope.

Usage: python src/eda/threshold_pool_twin.py --a fused_pool_all_blob5_t020 --b fused_pool_all_blob5_t010
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
REPS, FLOOR = 4000, 0.003


def load(name, feats, fold_of):
    pool = pd.read_parquet(OUT / f"{name}_labelled.parquet")
    ex = OUT / f"extra_features_{name.split('_')[-1]}.parquet"
    pool = pool.merge(pd.read_parquet(ex)[["cand_id"] + EXTRA], on="cand_id", how="left")
    pool["fold"] = pool["stem"].map(fold_of).astype(int)
    assert not pool[feats].isna().any().any(), f"{name}: missing features"
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--bag", type=int, default=1)
    a = ap.parse_args()
    feats = list(joblib.load(ROOT / "outputs/selector/selector_v5_shrunk_t020pox.joblib")["features"])
    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))

    res, parts = {}, {}
    for tag, name in (("A", a.a), ("B", a.b)):
        pool = load(name, feats, fold_of)
        P, kept = B.run(pool, feats, gt, fold_of, bool(a.bag))
        parts[tag] = P
        cat = lambda k: np.concatenate([P[F][k] for F in range(5)])
        per = [float(pq(*[P[F][k].sum() for k in ("tp", "fp", "fn", "iou")])) for F in range(5)]
        val = float(pq(cat("tp").sum(), cat("fp").sum(), cat("fn").sum(), cat("iou").sum()))
        res[name] = dict(pq=val, per_fold=per, kept=int(kept), candidates=len(pool),
                         positives=int(pool["y"].sum()))
        print(f"{name}: {len(pool)} aday, {int(pool['y'].sum())} pozitif, tutulan {kept}, "
              f"PQ {val:.4f}  fold {[round(x,4) for x in per]}", flush=True)

    dpf = [res[a.b]["per_fold"][i] - res[a.a]["per_fold"][i] for i in range(5)]
    d = res[a.b]["pq"] - res[a.a]["pq"]
    # paired image bootstrap on the pooled ratio
    A = {k: np.concatenate([parts["A"][F][k] for F in range(5)]) for k in ("tp", "fp", "fn", "iou")}
    Bp = {k: np.concatenate([parts["B"][F][k] for F in range(5)]) for k in ("tp", "fp", "fn", "iou")}
    rng = np.random.default_rng(0)
    n = len(A["tp"])
    boot = np.empty(REPS)
    for i in range(REPS):
        idx = rng.integers(0, n, n)
        boot[i] = (pq(Bp["tp"][idx].sum(), Bp["fp"][idx].sum(), Bp["fn"][idx].sum(), Bp["iou"][idx].sum())
                   - pq(A["tp"][idx].sum(), A["fp"][idx].sum(), A["fn"][idx].sum(), A["iou"][idx].sum()))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    verdict = "PAYS" if np.mean(dpf) >= FLOOR else "HURTS" if np.mean(dpf) <= -FLOOR else "SUB-FLOOR"
    print(f"\nB - A: havuzlanmis {d:+.4f}, fold ortalamasi {np.mean(dpf):+.4f}, "
          f"fold basina {[round(x,4) for x in dpf]}")
    print(f"  %95 CI [{lo:+.4f}, {hi:+.4f}]  P(d>0) {(boot>0).mean():.3f}  "
          f"+fold {sum(1 for x in dpf if x>0)}/5  -> {verdict}")
    print("  UYARI (ledger 38): tek bolunturlu CI belirsizligin ALT siniri; "
          "taban-alti bir sonuc >=3 bolunturde okunmadan terfi etmez.")
    res["delta"] = dict(pooled=d, mean_per_fold=float(np.mean(dpf)), per_fold=dpf,
                        ci=[float(lo), float(hi)], p=float((boot > 0).mean()), verdict=verdict)
    json.dump(res, open(OUT / "threshold_pool_twin.json", "w"), indent=1)
    print("wrote threshold_pool_twin.json")


if __name__ == "__main__":
    main()
