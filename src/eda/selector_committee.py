"""E198 -- the v2 question: can the arms that read zero be COMBINED into something that is not zero?

Naive stacking is measured and it fails: three sub-floor model changes together read -0.0004 and
six last-epoch checkpoints together -0.0024 (ledger 32e). Stacking changes the pipeline once for
every arm and their biases add. Averaging does the opposite, and the board already said so: the
one selector arm that transferred is bagging, which averages five copies of the SAME learner over
resampled images. Its natural generalisation has never been measured -- average over DIFFERENT
learner biases:

  base        the deployed classifier (31 leaves, min leaf 25, l2 1, lr 0.05)
  reg         E122's regularised classifier (15 leaves, min leaf 60, l2 5)          local +0.0014
  slow        lr 0.03 over 500 iterations                                           local +0.0006
  relational  the deployed classifier plus E195's thirteen neighbourhood features   local +0.0022

Each variant is itself bagged over five image bootstraps, each is mapped through its own inner
out-of-fold quantile reference (so the four scores live on the same [0,1] scale), and the committee
score is their mean. The gate stays at the deployed (0.40, 0.60): averaging uniform scores keeps
the quantile meaning, which is the property that makes this different from stacking decisions.

Read against the deployed bagged selector alone, exact scorer, paired image bootstrap, +-0.003
band. Also reported: the committee's agreement structure (how often the four variants disagree at
the gate), because a committee only helps where its members disagree.

Usage: python src/eda/selector_committee.py [--nbag 5]
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
from e115c_threshold_ci import per_image, pq  # noqa: E402
from extra_features import EXTRA  # noqa: E402
from relational_features import REL  # noqa: E402
from fit_selector_bagged import make_cls  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

OUT = ROOT / "outputs" / "eda"
GATE = (0.40, 0.60)
REPS, FLOOR, EQUIV = 4000, 0.003, 0.005


def fit_pairs(df, feats, nbag, rng, reg_arm, slow):
    stems = df["stem"].unique()
    by = {s: g for s, g in df.groupby("stem")}
    out = []
    for _ in range(nbag):
        boot = pd.concat([by[s] for s in rng.choice(stems, len(stems), replace=True)],
                         ignore_index=True)
        X, y = boot[feats].to_numpy(np.float64), boot["y"].to_numpy()
        pos = y == 1
        c = make_cls(reg_arm, slow).fit(X, y)
        r = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06).fit(
            X[pos], boot.loc[pos, "iou_sum"].to_numpy(np.float64))
        out.append((c, r))
    return out


def score(pairs, X):
    return np.mean([c.predict_proba(X)[:, 1] * np.clip(r.predict(X), 0.0, None)
                    for c, r in pairs], axis=0)


def oof_scores(pool, feats, nbag, reg_arm, slow, seed=0):
    """The deployed protocol: fit on four folds, inner leave-one-fold-out quantile reference,
    score the held-out fold. Returns the quantile-mapped score per candidate."""
    rng = np.random.default_rng(seed)
    p = np.zeros(len(pool))
    X = pool[feats].to_numpy(np.float64)
    fold = pool["fold"].to_numpy()
    for F in range(5):
        tr = fold != F
        inner = np.zeros(int(tr.sum()))
        sub = pool[tr]
        for gk in [k for k in range(5) if k != F]:
            itr = (sub["fold"] != gk).to_numpy()
            msk = (sub["fold"] == gk).to_numpy()
            inner[msk] = score(fit_pairs(sub[itr], feats, nbag, rng, reg_arm, slow),
                               X[tr][msk])
        ref = np.sort(inner)
        pairs = fit_pairs(sub, feats, nbag, rng, reg_arm, slow)
        p[fold == F] = np.searchsorted(ref, score(pairs, X[fold == F]), side="right") / len(ref)
        print(f"    fold {F} done", flush=True)
    return p


def evaluate(pool, keep, gt, stems, ncopy):
    sub = pool[keep]
    return per_image(sub[["cand_id", "rle", "stem"]], gt, stems, ncopy)[0]


def paired(A, Bp):
    d = (pq(Bp["tp"].sum(), Bp["fp"].sum(), Bp["fn"].sum(), Bp["iou"].sum())
         - pq(A["tp"].sum(), A["fp"].sum(), A["fn"].sum(), A["iou"].sum()))
    rng = np.random.default_rng(0)
    n = len(A["tp"])
    boot = np.empty(REPS)
    for i in range(REPS):
        idx = rng.integers(0, n, n)
        boot[i] = (pq(Bp["tp"][idx].sum(), Bp["fp"][idx].sum(), Bp["fn"][idx].sum(), Bp["iou"][idx].sum())
                   - pq(A["tp"][idx].sum(), A["fp"][idx].sum(), A["fn"][idx].sum(), A["iou"][idx].sum()))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    verdict = ("PAYS" if d >= FLOOR else "HURTS" if d <= -FLOOR else
               ("TIED" if lo > -EQUIV and hi < EQUIV else "UNDERPOWERED"))
    return float(d), float(lo), float(hi), float((boot > 0).mean()), verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbag", type=int, default=5)
    a = ap.parse_args()
    pool = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    pool = pool.merge(pd.read_parquet(OUT / "extra_features_t020.parquet")[["cand_id"] + EXTRA],
                      on="cand_id", how="left")
    pool = pool.merge(pd.read_parquet(OUT / "relational_features_t020.parquet"), on="cand_id",
                      how="left", validate="one_to_one")
    folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    pool["fold"] = pool["stem"].map(fold_of).astype(int)
    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt = gt[gt["stem"].isin(set(pool["stem"]))]
    stems = np.array(sorted(set(pool["stem"])))
    ncopy = gt.groupby("stem")["entry_id"].nunique().reindex(stems).fillna(1).to_numpy()
    base = list(joblib.load(ROOT / "outputs/selector/selector_v5_shrunk_t020pox.joblib")["features"])

    variants = {"base": (base, False, False), "reg": (base, True, False),
                "slow": (base, False, True), "relational": (base + REL, False, False)}
    P = {}
    for name, (feats, reg_arm, slow) in variants.items():
        print(f"  {name}: {len(feats)} features", flush=True)
        P[name] = oof_scores(pool, feats, a.nbag, reg_arm, slow)
    conf_ok = (pool["conf"] >= GATE[0]).to_numpy()
    keeps = {k: conf_ok & (v >= GATE[1]) for k, v in P.items()}
    committee = np.mean([P[k] for k in variants], axis=0)
    keeps["committee"] = conf_ok & (committee >= GATE[1])

    res, parts = {}, {}
    for k, keep in keeps.items():
        parts[k] = evaluate(pool, keep, gt, stems, ncopy)
        val = pq(parts[k]["tp"].sum(), parts[k]["fp"].sum(), parts[k]["fn"].sum(), parts[k]["iou"].sum())
        res[k] = dict(pq=float(val), kept=int(keep.sum()))
        print(f"{k:12s} kept {int(keep.sum()):5d}  PQ {val:.4f}", flush=True)
    for k in ("committee", "reg", "slow", "relational"):
        d, lo, hi, p, verdict = paired(parts["base"], parts[k])
        res[k] |= dict(delta_vs_base=d, ci=[lo, hi], p=p, verdict=verdict)
        print(f"  {k:12s} vs base: {d:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  P {p:.3f}  {verdict}")

    # a committee only helps where its members disagree
    band = conf_ok & (np.max([P[k] for k in variants], axis=0) >= 0.5) & \
           (np.min([P[k] for k in variants], axis=0) < GATE[1])
    disagree = conf_ok & (np.array([keeps[k] for k in variants]).sum(0) % 4 != 0)
    y = pool["y"].to_numpy()
    print(f"\nmembers disagree on {int(disagree.sum())} candidates ({disagree.mean():.1%} of the pool); "
          f"their truth rate {y[disagree].mean():.3f}; committee keeps {int((keeps['committee'] & disagree).sum())}")
    res["disagreement"] = dict(n=int(disagree.sum()), truth=float(y[disagree].mean()),
                               committee_keeps=int((keeps["committee"] & disagree).sum()))
    json.dump(res, open(OUT / "selector_committee.json", "w"), indent=1)
    print("wrote selector_committee.json")


if __name__ == "__main__":
    main()
