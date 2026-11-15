"""E210 -- which submission should take the second final slot next to sub036?

The private leaderboard scores the BETTER of the two selected submissions. The value of a second
slot is therefore not its own expected score but how much it adds to the first:

    gain(B) = E[ max(PQ_A, PQ_B) ] - E[ PQ_A ],     A = sub036

over the population the private board will be computed on. A candidate identical to A adds
nothing however good it is; a candidate that is weaker on average but fails on different images can
add more. This measures it on the out-of-fold images, which have the test set's structure (i.i.d.
split, same stations and years to within the composition shift of ledger 43): for each candidate
it rebuilds the deployed keep decision out of fold, scores every image with the exact metric, and
draws random image subsets of test size to estimate gain(B), P(PQ_B > PQ_A) and the expected
maximum. Also reported: how often the two keep the same candidate locally, the correlation of their
per-image PQ, and, on the test set itself, the share of sub036's submitted masks that the
candidate's CSV contains identically.

Candidates are the leaderboard's top six apart from sub036 plus the cheapest gate variant; each is
rebuilt with the recipe it was submitted with:

  sub042  bagged x5 (seed 0) + marginal-band cascade, fixed q 0.85
  sub043  bagged x5, bootstrap seed 1
  sub045  bagged x10
  sub037  unbagged + marginal-band cascade, fixed q 0.85
  sub033  unbagged
  sub040  unbagged, conf >= 0.45

Usage: python src/eda/final_pair_selection.py [--reps 4000]
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
from e115c_threshold_ci import per_image, pq  # noqa: E402
from extra_features import EXTRA  # noqa: E402

OUT = ROOT / "outputs" / "eda"
GATE, BAND, CASCADE_Q = (0.40, 0.60), (0.40, 0.60), 0.85
SUBS = ROOT / "outputs" / "submissions"


def bagged_oof(a, feats, nbag, seed):
    """Out-of-fold quantile-mapped scores of an image-bootstrap bagged selector: the protocol of
    outputs/eda/_bag5_oof_scores.py (which produced sub036's scores), with nbag and seed free."""
    rng = np.random.default_rng(seed)
    X = a[feats].to_numpy(np.float64)
    p = np.zeros(len(a))
    for F in range(5):
        trn_m = (a["fold"] != F).to_numpy()
        trn = a[trn_m]

        def models_for(sub):
            stems = sub["stem"].unique()
            by = {s: g for s, g in sub.groupby("stem")}
            return [B.fit_pair(pd.concat([by[s] for s in rng.choice(stems, len(stems), replace=True)],
                                         ignore_index=True), feats) for _ in range(nbag)]

        inner = np.zeros(len(trn))
        Xt = X[trn_m]
        for g in [k for k in range(5) if k != F]:
            itr, msk = (trn["fold"] != g).to_numpy(), (trn["fold"] == g).to_numpy()
            inner[msk] = B.raw_score(models_for(trn[itr]), Xt[msk])
        ref = np.sort(inner)
        te = ~trn_m
        p[te] = np.searchsorted(ref, B.raw_score(models_for(trn), X[te]), side="right") / len(ref)
        print(f"    nbag {nbag} seed {seed}: fold {F}", flush=True)
    return p


def cascade_admit(a, p, feats):
    """The deployed marginal-band cascade at fixed q 0.85, out of fold: a classifier fitted on the
    band rows of the other folds, admitting the top 15 % of each held-out fold's band."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    conf_ok = (a["conf"] >= GATE[0]).to_numpy()
    band = conf_ok & (p >= BAND[0]) & (p < BAND[1])
    X, y, f = a[feats].to_numpy(np.float64), a["y"].to_numpy(), a["fold"].to_numpy()
    admit = np.zeros(len(a), bool)
    for k in range(5):
        tr, te = band & (f != k), band & (f == k)
        if not te.any():
            continue
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                           min_samples_leaf=25, l2_regularization=1.0,
                                           random_state=42).fit(X[tr], y[tr])
        s = m.predict_proba(X[te])[:, 1]
        idx = np.flatnonzero(te)
        admit[idx[s >= np.quantile(s, CASCADE_Q)]] = True
    return admit


def test_overlap(ref_csv, cand_csv):
    """Share of the reference CSV's masks present byte-identically in the candidate CSV."""
    r, c = pd.read_csv(ref_csv), pd.read_csv(cand_csv)
    img = lambda d: d.iloc[:, 0].astype(str).str.rsplit("_", n=1).str[0]
    rs = set(zip(img(r), r.iloc[:, -1].astype(str)))
    cs = set(zip(img(c), c.iloc[:, -1].astype(str)))
    return len(rs & cs) / max(len(rs), 1), len(r), len(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4000)
    a_ = ap.parse_args()
    feats = list(joblib.load(ROOT / "outputs/selector/selector_v5_shrunk_t020pox.joblib")["features"])
    cfeats = list(joblib.load(ROOT / "outputs/selector/selector_v5_shrunkblob5prodts.joblib")["features"]) + list(EXTRA)
    a = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    a = a.merge(pd.read_parquet(OUT / "extra_features_t020.parquet")[["cand_id"] + EXTRA], on="cand_id")
    folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    a["fold"] = a["stem"].map(fold_of).astype(int)
    single = a[["cand_id"]].merge(pd.read_parquet(OUT / "t020pox_oof_scores.parquet"), on="cand_id")["p"].to_numpy()
    bag5 = a[["cand_id"]].merge(pd.read_parquet(OUT / "bag5_oof_scores.parquet"), on="cand_id")["p"].to_numpy()
    cache = OUT / "final_pair_oof_scores.parquet"
    if cache.exists():
        c = a[["cand_id"]].merge(pd.read_parquet(cache), on="cand_id")
        bag5_s1, bag10 = c["bag5_s1"].to_numpy(), c["bag10"].to_numpy()
    else:
        print("  bagged OOF, seed 1 (sub043) and ten bags (sub045):", flush=True)
        bag5_s1 = bagged_oof(a, feats, 5, 1)
        bag10 = bagged_oof(a, feats, 10, 0)
        pd.DataFrame({"cand_id": a["cand_id"], "bag5_s1": bag5_s1, "bag10": bag10}).to_parquet(cache, index=False)
    conf40 = (a["conf"] >= 0.40).to_numpy()
    conf45 = (a["conf"] >= 0.45).to_numpy()
    keeps = {
        "sub036 (A)": conf40 & (bag5 >= 0.60),
        "sub042": (conf40 & (bag5 >= 0.60)) | cascade_admit(a, bag5, cfeats),
        "sub043": conf40 & (bag5_s1 >= 0.60),
        "sub045": conf40 & (bag10 >= 0.60),
        "sub037": (conf40 & (single >= 0.60)) | cascade_admit(a, single, cfeats),
        "sub033": conf40 & (single >= 0.60),
        "sub040": conf45 & (single >= 0.60),
    }
    csv = {"sub036 (A)": "sub036_t020pox_bag5.csv", "sub042": "sub042_t020pox_bag5_cascade085.csv",
           "sub043": "sub043_t020pox_bag5_seed1.csv", "sub045": "sub045_t020pox_bag10.csv",
           "sub037": "sub037_t020pox_cascade085.csv", "sub033": "sub033_t020_photoorient_p060.csv",
           "sub040": "sub040_t020pox_c45.csv"}

    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    stems = np.array(sorted(fold_of))                   # all 707: an image with no kept mask is all FN
    gt = gt[gt["stem"].isin(set(stems))]
    ncopy = gt.groupby("stem")["entry_id"].nunique().reindex(stems).fillna(1).to_numpy()
    parts = {}
    for name, k in keeps.items():
        st = per_image(a[k][["cand_id", "rle", "stem"]], gt, stems, ncopy)[0]
        parts[name] = np.stack([st["tp"], st["fp"], st["fn"], st["iou"]], 1).astype(float)
        print(f"  {name:11s} kept {int(k.sum()):5d}  PQ {pq(*parts[name].sum(0)):.4f}", flush=True)

    A = parts["sub036 (A)"]
    rng = np.random.default_rng(0)
    n_img = len(stems)
    res = {}
    print(f"\n{'second slot':11s}{'PQ':>8s}{'shared':>8s}{'corr':>7s}"
          f"{'gain@90':>10s}{'P>A@90':>8s}{'gain@180':>10s}{'P>A@180':>9s}{'test same':>11s}")
    for name, Bp in parts.items():
        if name == "sub036 (A)":
            continue
        kA, kB = keeps["sub036 (A)"], keeps[name]
        shared = (kA & kB).sum() / max((kA | kB).sum(), 1)
        den = lambda P: P[:, 0] + P[:, 1] / 2 + P[:, 2] / 2
        ok = (den(A) > 0) & (den(Bp) > 0)
        pa, pb = A[ok, 3] / den(A)[ok], Bp[ok, 3] / den(Bp)[ok]
        corr = float(np.corrcoef(pa, pb)[0, 1])
        row = dict(pq=float(pq(*Bp.sum(0))), shared=float(shared), corr=corr)
        for n in (90, 180):
            g, win = np.empty(a_.reps), np.empty(a_.reps)
            for r in range(a_.reps):
                idx = rng.choice(n_img, n, replace=False)
                qa, qb = pq(*A[idx].sum(0)), pq(*Bp[idx].sum(0))
                g[r], win[r] = max(qa, qb) - qa, qb > qa
            row[f"gain_{n}"], row[f"p_better_{n}"] = float(g.mean()), float(win.mean())
        same, nr, nc = test_overlap(SUBS / csv["sub036 (A)"], SUBS / csv[name])
        row.update(test_same=float(same), test_rows=int(nc))
        res[name] = row
        print(f"{name:11s}{row['pq']:>8.4f}{shared:>8.3f}{corr:>7.3f}{row['gain_90']:>+10.4f}"
              f"{row['p_better_90']:>8.3f}{row['gain_180']:>+10.4f}{row['p_better_180']:>9.3f}{same:>11.3f}")
    res["sub036 (A)"] = dict(pq=float(pq(*A.sum(0))), kept=int(keeps["sub036 (A)"].sum()))
    json.dump(res, open(OUT / "final_pair_selection.json", "w"), indent=1)
    print("\ngain = E[max(PQ_A, PQ_B)] - E[PQ_A] over random image subsets of test size;"
          " 'test same' = share of sub036's submitted masks present identically in B's CSV")


if __name__ == "__main__":
    main()
