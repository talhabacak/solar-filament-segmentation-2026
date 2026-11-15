"""E194 -- is a set-level PQ objective actually a source of headroom for SELECTION?

Section 4.7(b) of the report names, as the most promising untried idea, "a learned bridge
trained end-to-end against the matching objective itself, a differentiable PQ surrogate over
candidate SETS rather than per-candidate value regression". The motivation is that PQ has a
global denominator, so keeping one candidate should change what the others are worth, and a
per-candidate rule cannot see that coupling.

Before building it, write the coupling down. For an image with c annotator copies, k kept
predictions and G ground-truth rows, the scorer's own accounting (verified in E115c and
reproduced in `per_image`) is

    tp = matched pairs,   fp = k*c - tp,   fn = G - tp

so the denominator of PQ, summed over images, is

    D = tp + (k*c - tp)/2 + (G - tp)/2 = (1/2) * sum_over_kept(c_i) + (1/2) * G_total

Every tp term cancels. The denominator does NOT depend on which candidates are kept, only on
how many copies they are charged against, and the second term is a constant of the dataset. The
numerator is sum_over_kept(iou_sum_i). So PQ as a function of the kept set K is

    PQ(K) = sum_K v_i / (0.5 * sum_K c_i + 0.5 * G)          v_i = iou_sum_i

which is a linear-fractional function of an indicator vector: SEPARABLE. Dinkelbach's theorem
then says the set maximising it is exactly

    K* = { i : v_i > 0.5 * c_i * PQ(K*) }

a per-candidate threshold on value minus cost -- which is the break-even rule this project
already derives from the metric and already deploys. If that is right, a set-level learner
cannot beat a per-candidate one given the same value estimates, and 4.7(b) is closed for the
selection stage before a line of it is written.

The argument has one assumption that the pipeline can violate: v_i and c_i must not depend on
what else is kept. They do, slightly, because kept candidates are made pixel-disjoint against
each other, so a neighbour's presence can carve a mask and move its IoU. That interaction is
the ONLY room a set-level objective has, and it is measurable rather than arguable.

Four readings, in order:

  1. IDENTITY   -- does the closed form reproduce the exact scorer on the production kept set?
                   Any gap is the de-overlap interaction, in PQ units.
  2. OPTIMALITY -- run Dinkelbach on the TRUE values to get the best set that exists, then try
                   to beat it with greedy add/remove and with random restarts. If nothing beats
                   it, the rule is optimal and the proof holds numerically.
  3. COUPLING   -- how much of the optimum is reachable by a per-IMAGE adaptive threshold (the
                   simplest thing a set-level learner would learn) over a global one? On the
                   true values this is an oracle; if the oracle is zero the learner is pointless.
  4. ESTIMATES  -- repeat 3 with the deployed selector's ESTIMATED scores, since a set-level
                   learner in practice sees estimates, not truth.

Usage: python src/eda/setlevel_pq_separability.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/eda", "src/models"):
    sys.path.insert(0, str(ROOT / p))
from filament.pq import score  # noqa: E402
from e115c_threshold_ci import per_image  # noqa: E402
from extra_features import EXTRA  # noqa: E402

OUT = ROOT / "outputs" / "eda"
BUNDLE = ROOT / "outputs" / "selector" / "selector_v5_shrunk_t020pox.joblib"


def log(*a):
    print(*a, flush=True)


def closed_form(v, c, g_total):
    """PQ(K) for the kept rows already selected into v and c."""
    d = 0.5 * float(c.sum()) + 0.5 * g_total
    return float(v.sum()) / d if d > 0 else 0.0


def dinkelbach(v, c, g_total, iters=50):
    """Maximise sum_K v / (0.5 sum_K c + 0.5 G) over subsets K.

    Dinkelbach: at value lam the best set is {i : v_i - 0.5 * c_i * lam > 0}; iterate until the
    achieved value stops moving. Converges superlinearly and is exact for linear-fractional
    objectives, so the returned set is the global maximum, not a local one.
    """
    lam = 0.0
    for _ in range(iters):
        keep = (v - 0.5 * c * lam) > 0
        new = closed_form(v[keep], c[keep], g_total)
        if abs(new - lam) < 1e-12:
            return keep, new
        lam = new
    return keep, lam


def greedy_probe(v, c, g_total, keep, rng, restarts=8, sweeps=3):
    """Try to beat Dinkelbach's set by local search, from its own answer and from random sets.

    If a linear-fractional optimum is really optimal, nothing here can improve on it. This is
    the numerical check on the proof: a single counterexample would refute it.
    """
    best = closed_form(v[keep], c[keep], g_total)
    starts = [keep.copy()] + [rng.random(len(v)) < p for p in np.linspace(0.2, 0.9, restarts)]
    for s in starts:
        cur, val = s.copy(), closed_form(v[s], c[s], g_total)
        for _ in range(sweeps):
            moved = False
            # a flip changes the objective in closed form, so score every flip at once
            num, den = float(v[cur].sum()), 0.5 * float(c[cur].sum()) + 0.5 * g_total
            sign = np.where(cur, -1.0, 1.0)
            gain = (num + sign * v) / (den + sign * 0.5 * c) - num / den
            order = np.argsort(-gain)
            for i in order[:2000]:
                if gain[i] <= 1e-15:
                    break
                cur[i] = not cur[i]
                nv = closed_form(v[cur], c[cur], g_total)
                if nv > val + 1e-15:
                    val, moved = nv, True
                    num, den = float(v[cur].sum()), 0.5 * float(c[cur].sum()) + 0.5 * g_total
                    sign = np.where(cur, -1.0, 1.0)
                    gain = (num + sign * v) / (den + sign * 0.5 * c) - num / den
                    order = np.argsort(-gain)
                else:
                    cur[i] = not cur[i]
            if not moved:
                break
        best = max(best, val)
    return best


def per_image_oracle(df, col, g_total, grid):
    """Best achievable by choosing a SEPARATE threshold on `col` for every image.

    This is the cleanest statement of what a set-level learner could add: per-image adaptation.
    It is a joint problem because the denominator is shared, so it is solved the same way --
    Dinkelbach on the outer value, with each image choosing its own best threshold at that value.
    """
    lam, groups = 0.0, list(df.groupby("stem", sort=False))
    for _ in range(60):
        num = den = 0.0
        for _stem, g in groups:
            s = g[col].to_numpy()
            v = g["iou_sum"].to_numpy(np.float64)
            cc = g["n_copies"].to_numpy(np.float64)
            best, bn, bd = -np.inf, 0.0, 0.0
            for t in grid:
                k = s >= t
                n_, d_ = v[k].sum(), 0.5 * cc[k].sum()
                obj = n_ - lam * d_             # Dinkelbach's surrogate, additive over images
                if obj > best:
                    best, bn, bd = obj, n_, d_
            num += bn
            den += bd
        new = num / (den + 0.5 * g_total)
        if abs(new - lam) < 1e-12:
            return new
        lam = new
    return lam


def global_threshold_best(df, col, g_total, grid):
    best = 0.0
    for t in grid:
        k = df[col].to_numpy() >= t
        best = max(best, closed_form(df["iou_sum"].to_numpy(np.float64)[k],
                                     df["n_copies"].to_numpy(np.float64)[k], g_total))
    return best


def oof_selector_scores(pool, feats):
    """The deployed protocol: fit on the other four folds, normalise through an inner LOFO
    quantile map built from those same folds, never the held-out fold's own rows."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    def fit_predict(trn, tst):
        X, y = trn[feats].to_numpy(np.float64), trn["y"].to_numpy()
        v = trn["iou_sum"].to_numpy(np.float64)
        c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                           min_samples_leaf=25, l2_regularization=1.0,
                                           random_state=42).fit(X, y)
        r = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                          random_state=42).fit(X[y == 1], v[y == 1])
        Z = tst[feats].to_numpy(np.float64)
        return c.predict_proba(Z)[:, 1] * np.clip(r.predict(Z), 0, None)

    out = []
    for F in range(5):
        trn, tst = pool[pool["fold"] != F], pool[pool["fold"] == F].copy()
        inner = np.zeros(len(trn))
        for g in [k for k in range(5) if k != F]:
            itr, msk = trn["fold"] != g, trn["fold"] == g
            inner[msk.to_numpy()] = fit_predict(trn[itr], trn[msk])
        ref = np.sort(inner)
        raw = fit_predict(trn, tst)
        tst["p"] = np.searchsorted(ref, raw, side="right") / len(ref)
        out.append(tst)
    return pd.concat(out)


def main():
    feats = list(joblib.load(BUNDLE)["features"])
    folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    g_total = float(len(gt))

    pool = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    extra = pd.read_parquet(OUT / "extra_features_t020.parquet")
    pool = pool.merge(extra[["cand_id"] + [c for c in EXTRA if c in extra.columns]],
                      on="cand_id", how="left")
    pool["fold"] = pool["stem"].map(fold_of).astype(int)
    missing = [f for f in feats if f not in pool.columns]
    assert not missing, f"pool lacks {len(missing)} selector features: {missing[:8]}"
    log(f"pool {len(pool)} candidates over {pool['stem'].nunique()} images; "
        f"{int(g_total)} ground-truth rows")

    res = {}

    # ---- 1. IDENTITY: closed form against the exact scorer, on the deployed kept set
    scored = oof_selector_scores(pool, feats)
    kept = scored[(scored["conf"] >= 0.40) & (scored["p"] >= 0.60)]
    exact = score(gt[["filament_id", "segmentation_rle"]],
                  kept[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id",
                                                           "rle": "segmentation_rle"}))
    cf = closed_form(kept["iou_sum"].to_numpy(np.float64),
                     kept["n_copies"].to_numpy(np.float64), g_total)
    log(f"[1] identity   exact scorer PQ {exact['pq']:.6f}   closed form {cf:.6f}   "
        f"gap {cf - exact['pq']:+.6f}")
    res["identity"] = dict(exact=float(exact["pq"]), closed_form=cf,
                           gap=float(cf - exact["pq"]), kept=int(len(kept)))

    # ---- 2. OPTIMALITY: the best set that exists, and an attempt to beat it
    v = pool["iou_sum"].to_numpy(np.float64)
    c = pool["n_copies"].to_numpy(np.float64)
    keep, opt = dinkelbach(v, c, g_total)
    rng = np.random.default_rng(0)
    probed = greedy_probe(v, c, g_total, keep, rng)
    thr = 0.5 * opt
    log(f"[2] optimality Dinkelbach PQ {opt:.6f} keeping {int(keep.sum())} of {len(v)}; "
        f"local search best {probed:.6f} (gap {probed - opt:+.2e})")
    log(f"    the optimal rule is a per-candidate threshold: keep iff iou_sum > "
        f"{thr:.4f} * copies  -- the deployed break-even, derived independently in section 1.2")
    res["optimality"] = dict(dinkelbach=opt, local_search=probed, kept=int(keep.sum()),
                             breakeven_per_copy=float(thr))

    # ---- 3. COUPLING: per-image adaptive threshold vs one global threshold, on TRUE value
    pool["true_ratio"] = v / np.maximum(c, 1e-9)
    grid = np.unique(np.quantile(pool["true_ratio"], np.linspace(0, 1, 201)))
    g_best = global_threshold_best(pool, "true_ratio", g_total, grid)
    i_best = per_image_oracle(pool, "true_ratio", g_total, grid)
    log(f"[3] coupling   true value: one global threshold {g_best:.6f}, "
        f"a free threshold per image {i_best:.6f}, oracle gain {i_best - g_best:+.6f}")
    res["coupling_true"] = dict(global_=g_best, per_image=i_best, gain=float(i_best - g_best))

    # ---- 4. ESTIMATES: the same contrast on what a learner actually sees
    gridp = np.unique(np.quantile(scored["p"], np.linspace(0, 1, 201)))
    g_est = global_threshold_best(scored, "p", g_total, gridp)
    i_est = per_image_oracle(scored, "p", g_total, gridp)
    log(f"[4] estimates  selector score: global {g_est:.6f}, per image {i_est:.6f}, "
        f"oracle gain {i_est - g_est:+.6f}")
    res["coupling_estimated"] = dict(global_=g_est, per_image=i_est, gain=float(i_est - g_est))

    json.dump(res, open(OUT / "setlevel_pq_separability.json", "w"), indent=1)
    log("SETLEVEL_DONE")


if __name__ == "__main__":
    main()
