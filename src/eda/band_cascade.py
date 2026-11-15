"""E138 -- a second-stage selector for the marginal band (a cascade).

E137 found the shape of the population the pool reaches and the selector drops:
907 true candidates, 29% of them within 0.05 of the gate, and a band
(p in [0.40, 0.60)) that is 38.3% true. A classifier trained only inside that
band reaches out-of-fold AUC 0.601 and its top fifth is 50.6% true -- above the
empirical break-even purity of ~45% measured in E128. That is small but it is
the first positive signal in the rejected population, and it is exactly the
cascade idea: the first stage ranks everything, a second stage re-ranks the
candidates the first stage is unsure about.

Protocol (honest, and stricter than the analysis that motivated it):
  - the band model for fold k is fitted ONLY on the other four folds' band
    candidates, so its scores on fold k are out of fold;
  - the admission rule is chosen on the training folds too: the quantile q of
    the band score that maximises summed exact-PQ gain there, swept over a grid;
  - the held-out fold's band candidates above that quantile are ADDED to the
    deployed keep set, and the exact scorer is run over all 703 images;
  - reported against the deployed baseline with the paired image bootstrap.

Bands: the pre-registered decision is the E120 one -- pays at >= +0.003, hurts
at <= -0.003, otherwise tied. A tie closes the rejected-true population for
good: it would mean the pool's unreached rows are unreachable by ranking.

Usage: python src/eda/band_cascade.py
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
sys.path.insert(0, str(ROOT / "src" / "models"))
from filament.pq import score  # noqa: E402
from e115c_threshold_ci import per_image, pq  # noqa: E402
from extra_features import EXTRA  # noqa: E402

OUT = ROOT / "outputs" / "eda"
GATE = (0.40, 0.60)
BAND = (0.40, 0.60)
REPS, FLOOR, EQUIV = 4000, 0.003, 0.005
QGRID = [1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5]   # 1.0 = admit nothing


def log(*a):
    print(*a, flush=True)


def main(scores="t020pox_oof_scores.parquet", tag=""):
    from sklearn.ensemble import HistGradientBoostingClassifier
    feats = list(joblib.load(ROOT / "outputs/selector/selector_v5_shrunkblob5prodts.joblib")["features"]) + list(EXTRA)
    a = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    a = a.merge(pd.read_parquet(OUT / "extra_features_t020.parquet")[["cand_id"] + EXTRA], on="cand_id", how="left")
    a = a.merge(pd.read_parquet(OUT / scores), on="cand_id", how="left")
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    a["fold"] = a["stem"].map(fold_of).astype(int)
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    g = gt[gt["stem"].isin(set(a["stem"]))]
    stems = np.array(sorted(set(a["stem"])))
    ncopy = g.groupby("stem")["entry_id"].nunique().reindex(stems).fillna(1).to_numpy()

    keep = a[(a["conf"] >= GATE[0]) & (a["p"] >= GATE[1])]
    band = a[(a["conf"] >= GATE[0]) & (a["p"] >= BAND[0]) & (a["p"] < BAND[1])].copy()
    log(f"deployed keep {len(keep)}; marginal band {len(band)} ({band['y'].mean():.1%} true)")

    def sc(df):
        return score(g[["filament_id", "segmentation_rle"]],
                     df[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"}))

    base = sc(keep)
    log(f"baseline PQ {base['pq']:.4f} (TP {base['tp']} FP {base['fp']} FN {base['fn']})")

    X, y, f = band[feats].to_numpy(np.float64), band["y"].to_numpy(), band["fold"].to_numpy()
    band["s"] = np.nan
    chosen = []
    for k in range(5):
        tr, te = np.where(f != k)[0], np.where(f == k)[0]
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                           min_samples_leaf=25, l2_regularization=1.0, random_state=42)
        m.fit(X[tr], y[tr])
        band.iloc[te, band.columns.get_loc("s")] = m.predict_proba(X[te])[:, 1]
        # choose the admission quantile on the TRAINING folds, by exact PQ there
        inner = np.zeros(len(tr))
        for j in [q for q in range(5) if q != k]:
            itr = np.where((f != k) & (f != j))[0]
            imk = f[tr] == j
            mi = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                min_samples_leaf=25, l2_regularization=1.0, random_state=1)
            mi.fit(X[itr], y[itr])
            inner[imk] = mi.predict_proba(X[tr][imk])[:, 1]
        btr = band.iloc[tr].assign(s_in=inner)
        ktr = keep[keep["fold"] != k]
        gtr = g[g["stem"].map(fold_of) != k]
        best_q, best_pq = 1.0, None
        for q in QGRID:
            add = btr[btr["s_in"] >= np.quantile(btr["s_in"], q)] if q < 1.0 else btr.iloc[0:0]
            trial = pd.concat([ktr, add])
            r = score(gtr[["filament_id", "segmentation_rle"]],
                      trial[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"}))
            if best_pq is None or r["pq"] > best_pq:
                best_q, best_pq = q, r["pq"]
        chosen.append(best_q)
        log(f"  fold {k}: admission quantile chosen on the training folds = {best_q} (their PQ {best_pq:.4f})")

    add_rows = []
    for k in range(5):
        bk = band[band["fold"] == k]
        q = chosen[k]
        if q < 1.0 and len(bk):
            add_rows.append(bk[bk["s"] >= np.quantile(bk["s"], q)])
    add = pd.concat(add_rows) if add_rows else band.iloc[0:0]
    log(f"\nadmitted {len(add)} candidates ({add['y'].mean():.1%} true)" if len(add) else "\nadmitted nothing")
    trial = pd.concat([keep, add])
    r = sc(trial)
    d = r["pq"] - base["pq"]
    log(f"cascade PQ {r['pq']:.4f} (TP {r['tp']} FP {r['fp']} FN {r['fn']}), delta {d:+.4f}")

    parts = {}
    for name, df in (("base", keep), ("cascade", trial)):
        st, rr = per_image(df, g, stems, ncopy)
        parts[name] = st
    rng = np.random.default_rng(0)
    n = len(stems)
    boot = np.empty(REPS)
    for i in range(REPS):
        idx = rng.integers(0, n, n)
        v = [pq(s["tp"][idx].sum(), s["fp"][idx].sum(), s["fn"][idx].sum(), s["iou"][idx].sum())
             for s in (parts["base"], parts["cascade"])]
        boot[i] = v[1] - v[0]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    verdict = "PAYS" if d >= FLOOR else "HURTS" if d <= -FLOOR else ("TIED" if lo > -EQUIV and hi < EQUIV else "UNDERPOWERED")
    log(f"95% CI [{lo:+.4f}, {hi:+.4f}], P(delta>0) {(boot > 0).mean():.3f} -> {verdict}")
    json.dump(dict(base=dict(pq=base["pq"], tp=base["tp"], fp=base["fp"], fn=base["fn"]),
                   cascade=dict(pq=r["pq"], tp=r["tp"], fp=r["fp"], fn=r["fn"]),
                   admitted=int(len(add)), admitted_purity=float(add["y"].mean()) if len(add) else None,
                   quantiles=chosen, delta=float(d), ci=[float(lo), float(hi)], verdict=verdict),
              open(OUT / f"band_cascade{tag}.json", "w"), indent=1, default=float)
    log(f"wrote {OUT / f'band_cascade{tag}.json'}")


if __name__ == "__main__":
    # optional: a different first-pass score file (e.g. the bagged selector's out-of-fold scores)
    # and a tag for the result json, so the cascade can be read on top of another selector
    main(*sys.argv[1:3])
