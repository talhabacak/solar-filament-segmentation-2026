"""Rank candidates by p(match) x expected IoU, not by p(match) alone.

The deployed selector predicts y = "this candidate matches at least one
ground-truth row" and thresholds it. The metric does not pay in matches: keeping
a candidate adds its summed IoU to the numerator and charges 0.5 per annotator
entry it misses. Two candidates with the same P(match) are therefore worth
different amounts, and the classifier cannot express that.

The two-stage score is

    s = p(y=1) * E[iou_sum | y=1]

with the second factor fit on positives only. Fitting iou_sum directly with a
squared-error loss has already failed here (-0.005 to -0.057): the target is 58%
exact zeros and the loss spends itself on them. Conditioning removes the zeros.

Why this is not the seventeenth ceiling-raise. Sixteen mechanisms raised the pool
or its oracle and delivered nothing, because something downstream had to capture
the extra reachability. This changes only the score being thresholded -- same
pool, same features, same policy space -- so the number measured IS the realised
number. There is nothing left to capture.

The protocol is paired and leak-free, which matters because it is what the first
attempt got wrong. Thresholds are chosen on the training half ONLY; taking
quantiles over the whole pool leaks the evaluation half into the policy and was
measured to inflate deltas by about a third. Both arms see identical splits, so
the split-to-split variance that dominates each arm cancels.

Usage: python src/eda/two_stage_selector.py --pool fused_pool_f0_blobens.parquet
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament.pq import score
from filament.rle import decode_rle, encode_mask
from selector import fast_pq
from selector_v5 import BASE, SRC, CROSS

OUT = ROOT / "outputs" / "eda"
SEL = ROOT / "outputs" / "selector"
FEATS = BASE + SRC + CROSS
GRID_C = (0.0, 0.2, 0.25, 0.3, 0.35, 0.4)


def log(*a):
    print(*a, flush=True)


def label_pool(pool_name, shrink, true_frac):
    from mask_shrink import shrink_frac, shrink_true_frac
    sh = shrink_true_frac if true_frac else shrink_frac
    cand = pd.read_parquet(SEL / pool_name)
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt_sub = gt[gt["file_name"].isin(set(cand["stem"] + ".jpeg"))]
    rle, areas = [], []
    for r in cand.itertuples():
        m2 = sh(decode_rle(r.rle).astype(bool), shrink)
        rle.append(encode_mask(m2.astype(np.uint8)))
        areas.append(int(m2.sum()))
    cand["rle"], cand["area"] = rle, areas
    cand["log10_area"] = np.log10(np.maximum(cand["area"], 1))
    pred = cand[["cand_id", "rle"]].rename(
        columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
    res = score(gt_sub[["filament_id", "segmentation_rle"]], pred, return_details=True)
    agg = res["matches"].groupby("pred_filament_id").agg(
        weight=("gt_filament_id", "size"),
        n_entries_hit=("annotator_image", "nunique"), iou_sum=("iou", "sum"))
    for c in ["weight", "n_entries_hit", "iou_sum"]:
        cand[c] = cand["cand_id"].map(agg[c]).fillna(0)
    cand["y"] = (cand["weight"] >= 1).astype(int)
    stems = gt_sub["file_name"].str.replace(".jpeg", "", regex=False)
    return cand, len(gt_sub), stems


def fit_scores(cand, seed):
    """Out-of-fold classifier probability and the two-stage score."""
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  HistGradientBoostingRegressor)
    from sklearn.model_selection import GroupKFold
    X = cand[FEATS].to_numpy(np.float64)
    y = cand["y"].to_numpy()
    v = cand["iou_sum"].to_numpy(np.float64)
    g = cand["stem"].to_numpy()
    p = np.zeros(len(y))
    e = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                           random_state=seed)
        c.fit(X[tr], y[tr])
        p[te] = c.predict_proba(X[te])[:, 1]
        pos = tr[y[tr] == 1]
        # the conditional expectation is fit on POSITIVES only: fitting iou_sum
        # unconditionally means 58% exact zeros and a squared-error loss that
        # spends itself on them, which is the form already measured to lose
        r = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                          random_state=seed)
        r.fit(X[pos], v[pos])
        e[te] = np.clip(r.predict(X[te]), 0.0, None)
    return p, p * e


def paired(cand, total_gt, gt_stems, cols, seed, n_rep=10):
    """PQ per held-out half, policy chosen on the other half ONLY."""
    files = np.array(sorted(cand["stem"].unique()))
    rng = np.random.default_rng(seed)
    out = {c: [] for c in cols}
    for _ in range(n_rep):
        perm = rng.permutation(len(files))
        half = set(files[perm[:len(files) // 2]])
        for tr_h, ev_h in ((half, set(files) - half), (set(files) - half, half)):
            ctr, cev = cand[cand["stem"].isin(tr_h)], cand[cand["stem"].isin(ev_h)]
            gt_tr = int(gt_stems.isin(tr_h).sum())
            gt_ev = total_gt - gt_tr
            for c in cols:
                # the threshold grid is built from the TRAINING half's own score
                # distribution -- quantiles over the whole pool would leak the
                # evaluation half into the policy
                qs = np.quantile(ctr[c], np.arange(0.05, 0.96, 0.025))
                best, best_pq = None, -1.0
                for cf in GRID_C:
                    for t in qs:
                        k = ctr[(ctr["conf"] >= cf) & (ctr[c] >= t)]
                        if k.empty:
                            continue
                        q = fast_pq(k, gt_tr)["pq"]
                        if q > best_pq:
                            best, best_pq = (cf, t), q
                k_ev = cev[(cev["conf"] >= best[0]) & (cev[c] >= best[1])]
                out[c].append(fast_pq(k_ev, gt_ev)["pq"])
    return {c: np.array(v) for c, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="fused_pool_f0_blobens.parquet")
    ap.add_argument("--shrink", type=float, default=0.10)
    ap.add_argument("--true-frac", action="store_true", default=True)
    ap.add_argument("--seeds", default="13,29,47,61,83,101")
    a = ap.parse_args()

    t0 = time.time()
    cand, total_gt, stems = label_pool(a.pool, a.shrink, a.true_frac)
    log(f"{a.pool}: {len(cand)} candidates, {int(cand['y'].sum())} positive, "
        f"{total_gt} ground-truth rows ({time.time()-t0:.0f}s)")
    log(f"iou_sum: {100*(cand['iou_sum'] == 0).mean():.1f}% exact zeros, "
        f"mean over positives {cand.loc[cand['y'] == 1, 'iou_sum'].mean():.3f}")

    seeds = [int(s) for s in a.seeds.split(",")]
    rows = []
    for sd in seeds:
        p, s2 = fit_scores(cand, sd)
        cand["p_cls"], cand["p_two"] = p, s2
        from scipy.stats import spearmanr
        rho_c = spearmanr(p, cand["iou_sum"]).statistic
        rho_2 = spearmanr(s2, cand["iou_sum"]).statistic
        r = paired(cand, total_gt, stems, ["p_cls", "p_two"], sd)
        d = r["p_two"] - r["p_cls"]
        se = d.std(ddof=1) / np.sqrt(len(d))
        rows.append(dict(seed=sd, cls=r["p_cls"].mean(), two=r["p_two"].mean(),
                         delta=d.mean(), se=se, t=d.mean() / se if se else np.nan,
                         wins=int((d > 0).sum()), n=len(d),
                         rho_cls=rho_c, rho_two=rho_2))
        log(f"  seed {sd:3d}: classifier {r['p_cls'].mean():.4f}  two-stage "
            f"{r['p_two'].mean():.4f}  delta {d.mean():+.4f} "
            f"(se {se:.4f}, t {rows[-1]['t']:+.2f}, wins {rows[-1]['wins']}/{len(d)})")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "two_stage_selector.csv", index=False)
    pooled = df["delta"].mean()
    pooled_se = df["delta"].std(ddof=1) / np.sqrt(len(df))
    log(f"\npooled over {len(df)} seeds: delta {pooled:+.4f} "
        f"+/- {pooled_se:.4f} (seed-to-seed), "
        f"{int((df['delta'] > 0).sum())}/{len(df)} seeds positive")
    log(f"  rank correlation with iou_sum: classifier "
        f"{df['rho_cls'].mean():.4f}, two-stage {df['rho_two'].mean():.4f}")
    log("\n  thresholds came from the training half only; taking them over the")
    log("  whole pool leaks the evaluation half into the policy and was measured")
    log("  elsewhere to inflate deltas by about a third.")

    json.dump(dict(pool=a.pool, pooled_delta=float(pooled),
                   pooled_se=float(pooled_se),
                   per_seed=df.to_dict("records")),
              open(OUT / "two_stage_selector.json", "w"), indent=1, default=float)
    log(f"\nwrote {OUT/'two_stage_selector.json'}")


if __name__ == "__main__":
    main()
