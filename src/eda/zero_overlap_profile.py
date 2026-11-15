"""What are the 282 ground-truth rows nothing we keep even touches?

They are the largest single failure block -- 18.7% of fold-0's ground truth, more
than the under-covered (9.5%) and over-covered (7.4%) near-misses combined -- and
the only one that post-processing cannot reach: there is no mask to adjust.

The first thing to establish is which stage lost them, because the answer picks
the remedy and the two are not close:

  dropped by the SELECTOR   a pool candidate overlaps the row, and the deployed
                            policy did not keep it. A selection problem, and the
                            selector is cross-fitted and its operating point is
                            flat from 0.26 to 0.60, so there is little to gain.
  never PROPOSED            no pool candidate overlaps it at all. A detection
                            problem, and the only kind of failure left that a
                            model change could address.

Then the population is characterised against the rows we do match, on appearance,
geometry, position, provenance and annotator agreement, with a grouped
cross-validated classifier to say which of those carry signal rather than which
merely correlate. And finally the semantic probability field is read underneath
them: a model that fires at 0.3 under a row it never proposed is failing at
extraction, while one that fires at 0.02 is failing at representation, and only
the second is a training problem.

Usage: python src/eda/zero_overlap_profile.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament.rle import decode_rle

OUT = ROOT / "outputs" / "eda"
SEL = ROOT / "outputs" / "selector"
SEM = ROOT / "outputs" / "semantic"
PROBS = "fold0_probs_cldice.npz"


def log(*a):
    print(*a, flush=True)


def boxed(m):
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return None
    return ((int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1),
            m[ys.min():ys.max() + 1, xs.min():xs.max() + 1], int(m.sum()))


def overlaps(a, b):
    (ay0, ay1, ax0, ax1), asub, _ = a
    (by0, by1, bx0, bx1), bsub, _ = b
    y0, y1 = max(ay0, by0), min(ay1, by1)
    x0, x1 = max(ax0, bx0), min(ax1, bx1)
    if y0 >= y1 or x0 >= x1:
        return False
    return bool(np.logical_and(asub[y0 - ay0:y1 - ay0, x0 - ax0:x1 - ax0],
                               bsub[y0 - by0:y1 - by0, x0 - bx0:x1 - bx0]).any())


def main():
    d = pd.read_parquet(OUT / "pixel_vs_instance_errors.parquet")
    g = d[d.kind == "gt"].copy()
    zero = set(g.loc[g["outcome"] == "no overlapping prediction", "id"])
    matched = set(g.loc[g["outcome"] == "matched", "id"])
    log(f"fold-0: {len(zero)} zero-overlap rows, {len(matched)} matched")

    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    f0 = set(folds.loc[folds["fold"] == 0, "file_name"]
             .str.replace(".jpeg", "", regex=False))
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt = gt[gt["stem"].isin(f0)]
    pool = pd.read_parquet(SEL / "fused_pool_all.parquet")
    pool = pool[pool["stem"].isin(f0)]

    # ---- 1. which stage lost them ------------------------------------------
    by_stem = {s: list(x["rle"]) for s, x in pool.groupby("stem")}
    rows, t0 = [], time.time()
    for stem, sub in gt[gt["filament_id"].isin(zero)].groupby("stem"):
        cands = [boxed(decode_rle(r).astype(bool)) for r in by_stem.get(stem, [])]
        cands = [c for c in cands if c is not None]
        for r in sub.itertuples():
            gm = boxed(decode_rle(r.segmentation_rle).astype(bool))
            hit = any(overlaps(gm, c) for c in cands) if gm else False
            rows.append(dict(filament_id=r.filament_id, in_pool=hit))
    stage = pd.DataFrame(rows)
    n_dropped = int(stage["in_pool"].sum())
    log(f"\n1. WHICH STAGE LOST THEM ({time.time()-t0:.0f}s)")
    log(f"   a pool candidate DOES overlap, the selector dropped it : "
        f"{n_dropped:4d} ({100*n_dropped/len(stage):.1f}%)")
    log(f"   nothing in the pool overlaps them at all               : "
        f"{len(stage)-n_dropped:4d} ({100*(1-n_dropped/len(stage)):.1f}%)")
    log("   the second group is the only one a model change can reach.")

    # ---- 2. what distinguishes them ----------------------------------------
    prof = pd.read_parquet(OUT / "error_profile_fn_gt.parquet")
    prof = prof[prof["filament_id"].isin(zero | matched)].copy()
    prof["grp"] = np.where(prof["filament_id"].isin(zero), "zero-overlap", "matched")
    cols = [c for c in ["contrast", "mask_area", "spine_len", "width_proxy",
                        "eccentricity", "solidity", "n_cc", "depth", "edge_grad",
                        "fg_min", "r_norm", "abs_lat", "n_gt_image", "nn_dist"]
            if c in prof.columns]
    log(f"\n2. HOW THEY DIFFER FROM MATCHED ROWS (medians)")
    t = prof.groupby("grp")[cols].median().T
    t["ratio"] = t["zero-overlap"] / t["matched"].replace(0, np.nan)
    log(t.to_string(float_format=lambda v: f"{v:.3f}"))

    log("\n   consensus tier composition (column %):")
    ct = pd.crosstab(prof["consensus_tier"], prof["grp"], normalize="columns") * 100
    log("   " + ct.to_string(float_format=lambda v: f"{v:5.1f}").replace("\n", "\n   "))

    if "site_name" in prof.columns:
        st = pd.crosstab(prof["site_name"], prof["grp"], normalize="columns") * 100
        log("\n   station composition (column %):")
        log("   " + st.to_string(float_format=lambda v: f"{v:5.1f}").replace("\n", "\n   "))

    # which features carry signal, not merely correlate
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.inspection import permutation_importance
    sub = prof.dropna(subset=cols)
    X = sub[cols].to_numpy(np.float64)
    y = (sub["grp"] == "zero-overlap").astype(int).to_numpy()
    grp = sub["stem"].to_numpy()
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, grp):
        m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y, oof)
    log(f"\n   grouped-CV separability of zero-overlap from matched: AUC {auc:.4f}")
    m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06).fit(X, y)
    imp = permutation_importance(m, X, y, n_repeats=6, random_state=0)
    order = np.argsort(-imp.importances_mean)
    log("   permutation importance (what actually carries it):")
    for i in order[:8]:
        log(f"     {cols[i]:18s} {imp.importances_mean[i]:+.4f}")

    # ---- 3. does the model fire under them at all? -------------------------
    z = np.load(SEM / PROBS)
    gtr = gt.set_index("filament_id")["segmentation_rle"]
    stats, t0 = [], time.time()
    for stem, sub2 in gt[gt["filament_id"].isin(zero | matched)].groupby("stem"):
        if stem not in z:
            continue
        prob = np.asarray(z[stem]).astype(np.float32) / 255.0
        for r in sub2.itertuples():
            m_ = decode_rle(gtr.loc[r.filament_id]).astype(bool)
            if not m_.any():
                continue
            pv = prob[m_]
            stats.append(dict(filament_id=r.filament_id,
                              grp=("zero-overlap" if r.filament_id in zero else "matched"),
                              p_mean=float(pv.mean()), p_max=float(pv.max()),
                              p_p90=float(np.percentile(pv, 90)),
                              frac_over_30=float((pv >= 0.3).mean())))
    s = pd.DataFrame(stats)
    log(f"\n3. DOES THE SEMANTIC FIELD FIRE UNDER THEM? ({time.time()-t0:.0f}s)")
    log(s.groupby("grp")[["p_mean", "p_max", "p_p90", "frac_over_30"]].median()
        .to_string(float_format=lambda v: f"{v:.3f}"))
    zo = s[s.grp == "zero-overlap"]
    log(f"\n   zero-overlap rows with max probability below 0.3: "
        f"{int((zo['p_max'] < 0.3).sum())} of {len(zo)} "
        f"({100*(zo['p_max'] < 0.3).mean():.1f}%)")
    log("   those are invisible to the model, not merely unextracted; the rest")
    log("   are lit but never survived into a component the pool kept.")

    s.to_parquet(OUT / "zero_overlap_profile.parquet", index=False)
    json.dump(dict(n_zero=len(zero), dropped_by_selector=n_dropped,
                   never_proposed=len(stage) - n_dropped, separability_auc=float(auc),
                   invisible=int((zo["p_max"] < 0.3).sum())),
              open(OUT / "zero_overlap_profile.json", "w"), indent=1)
    log(f"\nwrote {OUT/'zero_overlap_profile.json'}")


if __name__ == "__main__":
    main()
