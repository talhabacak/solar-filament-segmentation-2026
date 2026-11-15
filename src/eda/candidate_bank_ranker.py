"""E096 v1 -- asymmetric, cluster-relative variant ranker on the candidate bank.

The capture study (ledger §10b) showed that a naive "predict each variant's IoU
and take the argmax" learner LOSES (-0.020 PQ on fold 0) even though its
predictions correlate with IoU (Spearman 0.60): a wrong deviation swaps a
matched vote50 for an unmatched variant, and that costs more than a right
deviation gains. Three changes, each dictated by that result:

  1. the target is the DELTA against the production mask,
       d = IoU(variant) - IoU(vote50),  per cluster,
     and the decision is asymmetric: vote50 by default, deviate to the
     argmax-d variant only if the predicted d exceeds a margin m > 0;
  2. features are CLUSTER-RELATIVE: every variant is described against its
     own vote50 (area ratio, semantic-probability difference, agreement
     difference) plus cluster-level structure that the bank already carries or
     that is cheap to derive (member count, detector/semantic origin, the
     union/vote50 and vote65/vote50 area ratios as member-disagreement proxies,
     mean width of the vote50 mask, normalised radius);
  3. the margin is chosen on the TRAINING images only, by nested group
     cross-fit, maximising a PQ-shaped utility rather than raw IoU:
       u = c(IoU_variant) - c(IoU_vote50),  c(x) = x if x > 0.5 else 0,
     so a deviation that drops a match below 0.5 is charged its full loss.

Reject decision is held FIXED at the oracle-production arm (vote50 IoU > 0.5),
exactly as in the capture study, so the number is comparable to §10b and to
the legal-oracle ceiling: the question is still only WHICH mask per kept
cluster. Scores use the exact scorer on the fold's GT rows.

Pre-registered gate (ledger §10): >= 25% of the legal-ceiling gain on the
fold, i.e. >= +0.012 PQ on fold 0, before anything moves toward the selector.

Usage: python src/eda/candidate_bank_ranker.py --fold 0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "eda"))

from filament.pq import score
from filament.rle import decode_rle
from mask_shrink import mean_width_px
from candidate_bank_capture import legal_pass

OUT = ROOT / "outputs" / "eda"
VARIANTS = ["vote65", "top", "semtop", "union", "semcut", "er_vote", "er_top"]


def log(*a):
    print(*a, flush=True)


def util(iou):
    return np.where(iou > 0.5, iou, 0.0)


def build_table(vb, dec, extra_cols=()):
    """One row per (cluster, non-vote50 variant) with cluster-relative
    features and the delta target. Clusters without a vote50 row are dropped
    (the production policy cannot emit them anyway)."""
    base = vb[vb["variant"] == "vote50"].set_index("cluster_id")
    vb = vb[vb["cluster_id"].isin(base.index) & (vb["variant"] != "vote50")].copy()
    b = base.loc[vb["cluster_id"]]
    # width of the production mask: 1 decode per cluster (~1.9k), cached
    width = {cid: mean_width_px(dec(int(i))) for cid, i in zip(base.index, base["row"])}
    # member-disagreement proxies from the bank itself (no decode)
    piv = vb.pivot_table(index="cluster_id", columns="variant", values="area", aggfunc="first")
    union_ratio = (piv["union"] / base["area"]).reindex(vb["cluster_id"]).to_numpy()
    v65_ratio = (piv["vote65"] / base["area"]).reindex(vb["cluster_id"]).to_numpy()
    X = pd.DataFrame({
        "v_idx": vb["variant"].map({v: i for i, v in enumerate(VARIANTS)}).to_numpy(),
        "area_ratio": vb["area_ratio_to_vote50"].to_numpy(),
        "log_area_b": np.log1p(b["area"].to_numpy()),
        "width_b": np.array([width[c] for c in vb["cluster_id"]]),
        "r_norm": b["r_norm"].to_numpy(),
        "conf_b": b["conf"].to_numpy(),
        "n_members": b["n_members"].to_numpy(),
        "n_det": b["n_det"].to_numpy(),
        "from_det": b["from_det"].to_numpy(),
        "from_sem": b["from_sem"].to_numpy(),
        "det_conf_mean": b["det_conf_mean"].to_numpy(),
        "union_ratio": union_ratio,
        "v65_ratio": v65_ratio,
        "sem_prob_b": b["sem_prob_mean"].to_numpy(),
        "d_sem_prob": vb["sem_prob_mean"].to_numpy() - b["sem_prob_mean"].to_numpy(),
        "d_sem_p90": vb["sem_prob_p90"].to_numpy() - b["sem_prob_p90"].to_numpy(),
        "d_sem_frac": vb["sem_frac_over_50"].to_numpy() - b["sem_frac_over_50"].to_numpy(),
    }, index=vb.index)
    # S002: boundary-band probability profile, raw for the variant and as the
    # delta against the cluster's vote50 (both, since the ranker is cluster-relative)
    for c in extra_cols:
        X[c] = vb[c].to_numpy()
        X["d_" + c] = vb[c].to_numpy() - b[c].to_numpy()
    d = vb["iou"].to_numpy() - b["iou"].to_numpy()
    u = util(vb["iou"].to_numpy()) - util(b["iou"].to_numpy())
    meta = pd.DataFrame({
        "stem": vb["stem"].to_numpy(), "cluster_id": vb["cluster_id"].to_numpy(),
        "variant": vb["variant"].to_numpy(), "iou": vb["iou"].to_numpy(),
        "iou_b": b["iou"].to_numpy(), "conf_b": b["conf"].to_numpy(),
        "row_b": b["row"].to_numpy(), "d": d, "u": u,
    }, index=vb.index)
    return X, meta


def fit_predict(X, y, grp, tr, te, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                      min_samples_leaf=40, l2_regularization=1.0, random_state=seed)
    m.fit(X[tr], y[tr])
    return m.predict(X[te])


def choose_margin(pred, meta_tr, grid):
    """Utility of the asymmetric policy on the training images at each margin:
    per cluster, take the argmax-pred variant; deviate iff pred > m; sum u."""
    df = meta_tr.assign(pred=pred)
    best = df.loc[df.groupby("cluster_id")["pred"].idxmax()]
    vals = [(m, float(best.loc[best["pred"] > m, "u"].sum())) for m in grid]
    m_star = max(vals, key=lambda t: t[1])[0]
    return m_star, vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--extra", default=None,
                    help="join outputs/eda/variant_bank_fold{k}_<extra>.parquet and add its columns "
                         "(raw + delta vs vote50) to the ranker; 'ring' = S002 boundary-band profile")
    a = ap.parse_args()
    vb = pd.read_parquet(OUT / f"variant_bank_fold{a.fold}.parquet")
    extra_cols = ()
    if a.extra:
        ex = pd.read_parquet(OUT / f"variant_bank_fold{a.fold}_{a.extra}.parquet")
        extra_cols = tuple(ex.columns)
        vb = vb.join(ex)
        log(f"  extra feature family '{a.extra}': {len(extra_cols)} columns")
    vb["row"] = vb.index
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt = gt[gt["stem"].isin(set(vb["stem"]))][["filament_id", "segmentation_rle"]]
    rle_of = dict(zip(vb.index, vb["rle"]))

    def dec(i):
        return decode_rle(rle_of[i]).astype(bool)

    X, meta = build_table(vb, dec, extra_cols)
    log(f"fold {a.fold}: {meta['cluster_id'].nunique()} clusters with a vote50 row, "
        f"{len(meta)} candidate deviations, {len(gt)} GT rows")
    log(f"  delta target: mean {meta['d'].mean():+.4f}, >0 in {100*(meta['d']>0).mean():.1f}% of rows; "
        f"utility >0 in {100*(meta['u']>0).mean():.1f}%, <0 in {100*(meta['u']<0).mean():.1f}%")

    from sklearn.model_selection import GroupKFold
    Xn = X.to_numpy(np.float64)
    y = meta["d"].to_numpy()
    grp = meta["stem"].to_numpy()
    grid = np.round(np.arange(0.0, 0.31, 0.01), 2)
    pred = np.zeros(len(meta))
    margins = []
    for tr, te in GroupKFold(n_splits=a.splits).split(Xn, y, grp):
        pred[te] = fit_predict(Xn, y, grp, tr, te)
        # nested: out-of-fold predictions INSIDE the training part choose m
        inner = np.zeros(len(tr))
        for itr, ite in GroupKFold(n_splits=a.splits).split(Xn[tr], y[tr], grp[tr]):
            inner[ite] = fit_predict(Xn[tr], y[tr], grp[tr], itr, ite, seed=1)
        m_star, _ = choose_margin(inner, meta.iloc[tr], grid)
        margins.append(m_star)
    meta["pred"] = pred
    m_star = float(np.median(margins))
    from scipy.stats import spearmanr
    log(f"  OOF Spearman(pred d, d) = {spearmanr(pred, y).statistic:.3f}; "
        f"nested margins per split {margins} -> m* = {m_star:.2f}")

    # --- decisions under the FIXED production reject (vote50 IoU > 0.5)
    kept = meta[meta["iou_b"] > 0.5]
    best = kept.loc[kept.groupby("cluster_id")["pred"].idxmax()]
    base = vb[vb["variant"] == "vote50"]
    base = base[base["iou"] > 0.5]

    def emit(policy_margin):
        """Production mask per kept cluster, replaced by the argmax variant
        (bank row index = best.index) wherever the predicted delta beats the
        margin. The de-overlap key stays the production confidence so the
        legal pass orders identically across arms."""
        dev = best[best["pred"] > policy_margin]
        dev_row = dict(zip(dev["cluster_id"], dev.index))
        chosen = []
        for _, r in base.iterrows():
            i = dev_row.get(r["cluster_id"], int(r["row"]))
            chosen.append((r["stem"], float(r["conf"]), dec(int(i))))
        return chosen, dev

    res = {}
    log(f"\n  {'arm':<16}{'preds':>7}{'PQ':>9}{'SQ':>9}{'RQ':>9}{'FN':>6}{'FP':>6}{'dev':>6}{'dev+':>6}{'dev-':>6}")
    for name, m in [("prod", 9.9), (f"asym_m{m_star:.2f}", m_star), ("asym_m0", 0.0)]:
        chosen, dev = emit(m)
        pdf = legal_pass(chosen, None)
        r = score(gt, pdf)
        nplus = int((dev["u"] > 0).sum()) if len(dev) else 0
        nminus = int((dev["u"] < 0).sum()) if len(dev) else 0
        res[name] = dict(n=len(pdf), pq=r["pq"], sq=r["sq"], rq=r["rq"], fn=r["fn"], fp=r["fp"],
                         deviations=int(len(dev)), dev_help=nplus, dev_hurt=nminus, margin=m)
        log(f"  {name:<16}{len(pdf):>7}{r['pq']:>9.4f}{r['sq']:>9.4f}{r['rq']:>9.4f}{r['fn']:>6}{r['fp']:>6}"
            f"{len(dev):>6}{nplus:>6}{nminus:>6}")

    # what the margin buys: realised utility of deviations at each margin (OOF)
    log("\n  OOF realised utility of the deviation set by margin (kept clusters):")
    for m in [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2]:
        dv = best[best["pred"] > m]
        log(f"    m={m:.2f}  n={len(dv):>4}  sum u={dv['u'].sum():+.2f}  sum d={dv['d'].sum():+.2f}  "
            f"help {int((dv['u']>0).sum()):>3}  hurt {int((dv['u']<0).sum()):>3}")

    oracle = json.load(open(OUT / f"candidate_bank_oracle_fold{a.fold}.json"))
    ceil = oracle["res"]["legal"]["pq"] - oracle["res"]["prod"]["pq"]
    log(f"\n  legal ceiling on this fold: +{ceil:.4f}; gate = 25% -> +{0.25*ceil:.4f}")
    for k, v in res.items():
        d = v["pq"] - res["prod"]["pq"]
        log(f"  {k:<16}{d:>+9.4f}   {100*d/ceil:>5.0f}% of ceiling")
    json.dump(dict(fold=a.fold, res=res, margin=m_star, margins=margins,
                   spearman=float(spearmanr(pred, y).statistic), ceiling=ceil,
                   features=list(X.columns)),
              open(OUT / f"candidate_bank_ranker_fold{a.fold}{('_' + a.extra) if a.extra else ''}.json", "w"), indent=1)
    log(f"\nwrote {OUT/f'candidate_bank_ranker_fold{a.fold}.json'}")


if __name__ == "__main__":
    main()
