"""E096 v2 -- RESCUE ranker on the candidate bank.

The decomposition of the fold-0 legal-oracle gain (ledger §10c) put 65% of
the ceiling's utility in 84 clusters whose production mask (vote50) does not
reach IoU 0.5 while some other variant (mostly semcut or union) does. Both the
capture study and the v1 ranker held the reject fixed at "vote50 IoU > 0.5",
so they could never touch those clusters; they competed in the matched arena,
which holds 35% of the ceiling and is 71% annotator-style noise (second-best
variant within 0.02 IoU of the best).

This asks the right question: among clusters the production mask MISSES, can
a model that cannot see the ground truth find the ones some other variant
HITS, at a precision above break-even?

Marginal PQ of adding one prediction to a pool at PQ0 with denominator D:
  true rescue with IoU s:  (s - PQ0/2) / D      (TP+1, FN-1 -> D + 1/2)
  false rescue:            (-PQ0/2) / D         (FP+1        -> D + 1/2)
so the break-even precision is PQ0 / (2 s); at PQ0 0.536 and s ~ 0.6 that is
~45%, against a base rate of 84/1002 = 8.4%. The threshold on the rescue
probability is chosen on the TRAINING images only (nested group cross-fit),
maximising sum(s_hat - PQ0/2) over rescued clusters with s_hat the realised
utility -- i.e. the same PQ-shaped objective the exact scorer will apply.

Two models, one hypothesis ("rescue is learnable above break-even"):
  R  cluster-level classifier: P(any non-vote50 variant reaches IoU > 0.5),
     from the vote50 row's features, the production mask's width, and the
     semantic/area features of semcut, union, er_vote and vote65 relative to
     vote50 (no GT);
  V  which variant to emit for a rescued cluster: the per-variant IoU
     regressor of the capture study (OOF Spearman 0.60), argmax over the
     non-vote50 variants.

Arms (all with the exact scorer, fixed production reject for the base pool):
  prod        vote50 for clusters with vote50 IoU > 0.5        (§10b prod)
  rescue      prod + rescued clusters' chosen variant, legal pass
  rescue_orc  prod + the 84 true rescues with their best variant (upper bound
              of this policy on this fold; equals the ceiling's rescue part)

Usage: python src/eda/candidate_bank_rescue.py --fold 0
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
from candidate_bank_capture import legal_pass, FEATS

OUT = ROOT / "outputs" / "eda"
REL = ["semcut", "union", "er_vote", "vote65"]
RELF = ["area_ratio_to_vote50", "sem_prob_mean", "sem_prob_p90", "sem_frac_over_50"]


def log(*a):
    print(*a, flush=True)


def hgb_cls(seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=30, l2_regularization=1.0,
                                          class_weight="balanced", random_state=seed)


def hgb_reg(seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=seed)


def cluster_table(vb, dec):
    base = vb[vb["variant"] == "vote50"].set_index("cluster_id")
    cols = {f"b_{c}": base[c].to_numpy() for c in
            ["area", "r_norm", "conf", "n_members", "n_det", "from_det", "from_sem",
             "det_conf_mean", "sem_prob_mean", "sem_prob_p90", "sem_frac_over_50"]}
    cols["b_log_area"] = np.log1p(base["area"].to_numpy())
    cols["b_width"] = np.array([mean_width_px(dec(int(i))) for i in base["row"]])
    for v in REL:
        r = vb[vb["variant"] == v].set_index("cluster_id").reindex(base.index)
        for f in RELF:
            x = r[f].to_numpy()
            if f != "area_ratio_to_vote50":
                x = x - base[f].to_numpy()
            cols[f"{v}_{f}"] = x
    X = pd.DataFrame(cols, index=base.index)
    # targets
    non = vb[(vb["variant"] != "vote50") & vb["cluster_id"].isin(base.index)]
    best = non.loc[non.groupby("cluster_id")["iou"].idxmax()].set_index("cluster_id").reindex(base.index)
    meta = pd.DataFrame({
        "stem": base["stem"], "iou_b": base["iou"], "conf_b": base["conf"], "row_b": base["row"],
        "best_iou": best["iou"], "best_variant": best["variant"], "best_row": best["row"],
    }, index=base.index)
    return X, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--splits", type=int, default=5)
    a = ap.parse_args()
    vb = pd.read_parquet(OUT / f"variant_bank_fold{a.fold}.parquet")
    vb["row"] = vb.index
    vb["v_idx"] = vb["variant"].map({v: i for i, v in enumerate(sorted(vb["variant"].unique()))})
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt = gt[gt["stem"].isin(set(vb["stem"]))][["filament_id", "segmentation_rle"]]
    rle_of = dict(zip(vb.index, vb["rle"]))

    def dec(i):
        return decode_rle(rle_of[i]).astype(bool)

    X, meta = cluster_table(vb, dec)
    miss = meta["iou_b"] <= 0.5
    y_resc = ((meta["best_iou"] > 0.5) & miss).to_numpy()
    log(f"fold {a.fold}: {len(meta)} clusters; production misses {int(miss.sum())}, "
        f"of which {int(y_resc.sum())} are rescuable by some variant "
        f"(base rate {100*y_resc.sum()/miss.sum():.1f}%)")
    log("  rescuable clusters' best variant: " +
        ", ".join(f"{k} {v}" for k, v in meta.loc[y_resc, 'best_variant'].value_counts().items()))
    log(f"  rescuable clusters' best IoU: median {meta.loc[y_resc, 'best_iou'].median():.3f}, "
        f"mean {meta.loc[y_resc, 'best_iou'].mean():.3f}")

    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score, average_precision_score
    # --- R: rescue classifier, trained and applied on MISSED clusters only
    Xm = X[miss].to_numpy(np.float64)
    ym = y_resc[miss.to_numpy()]
    gm = meta.loc[miss, "stem"].to_numpy()
    mm = meta[miss]
    p = np.zeros(len(ym))
    thr = []
    PQ0 = None  # filled after the prod arm is scored; threshold uses utility below

    def utility_at(prob, sub, t, pq0):
        pick = prob >= t
        s = sub["best_iou"].to_numpy()
        hit = (s > 0.5) & pick
        fal = (s <= 0.5) & pick
        return float((s[hit] - pq0 / 2).sum() - (pq0 / 2) * fal.sum())

    grid = np.round(np.arange(0.3, 0.96, 0.02), 2)
    pq0_guess = 0.536
    for tr, te in GroupKFold(n_splits=a.splits).split(Xm, ym, gm):
        m = hgb_cls().fit(Xm[tr], ym[tr])
        p[te] = m.predict_proba(Xm[te])[:, 1]
        inner = np.zeros(len(tr))
        for itr, ite in GroupKFold(n_splits=a.splits).split(Xm[tr], ym[tr], gm[tr]):
            inner[ite] = hgb_cls(seed=1).fit(Xm[tr][itr], ym[tr][itr]).predict_proba(Xm[tr][ite])[:, 1]
        vals = [(t, utility_at(inner, mm.iloc[tr], t, pq0_guess)) for t in grid]
        thr.append(max(vals, key=lambda z: z[1])[0])
    t_star = float(np.median(thr))
    mm = mm.assign(p=p)
    log(f"\n  R (OOF): AUC {roc_auc_score(ym, p):.3f}, AP {average_precision_score(ym, p):.3f} "
        f"(base {ym.mean():.3f}); nested thresholds {thr} -> t* = {t_star:.2f}")
    log("  OOF precision/recall of the rescue set by threshold:")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pick = p >= t
        n = int(pick.sum())
        prec = ym[pick].mean() if n else float("nan")
        log(f"    t={t:.2f}  n={n:>4}  precision {100*prec:5.1f}%  recall {100*(ym & pick).sum()/ym.sum():5.1f}%  "
            f"utility {utility_at(p, mm, t, pq0_guess):+.2f}")

    # --- V: which variant for a rescued cluster (per-variant IoU regressor, OOF)
    non = vb[(vb["variant"] != "vote50")]
    Xv = non[FEATS].to_numpy(np.float64)
    yv = non["iou"].to_numpy()
    gv = non["stem"].to_numpy()
    pv = np.zeros(len(non))
    for tr, te in GroupKFold(n_splits=a.splits).split(Xv, yv, gv):
        pv[te] = hgb_reg().fit(Xv[tr], yv[tr]).predict(Xv[te])
    non = non.assign(pred_iou=pv)
    choice = non.loc[non.groupby("cluster_id")["pred_iou"].idxmax()].set_index("cluster_id")
    agree = (choice.loc[mm.index[ym], "variant"].to_numpy() == mm.loc[ym, "best_variant"].to_numpy()).mean()
    hit_if_chosen = (choice.loc[mm.index[ym], "iou"] > 0.5).mean()
    log(f"\n  V: on the {int(ym.sum())} rescuable clusters the chosen variant is the oracle's in "
        f"{100*agree:.0f}% and reaches IoU > 0.5 in {100*hit_if_chosen:.0f}%")

    # --- arms
    base_keep = meta[~miss]
    prod = [(r["stem"], float(r["conf_b"]), dec(int(r["row_b"]))) for _, r in base_keep.iterrows()]

    def run(name, extra, note=""):
        pdf = legal_pass(prod + extra, None)
        r = score(gt, pdf)
        res[name] = dict(n=len(pdf), pq=r["pq"], sq=r["sq"], rq=r["rq"], fn=r["fn"], fp=r["fp"], added=len(extra))
        log(f"  {name:<12}{len(pdf):>7}{r['pq']:>9.4f}{r['sq']:>9.4f}{r['rq']:>9.4f}{r['fn']:>6}{r['fp']:>6}{len(extra):>7}  {note}")
        return r["pq"]

    res = {}
    log(f"\n  {'arm':<12}{'preds':>7}{'PQ':>9}{'SQ':>9}{'RQ':>9}{'FN':>6}{'FP':>6}{'added':>7}")
    pq_prod = run("prod", [])
    picked = mm[mm["p"] >= t_star]
    extra = []
    for cid, r in picked.iterrows():
        c = choice.loc[cid]
        # de-overlap key: the rescue's own predicted IoU scaled below production
        # confidence range, so rescues never displace a production mask
        extra.append((r["stem"], float(c["pred_iou"]) * 0.5, dec(int(c["row"]))))
    n_true = int((picked["best_iou"] > 0.5).sum())
    run("rescue", extra, f"t*={t_star:.2f}, {n_true} of {len(picked)} rescuable, "
                         f"{int((choice.loc[picked.index, 'iou'] > 0.5).sum())} actually hit")
    orc = mm[ym]
    extra_o = [(r["stem"], 0.5, dec(int(r["best_row"]))) for _, r in orc.iterrows()]
    run("rescue_orc", extra_o, "upper bound: true rescues, oracle variant")

    oracle = json.load(open(OUT / f"candidate_bank_oracle_fold{a.fold}.json"))
    ceil = oracle["res"]["legal"]["pq"] - oracle["res"]["prod"]["pq"]
    log(f"\n  legal ceiling +{ceil:.4f}; gate 25% -> +{0.25*ceil:.4f}")
    for k, v in res.items():
        d = v["pq"] - pq_prod
        log(f"  {k:<12}{d:>+9.4f}   {100*d/ceil:>5.0f}% of ceiling")
    json.dump(dict(fold=a.fold, res=res, t_star=t_star, thresholds=thr,
                   auc=float(roc_auc_score(ym, p)), ap=float(average_precision_score(ym, p)),
                   base_rate=float(ym.mean()), ceiling=ceil, features=list(X.columns)),
              open(OUT / f"candidate_bank_rescue_fold{a.fold}.json", "w"), indent=1)
    log(f"\nwrote {OUT/f'candidate_bank_rescue_fold{a.fold}.json'}")


if __name__ == "__main__":
    main()
