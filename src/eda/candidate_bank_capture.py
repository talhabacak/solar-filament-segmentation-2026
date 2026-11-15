"""E095 / E096-lite -- how much of the legal-oracle gain is capturable without GT?

The oracle (candidate_bank_oracle.py) chose, per cluster, the variant with the
best ground-truth IoU: +0.0468 PQ and +132 rows on fold 0. That is the ceiling.
This asks what a rule that cannot see the ground truth recovers, on the same
variant bank, with the same legal (non-overlap) pass and the same exact scorer.

Arms, each a single hypothesis:
  prod        vote50 everywhere, greedy de-overlap by confidence (production)
  R_erode     er_vote everywhere (the oracle picked it 336 times)
  R_union     union everywhere
  R_thin      er_vote if the vote50 mask is thin (mean width < 9 px) else vote50
  R_disagree  union if the members disagree (area_ratio(union/vote50) > 1.5)
              else vote50
  L_iou       LEARNED: an out-of-fold (GroupKFold by image) regressor predicts
              each variant's IoU from cluster/variant features; per cluster take
              the argmax, reject if the prediction is below a threshold chosen on
              the training folds only. This is E096-lite -- one feature group,
              one model, no tuning beyond the reject threshold.

Every arm keeps the reject decision of the production policy out of scope: the
question here is only WHICH mask per cluster, so all arms keep the same
clusters the production keeps (those whose vote50 mask survives), except L_iou
which may also reject. Scores are on fold 0's 1510 GT rows.

Usage: python src/eda/candidate_bank_capture.py --fold 0
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

from filament import IMG_H, IMG_W
from filament.pq import score
from filament.rle import decode_rle, encode_mask
from mask_shrink import mean_width_px

OUT = ROOT / "outputs" / "eda"
FEATS = ["area", "r_norm", "conf", "n_members", "n_det", "from_det", "from_sem",
         "det_conf_mean", "sem_prob_mean", "sem_prob_p90", "sem_frac_over_50",
         "area_ratio_to_vote50", "v_idx"]


def log(*a):
    print(*a, flush=True)


def legal_pass(chosen, order_key):
    """chosen: list of (stem, key, mask). Non-overlap by descending key; a mask
    that collides is carved and kept if anything is left. Returns pred rows."""
    preds = []
    by_stem = {}
    for stem, key, m in chosen:
        by_stem.setdefault(stem, []).append((key, m))
    for stem, lst in by_stem.items():
        lst.sort(key=lambda t: -t[0])
        taken = np.zeros((IMG_H, IMG_W), bool)
        k = 0
        for key, m in lst:
            mm = m & ~taken
            if mm.sum() < 64:
                continue
            taken |= mm
            k += 1
            preds.append((f"{stem}_{k}", encode_mask(mm.astype(np.uint8))))
    return pd.DataFrame(preds, columns=["filament_id", "segmentation_rle"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    a = ap.parse_args()
    vb = pd.read_parquet(OUT / f"variant_bank_fold{a.fold}.parquet")
    vb["v_idx"] = vb["variant"].map({v: i for i, v in enumerate(sorted(vb["variant"].unique()))})
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt = gt[gt["stem"].isin(set(vb["stem"]))]
    gt_df = gt[["filament_id", "segmentation_rle"]]
    log(f"fold {a.fold}: {vb['cluster_id'].nunique()} clusters, {len(vb)} variants, "
        f"{len(gt)} GT rows")

    # masks are decoded on demand: 14,832 variants at 4 MiB each is 59 GB,
    # which killed the first run. Only the chosen mask per cluster is kept.
    rle_of = dict(zip(vb.index, vb["rle"]))

    def dec(i):
        return decode_rle(rle_of[i]).astype(bool)

    piv = {cid: g for cid, g in vb.groupby("cluster_id")}

    def pick(rule):
        """Reject decision held FIXED at the oracle's production arm: a cluster
        is kept iff its vote50 mask matches ground truth at IoU > 0.5. So every
        rule arm keeps exactly the clusters production-oracle keeps, and the
        arms differ only in WHICH variant they emit for those clusters."""
        chosen = []
        for cid, g in piv.items():
            base = g[g["variant"] == "vote50"]
            if base.empty:
                continue
            b = base.iloc[0]
            if b["iou"] <= 0.5:
                continue
            v = rule(g, b)
            if v is None:
                continue
            row = g[g["variant"] == v]
            if row.empty:
                row = base
            r = row.iloc[0]
            chosen.append((r["stem"], float(r["conf"]), dec(row.index[0])))
        return chosen

    def width_of(idx):
        return mean_width_px(dec(idx))

    rules = {
        "prod":       lambda g, b: "vote50",
        "R_erode":    lambda g, b: "er_vote",
        "R_union":    lambda g, b: "union",
        "R_thin":     lambda g, b: "er_vote" if width_of(b.name) < 9.0 else "vote50",
        "R_disagree": lambda g, b: ("union" if (g.loc[g["variant"] == "union", "area_ratio_to_vote50"].max() > 1.5)
                                    else "vote50"),
    }
    res = {}
    log(f"\n  {'arm':<12}{'preds':>7}{'PQ':>9}{'SQ':>9}{'RQ':>9}{'FN':>6}{'FP':>6}")
    for name, rule in rules.items():
        pdf = legal_pass(pick(rule), None)
        r = score(gt_df, pdf)
        res[name] = dict(n=len(pdf), pq=r["pq"], sq=r["sq"], rq=r["rq"], fn=r["fn"], fp=r["fp"])
        log(f"  {name:<12}{len(pdf):>7}{r['pq']:>9.4f}{r['sq']:>9.4f}{r['rq']:>9.4f}{r['fn']:>6}{r['fp']:>6}")

    # --- L_iou: out-of-fold IoU regressor over variants, argmax per cluster
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    X = vb[FEATS].to_numpy(np.float64)
    y = vb["iou"].to_numpy(np.float64)
    grp = vb["stem"].to_numpy()
    pred = np.zeros(len(vb))
    thr_per_fold = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, grp):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
        # reject threshold chosen on the TRAINING part only: the tau that
        # maximises (#chosen with iou>0.5) - 0.5*(#chosen with iou<=0.5)
        ptr = m.predict(X[tr])
        best_t, best_v = 0.0, -1e9
        for t in np.arange(0.2, 0.7, 0.02):
            keep = ptr >= t
            val = int(((y[tr] > 0.5) & keep).sum()) - 0.5 * int(((y[tr] <= 0.5) & keep).sum())
            if val > best_v:
                best_v, best_t = val, t
        thr_per_fold.append(best_t)
    vb["pred_iou"] = pred
    tau = float(np.median(thr_per_fold))
    chosen = []
    for cid, g in piv.items():
        gg = vb.loc[g.index]
        j = gg["pred_iou"].idxmax()
        if gg.loc[j, "pred_iou"] < tau:
            continue
        chosen.append((gg.loc[j, "stem"], float(gg.loc[j, "pred_iou"]), dec(j)))
    pdf = legal_pass(chosen, None)
    r = score(gt_df, pdf)
    res["L_iou"] = dict(n=len(pdf), pq=r["pq"], sq=r["sq"], rq=r["rq"], fn=r["fn"], fp=r["fp"], tau=tau)
    log(f"  {'L_iou':<12}{len(pdf):>7}{r['pq']:>9.4f}{r['sq']:>9.4f}{r['rq']:>9.4f}{r['fn']:>6}{r['fp']:>6}"
        f"   (OOF regressor, LEARNED reject tau {tau:.2f}, argmax variant)")
    # same learned choice, but with the fixed oracle-production reject, so the
    # choice effect is read separately from the reject effect
    chosen = []
    for cid, g in piv.items():
        gg = vb.loc[g.index]
        b = gg[gg["variant"] == "vote50"]
        if b.empty or b.iloc[0]["iou"] <= 0.5:
            continue
        j = gg["pred_iou"].idxmax()
        chosen.append((gg.loc[j, "stem"], float(gg.loc[j, "pred_iou"]), dec(j)))
    pdf = legal_pass(chosen, None)
    r = score(gt_df, pdf)
    res["L_iou_fixrej"] = dict(n=len(pdf), pq=r["pq"], sq=r["sq"], rq=r["rq"], fn=r["fn"], fp=r["fp"])
    log(f"  {'L_iou_fixrej':<12}{len(pdf):>7}{r['pq']:>9.4f}{r['sq']:>9.4f}{r['rq']:>9.4f}{r['fn']:>6}{r['fp']:>6}"
        f"   (learned choice, FIXED production reject)")
    from scipy.stats import spearmanr
    log(f"\n  OOF Spearman(pred_iou, iou) over variants: {spearmanr(pred, y).statistic:.3f}")
    top_acc = np.mean([vb.loc[g.index, 'pred_iou'].idxmax() == vb.loc[g.index, 'iou'].idxmax()
                       for _, g in piv.items()])
    log(f"  argmax agrees with the oracle's variant in {100*top_acc:.1f}% of clusters")

    oracle = json.load(open(OUT / f"candidate_bank_oracle_fold{a.fold}.json"))
    prod_pq, legal_pq = oracle["res"]["prod"]["pq"], oracle["res"]["legal"]["pq"]
    log(f"\n  oracle ceiling on this fold: prod {prod_pq:.4f} -> legal {legal_pq:.4f} (+{legal_pq-prod_pq:.4f})")
    log(f"  {'arm':<12}{'delta vs prod':>14}{'share of ceiling':>18}")
    for k, v in res.items():
        d = v["pq"] - res["prod"]["pq"]
        log(f"  {k:<12}{d:>+14.4f}{100*d/max(legal_pq-prod_pq,1e-9):>17.0f}%")
    json.dump(dict(fold=a.fold, res=res, ceiling=dict(prod=prod_pq, legal=legal_pq)),
              open(OUT / f"candidate_bank_capture_fold{a.fold}.json", "w"), indent=1)
    log(f"\nwrote {OUT/f'candidate_bank_capture_fold{a.fold}.json'}")


if __name__ == "__main__":
    main()
