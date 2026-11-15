"""What are the candidates a looser limb threshold ADMITS -- and what separates
the true ones from the false ones?

E119 loosened p_oof at r > R0 and recovered true positives at the cost of as
many false positives. Talha's point: that means the rows are recoverable; the
question is whether the admitted FPs carry a simple signature. This takes the
admittable limb set -- candidates with conf >= 0.40, r_norm > 0.75, and
honest p_oof in [0.30, 0.60) (rejected today, admitted by some E119 setting) --
splits it into TP (matches a GT row at IoU > 0.5) and FP, and reports:

  1. counts by p_oof band and by radius band, TP share in each
  2. per feature (the selector's 32 + the E120 limb-relative ones if built):
     median TP vs FP, single-feature AUC on the admittable set
  3. an out-of-fold depth-2 decision tree (GroupKFold by stem) on the
     admittable set: can a two-split rule reach a precision worth keeping?
     Break-even precision for admitting a candidate is PQ0/(2*IoU) ~ 0.44/1.2
     ~ 37% here (adding a TP with IoU s: +s - PQ0/2 ; adding an FP: -PQ0/2)
  4. the same for the DISC's admittable set, as the comparison

Everything from cached artefacts; no labels are used as features.

Usage: python src/eda/limb_admitted_anatomy.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "eda"
BUNDLE = ROOT / "outputs" / "selector" / "selector_v5_shrunkblob5prodts.joblib"
NEW = ["img_dprime", "band_contrast", "flat_dprime", "elong_radial"]


def log(*a):
    print(*a, flush=True)


def auc(y, x):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y, x))
    except Exception:
        return float("nan")


def anatomy(df, feats, name, pq0=0.436):
    y = df["y"].to_numpy()
    log(f"\n=== {name}: {len(df)} admittable candidates, TP {int(y.sum())} ({100*y.mean():.1f}%) ===")
    log("  by p_oof band:")
    for lo, hi in [(0.3, 0.4), (0.4, 0.5), (0.5, 0.55), (0.55, 0.6)]:
        g = df[(df["p_oof"] >= lo) & (df["p_oof"] < hi)]
        if len(g):
            log(f"    [{lo:.2f},{hi:.2f})  n {len(g):>4}  TP {int(g['y'].sum()):>4}  TP share {100*g['y'].mean():5.1f}%  mean IoU of TP {g.loc[g['y']==1,'iou_sum'].mean() if g['y'].sum() else float('nan'):.3f}")
    log(f"  break-even precision for admitting ~ {100*pq0/(2*max(df.loc[df['y']==1,'iou_sum'].mean(),1e-6)):.0f}%")
    rows = []
    for f in feats:
        if f not in df:
            continue
        a, b = df.loc[df["y"] == 1, f], df.loc[df["y"] == 0, f]
        rows.append(dict(feature=f, tp_med=float(a.median()), fp_med=float(b.median()), auc=auc(y, df[f].to_numpy())))
    t = pd.DataFrame(rows)
    t["sep"] = (t["auc"] - 0.5).abs()
    t = t.sort_values("sep", ascending=False)
    log("  single-feature separation (AUC, TP vs FP), top 12:")
    for r in t.head(12).itertuples():
        log(f"    {r.feature:<22} AUC {r.auc:.3f}   median TP {r.tp_med:9.3f}  FP {r.fp_med:9.3f}")
    # OOF depth-2 tree
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.model_selection import GroupKFold
    X = df[[f for f in feats if f in df]].to_numpy(np.float64)
    used = [f for f in feats if f in df]
    grp = df["stem"].to_numpy()
    pred = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits=5).split(X, y, grp):
        clf = DecisionTreeClassifier(max_depth=2, min_samples_leaf=20, random_state=0).fit(X[tr], y[tr])
        pred[te] = clf.predict_proba(X[te])[:, 1]
    best = None
    for thr in np.arange(0.3, 0.9, 0.05):
        keep = pred >= thr
        if keep.sum() < 10:
            continue
        prec = y[keep].mean()
        gain = (df.loc[keep & (y == 1), "iou_sum"].sum() - pq0 / 2 * keep.sum())  # numerator gain minus denominator cost
        if best is None or gain > best[3]:
            best = (thr, int(keep.sum()), float(prec), float(gain))
    clf_all = DecisionTreeClassifier(max_depth=2, min_samples_leaf=20, random_state=0).fit(X, y)
    log("  OOF depth-2 tree on the admittable set:")
    if best:
        log(f"    best OOF cut: keep {best[1]} candidates at tree-prob >= {best[0]:.2f}, precision {100*best[2]:.0f}%, "
            f"utility {best[3]:+.1f} (IoU gained minus PQ0/2 per admitted; >0 means it would pay)")
    log("    tree (fit on all, for reading only):")
    for line in export_text(clf_all, feature_names=used, max_depth=2).splitlines():
        log("      " + line)
    return t, best


def main():
    import joblib
    feats = list(joblib.load(BUNDLE)["features"])
    cand = pd.read_parquet(OUT / "e115b_labelled_pool.parquet")
    po = pd.read_parquet(OUT / "e115b_p_oof.parquet").set_index("cand_id")["p_oof"]
    cand["p_oof"] = cand["cand_id"].map(po)
    fcache = OUT / "limb_relative_features.parquet"
    if fcache.exists():
        cand = cand.merge(pd.read_parquet(fcache), on="cand_id", how="left")
        feats = feats + NEW
        log("limb-relative features included")
    adm = cand[(cand["conf"] >= 0.40) & (cand["p_oof"] >= 0.30) & (cand["p_oof"] < 0.60)]
    limb = adm[adm["r_norm"] > 0.75]
    disc = adm[adm["r_norm"] <= 0.75]
    res = {}
    for df, name in ((limb, "LIMB r > 0.75"), (disc, "DISC r <= 0.75")):
        t, best = anatomy(df, feats, name)
        res[name] = dict(n=len(df), tp=int(df["y"].sum()), top=t.head(12).to_dict("records"), best_cut=best)
    json.dump(res, open(OUT / "limb_admitted_anatomy.json", "w"), indent=1, default=float)
    log(f"\nwrote {OUT/'limb_admitted_anatomy.json'}")


if __name__ == "__main__":
    main()
