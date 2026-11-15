"""E219 -- what does the unlabelled test set let us predict about our private score?

Observation (docs/21, docs/34): local cross-fitted PQ has sat 0.043-0.050 above the public board
for EVERY pipeline generation -- sub004/sub007 (-0.050/-0.049) before the selector and the
five-model ensemble existed, sub036 now (-0.043) -- and other honest teams show the same sign
(0.46 -> 0.40, 0.406 -> 0.37). A constant, pipeline-independent offset points at the test side,
not at our machinery. Composition (year, hf, station, halo; ledger §51) explains only -0.003.

Four readings, all label-free on test (the per-image annotator counts that exist in the
quarantined public MAGFiLO metadata are NOT used anywhere here):

A  Annotator effect, local. Per annotator entry (code GGBBAA = group, batch, slot) the pooled PQ
   of sub036's out-of-fold predictions, and -- to separate annotator style from image difficulty
   -- the PAIRED reading on images drawn by two or more annotators: the same masks scored against
   different annotators of the same image. How far can annotator mix alone move PQ?
B  Image-level shift. A classifier separating train from test on image-only quantities (hf,
   disc statistics, limb width, halo step, year, month, station); cross-validated AUC, and the
   out-of-fold PQ reweighted by the density ratio = the composition-predicted test PQ.
C  Halves. Random 90/90 splits of test-sized subsets of the OOF images: spread of public-private
   differences, so a public reading can be turned into a private interval.
D  Robustness of the final pair. sub036 and sub042 per annotator group, year band and hf
   quartile, out of fold.

Usage: python src/eda/e219_test_generalisation.py
Output: outputs/eda/e219_test_generalisation.json (+ _entries.parquet)
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
from e115c_threshold_ci import pq  # noqa: E402

OUT = ROOT / "outputs" / "eda"
RNG = np.random.default_rng(0)


def pooled(df):
    d = df["tp"].sum() + 0.5 * df["fp"].sum() + 0.5 * df["fn"].sum()
    return df["tp_iou_sum"].sum() / d if d else np.nan


def boot_ci(df, by="stem", reps=2000):
    stems = df[by].unique()
    g = {s: x for s, x in df.groupby(by)}
    v = [pooled(pd.concat([g[s] for s in RNG.choice(stems, len(stems))])) for _ in range(reps)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def entries_for(keep_mask, a, gt):
    pred = a.loc[keep_mask, ["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
    e = score(gt[["filament_id", "segmentation_rle"]], pred, return_details=True)["entries"]
    e["entry"] = e["annotator_image"].str.split("-", n=1).str[0]
    e["stem"] = e["annotator_image"].str.split("-", n=1).str[1]
    e["group"] = e["entry"].str[:2]
    e["gb"] = e["entry"].str[:4]
    return e


def main():
    res = {}
    a = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet", columns=["stem", "cand_id", "rle", "conf"])
    a = a.merge(pd.read_parquet(OUT / "bag5_oof_scores.parquet"), on="cand_id")
    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    keep036 = ((a["conf"] >= 0.40) & (a["p"] >= 0.60)).to_numpy()
    e = entries_for(keep036, a, gt)
    nent = e.groupby("stem")["entry"].nunique()
    e["n_entries"] = e["stem"].map(nent)
    q = pd.read_parquet(OUT / "image_quality.parquet")
    e = e.merge(q[["stem", "year", "site", "hf"]], on="stem", how="left")
    res["guard_pooled_pq"] = float(pooled(e))
    print(f"guard: pooled PQ from entries {res['guard_pooled_pq']:.4f} (sub036 OOF 0.4431)")

    # ---- A. annotator effect -------------------------------------------------------------
    rows = []
    for g, x in e.groupby("group"):
        rows.append(dict(group=g, entries=len(x), images=x["stem"].nunique(), pq=float(pooled(x)),
                         ci=boot_ci(x), years=f"{int(x.year.min())}-{int(x.year.max())}",
                         gt_per_entry=float(x["n_gt"].mean())))
    A_groups = pd.DataFrame(rows).sort_values("pq")
    print("\nA. pooled PQ by annotator group\n", A_groups.to_string(index=False))
    # paired: on multi-annotator images, entry PQ minus the image's mean entry PQ (same masks)
    m = e[e["n_entries"] > 1].copy()
    m["epq"] = m["tp_iou_sum"] / (m["tp"] + 0.5 * m["fp"] + 0.5 * m["fn"]).clip(lower=1e-9)
    m["dev"] = m["epq"] - m.groupby("stem")["epq"].transform("mean")
    paired = (m.groupby("entry").agg(n=("dev", "size"), dev=("dev", "mean"), dev_sd=("dev", "std"))
              .query("n >= 15").sort_values("dev"))
    paired["se"] = paired["dev_sd"] / np.sqrt(paired["n"])
    print("\nA2. paired deviation per annotator (same image, same masks; n >= 15 images)\n",
          paired.round(4).to_string())
    pg = m.groupby("group").agg(n=("dev", "size"), dev=("dev", "mean")).round(4)
    print("\nA3. paired deviation per group\n", pg.to_string())
    # spread a different annotator mix could produce: image-weighted, paired effects only
    res["A"] = dict(groups=A_groups.to_dict("records"), paired_by_annotator=paired.reset_index().to_dict("records"),
                    paired_by_group=pg.reset_index().to_dict("records"),
                    paired_annotator_range=[float(paired["dev"].min()), float(paired["dev"].max())],
                    paired_annotator_sd=float(paired["dev"].std()))

    # ---- B. image-level shift -------------------------------------------------------------
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_auc_score
    feats = ["hf", "disk_sd", "disk_mean", "limb_width", "limb_ok", "step", "year", "month"]
    X = q[feats].copy()
    for s in sorted(q["site"].unique()):
        X[f"site_{s}"] = (q["site"] == s).astype(int)
    y = (q["split"] == "test").astype(int).to_numpy()
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20)
    pr = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    auc = roc_auc_score(y, pr)
    q["p_test"] = pr
    tr = q[q["split"] == "train"].copy()
    w = (tr["p_test"] / (1 - tr["p_test"])).clip(upper=np.quantile(tr["p_test"] / (1 - tr["p_test"]), 0.99))
    w = w / w.mean()
    tr["w"] = w
    img = e.groupby("stem")[["tp", "fp", "fn", "tp_iou_sum"]].sum().reset_index().merge(tr[["stem", "w"]], on="stem")
    num = (img["tp_iou_sum"] * img["w"]).sum()
    den = ((img["tp"] + 0.5 * img["fp"] + 0.5 * img["fn"]) * img["w"]).sum()
    pq_w = num / den
    ess = img["w"].sum() ** 2 / (img["w"] ** 2).sum()
    boots = []
    for _ in range(2000):
        s = img.sample(len(img), replace=True, random_state=int(RNG.integers(1e9)))
        boots.append((s["tp_iou_sum"] * s["w"]).sum() / ((s["tp"] + 0.5 * s["fp"] + 0.5 * s["fn"]) * s["w"]).sum())
    res["B"] = dict(auc_train_vs_test=float(auc), pq_unweighted=float(pooled(e)), pq_reweighted=float(pq_w),
                    ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                    effective_n=float(ess))
    print(f"\nB. train-vs-test image AUC {auc:.3f}; OOF PQ {pooled(e):.4f} -> reweighted to test mix {pq_w:.4f} "
          f"CI {res['B']['ci95']}; effective n {ess:.0f}/707")

    # ---- C. halves ----------------------------------------------------------------------
    A_ = img[["tp", "fp", "fn", "tp_iou_sum"]].to_numpy(float)
    d_half, d_full = [], []
    for _ in range(20000):
        idx = RNG.choice(len(A_), 180, replace=False)
        h1, h2 = idx[:90], idx[90:]
        p1, p2, pf = pq(*A_[h1, :3].sum(0), A_[h1, 3].sum()), pq(*A_[h2, :3].sum(0), A_[h2, 3].sum()), \
            pq(*A_[idx, :3].sum(0), A_[idx, 3].sum())
        d_half.append(p2 - p1)
        d_full.append(pf - p1)
    res["C"] = dict(sd_private_minus_public=float(np.std(d_half)), sd_full_minus_public=float(np.std(d_full)),
                    private_given_public_040=[0.40 - 1.96 * float(np.std(d_half)), 0.40 + 1.96 * float(np.std(d_half))])
    print(f"\nC. 90/90 halves: sd(private - public) {np.std(d_half):.4f}; given public 0.40 the private 95 % "
          f"interval (no systematic shift between halves) {res['C']['private_given_public_040']}")

    # ---- D. final pair robustness -------------------------------------------------------
    try:
        from final_pair_selection import cascade_admit
        from extra_features import EXTRA
        b = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
        b = b.merge(pd.read_parquet(OUT / "extra_features_t020.parquet")[["cand_id"] + EXTRA], on="cand_id")
        folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
        fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
        b["fold"] = b["stem"].map(fold_of).astype(int)
        bag5 = b[["cand_id"]].merge(pd.read_parquet(OUT / "bag5_oof_scores.parquet"), on="cand_id")["p"].to_numpy()
        cfeats = list(joblib.load(ROOT / "outputs/selector/selector_v5_shrunkblob5prodts.joblib")["features"]) + list(EXTRA)
        k042 = ((b["conf"] >= 0.40).to_numpy() & (bag5 >= 0.60)) | cascade_admit(b, bag5, cfeats)
        e42 = entries_for(k042, b, gt).merge(q[["stem", "year", "hf"]], on="stem", how="left")
        hfq = pd.qcut(q.loc[q.split == "train", "hf"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
        hfmap = dict(zip(q.loc[q.split == "train", "stem"], hfq))
        out = []
        for name, key in (("group", lambda d: d["group"]),
                          ("years", lambda d: pd.cut(d["year"], [2010, 2013, 2017, 2023], labels=["2011-13", "2014-17", "2018-22"])),
                          ("hf", lambda d: d["stem"].map(hfmap))):
            for lvl in sorted(key(e).dropna().unique()):
                x36, x42 = e[key(e) == lvl], e42[key(e42) == lvl]
                out.append(dict(axis=name, level=str(lvl), pq036=float(pooled(x36)), pq042=float(pooled(x42)),
                                delta=float(pooled(x42) - pooled(x36))))
        D = pd.DataFrame(out)
        res["D"] = D.to_dict("records")
        res["D_summary"] = dict(pq036=float(pooled(e)), pq042=float(pooled(e42)),
                                levels_where_042_wins=int((D["delta"] > 0).sum()), levels=len(D))
        print("\nD. sub042 - sub036 by subgroup\n", D.round(4).to_string(index=False))
    except Exception as ex:  # D is optional; A-C stand alone
        print("D skipped:", repr(ex))

    e.to_parquet(OUT / "e219_entries.parquet", index=False)
    json.dump(res, open(OUT / "e219_test_generalisation.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
