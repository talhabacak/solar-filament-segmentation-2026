"""E197 -- is the five-fold split itself a source of the deltas we keep reading?

Every local decision in this project is a pooled five-fold cross-fitted PQ on ONE partition
(data/splits/folds_v1.parquet, GroupKFold by image, stratified by station and contrast). The
partition is balanced on what it was built to balance: images per fold 141-142, copies per image
1.620-1.645, the 1/2/3-annotator mix within a percentage point. It is NOT balanced on difficulty:
the candidate pool's positive rate runs 0.372 (fold 4) to 0.492 (fold 1), candidates per image
13.2 to 15.7, ground-truth rows 1510 to 1698.

That matters because the metric pools over folds: a fold with more candidates and a lower positive
rate contributes a different denominator. If an arm's delta moves by as much as the delta itself
when the partition is reshuffled, then every reading below the +-0.003 band is partition noise and
no smarter combination of such arms can be trusted.

This measures exactly that. The same arm (the thirteen relational features of E195, whose reading
on the deployed machinery was +0.0022 and whose board transfer was #9) is re-read under:

  prod      the production partition
  shuf1-3   GroupKFold by image with a shuffled image order (same construction, new assignment)
  bal       folds balanced on copies per image AND image contrast quartile

Reported per partition: pooled PQ of the deployed selector, the arm's delta, and the per-fold
deltas. The spread across partitions is the number that decides whether a sub-floor local reading
means anything at all.

Usage: python src/eda/fold_design_audit.py [--arm relational|bagging] [--parts 5]
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
from e115c_threshold_ci import pq  # noqa: E402
from extra_features import EXTRA  # noqa: E402
from relational_features import REL  # noqa: E402

OUT = ROOT / "outputs" / "eda"


def partitions(stems_df, n_parts):
    """The production partition plus alternatives, each a dict stem -> fold."""
    from sklearn.model_selection import GroupKFold
    out = {}
    folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
    folds["stem"] = folds["file_name"].str.replace(".jpeg", "", regex=False)
    out["prod"] = dict(zip(folds["stem"], folds["fold"]))
    stems = stems_df["stem"].to_numpy()
    for s in range(1, n_parts - 1):
        rng = np.random.default_rng(s)
        order = rng.permutation(len(stems))
        assign = np.empty(len(stems), int)
        for k, idx in enumerate(np.array_split(order, 5)):
            assign[idx] = k
        out[f"shuf{s}"] = dict(zip(stems, assign))
    # balanced on the two observable quantities that enter the metric's denominator and the
    # image's difficulty: annotator copies and image contrast
    d = stems_df.copy()
    d["cq"] = pd.qcut(d["contrast"], 4, labels=False, duplicates="drop")
    d = d.sort_values(["copies", "cq"]).reset_index(drop=True)
    d["fold"] = [i % 5 for i in range(len(d))]          # deal round-robin inside each stratum
    out["bal"] = dict(zip(d["stem"], d["fold"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="relational", choices=["relational", "bagging"])
    ap.add_argument("--parts", type=int, default=5)
    a = ap.parse_args()

    pool = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    pool = pool.merge(pd.read_parquet(OUT / "extra_features_t020.parquet")[["cand_id"] + EXTRA],
                      on="cand_id", how="left")
    pool = pool.merge(pd.read_parquet(OUT / "relational_features_t020.parquet"), on="cand_id",
                      how="left", validate="one_to_one")
    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    base = list(joblib.load(ROOT / "outputs/selector/selector_v5_shrunk_t020pox.joblib")["features"])

    stems_df = (pool.groupby("stem").agg(contrast=("img_contrast", "mean")).reset_index()
                .merge(gt.groupby("stem")["entry_id"].nunique().rename("copies").reset_index(),
                       on="stem", how="left").fillna({"copies": 1}))
    parts = partitions(stems_df, a.parts)

    arms = ({"A": (base, False), "B": (base + REL, False)} if a.arm == "relational"
            else {"A": (base, False), "B": (base, True)})
    print(f"arm: {a.arm}; partitions: {list(parts)}", flush=True)
    res = {}
    for pname, fold_of in parts.items():
        pool["fold"] = pool["stem"].map(fold_of).astype(int)
        vals, parts_out = {}, {}
        for k, (feats, bag) in arms.items():
            parts_out[k], _ = B.run(pool, feats, gt, fold_of, bag)
            cat = lambda key, P=parts_out[k]: np.concatenate([P[F][key] for F in range(5)])
            vals[k] = float(pq(cat("tp").sum(), cat("fp").sum(), cat("fn").sum(), cat("iou").sum()))
        per_fold = [float(pq(*[parts_out["B"][F][k].sum() for k in ("tp", "fp", "fn", "iou")])
                          - pq(*[parts_out["A"][F][k].sum() for k in ("tp", "fp", "fn", "iou")]))
                    for F in range(5)]
        res[pname] = dict(base=vals["A"], arm=vals["B"], delta=vals["B"] - vals["A"],
                          per_fold=per_fold)
        print(f"  {pname:6s} base {vals['A']:.4f}  arm {vals['B']:.4f}  "
              f"delta {vals['B']-vals['A']:+.4f}  per fold {[round(x,4) for x in per_fold]}", flush=True)

    d = np.array([r["delta"] for r in res.values()])
    b = np.array([r["base"] for r in res.values()])
    print(f"\ndelta across partitions: mean {d.mean():+.4f}, sd {d.std(ddof=1):.4f}, "
          f"range [{d.min():+.4f}, {d.max():+.4f}]")
    print(f"base PQ across partitions: mean {b.mean():.4f}, sd {b.std(ddof=1):.4f}, "
          f"range [{b.min():.4f}, {b.max():.4f}]")
    res["summary"] = dict(delta_mean=float(d.mean()), delta_sd=float(d.std(ddof=1)),
                          base_mean=float(b.mean()), base_sd=float(b.std(ddof=1)))
    json.dump(res, open(OUT / f"fold_design_audit_{a.arm}.json", "w"), indent=1)
    print(f"wrote fold_design_audit_{a.arm}.json")


if __name__ == "__main__":
    main()
