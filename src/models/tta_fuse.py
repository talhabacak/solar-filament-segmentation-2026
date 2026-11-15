"""TTA with consensus fusion (not naive pooling), measured on one fold.

Naive view pooling was measured to add +53 TP but +496 FP (duplicate slivers
survive greedy de-overlap). Here the views are treated as ensemble members:
cluster masks at IoU>0.5, fuse each cluster into a conf-weighted consensus mask,
score = mean_conf * sqrt(support/n_views), then de-overlap and sweep.

Usage: python src/models/tta_fuse.py --tag _e013tta --fold 0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament.pq import score
from filament.rle import encode_mask
from ensemble import ensemble_image
from selector import FEATURES, mask_features

R_MAX = 0.97


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_e013tta")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--confs", default="0.3,0.4,0.5")
    ap.add_argument("--areas", default="128,256")
    ap.add_argument("--selector", default=None,
                    help="joblib path; when set, sweeps p_sel instead of area")
    ap.add_argument("--psel", default="0.0,0.15,0.25,0.35")
    a = ap.parse_args()

    preds = pd.read_parquet(ROOT / "outputs" / "oof_cache" /
                            f"fold{a.fold}_preds{a.tag}.parquet")
    preds = preds[(preds["conf"] >= 0.05) & (preds["r_norm"] <= R_MAX)].copy()
    preds["model"] = preds["view"]
    n_views = preds["view"].nunique()
    print(f"{len(preds)} candidates over {preds['stem'].nunique()} images, "
          f"{n_views} views {sorted(preds['view'].unique())}", flush=True)

    use_sel = a.selector is not None
    rows = []
    for stem, g in preds.groupby("stem"):
        clusters = ensemble_image(g, n_models=n_views)
        for k, d in enumerate(clusters, start=1):
            row = dict(stem=stem, cand_id=f"{stem}_{k}", support=d["support"],
                       mean_conf=d["mean_conf"], max_conf=d["max_conf"],
                       score=d["score"], area=int(d["mask"].sum()),
                       rle=encode_mask(d["mask"].astype(np.uint8)))
            if use_sel:
                f = mask_features(d["mask"])
                row.update({k2: v for k2, v in f.items() if k2 != "area"})
                row["n_cands_img"] = len(clusters)
                row["rank_conf"] = float(k)
                row["conf"] = d["mean_conf"]
            rows.append(row)
    cand = pd.DataFrame(rows)
    if use_sel:
        import joblib
        bundle = joblib.load(ROOT / a.selector)
        model = bundle["model"] if isinstance(bundle, dict) else bundle
        cand["log10_area"] = np.log10(cand["area"])
        cand["p_sel"] = model.predict_proba(cand[FEATURES].to_numpy(np.float64))[:, 1]
    print(f"fused clusters: {len(cand)} "
          f"(support dist {cand['support'].value_counts().sort_index().to_dict()})", flush=True)

    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    val_files = sorted(folds.loc[folds["fold"] == a.fold, "file_name"])
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt_val = gt[gt["file_name"].isin(val_files)][["filament_id", "segmentation_rle"]]

    gt_cache: dict = {}
    out = []
    second_axis = ([float(x) for x in a.psel.split(",")] if use_sel
                   else [int(x) for x in a.areas.split(",")])
    for conf_t in [float(x) for x in a.confs.split(",")]:
        for area_t in second_axis:
            for kmin in [1, 2]:
                if use_sel:
                    k = cand[(cand["mean_conf"] >= conf_t) & (cand["p_sel"] >= area_t)
                             & (cand["support"] >= kmin)]
                else:
                    k = cand[(cand["mean_conf"] >= conf_t) & (cand["area"] >= area_t)
                             & (cand["support"] >= kmin)]
                pred_df = k[["cand_id", "rle"]].rename(
                    columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
                if pred_df.empty:
                    continue
                r = score(gt_val, pred_df, gt_cache=gt_cache)
                out.append(dict(conf=conf_t, area=area_t, support_min=kmin,
                                **{kk: v for kk, v in r.items()
                                   if kk in ("pq", "sq", "rq", "tp", "fp", "fn")}))
                print(f"conf>={conf_t} area>={area_t} support>={kmin}: "
                      f"PQ={r['pq']:.4f} SQ={r['sq']:.4f} RQ={r['rq']:.4f} "
                      f"TP={r['tp']} FP={r['fp']} FN={r['fn']}", flush=True)
    df = pd.DataFrame(out).sort_values("pq", ascending=False)
    df.to_csv(ROOT / "outputs" / "oof_cache" / f"fold{a.fold}_ttafuse{a.tag}.csv", index=False)
    print("\nbest:\n", df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
