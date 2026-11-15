"""E112 -- the champion's PQ broken down by slice, with the exact scorer.

sub020's recipe (five-fold blob member at half the field, production one-pixel
erosion, two-stage selector at conf 0.40 / p 0.55) is applied to the full
out-of-fold pool it was fitted on, and the kept set is scored per slice. The
selector application is IN-SAMPLE here -- the bundle was fitted on this pool --
so absolute numbers sit a little above the cross-fit (0.4368); what this table
is for is the SHAPE: which folds, sites, years, radii and widths carry the
loss, and whether any slice collapses. A promotion candidate is later run
through the same script and compared slice by slice.

Slices
  image-level : fold, site (GONG station letter in the stem), year
  instance    : radial band of the GT centroid, GT mean width, GT area
For image-level slices the exact scorer is run on that subset of images; for
instance-level slices the per-GT match table from score(return_details=True)
gives reach (matched at IoU > 0.5) and matched IoU, since PQ itself is pooled
over images and does not decompose by instance attribute.

Usage: python src/eda/champion_stress.py
"""

import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament.pq import score
from filament.rle import decode_rle, encode_mask
from mask_shrink import shrink_frac, mean_width_px

OUT = ROOT / "outputs" / "eda"
SEL = ROOT / "outputs" / "selector"
DISK_C, DISK_R = 1023.5, 900.0
POOL, BUNDLE = "fused_pool_all_blob5.parquet", "selector_v5_shrunkblob5prodts.joblib"


import argparse

def log(*a):
    print(*a, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--p", type=float, default=None)
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    b = joblib.load(SEL / BUNDLE)
    pol = dict(b["policy"])
    if args.conf is not None:
        pol["conf"] = args.conf
    if args.p is not None:
        pol["p"] = args.p

    tag = args.tag or ("sub020" if pol == b["policy"] else f"sub_c{pol['conf']:.2f}_p{pol['p']:.2f}")

    sub020_cache = OUT / "kept_sub020_shrunk.parquet"
    if sub020_cache.exists() and pol["p"] >= 0.55 and pol["conf"] >= 0.40:
        log(f"Fast path: loading pre-shrunk candidates from {sub020_cache.name}")
        cand = pd.read_parquet(sub020_cache)
        keep = cand[(cand["conf"] >= pol["conf"]).to_numpy() & (cand["p_sel"] >= pol["p"])].copy()
    else:
        cand = pd.read_parquet(SEL / POOL)
        log(f"Pre-shrinking {len(cand)} candidates...")
        rles, areas = [], []
        for r in cand["rle"]:
            m = shrink_frac(decode_rle(r).astype(bool), 0.05)
            rles.append(encode_mask(m.astype(np.uint8)))
            areas.append(int(m.sum()))
        cand["rle"], cand["area"] = rles, areas
        cand["log10_area"] = np.log10(np.maximum(cand["area"], 1))
        X = cand[b["features"]].to_numpy(np.float64)
        p = b["model"].predict_proba(X)[:, 1]
        if b.get("iou_model") is not None:
            p = p * np.clip(b["iou_model"].predict(X), 0.0, None)
            ref = b.get("two_stage_cdf")
            if ref is not None:
                p = np.searchsorted(ref, p, side="right") / len(ref)
        keep = cand[(cand["conf"] >= pol["conf"]).to_numpy() & (p >= pol["p"])].copy()

    log(f"Applied policy: {pol}, tag: {tag}")
    log(f"kept {len(keep)} instances over {keep['stem'].nunique()} images")

    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt = gt[gt["stem"].isin(set(cand["stem"]))].copy()
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    gt["fold"] = gt["stem"].map(fold_of)
    gt["site"] = gt["stem"].str[-2]          # ...Bh / Ch / Lh / Mh / Th / Uh
    gt["year"] = gt["stem"].str[:4]
    pred = keep[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"})

    # overall + per-GT match table
    res = score(gt[["filament_id", "segmentation_rle"]], pred, return_details=True)
    log(f"\nOVERALL (in-sample selector on the OOF pool): PQ {res['pq']:.4f}  "
        f"SQ {res['sq']:.4f}  RQ {res['rq']:.4f}  TP {res['tp']} FP {res['fp']} FN {res['fn']}")
    m = res["matches"]
    best = m.groupby("gt_filament_id")["iou"].max()
    gt["best_iou"] = gt["filament_id"].map(best).fillna(0.0)
    gt["reached"] = gt["best_iou"] > 0.5

    rows = []

    def image_slice(name, key):
        log(f"\n{name}")
        log(f"  {'slice':>10}{'imgs':>6}{'GT':>6}{'PQ':>9}{'SQ':>8}{'RQ':>8}{'FN':>6}{'FP':>6}")
        for v, g in gt.groupby(key):
            stems = set(g["stem"])
            pp = keep[keep["stem"].isin(stems)][["cand_id", "rle"]].rename(
                columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
            r = score(g[["filament_id", "segmentation_rle"]], pp)
            rows.append(dict(axis=name, slice=str(v), images=len(stems), gt=len(g),
                             pq=r["pq"], sq=r["sq"], rq=r["rq"], fn=r["fn"], fp=r["fp"]))
            log(f"  {str(v):>10}{len(stems):>6}{len(g):>6}{r['pq']:>9.4f}{r['sq']:>8.4f}"
                f"{r['rq']:>8.4f}{r['fn']:>6}{r['fp']:>6}")

    image_slice("by FOLD", "fold")
    image_slice("by SITE", "site")
    image_slice("by YEAR", "year")

    # instance-level attributes
    gt_attr_cache = OUT / "gt_slice_attributes.parquet"
    if gt_attr_cache.exists():
        attr_df = pd.read_parquet(gt_attr_cache)
        gt["r"] = attr_df["r"].to_numpy()
        gt["width"] = attr_df["width"].to_numpy()
        gt["area"] = attr_df["area"].to_numpy()
    else:
        rr, ww, aa = [], [], []
        for r in gt["segmentation_rle"]:
            mk = decode_rle(r).astype(bool)
            ys, xs = np.nonzero(mk)
            rr.append(float(np.hypot(xs.mean() - DISK_C, ys.mean() - DISK_C) / DISK_R) if len(xs) else 1.0)
            ww.append(mean_width_px(mk) if len(xs) else 0.0)
            aa.append(int(mk.sum()))
        gt["r"], gt["width"], gt["area"] = rr, ww, aa
        pd.DataFrame({"filament_id": gt["filament_id"], "r": rr, "width": ww, "area": aa}).to_parquet(gt_attr_cache, index=False)

    gt["rband"] = pd.cut(gt["r"], [0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.01])
    gt["wband"] = pd.qcut(gt["width"], 5, duplicates="drop")
    gt["aband"] = pd.qcut(gt["area"], 5, duplicates="drop")

    def inst_slice(name, key):
        log(f"\n{name}  (reach = matched at IoU > 0.5; IoU = mean over reached)")
        log(f"  {'slice':>16}{'GT':>6}{'reach':>8}{'mean IoU':>10}")
        for v, g in gt.groupby(key, observed=True):
            rch = g["reached"].mean()
            iou = g.loc[g["reached"], "best_iou"].mean() if g["reached"].any() else 0.0
            rows.append(dict(axis=name, slice=str(v), gt=len(g), reach=rch, iou=iou))
            log(f"  {str(v):>16}{len(g):>6}{100*rch:>7.1f}%{iou:>10.3f}")

    inst_slice("by RADIUS", "rband")
    inst_slice("by GT WIDTH (px)", "wband")
    inst_slice("by GT AREA (px)", "aband")

    df = pd.DataFrame(rows)
    out_csv = OUT / f"champion_stress_{tag}.csv"
    out_json = OUT / f"champion_stress_{tag}.json"
    df.to_csv(out_csv, index=False)
    worst_fold = df[df.axis == "by FOLD"].sort_values("pq").iloc[0]
    worst_site = df[df.axis == "by SITE"].sort_values("pq").iloc[0]
    log(f"\nworst fold {worst_fold['slice']} PQ {worst_fold['pq']:.4f}; "
        f"worst site {worst_site['slice']} PQ {worst_site['pq']:.4f}; overall {res['pq']:.4f}")
    json.dump(dict(overall=dict(pq=res["pq"], sq=res["sq"], rq=res["rq"]),
                   worst_fold=dict(worst_fold), worst_site=dict(worst_site)),
              open(out_json, "w"), indent=1, default=float)
    log(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
