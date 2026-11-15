"""OOF error taxonomy: attribute PQ loss to failure modes.

For a cached OOF prediction set (sweep_oof cache parquet) at chosen thresholds:
per annotator-image entry, classify every FN GT and FP prediction:

FN classes:
  fragmented   -- >=2 preds overlap this GT and their union would clear IoU 0.5
  partial      -- one pred overlaps 0.1<IoU<=0.5 (boundary/extent miss)
  merged       -- an overlapping pred covers this GT together with another GT
                  (many-to-one), stealing the match
  missed       -- no pred overlap above 0.05 (pure recall failure)
FP classes:
  hallucination -- no GT overlap (>0.05)
  fragment      -- overlaps a GT that another pred matched (split debris)
  partial_pair  -- overlaps an FN GT (the other side of 'partial'/'fragmented')
  merge         -- overlaps >=2 GTs substantially (>0.1 each)

Also stratifies FN by GT area quartile, width quartile, radial bin.

Usage: python src/eda/error_taxonomy.py --preds outputs/oof_cache/fold0_preds.parquet \
        --fold 0 --conf 0.4 --min-area 128 [--tag e011]
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament import IMG_H, IMG_W
from filament.pq import _entry_iou_matrix
from filament.rle import decode_rle, rles_to_layers

DISK_C, DISK_R = 1023.5, 900.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--min-area", type=int, default=128)
    ap.add_argument("--r-max", type=float, default=0.97)
    ap.add_argument("--tag", default="model")
    a = ap.parse_args()

    preds = pd.read_parquet(a.preds)
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    val_files = sorted(folds.loc[folds["fold"] == a.fold, "file_name"])
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt_val = gt[gt["file_name"].isin(val_files)]
    morph = pd.read_parquet(ROOT / "outputs" / "eda" / "morphology_instances.parquet")
    morph = morph.set_index("ann_id")

    # postprocess predictions per stem exactly like sweep/submission
    pred_masks_by_stem = {}
    for stem, g in preds.groupby("stem"):
        g = g[(g["conf"] >= a.conf) & (g["r_norm"] <= a.r_max)].sort_values("conf", ascending=False)
        taken = np.zeros((IMG_H, IMG_W), dtype=bool)
        kept = []
        for r in g.itertuples():
            m = decode_rle(r.rle).astype(bool) & ~taken
            if m.sum() < a.min_area:
                continue
            taken |= m
            kept.append(m)
        pred_masks_by_stem[stem] = kept

    fn_class = Counter()
    fp_class = Counter()
    fn_rows = []

    for entry_id, sub in gt_val.groupby("entry_id"):
        stem = sub["file_name"].iloc[0].rsplit(".", 1)[0]
        kept = pred_masks_by_stem.get(stem, [])
        gt_layers = rles_to_layers(sub["segmentation_rle"].tolist())
        n_gt = len(sub)
        if not kept:
            fn_class["missed"] += n_gt
            for r in sub.itertuples():
                fn_rows.append((r.ann_id, "missed"))
            continue
        pred_layers = np.stack(kept).astype(np.uint8)
        iou = _entry_iou_matrix(gt_layers, pred_layers)      # (n_gt, n_pred)
        hit = iou > 0.5
        gt_matched = hit.any(axis=1)
        pred_matched = hit.any(axis=0)

        # overlap (any intersection) matrix at low threshold
        ov = iou > 0.05

        for gi in range(n_gt):
            if gt_matched[gi]:
                continue
            over_preds = np.nonzero(ov[gi])[0]
            cls = "missed"
            if len(over_preds) >= 2:
                union = np.zeros((IMG_H, IMG_W), dtype=bool)
                for pj in over_preds:
                    union |= pred_layers[pj].astype(bool)
                g_mask = gt_layers[gi].astype(bool)
                u_iou = (union & g_mask).sum() / max((union | g_mask).sum(), 1)
                cls = "fragmented" if u_iou > 0.5 else "partial"
            elif len(over_preds) == 1:
                pj = over_preds[0]
                # merged: that pred also overlaps another GT at least as much
                others = [g2 for g2 in range(n_gt) if g2 != gi and ov[g2, pj]]
                if others and pred_matched[pj]:
                    cls = "merged"
                elif others:
                    cls = "merged" if iou[gi, pj] < max(iou[g2, pj] for g2 in others) else "partial"
                else:
                    cls = "partial"
            fn_class[cls] += 1
            fn_rows.append((sub["ann_id"].iloc[gi], cls))

        for pj in range(len(kept)):
            if pred_matched[pj]:
                continue
            over_gts = np.nonzero(ov[:, pj])[0]
            if len(over_gts) == 0:
                fp_class["hallucination"] += 1
            elif len(over_gts) >= 2 and sum(iou[g2, pj] > 0.1 for g2 in over_gts) >= 2:
                fp_class["merge"] += 1
            elif any(gt_matched[g2] for g2 in over_gts):
                fp_class["fragment"] += 1
            else:
                fp_class["partial_pair"] += 1

    # stratify FN by GT properties
    fn_df = pd.DataFrame(fn_rows, columns=["ann_id", "cls"]).set_index("ann_id")
    fn_df = fn_df.join(morph[["poly_area", "width_proxy", "r_norm"]], how="left")
    strata = {}
    for col, name in [("poly_area", "area"), ("width_proxy", "width"), ("r_norm", "radial")]:
        edges = morph[col].quantile([0, 0.25, 0.5, 0.75, 1.0]).values
        bins = pd.cut(fn_df[col], bins=edges, labels=["Q1", "Q2", "Q3", "Q4"], include_lowest=True)
        strata[name] = pd.crosstab(bins, fn_df["cls"]).to_dict()

    total_fn = sum(fn_class.values())
    out = dict(
        tag=a.tag, conf=a.conf, min_area=a.min_area,
        fn_total=total_fn, fn_classes=dict(fn_class),
        fn_class_pct={k: round(100 * v / max(total_fn, 1), 1) for k, v in fn_class.items()},
        fp_total=sum(fp_class.values()), fp_classes=dict(fp_class),
        fn_strata=strata,
    )
    dest = ROOT / "outputs" / "eda" / f"error_taxonomy_{a.tag}.json"
    with open(dest, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items() if k != "fn_strata"}, indent=1))
    print("FN by width quartile:", {k: dict(v) for k, v in
          pd.DataFrame(strata["width"]).T.items()} if strata else {})
    print("saved ->", dest)


if __name__ == "__main__":
    main()
