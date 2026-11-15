"""Test if filtering out heavily-carved candidates during/after greedy de-overlap
improves PQ by removing fragments that became false positives.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament import IMG_H, IMG_W
from filament.pq import score
from filament.rle import decode_rle, encode_mask
from mask_shrink import shrink_frac

OUT = ROOT / "outputs" / "eda"
SEL = ROOT / "outputs" / "selector"
POOL, BUNDLE = "fused_pool_all_blob5.parquet", "selector_v5_shrunkblob5prodts.joblib"


def main():
    cand = pd.read_parquet(SEL / POOL)
    b = joblib.load(SEL / BUNDLE)
    sel = b["model"]
    conf_min, p_min = b["policy"]["conf"], b["policy"]["p"]
    feats = b["features"]

    # Focus on fold 0 first
    cand_f0 = cand[cand["fold"] == 0].copy()
    gt_all = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    val_files = set(folds.loc[folds["fold"] == 0, "file_name"])
    gt_f0 = gt_all[gt_all["file_name"].isin(val_files)][["filament_id", "segmentation_rle"]]

    print(f"Fold 0: {len(cand_f0)} candidates, {len(gt_f0)} GT rows", flush=True)

    # Apply shrink as sub020 does
    rles, areas = [], []
    for r in cand_f0["rle"]:
        m = shrink_frac(decode_rle(r).astype(bool), 0.05)
        rles.append(encode_mask(m.astype(np.uint8)))
        areas.append(int(m.sum()))
    cand_f0["rle"] = rles
    cand_f0["area"] = areas
    cand_f0["log10_area"] = np.log10(np.maximum(areas, 1))

    X = cand_f0[feats].to_numpy(dtype=np.float64)
    probs = sel.predict_proba(X)[:, 1]
    cand_f0["p_sel"] = probs

    # Baseline selection before de-overlap
    sub = cand_f0[(cand_f0["conf"] >= conf_min) & (cand_f0["p_sel"] >= p_min)].copy()
    print(f"Selected candidates before de-overlap: {len(sub)}", flush=True)

    # Sweep min_kept_ratio (fraction of original area that must remain after carving)
    # 0.0 means current behavior (standard greedy de-overlap with area >= 64)
    ratios = [0.0, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]

    print(f"\n{'min_kept_ratio':<16}{'preds':>7}{'PQ':>9}{'SQ':>9}{'RQ':>9}{'TP':>6}{'FP':>6}{'FN':>6}{'dropped':>9}", flush=True)
    for mkr in ratios:
        preds = []
        n_dropped = 0
        for stem, g in sub.groupby("stem"):
            g = g.sort_values("conf", ascending=False)
            taken = np.zeros((IMG_H, IMG_W), bool)
            k = 0
            for r in g.itertuples():
                orig_m = decode_rle(r.rle).astype(bool)
                orig_area = orig_m.sum()
                mm = orig_m & ~taken
                carved_area = mm.sum()
                if carved_area < 64:
                    continue
                ratio = carved_area / orig_area
                if ratio < mkr:
                    n_dropped += 1
                    continue
                taken |= mm
                k += 1
                preds.append((f"{stem}_{k}", encode_mask(mm.astype(np.uint8))))

        pdf = pd.DataFrame(preds, columns=["filament_id", "segmentation_rle"])
        res = score(gt_f0, pdf)
        print(f"{mkr:<16.2f}{len(pdf):>7}{res['pq']:>9.4f}{res['sq']:>9.4f}{res['rq']:>9.4f}{res['tp']:>6}{res['fp']:>6}{res['fn']:>6}{n_dropped:>9}", flush=True)


if __name__ == "__main__":
    main()
