"""Latitude and the errors -- Talha's reading of the TP/FN/FP map.

Filaments follow the butterfly diagram (activity belts at |lat| ~ 15-40 deg,
few at the equator, polar-crown filaments sparse at high latitude), so the
DENSITY of TP, FN and FP all peak in the belts. The question is whether the
ERROR RATES do: is a filament at the equator or at high latitude more likely
to be missed, or a candidate there more likely to be false, than in the
belts? And does the selector already use it -- `abs_lat` (|y - cy| / R, the
image-frame latitude proxy) is one of its 32 features.

Measures (production policy on the honest OOF pool, GT over all folds_v1
stems):
  1. miss rate FN/(TP+FN) and FP share by |lat| band, and on a (|lat|, r) grid
  2. among pool candidates (conf >= 0.40): TP share by |lat| band, and the
     single-feature AUC of abs_lat for TP vs FP (does latitude separate?)
  3. the selector WITHOUT abs_lat: E115b's OOF recipe refit on 31 features,
     exact PQ per fold against the 32-feature control -- if PQ does not move,
     latitude carries no information the other features lack
  4. partial dependence of the 32-feature OOF score on abs_lat (median p_oof
     by |lat| band at fixed TP/FP), to see how the selector uses it

Usage: python src/eda/latitude_structure.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "eda"))

from filament.pq import score
from filament.rle import decode_rle
from e115c_threshold_ci import oof_scores

OUT = ROOT / "outputs" / "eda"
BUNDLE = ROOT / "outputs" / "selector" / "selector_v5_shrunkblob5prodts.joblib"
DISK_C, DISK_R = 1023.5, 900.0
LAT = [0.0, 0.15, 0.30, 0.50, 0.75, 1.01]


def log(*a):
    print(*a, flush=True)


def main():
    import joblib
    feats = list(joblib.load(BUNDLE)["features"])
    cand = pd.read_parquet(OUT / "e115b_labelled_pool.parquet")
    po = pd.read_parquet(OUT / "e115b_p_oof.parquet").set_index("cand_id")["p_oof"]
    cand["p_oof"] = cand["cand_id"].map(po)
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    cand["fold"] = cand["stem"].map(fold_of).astype(int)
    gt["fold"] = gt["stem"].map(fold_of)
    gt = gt[gt["fold"].notna()].copy()
    gt["fold"] = gt["fold"].astype(int)

    # --- 1. GT rows: centroid lat/r and status under the production policy
    em = cand[(cand["conf"] >= 0.40) & (cand["p_oof"] >= 0.60)]
    pred = em[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
    res = score(gt[["filament_id", "segmentation_rle"]], pred, return_details=True)
    hit_gt = set(res["matches"]["gt_filament_id"])
    hit_pred = set(res["matches"]["pred_filament_id"])
    lat, rad = [], []
    for rle in gt["segmentation_rle"]:
        m = decode_rle(rle); ys, xs = np.nonzero(m)
        lat.append(abs(ys.mean() - DISK_C) / DISK_R if len(xs) else np.nan)
        rad.append(np.hypot(xs.mean() - DISK_C, ys.mean() - DISK_C) / DISK_R if len(xs) else np.nan)
    gt["lat"], gt["r"] = lat, rad
    gt["tp"] = gt["filament_id"].isin(hit_gt)
    em = em.assign(fp=~em["cand_id"].isin(hit_pred))
    log(f"production policy: PQ {res['pq']:.4f} TP {res['tp']} FP {res['fp']} FN {res['fn']}")
    log(f"\n  {'|lat| band':<12}{'GT rows':>8}{'miss rate':>10}{'emitted':>9}{'FP share':>9}")
    out = {"by_lat": []}
    for lo, hi in zip(LAT[:-1], LAT[1:]):
        g = gt[(gt["lat"] > lo) & (gt["lat"] <= hi)]
        e = em[(em["abs_lat"] > lo) & (em["abs_lat"] <= hi)]
        rec = dict(band=f"({lo},{hi}]", rows=len(g), miss=1 - g["tp"].mean() if len(g) else np.nan,
                   emitted=len(e), fp_share=e["fp"].mean() if len(e) else np.nan)
        out["by_lat"].append(rec)
        log(f"  {rec['band']:<12}{rec['rows']:>8}{100*rec['miss']:>9.1f}%{rec['emitted']:>9}{100*rec['fp_share']:>8.1f}%")
    log("\n  miss rate on the (|lat|, r) grid  [rows]:")
    RB = [0, 0.5, 0.8, 1.01]
    hdr = "  " + " " * 12 + "".join(f"{f'r({RB[j]},{RB[j+1]}]':>18}" for j in range(3))
    log(hdr)
    for lo, hi in zip(LAT[:-1], LAT[1:]):
        cells = []
        for j in range(3):
            g = gt[(gt["lat"] > lo) & (gt["lat"] <= hi) & (gt["r"] > RB[j]) & (gt["r"] <= RB[j + 1])]
            cells.append(f"{100*(1-g['tp'].mean()):5.1f}% [{len(g):4d}]" if len(g) >= 20 else f"{'--':>7} [{len(g):4d}]")
        log(f"  ({lo},{hi}]".ljust(14) + "".join(f"{c:>18}" for c in cells))

    # --- 2. candidates: TP share by lat, AUC of abs_lat
    from sklearn.metrics import roc_auc_score
    pool = cand[cand["conf"] >= 0.40]
    log("\n  pool candidates (conf >= 0.40): TP share and median p_oof by |lat| band")
    for lo, hi in zip(LAT[:-1], LAT[1:]):
        g = pool[(pool["abs_lat"] > lo) & (pool["abs_lat"] <= hi)]
        if len(g):
            log(f"    ({lo},{hi}]  n {len(g):>5}  TP share {100*g['y'].mean():5.1f}%  p_oof TP {g.loc[g['y']==1,'p_oof'].median():.3f}  FP {g.loc[g['y']==0,'p_oof'].median():.3f}")
    log(f"  single-feature AUC(abs_lat) for TP vs FP: {roc_auc_score(pool['y'], pool['abs_lat']):.3f}; "
        f"AUC(r_norm): {roc_auc_score(pool['y'], pool['r_norm']):.3f}")

    # --- 3. leave-abs_lat-out refit
    log("\n  OOF selector without abs_lat (31 features):")
    cand["p_nolat"] = oof_scores(cand, [f for f in feats if f != "abs_lat"], cand["fold"].to_numpy())
    d = []
    log(f"  {'fold':<5}{'32 feats':>9}{'no abs_lat':>11}{'Δ':>9}")
    for f in range(5):
        def ex(pcol):
            sub = cand[(cand["fold"] == f) & (cand["conf"] >= 0.40) & (cand[pcol] >= 0.60)]
            p = sub[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
            return score(gt[gt["fold"] == f][["filament_id", "segmentation_rle"]], p)["pq"]
        a, b = ex("p_oof"), ex("p_nolat")
        d.append(b - a)
        log(f"  {f:<5}{a:>9.4f}{b:>11.4f}{b-a:>+9.4f}")
    log(f"  mean Δ (drop abs_lat) {np.mean(d):+.4f}")
    out.update(auc_lat=float(roc_auc_score(pool["y"], pool["abs_lat"])), drop_lat_delta=[float(x) for x in d])
    json.dump(out, open(OUT / "latitude_structure.json", "w"), indent=1, default=float)
    log(f"\nwrote {OUT/'latitude_structure.json'}")


if __name__ == "__main__":
    main()
