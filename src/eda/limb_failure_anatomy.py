"""Where exactly does the limb fail: proposal (the model never puts a candidate
on the filament), selection (a candidate exists and matches but is rejected),
or boundary (a candidate is emitted but its IoU falls under 0.5)?

Talha's question: is the limb a model problem or a selector problem? The
E112 stress table gives reach per radius band (21% at r > 0.9 against ~66%
inside) but does not say WHICH stage loses the rows. This decomposes every
GT row of the honest out-of-fold pool by its centroid radius:

  reached        matched (IoU > 0.5) by an EMITTED candidate under the
                 production policy (conf >= 0.40, p_oof >= 0.60)
  selectable     matched by SOME pool candidate, but that candidate was not
                 emitted -> the selector's loss (fixable without retraining)
  boundary       the best pool candidate overlaps the row but under IoU 0.5
                 (max IoU in (0.1, 0.5]) -> mask quality / shrink / merge
  never-proposed no pool candidate reaches IoU 0.1 -> the model's loss

plus the annotator-copy count of the row's image (limb rows may sit on
images with fewer copies) and the row's mean local width, so the limb's
failure profile can be compared with the disc's. All from cached artefacts;
CPU; the exact scorer's match table at the "everything emitted" policy gives
the per-GT best IoU.

Usage: python src/eda/limb_failure_anatomy.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "eda"))

from filament.pq import score
from filament.rle import decode_rle

OUT = ROOT / "outputs" / "eda"
DISK_C, DISK_R = 1023.5, 900.0
BANDS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]


def log(*a):
    print(*a, flush=True)


def main():
    cand = pd.read_parquet(OUT / "e115b_labelled_pool.parquet")
    po = pd.read_parquet(OUT / "e115b_p_oof.parquet").set_index("cand_id")["p_oof"]
    cand["p_oof"] = cand["cand_id"].map(po)
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fs = set(folds["file_name"].str.replace(".jpeg", "", regex=False))
    gt = gt[gt["stem"].isin(fs)].copy()          # all folds_v1 stems: fixed scope
    log(f"GT rows {len(gt)}, pool {len(cand)}")

    # per-GT geometry: centroid radius, mean local width, image copy count
    r, w = [], []
    for rle in gt["segmentation_rle"]:
        m = decode_rle(rle).astype(np.uint8)
        ys, xs = np.nonzero(m)
        r.append(float(np.hypot(xs.mean() - DISK_C, ys.mean() - DISK_C) / DISK_R) if len(xs) else np.nan)
        d = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        w.append(float(2 * d[m > 0].mean()) if len(xs) else np.nan)
    gt["r"], gt["width"] = r, w
    gt["copies"] = gt["stem"].map(gt.groupby("stem")["entry_id"].nunique())
    log("geometry done")

    def matches(sub, thr_note):
        pred = sub[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
        res = score(gt[["filament_id", "segmentation_rle"]], pred, return_details=True)
        m = res["matches"]
        log(f"  {thr_note}: PQ {res['pq']:.4f} TP {res['tp']} FP {res['fp']} FN {res['fn']}")
        return set(m["gt_filament_id"])

    # 1) emitted under the production policy
    emitted = cand[(cand["conf"] >= 0.40) & (cand["p_oof"] >= 0.60)]
    reached = matches(emitted, "production policy")
    # 2) matched by ANY pool candidate (everything emitted; the scorer is strict IoU > 0.5)
    anyhit = matches(cand, "whole pool")
    # 3) best overlap under 0.5: per-GT max IoU with any candidate, cheap via
    #    the bank-style bbox check is heavy; approximate with the scorer's
    #    match table at a lowered threshold is not available, so use the
    #    candidate labels: a GT row 'touched' if any candidate on its image
    #    has iou_sum > 0 -- too coarse. Instead compute max IoU directly for
    #    the unmatched rows only (their number is what we need).
    unmatched = gt[~gt["filament_id"].isin(anyhit)]
    by_stem = {s: g for s, g in cand.groupby("stem")}
    best = {}
    for stem, g in unmatched.groupby("stem"):
        cs = by_stem.get(stem)
        if cs is None:
            for fid in g["filament_id"]:
                best[fid] = 0.0
            continue
        cmasks = [decode_rle(x).astype(bool) for x in cs["rle"]]
        for fid, rle in zip(g["filament_id"], g["segmentation_rle"]):
            gm = decode_rle(rle).astype(bool)
            ga = gm.sum()
            b = 0.0
            for cm in cmasks:
                inter = (gm & cm).sum()
                if inter:
                    b = max(b, inter / (ga + cm.sum() - inter))
            best[fid] = float(b)
    gt["best_iou_unmatched"] = gt["filament_id"].map(best)
    log("unmatched rows' best IoU done")

    def status(row):
        f = row["filament_id"]
        if f in reached:
            return "reached"
        if f in anyhit:
            return "selectable"
        return "boundary" if row["best_iou_unmatched"] > 0.1 else "never-proposed"
    gt["status"] = gt.apply(status, axis=1)

    log(f"\n  {'band':<12}{'rows':>6}{'reached':>9}{'selectable':>12}{'boundary':>10}{'never':>8}{'copies':>8}{'width':>7}")
    rows = []
    for lo, hi in BANDS:
        g = gt[(gt["r"] > lo) & (gt["r"] <= hi)]
        n = len(g)
        c = g["status"].value_counts()
        rec = dict(band=f"({lo},{hi}]", rows=n,
                   reached=c.get("reached", 0) / n, selectable=c.get("selectable", 0) / n,
                   boundary=c.get("boundary", 0) / n, never=c.get("never-proposed", 0) / n,
                   copies=g["copies"].mean(), width=g["width"].median())
        rows.append(rec)
        log(f"  {rec['band']:<12}{n:>6}{100*rec['reached']:>8.1f}%{100*rec['selectable']:>11.1f}%"
            f"{100*rec['boundary']:>9.1f}%{100*rec['never']:>7.1f}%{rec['copies']:>8.2f}{rec['width']:>7.1f}")
    g = gt
    log(f"  {'ALL':<12}{len(g):>6}{100*(g['status']=='reached').mean():>8.1f}%"
        f"{100*(g['status']=='selectable').mean():>11.1f}%{100*(g['status']=='boundary').mean():>9.1f}%"
        f"{100*(g['status']=='never-proposed').mean():>7.1f}%")
    pd.DataFrame(rows).to_csv(OUT / "limb_failure_anatomy.csv", index=False)
    gt[["filament_id", "stem", "r", "width", "copies", "status", "best_iou_unmatched"]].to_parquet(
        OUT / "limb_failure_anatomy_rows.parquet", index=False)
    log(f"\nwrote {OUT/'limb_failure_anatomy.csv'} and _rows.parquet")


if __name__ == "__main__":
    main()
