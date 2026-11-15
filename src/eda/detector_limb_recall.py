"""Detector lane at the limb -- the precondition named in ledger §16c.

The rejected-but-true limb candidates carry detector members at half the
confidence of the emitted ones. Before any detector-side work, measure the
detector alone: for every GT row (all folds_v1 stems), does the fold's
held-out YOLO detector (e013, conf >= 0.05 cache) produce a mask that hits
the row at IoU > 0.5, and at what confidence? By radius band:

  recall@0.05   share of GT rows hit by ANY cached detection (conf >= 0.05)
  recall@0.25   share hit by a detection with conf >= 0.25 (about where the
                selector starts trusting a member, §16c: emitted TP det_conf
                median 0.60-0.68, rejected 0.24-0.28)
  hit conf      median max-conf of the hitting detections
  IoU           median IoU of the best hit

Reading: if recall@0.05 at r > 0.8 is close to the disc's, the detector SEES
the limb filaments but scores them low -> a calibration/confidence problem
(cheap: limb-aware conf, or the selector's det features re-weighted by band);
if recall@0.05 itself collapses at the limb, the detector does not propose
there -> detector training must change (limb augmentation / resolution /
loss), which is the expensive road. Both outcomes are written down before
the run.

Usage: python src/eda/detector_limb_recall.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from filament.rle import decode_rle

OUT = ROOT / "outputs" / "eda"
CACHE = ROOT / "outputs" / "oof_cache"
DISK_C, DISK_R = 1023.5, 900.0
BANDS = [(0.0, 0.5), (0.5, 0.8), (0.8, 0.9), (0.9, 1.01)]


def log(*a):
    print(*a, flush=True)


def main():
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    gt = gt[gt["stem"].isin(fold_of)].copy()
    det = pd.concat([pd.read_parquet(CACHE / f"fold{k}_preds_e013.parquet").assign(fold=k) for k in range(5)],
                    ignore_index=True)
    if "view" in det.columns:
        det = det[det["view"] == "o"]
    log(f"GT rows {len(gt)}; detections {len(det)} (conf >= {det['conf'].min():.2f}) over {det['stem'].nunique()} stems")

    rows = []
    by_stem = {s: g for s, g in det.groupby("stem")}
    for i, (stem, g) in enumerate(gt.groupby("stem")):
        d = by_stem.get(stem)
        dm = [(decode_rle(r).astype(bool), float(c)) for r, c in zip(d["rle"], d["conf"])] if d is not None else []
        for r in g.itertuples():
            m = decode_rle(r.segmentation_rle).astype(bool)
            ys, xs = np.nonzero(m)
            if not len(xs):
                continue
            rr = float(np.hypot(xs.mean() - DISK_C, ys.mean() - DISK_C) / DISK_R)
            ga = m.sum()
            best_iou, best_conf, conf_hits = 0.0, 0.0, []
            for cm, c in dm:
                inter = (m & cm).sum()
                if not inter:
                    continue
                iou = inter / (ga + cm.sum() - inter)
                if iou > 0.5:
                    conf_hits.append(c)
                if iou > best_iou:
                    best_iou, best_conf = iou, c
            rows.append(dict(filament_id=r.filament_id, r=rr, best_iou=best_iou,
                             hit=bool(conf_hits), hit_conf=max(conf_hits) if conf_hits else np.nan,
                             hit25=any(c >= 0.25 for c in conf_hits)))
        if (i + 1) % 100 == 0:
            log(f"  {i+1} images")
    t = pd.DataFrame(rows)
    t.to_parquet(OUT / "detector_limb_recall_rows.parquet", index=False)
    log(f"\n  {'band':<10}{'rows':>6}{'recall@0.05':>13}{'recall@0.25':>13}{'hit conf med':>14}{'best IoU med':>14}{'touched(IoU>0.1)':>18}")
    summ = []
    for lo, hi in BANDS + [(0.0, 1.01)]:
        g = t[(t["r"] > lo) & (t["r"] <= hi)]
        rec = dict(band=f"({lo},{hi}]", rows=len(g), recall05=g["hit"].mean(), recall25=g["hit25"].mean(),
                   hit_conf=g["hit_conf"].median(), best_iou=g["best_iou"].median(),
                   touched=(g["best_iou"] > 0.1).mean())
        summ.append(rec)
        log(f"  {rec['band']:<10}{rec['rows']:>6}{100*rec['recall05']:>12.1f}%{100*rec['recall25']:>12.1f}%"
            f"{rec['hit_conf']:>14.3f}{rec['best_iou']:>14.3f}{100*rec['touched']:>17.1f}%")
    pd.DataFrame(summ).to_csv(OUT / "detector_limb_recall.csv", index=False)
    log(f"\nwrote {OUT/'detector_limb_recall.csv'}")


if __name__ == "__main__":
    main()
