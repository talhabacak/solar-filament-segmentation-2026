"""If we could paint the exact annotator mask wherever we detect, what then?

This bounds every remaining "better mask model" idea at once — SAM, DINOv2, a
bigger refiner, more training — because all of them can only improve the step
that turns a detection into pixels. None of them can find a filament the
detector missed, and none can rescue a detection that is in the wrong place.

The evidence that this needs checking: the seed refiner reaches mean IoU 0.708
and clears IoU>0.5 on 97% of instances when it is given GT boxes, yet gains only
+0.003 PQ when given detector boxes. That gap says the box, not the mask step,
may be what binds.

Method: take each detector candidate, find the GT instance it overlaps most, and
replace its mask with that GT mask exactly. De-overlap and score. The result is
an upper bound no mask model can pass, given this detector.

Two variants of the association, because they bound different things:
  best-IoU     replace with the GT the candidate already overlaps most
  best-centre  replace with the GT whose centroid is nearest, which also
               rescues candidates whose mask is wrong but whose location is right

Usage: python src/eda/perfect_mask_ceiling.py --fold 0
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament import IMG_H, IMG_W
from filament.pq import score, _entry_iou_matrix
from filament.rle import decode_rle, encode_mask, rles_to_layers
from semantic_candidates import R_MAX

OUT = ROOT / "outputs" / "eda"


def log(*a):
    print(*a, flush=True)


def centroid(m):
    ys, xs = np.nonzero(m)
    return (xs.mean(), ys.mean()) if len(ys) else (np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--confs", default="0.20,0.28,0.40")
    a = ap.parse_args()

    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    stems = [f.rsplit(".", 1)[0]
             for f in sorted(folds.loc[folds["fold"] == a.fold, "file_name"])]
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt_f = gt[gt["file_name"].isin({s + ".jpeg" for s in stems})]
    N = len(gt_f)
    gt_f = gt_f.copy()
    gt_f["stem"] = gt_f["file_name"].str.replace(".jpeg", "", regex=False)

    det_all = pd.read_parquet(ROOT / "outputs" / "oof_cache" / f"fold{a.fold}_preds_e013.parquet")
    det_all = det_all[det_all["r_norm"] <= R_MAX]
    if "view" in det_all.columns:
        det_all = det_all[det_all["view"] == "o"]

    gt_cache, res = {}, {}
    for conf in [float(c) for c in a.confs.split(",")]:
        sel = det_all[det_all["conf"] >= conf]
        rows_raw, rows_iou, rows_cen = [], [], []
        t0 = time.time()
        for stem, g in sel.groupby("stem", sort=False):
            gg = gt_f[gt_f["stem"] == stem]
            if gg.empty:
                continue
            # unique GT masks in this image (union over annotator copies is not
            # wanted here — each annotator's own drawing is a legitimate target)
            gl = rles_to_layers(gg["segmentation_rle"].tolist())
            dm = np.stack([decode_rle(r) for r in g["rle"]]).astype(np.uint8)
            iou = _entry_iou_matrix(gl, dm)                 # (n_gt, n_det)
            gc = np.array([centroid(x.astype(bool)) for x in gl])
            dc = np.array([centroid(x.astype(bool)) for x in dm])
            d2 = np.hypot(gc[:, None, 0] - dc[None, :, 0],
                          gc[:, None, 1] - dc[None, :, 1])
            confs = g["conf"].to_numpy()
            order = np.argsort(-confs)

            def emit(pick, sink):
                taken = np.zeros((IMG_H, IMG_W), bool)
                k = 0
                for j in order:
                    gi = pick(j)
                    if gi is None:
                        continue
                    m = gl[gi].astype(bool) & ~taken
                    if m.sum() < 64:
                        continue
                    taken |= m
                    k += 1
                    sink.append((f"{stem}_{k}", encode_mask(m.astype(np.uint8))))

            taken = np.zeros((IMG_H, IMG_W), bool)
            k = 0
            for j in order:
                m = dm[j].astype(bool) & ~taken
                if m.sum() < 64:
                    continue
                taken |= m
                k += 1
                rows_raw.append((f"{stem}_{k}", encode_mask(m.astype(np.uint8))))

            emit(lambda j: int(np.argmax(iou[:, j])) if iou[:, j].max() > 0 else None,
                 rows_iou)
            emit(lambda j: int(np.argmin(d2[:, j])), rows_cen)

        for name, rows in (("detector masks (actual)", rows_raw),
                           ("perfect mask, best-IoU assoc", rows_iou),
                           ("perfect mask, nearest-centre", rows_cen)):
            df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
            r = score(gt_f[["filament_id", "segmentation_rle"]], df,
                      gt_cache=gt_cache, return_details=True)
            hit = r["matches"]["gt_filament_id"].nunique()
            log(f"conf>={conf:.2f}  {name:30s} preds {len(df):5d}  "
                f"reach {hit:4d}/{N}={hit/N:.4f}  PQ {r['pq']:.4f} SQ {r['sq']:.4f} "
                f"TP {r['tp']:4d} FP {r['fp']:5d} FN {r['fn']:4d}")
            res[f"{conf}|{name}"] = dict(n=len(df), reach=hit / N, pq=r["pq"],
                                         sq=r["sq"], tp=r["tp"], fp=r["fp"],
                                         fn=r["fn"])
        log(f"  ({time.time()-t0:.0f}s)")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"perfect_mask_ceiling_fold{a.fold}.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    log(f"saved -> {OUT / f'perfect_mask_ceiling_fold{a.fold}.json'}")


if __name__ == "__main__":
    main()
