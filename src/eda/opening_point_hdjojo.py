"""Is the public 0.40 recipe's advantage in the MODEL or in the OPERATING POINT?

HDJoJo's inference notebook is a verified data point: written after the 12 August
rescore, and another team ran it 23 seconds after its last run and scored 0.40.
It is a single yolov8x-seg at 1792 with conf 0.3, NMS iou 0.0, max_det 100 and a
subtract-as-you-go de-overlap. No folds, no ensemble, no selector, no TTA. Our
far more elaborate pipeline scores 0.38.

Before spending ten hours training a yolov8x to find out, the cheap half of the
question can be answered with assets we already have: apply THEIR operating point
to OUR five e013 yolo11m folds, out-of-fold and honest, scored with the
organiser-parity scorer. If our own detectors under their settings jump towards
0.40, the difference was never the backbone.

Four things separate the two operating points and each is varied here:

  conf        they keep everything above 0.3 raw confidence; we pool from 0.05
              and then let a 32-feature selector decide.
  NMS iou     0.0 suppresses a box overlapping ANY higher-scoring box at all --
              maximally aggressive. Ours is swept, typically 0.05-0.7.
  max_det     100 vs the 300 default.
  de-overlap  they sort by `boxes.cls`, which for nc=1 is all zeros, so argsort
              is stable and [::-1] reverses it: detections are processed in
              REVERSE confidence order, giving the WEAKEST detection priority
              over the pixels it shares. Probably unintended. Both orders are
              measured, since if it matters at all it matters in our favour to
              know.

Everything is scored per fold on that fold's held-out images, then pooled -- the
same honest OOF our own numbers use, so the comparison is like for like.

Usage: python src/eda/opening_point_hdjojo.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament import IMG_H, IMG_W
from filament.pq import score
from filament.rle import encode_mask

OUT = ROOT / "outputs" / "eda"
IMDIR = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
RUNS = ROOT / "outputs" / "runs"
OURS_OOF = 0.4178          # our best single-model YOLO OOF (E015 union), docs/15


def log(*a):
    print(*a, flush=True)


def deoverlap(masks, order):
    """Subtract-as-you-go, exactly as the public notebook does it.

    Note it accumulates the ORIGINAL mask, not the cleaned one, so a later mask
    is cut by the union of every earlier mask rather than by what survived.
    """
    acc = np.zeros(masks.shape[1:], np.uint8)
    out = []
    for i in order:
        m = masks[i]
        out.append(m & ~acc)
        acc |= m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--folds", default="0,1,2,3,4")
    a = ap.parse_args()

    from ultralytics import YOLO
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    gt_all = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    paths = {p.stem: p for p in IMDIR.rglob("*.jpeg")}

    # (conf, nms_iou, max_det, order) -- 'theirs' is reverse-confidence
    # Trimmed to the informative axes after a 12-image probe showed max_det and
    # the de-overlap order to be inert (identical PQ to 4 decimals), so the full
    # five-fold run spends its time on conf and NMS instead.
    GRID = [
        (0.30, 0.00, 100, "theirs"),     # their exact settings
        (0.30, 0.70, 100, "byconf"),     # their conf, ultralytics-default NMS
        (0.20, 0.00, 100, "byconf"),
        (0.10, 0.00, 100, "byconf"),
        (0.40, 0.00, 100, "byconf"),
    ]
    raw = {g: [] for g in GRID}
    stems_used = []

    for f in [int(x) for x in a.folds.split(",")]:
        w = RUNS / f"e013_yolo11m_f{f}_1792" / "weights" / "best.pt"
        if not w.exists():
            log(f"fold {f}: no weights, skipping")
            continue
        model = YOLO(str(w))
        va = sorted(folds.loc[folds["fold"] == f, "file_name"].unique())
        va = [v.rsplit(".", 1)[0] for v in va]
        if a.limit:
            va = va[:a.limit]
        t0 = time.time()
        for stem in va:
            stems_used.append(stem)
            for (conf, nms, mdet, order) in GRID:
                r = model.predict(str(paths[stem]), imgsz=1792, conf=conf,
                                  iou=nms, max_det=mdet, verbose=False)[0]
                if r.masks is None:
                    continue
                md = (r.masks.data.cpu().numpy() > 0.5).astype(np.uint8)
                cf = r.boxes.conf.cpu().numpy()
                idx = (np.argsort(np.zeros(len(cf)))[::-1] if order == "theirs"
                       else np.argsort(cf)[::-1])
                for i, m in enumerate(deoverlap(md, idx)):
                    mr = cv2.resize(m, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
                    if mr.sum() < 5:
                        continue
                    raw[(conf, nms, mdet, order)].append(
                        (f"{stem}_{i+1}", encode_mask(mr)))
        log(f"fold {f}: {len(va)} images ({time.time()-t0:.0f}s)")
        del model

    gt = gt_all[gt_all["file_name"].isin({s + ".jpeg" for s in stems_used})]
    log(f"\npooled OOF over {len(stems_used)} images, "
        f"{gt['entry_id'].nunique()} annotator entries\n")
    log(f"{'conf':>5s} {'nms':>5s} {'maxdet':>7s} {'order':>7s} "
        f"{'PQ':>7s} {'SQ':>7s} {'RQ':>7s} {'TP':>6s} {'FP':>6s} {'FN':>6s} {'n':>6s}")
    rows = []
    for g in GRID:
        pred = raw[g]
        if not pred:
            continue
        pdf = pd.DataFrame(pred, columns=["filament_id", "segmentation_rle"])
        s = score(gt[["filament_id", "segmentation_rle"]], pdf)
        rows.append(dict(conf=g[0], nms=g[1], max_det=g[2], order=g[3],
                         n_pred=len(pdf), **{k: s[k] for k in
                                             ("pq", "sq", "rq", "tp", "fp", "fn")}))
        log(f"{g[0]:5.2f} {g[1]:5.2f} {g[2]:7d} {g[3]:>7s} "
            f"{s['pq']:7.4f} {s['sq']:7.4f} {s['rq']:7.4f} "
            f"{s['tp']:6d} {s['fp']:6d} {s['fn']:6d} {len(pdf):6d}")

    df = pd.DataFrame(rows).sort_values("pq", ascending=False)
    best = df.iloc[0]
    log(f"\nbest of their-style operating points: PQ {best.pq:.4f} "
        f"(conf {best.conf}, nms {best.nms}, order {best.order})")
    log(f"our recorded single-model YOLO OOF: {OURS_OOF:.4f}  "
        f"({best.pq - OURS_OOF:+.4f})")
    log(f"our full two-lane + selector pipeline: 0.4286 pooled  "
        f"({best.pq - 0.4286:+.4f})")
    log("\nreading: if their operating point lifts OUR detectors materially, the "
        "public 0.40 is about post-processing, not the backbone -- and a "
        "yolov8x run is not the experiment to spend ten hours on.")

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "operating_point_hdjojo.csv", index=False)
    log(f"\nsaved -> {OUT/'operating_point_hdjojo.csv'}")


if __name__ == "__main__":
    main()
