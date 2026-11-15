"""Fold-0 PQ for the Mask2Former lane -- the gate that decides whether it lives.

Queries to instances: each of the 100 queries carries a class logit pair
(filament / no-object) and a mask logit map at 1/4 resolution. A query becomes a
candidate when P(filament) clears a score threshold; its mask is the sigmoid map
thresholded at 0.5, upsampled to 2048. Candidates are then made pixel-disjoint
in score order (the metric requires disjointness) and small remnants dropped --
the same convention as every other lane, so the comparison is fair.

Score sweep instead of a fixed 0.5: a freshly fine-tuned M2F's calibration is
unknown, and the operating-point lesson from the detector lanes (their conf 0.3
/ NMS 0.0 was well-chosen for the metric) says the threshold IS part of the
measurement.

Context numbers on the same fold: production pipeline 0.4383, plain yolo11m
0.4193-0.4235, tiled CC lane 0.4277. The lane earns further work if it lands in
that band WITH a different error profile (checked via TP overlap against the
production kept set), or beats it outright.

Usage: python src/models/eval_m2f.py --weights outputs/runs_m2f/m2f_swint_f0_1024/last.pt
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament.pq import score
from filament.rle import encode_mask

CACHE = ROOT / "data" / "semantic_cache"
S = 1024
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--scores", default="0.5,0.7,0.85")
    ap.add_argument("--min-area", type=int, default=64)
    a = ap.parse_args()

    from transformers import Mask2FormerForUniversalSegmentation
    ck = torch.load(a.weights, map_location="cpu", weights_only=False)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        ck["hf_name"], num_labels=1, ignore_mismatched_sizes=True)
    model.load_state_dict(ck["model"])
    model = model.cuda().eval()

    import json
    idx = json.loads((CACHE / f"index_{S}.json").read_text())
    imgs = np.load(CACHE / f"images_{S}.npy", mmap_mode="r")
    frow = {f: i for i, f in enumerate(idx["files"])}
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    files = sorted(folds.loc[folds["fold"] == a.fold, "file_name"].unique())
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt = gt[gt["file_name"].isin(files)]
    log(f"fold {a.fold}: {len(files)} images, {len(gt)} GT rows")

    scores_t = [float(s) for s in a.scores.split(",")]
    preds = {t: [] for t in scores_t}
    t0 = time.time()
    with torch.no_grad():
        for n, fn in enumerate(files):
            img = np.asarray(imgs[frow[fn]])
            x = (np.stack([img] * 3, -1).astype(np.float32) - MEAN) / STD
            x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].cuda()
            out = model(pixel_values=x)
            cls = out.class_queries_logits[0].softmax(-1)[:, 0]      # P(filament)
            masks = out.masks_queries_logits[0].sigmoid()            # (Q, h, w)
            masks = torch.nn.functional.interpolate(
                masks[None], size=(2048, 2048), mode="bilinear")[0]
            cls_np = cls.float().cpu().numpy()
            stem = fn.rsplit(".", 1)[0]
            for thr in scores_t:
                order = np.argsort(-cls_np)
                taken = np.zeros((2048, 2048), bool)
                k = 0
                for qi in order:
                    if cls_np[qi] < thr:
                        break
                    m = (masks[qi] > 0.5).cpu().numpy()
                    m = m & ~taken
                    if m.sum() < a.min_area:
                        continue
                    taken |= m
                    k += 1
                    preds[thr].append((f"{stem}_{k}", encode_mask(m.astype(np.uint8))))
            if (n + 1) % 40 == 0:
                log(f"  {n+1}/{len(files)} ({time.time()-t0:.0f}s)")

    for thr in scores_t:
        pdf = pd.DataFrame(preds[thr], columns=["filament_id", "segmentation_rle"])
        if pdf.empty:
            log(f"score>={thr}: no predictions")
            continue
        s = score(gt[["filament_id", "segmentation_rle"]], pdf)
        log(f"score>={thr}: PQ {s['pq']:.4f} SQ {s['sq']:.4f} RQ {s['rq']:.4f}  "
            f"TP {s['tp']} FP {s['fp']} FN {s['fn']}  n={len(pdf)}")
    log("\ncontext (same fold): production pipeline 0.4383, yolo11m 0.4193-0.4235, "
        "tiled CC 0.4277")


if __name__ == "__main__":
    main()
