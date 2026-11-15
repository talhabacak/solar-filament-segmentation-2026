"""The gate: does the tiled lane beat our own oracle on the real metric?

Pixel-Dice is the training signal, not the score. This runs the actual thing:
blended mosaic -> binary -> connected components -> official-parity PQ against
every annotator copy, exactly as the submission would be scored.

The bar is deliberately harsh. The tiled lane does not have to beat our realised
0.4286; it has to beat our ORACLE, 0.5078 -- what the existing pipeline would
score if its selector were perfect. Anything below that and the pivot has not
bought a new ceiling, it has only traded one plateau for another, and docs/17
says to find that out early rather than after five folds.

Post-processing is swept rather than assumed, over the three knobs that turn a
probability map into instances:

  threshold     where the map becomes binary
  min_area      drops specks. Our GT has real instances down to a few hundred px,
                so this cannot be set generously without deleting true positives.
  close_kernel  morphological closing rejoins a thread broken by a faint gap.
                This is aimed straight at the metric's one-to-many penalty: a
                filament labelled as two components scores neither at IoU > 0.5.

Every combination is scored with the same scorer the submission uses, so nothing
about the sweep can flatter it.

Usage:
  python src/models/eval_tiled.py --weights outputs/runs_tiled/<name>/best.pt --fold 0
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

import tiling
from tiling import FULL
from filament.pq import score
from filament.rle import encode_mask

CACHE = ROOT / "data" / "tile_cache"
OUT = ROOT / "outputs" / "eda"
ORACLE = 0.5078          # the existing pipeline's ceiling, docs/15
REALISED = 0.4286        # what it actually scores


def log(*a):
    print(*a, flush=True)


def instances(prob, thr, min_area, close_k):
    """Probability map -> list of disjoint boolean masks.

    Connected components are inherently pixel-disjoint, which is what the metric
    requires; no de-overlap step is needed here, unlike the detector lane where
    duplicate proposals had to be resolved.
    """
    b = (prob > thr).astype(np.uint8)
    if close_k:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    return [(lab == i) for i in range(1, n)
            if stats[i, cv2.CC_STAT_AREA] >= min_area]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    import segmentation_models_pytorch as smp
    ck = torch.load(a.weights, map_location="cuda", weights_only=False)
    enc = a.encoder or ck["encoder"]
    model = smp.Unet(encoder_name=enc, encoder_weights=None, in_channels=3,
                     classes=1, decoder_attention_type="scse").cuda().eval()
    model.load_state_dict(ck["model"])
    log(f"{Path(a.weights).parent.name}: {enc}, val pixel-Dice "
        f"{ck.get('val_dice', float('nan')):.4f}, tta={a.tta}")

    idx = json.loads((CACHE / f"index_{FULL}.json").read_text())
    imgs = np.load(CACHE / f"images_{FULL}.npy", mmap_mode="r")
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"], folds["fold"]))
    files = [f for f in idx["files"] if fold_of[f] == a.fold]
    if a.limit:
        files = files[:a.limit]
    frow = {f: i for i, f in enumerate(idx["files"])}
    log(f"fold {a.fold}: {len(files)} images")

    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt = gt[gt["file_name"].isin(files)]
    log(f"  {len(gt)} GT instances over "
        f"{gt['entry_id'].nunique()} annotator entries")

    t0 = time.time()
    probs = {}
    for n, fn in enumerate(files):
        probs[fn] = tiling.mosaic_predict(model, np.asarray(imgs[frow[fn]]),
                                          tta=a.tta)
        if (n + 1) % 40 == 0:
            log(f"  {n+1}/{len(files)} mosaics ({time.time()-t0:.0f}s)")
    log(f"  mosaics done ({time.time()-t0:.0f}s)")

    THRS = (0.3, 0.4, 0.5, 0.6, 0.7)
    AREAS = (100, 200, 400)
    CLOSES = (0, 5, 9)
    rows = []
    for thr in THRS:
        for ma in AREAS:
            for ck_ in CLOSES:
                pred = []
                for fn in files:
                    stem = fn.rsplit(".", 1)[0]
                    for i, m in enumerate(instances(probs[fn], thr, ma, ck_)):
                        pred.append((f"{stem}_{i}", encode_mask(m.astype(np.uint8))))
                if not pred:
                    continue
                pdf = pd.DataFrame(pred, columns=["filament_id", "segmentation_rle"])
                s = score(gt[["filament_id", "segmentation_rle"]], pdf)
                rows.append(dict(thr=thr, min_area=ma, close=ck_, n_pred=len(pdf),
                                 **{k: s[k] for k in ("pq", "sq", "rq", "tp", "fp", "fn")}))
                log(f"  thr {thr} area {ma:4d} close {ck_}: PQ {s['pq']:.4f} "
                    f"SQ {s['sq']:.4f} RQ {s['rq']:.4f}  "
                    f"TP {s['tp']} FP {s['fp']} FN {s['fn']}  n={len(pdf)}")
    df = pd.DataFrame(rows).sort_values("pq", ascending=False)

    best = df.iloc[0]
    log(f"\nBEST: PQ {best.pq:.4f} at thr {best.thr} min_area {int(best.min_area)} "
        f"close {int(best.close)}")
    log(f"  vs existing pipeline REALISED {REALISED:.4f}  "
        f"({best.pq - REALISED:+.4f})")
    log(f"  vs existing pipeline ORACLE   {ORACLE:.4f}  "
        f"({best.pq - ORACLE:+.4f})  <- this is the gate")
    log(f"\n  {'PASSES' if best.pq > ORACLE else 'DOES NOT PASS'} the gate. "
        + ("The tiled lane raises the ceiling, so it becomes the main line."
           if best.pq > ORACLE else
           "Below our own oracle: the pivot has not bought a new ceiling yet."))

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "eval_tiled_sweep.csv", index=False)
    json.dump(dict(best=best.to_dict(), oracle=ORACLE, realised=REALISED,
                   weights=str(a.weights), tta=a.tta),
              open(OUT / "eval_tiled.json", "w"), indent=1, default=float)
    log(f"\nsaved -> {OUT/'eval_tiled_sweep.csv'}")


if __name__ == "__main__":
    main()
