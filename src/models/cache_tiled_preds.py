"""Cache the native-resolution tiled lane's instances in the oof_cache schema.

Why this lane deserves a second look. `eval_tiled.py` gated it against the wrong
bar. It required the tiled lane to beat our ORACLE (0.5078) before the pivot
counted, and reported "DOES NOT PASS" at 0.4277. But 0.4277 was scored by a
threshold and connected components alone -- no detector ensemble, no candidate
pool, no selector -- against a full five-model pipeline that realises 0.4286.
A single lane matching the whole pipeline is not a failure; it is the strongest
standalone result we have, tested against a bar meant for a replacement rather
than for a source.

The question this reopens is the one the M2F test asked and answered
affirmatively: does the lane MISS DIFFERENT INSTANCES? That is the only property
that has ever added at the pipeline level here. The tiled lane has the strongest
prior reason to differ of anything we have built -- it sees pixels at native
2048 in 1056-wide tiles, where a filament is 10-12 px across, instead of the
768 raster where the same structure is 4-4.5 px and the decoder's smoothing sets
the boundary.

Schema matches sweep_oof and cache_m2f_preds (stem, conf, area, r_norm, rle),
masks NOT de-overlapped because the pool build does that for every source.
Connected components are already pixel-disjoint within an image.

conf carries the mean predicted probability inside the component, which is this
lane's analogue of a detection score and keeps the selector's confidence
features meaningful.

Post-processing is fixed at the operating point `eval_tiled.py` swept to
(threshold 0.7, min_area 400, closing 9), NOT swept again here. Emitting several
thresholds as separate candidates is precisely what failed before: the pool
de-overlaps greedily at construction, so nested variants of one filament steal
each other's pixels and corrupt the good candidate's mask before any selector
sees it (ledger, multithreshold semantic extraction, -0.0161).

Usage:
  python src/models/cache_tiled_preds.py \
      --weights outputs/runs_tiled/tiled_b3_f0_1056/best.pt --fold 0
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
from filament.rle import encode_mask

CACHE = ROOT / "data" / "tile_cache"
OOF = ROOT / "outputs" / "oof_cache"
DISK_C = 1023.5
DISK_R = 900.0
THR, MIN_AREA, CLOSE_K = 0.7, 400, 9


def log(*a):
    print(*a, flush=True)


def instances(prob, thr=THR, min_area=MIN_AREA, close_k=CLOSE_K):
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
    ap.add_argument("--tag", default="_tiled")
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
    frow = {f: i for i, f in enumerate(idx["files"])}
    log(f"fold {a.fold}: {len(files)} images")

    rows, t0 = [], time.time()
    for n, fn in enumerate(files):
        prob = tiling.mosaic_predict(model, np.asarray(imgs[frow[fn]]), tta=a.tta)
        stem = fn.rsplit(".", 1)[0]
        for m in instances(prob):
            ys, xs = np.nonzero(m)
            rows.append(dict(
                stem=stem, conf=float(prob[m].mean()), area=int(m.sum()),
                r_norm=float(np.hypot(xs.mean() - DISK_C,
                                      ys.mean() - DISK_C) / DISK_R),
                rle=encode_mask(m.astype(np.uint8))))
        if (n + 1) % 40 == 0:
            log(f"  {n+1}/{len(files)} images, {len(rows)} candidates "
                f"({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    out = OOF / f"fold{a.fold}_preds{a.tag}.parquet"
    df.to_parquet(out, index=False)
    log(f"\nwrote {out}: {len(df)} candidates over {df['stem'].nunique()} images "
        f"({len(df)/max(df['stem'].nunique(),1):.1f} per image, "
        f"{time.time()-t0:.0f}s)")
    log(f"  conf {df['conf'].min():.3f}-{df['conf'].max():.3f} "
        f"(median {df['conf'].median():.3f}), "
        f"area median {df['area'].median():.0f}")


if __name__ == "__main__":
    main()
