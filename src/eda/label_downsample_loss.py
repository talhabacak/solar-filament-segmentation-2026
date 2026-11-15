"""E179 -- how much of the LABEL does the training cache destroy?

Talha, 2026-09-13: "what pooling-like layers do we have, could better choices be made, they may
be causing information loss." The encoder turned out to have no pooling at all (five strided
convolutions, four of them depthwise) and the decoder upsamples with nearest. But the biggest
resampling in the pipeline is neither: it is the TARGET. The cache rasterises every annotator
mask at 2048, area-averages it to the training size, and thresholds at 0.5
(`build_semantic_cache.py`: `cv2.resize(m, (S, S), INTER_AREA) > 0.5`).

Area-averaging is the right filter, but the 0.5 threshold is a decision, and for a structure
1-2 px wide at 2048 it can delete pixels the annotator drew: at 1536 a pixel covers 1.78 source
pixels, so a 2 px ribbon crossing a pixel diagonally may occupy less than half of it. If that
happens often, the model is being trained to miss exactly the thin population the error atlas
says we lose -- and no architectural change can recover a target that was never shown.

Measured per ground-truth instance (connected component of an annotator entry at 2048):
  area ratio      area at S, scaled back, over area at 2048
  survives        does the instance still exist at S at all
  fragments       connected components at S against 1 at 2048 (a broken filament)
  IoU             the mask at S, upsampled back to 2048, against the original

and the same for the alternative thresholds 0.25 and 0.35, which is the cheap fix if the loss
is real. Reported overall and by drawn width, the axis that should drive it.

Usage: python src/eda/label_downsample_loss.py [--size 1536] [--limit 150]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from filament.rle import decode_rle  # noqa: E402

OUT = ROOT / "outputs" / "eda"


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=1536)
    ap.add_argument("--limit", type=int, default=150)
    a = ap.parse_args()
    S = a.size
    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    stems = sorted(gt["stem"].unique())
    step = max(len(stems) // a.limit, 1)
    stems = stems[::step][: a.limit]
    THRS = (0.5, 0.35, 0.25)
    rows = []
    for n, stem in enumerate(stems):
        for rle in gt.loc[gt["stem"] == stem, "segmentation_rle"]:
            m = decode_rle(rle).astype(np.uint8)
            area0 = int(m.sum())
            if area0 < 32:
                continue
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            skel = max(sum(cv2.arcLength(c, True) for c in cnts) / 2.0, 1.0)
            small = cv2.resize(m.astype(np.float32), (S, S), interpolation=cv2.INTER_AREA)
            rec = dict(stem=stem, area0=area0, width=area0 / skel)
            for t in THRS:
                b = (small > t).astype(np.uint8)
                area1 = int(b.sum()) * (2048 / S) ** 2
                up = cv2.resize(b, (2048, 2048), interpolation=cv2.INTER_NEAREST)
                inter = int((up & m).sum())
                union = int((up | m).sum())
                ncomp = cv2.connectedComponents(b, connectivity=8)[0] - 1
                rec[f"ratio_{t}"] = area1 / area0
                rec[f"iou_{t}"] = inter / max(union, 1)
                rec[f"comps_{t}"] = ncomp
                rec[f"gone_{t}"] = int(b.sum() == 0)
            rows.append(rec)
        if (n + 1) % 30 == 0:
            log(f"  {n + 1}/{len(stems)} images, {len(rows)} instances")
    d = pd.DataFrame(rows)
    d.to_parquet(OUT / f"label_downsample_{S}.parquet", index=False)
    log(f"\n{len(d)} ground-truth instances at 2048, resampled to {S}")
    for t in THRS:
        log(f"\nthreshold {t}:")
        log(f"  area kept   median {d[f'ratio_{t}'].median():.3f}   p5 {d[f'ratio_{t}'].quantile(.05):.3f}")
        log(f"  IoU with the 2048 mask   median {d[f'iou_{t}'].median():.3f}   p5 {d[f'iou_{t}'].quantile(.05):.3f}")
        log(f"  instances deleted entirely: {int(d[f'gone_{t}'].sum())} ({d[f'gone_{t}'].mean():.2%})")
        log(f"  instances broken into >1 component: {int((d[f'comps_{t}'] > 1).sum())} ({(d[f'comps_{t}'] > 1).mean():.2%})")
    log("\nby drawn width (area / half-perimeter), production threshold 0.5:")
    g = d.assign(b=pd.cut(d["width"], [0, 6, 8, 11, 14, 999])).groupby("b", observed=True).agg(
        n=("area0", "size"), area_kept=("ratio_0.5", "median"), iou=("iou_0.5", "median"),
        deleted=("gone_0.5", "mean"), broken=("comps_0.5", lambda x: (x > 1).mean()))
    log(g.round(3).to_string())
    log("\nthe same, threshold 0.25:")
    g2 = d.assign(b=pd.cut(d["width"], [0, 6, 8, 11, 14, 999])).groupby("b", observed=True).agg(
        n=("area0", "size"), area_kept=("ratio_0.25", "median"), iou=("iou_0.25", "median"),
        deleted=("gone_0.25", "mean"), broken=("comps_0.25", lambda x: (x > 1).mean()))
    log(g2.round(3).to_string())
    json.dump({f"thr_{t}": dict(area_kept=float(d[f"ratio_{t}"].median()), iou=float(d[f"iou_{t}"].median()),
                                deleted=float(d[f"gone_{t}"].mean()), broken=float((d[f"comps_{t}"] > 1).mean()))
               for t in THRS} | {"n": len(d), "size": S},
              open(OUT / f"label_downsample_{S}.json", "w"), indent=1)
    log("\nLABEL_LOSS_DONE")


if __name__ == "__main__":
    main()
