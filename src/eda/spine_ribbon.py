"""Is an annotated filament just a constant-width ribbon around its spine?

MAGFiLO ships a `spine` polyline with every one of the 8199 annotations (100%
coverage) and we have never used it. Area divided by spine length is ~11 px for
all three classes (Left 10.92, Right 11.10, Unidentifiable 11.23) — U filaments
are shorter, not thinner — which is what you would see if annotators traced an
axis and swept a fixed-width brush along it.

If that is what the masks are, the mask is nearly determined by the spine, and
predicting a 1-D curve is a far easier problem than predicting a 2-D region.
Our SQ is 0.67, so an accurate ribbon model would be worth a lot.

This measures the ceiling of that idea honestly: take the TRUE spine, rasterise
it, dilate by a radius, and compare with the TRUE mask. Whatever IoU that
reaches is an upper bound on any spine-first pipeline, because a predicted spine
can only be worse than the annotated one.

Three radius policies, to separate "the ribbon model is right" from "the width
is predictable":
  global     one radius for the whole dataset      (what a fixed brush implies)
  per-class  one radius per category
  oracle     the best radius for each instance     (ceiling if width were known)

Work happens in a padded bbox crop, not at 2048^2, so this stays cheap.

Usage: python src/eda/spine_ribbon.py --sample 800
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

ANN = (ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" /
       "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
OUT = ROOT / "outputs" / "eda"
PAD = 40


def log(*a):
    print(*a, flush=True)


def crop_masks(ann, radii):
    """GT mask and spine-dilation masks for one annotation, in a bbox crop."""
    x, y, w, h = ann["bbox"]
    x0, y0 = int(x) - PAD, int(y) - PAD
    W, H = int(w) + 2 * PAD, int(h) + 2 * PAD
    if W <= 0 or H <= 0:
        return None

    gt = np.zeros((H, W), np.uint8)
    for poly in ann["segmentation"]:
        p = np.array(poly, np.float64).reshape(-1, 2)
        p[:, 0] -= x0
        p[:, 1] -= y0
        cv2.fillPoly(gt, [np.round(p).astype(np.int32)], 1)
    if gt.sum() == 0:
        return None

    sp = np.array(ann["spine"], np.float64).reshape(-1, 2)
    sp[:, 0] -= x0
    sp[:, 1] -= y0
    line = np.zeros((H, W), np.uint8)
    cv2.polylines(line, [np.round(sp).astype(np.int32)], False, 1, 1)
    if line.sum() == 0:
        return None

    dist = cv2.distanceTransform(1 - line, cv2.DIST_L2, 5)
    out = {}
    for r in radii:
        out[r] = (dist <= r).astype(np.uint8)
    return gt, out


def iou(a, b):
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=800)
    a = ap.parse_args()

    d = json.load(open(ANN))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    anns = d["annotations"]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(anns), size=min(a.sample, len(anns)), replace=False)

    radii = [round(r, 1) for r in np.arange(2.0, 12.5, 0.5)]
    rows = []
    for n, i in enumerate(idx):
        ann = anns[int(i)]
        got = crop_masks(ann, radii)
        if got is None:
            continue
        gt, dil = got
        ious = {r: iou(gt, m) for r, m in dil.items()}
        best_r = max(ious, key=ious.get)
        rows.append(dict(cat=cats[ann["category_id"]], area=ann["area"],
                         best_r=best_r, best_iou=ious[best_r],
                         **{f"iou_{r}": v for r, v in ious.items()}))
        if (n + 1) % 200 == 0:
            log(f"  {n+1}/{len(idx)}")
    df = pd.DataFrame(rows)
    log(f"\nmeasured {len(df)} annotations")

    # global radius: the one maximising mean IoU over everything
    means = {r: df[f"iou_{r}"].mean() for r in radii}
    g_r = max(means, key=means.get)
    log(f"\nglobal best radius {g_r} px -> mean IoU {means[g_r]:.4f}, "
        f"median {df[f'iou_{g_r}'].median():.4f}, "
        f"frac above 0.5 {float((df[f'iou_{g_r}']>0.5).mean()):.3f}")
    log("  radius sweep: " + "  ".join(f"{r}:{means[r]:.3f}" for r in radii[::2]))

    log("\nper class:")
    for c, g in df.groupby("cat"):
        cm = {r: g[f"iou_{r}"].mean() for r in radii}
        cr = max(cm, key=cm.get)
        log(f"  {c:16s} n={len(g):4d}  best r {cr:4.1f} -> mean IoU {cm[cr]:.4f}  "
            f"| oracle-per-instance mean IoU {g['best_iou'].mean():.4f}")

    log(f"\noracle per-instance radius: mean IoU {df['best_iou'].mean():.4f}, "
        f"median {df['best_iou'].median():.4f}, "
        f"frac above 0.5 {float((df['best_iou']>0.5).mean()):.3f}")
    log(f"best_r spread: p10 {df['best_r'].quantile(.1):.1f} "
        f"p50 {df['best_r'].median():.1f} p90 {df['best_r'].quantile(.9):.1f}")

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "spine_ribbon.parquet", index=False)
    with open(OUT / "spine_ribbon.json", "w") as f:
        json.dump(dict(n=len(df), global_radius=g_r, global_mean_iou=means[g_r],
                       oracle_mean_iou=float(df["best_iou"].mean()),
                       radius_sweep={str(k): float(v) for k, v in means.items()}),
                  f, indent=1)
    log(f"saved -> {OUT/'spine_ribbon.json'}")


if __name__ == "__main__":
    main()
