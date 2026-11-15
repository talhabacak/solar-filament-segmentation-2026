"""Figure for report §5.4: barbs of annotated filaments and our masks.

Uses the per-copy table of e220_fine_structure.py (barb definition there: thinner than the body AND
farther from the annotated spine than half a body width). Picks matched copies (our IoU > 0.5) from
distinct images with the most barb pixels, three where our mask recovers the barbs best and three
where it recovers them least, and draws the image, the annotator's outline, our outline, the spine
and the barb pixels.

Usage: python src/eda/e220b_barb_figure.py
Output: docs/report/figures/fig4_fine_structure.png
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/eda"):
    sys.path.insert(0, str(ROOT / p))
from filament.rle import decode_rle  # noqa: E402
from e220_fine_structure import lateral, thin_protrusions  # noqa: E402

OUT = ROOT / "outputs" / "eda"
IMG = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
ANN = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
MARGIN = 25


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = pd.read_parquet(OUT / "e220_fine_structure.parquet")
    m = r[(r["ours_iou"] > 0.5) & (r["lateral_px"] >= 60)].sort_values("lateral_px", ascending=False)
    m = m.drop_duplicates("stem")
    best = m.sort_values("ours_lat", ascending=False).head(3)
    worst = m.sort_values("ours_lat").head(3)
    picked = pd.concat([best, worst])

    d = json.load(open(ANN, encoding="utf-8"))
    spine_of = {a["id"]: np.asarray(a.get("spine", []), np.float32).reshape(-1, 2) for a in d["annotations"]}
    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet")
    gtr = gt.set_index("filament_id")
    pool = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet", columns=["stem", "cand_id", "rle", "conf"])
    pool = pool.merge(pd.read_parquet(OUT / "bag5_oof_scores.parquet"), on="cand_id")
    keep = pool[(pool["conf"] >= 0.40) & (pool["p"] >= 0.60)]

    fig, axes = plt.subplots(2, 3, figsize=(10.4, 7.2))
    for ax, row in zip(axes.ravel(), picked.itertuples()):
        G = decode_rle(gtr.loc[row.fid, "segmentation_rle"]).astype(bool)
        sp = spine_of[gtr.loc[row.fid, "ann_id"]]
        (y0, y1, x0, x1), g, _, h, dist = lateral(G, sp)
        L = thin_protrusions(g) & (dist > h + 2)
        km = [decode_rle(x).astype(bool) for x in keep.loc[keep["stem"] == row.stem, "rle"]]
        best_m, best_iou = None, 0.0
        for k in km:
            i = (k & G).sum()
            if i:
                iou = i / (k | G).sum()
                if iou > best_iou:
                    best_m, best_iou = k, iou
        Y0, Y1 = max(0, y0 - MARGIN), min(G.shape[0], y1 + MARGIN)
        X0, X1 = max(0, x0 - MARGIN), min(G.shape[1], x1 + MARGIN)
        im = cv2.imread(str(IMG / f"{row.stem}.jpeg"), cv2.IMREAD_GRAYSCALE)[Y0:Y1, X0:X1]
        Lf = np.zeros(G.shape, bool)
        Lf[y0:y1, x0:x1] = L
        Lc, Gc, Pc = Lf[Y0:Y1, X0:X1], G[Y0:Y1, X0:X1], best_m[Y0:Y1, X0:X1]
        ax.imshow(im, cmap="gray")
        ov = np.zeros((*Lc.shape, 4))
        ov[Lc] = (0.10, 0.75, 0.95, 0.65)
        ax.imshow(ov)
        ax.contour(Gc, levels=[0.5], colors="#2ca02c", linewidths=1.1)
        ax.contour(Pc, levels=[0.5], colors="#d62728", linewidths=1.1, linestyles="--")
        ax.plot(sp[:, 0] - X0, sp[:, 1] - Y0, color="#f1c40f", lw=0.8)
        rec = (Lc & Pc).sum() / max(Lc.sum(), 1)
        ax.set_title(f"{row.stem}: {rec:.0%} of barb pixels recovered", fontsize=7.5)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Barbs (shaded) of annotated filaments with the annotator's outline (solid), our mask (dashed) "
                 "and the spine (thin line).\nTop row: best recovered; bottom row: least recovered.", fontsize=9)
    fig.tight_layout()
    out = ROOT / "docs" / "report" / "figures" / "fig4_fine_structure.png"
    fig.savefig(out, dpi=200)
    print("figure ->", out, "| picked:", list(picked["stem"]))


if __name__ == "__main__":
    main()
