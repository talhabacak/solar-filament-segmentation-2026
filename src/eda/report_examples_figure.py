"""Draw the "examples of the segmentations" graphic of the technical report.

Shows the first final submission on test images: one full-disc frame with every predicted
instance outlined, and enlarged regions from that frame and from two other frames. Only our own
predictions are drawn; no test annotation is read anywhere.

The frames are chosen by a fixed rule, not by eye: the full-disc frame is the test image with the
most predicted instances, and the other two are the next two images in that ranking that come
from different years. Each enlarged region is centred on the largest predicted instance of its
frame that lies within r/R < 0.8.

Input : outputs/submissions/sub036_t020pox_bag5.csv and the competition test images
        (data/MAGFiLO_1.0_Kaggle_2026/test/test_images, not part of the repository)
Output: report/overleaf/figures/examples.png (300 dpi)

Usage: python src/eda/report_examples_figure.py [--test-dir path/to/test_images]
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from filament.rle import decode_rle  # noqa: E402

TEST = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "test" / "test_images"
SUB = ROOT / "outputs" / "submissions" / "sub036_t020pox_bag5.csv"
OUT = ROOT / "report" / "overleaf" / "figures"
CX, CY, R = 1023.6, 1024.6, 899.95          # measured disc geometry (src/eda/disk_refit.py)
CROP = 420                                  # side of an enlarged region, px
COLORS = ["#ff3b30", "#34c759", "#0a84ff", "#ffcc00", "#ff2d92", "#30d5c8", "#ff9500", "#bf5af2"]


def masks_of(sub, stem):
    return [decode_rle(r) for r in sub.loc[sub["stem"] == stem, "segmentation_rle"]]


def outline(ax, masks, lw, x0=0, y0=0):
    """Draw each instance's outer contour in a cycling colour, shifted into crop coordinates."""
    for k, m in enumerate(masks):
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cs:
            c = c[:, 0, :].astype(float)
            c = np.vstack([c, c[:1]])
            ax.plot(c[:, 0] - x0, c[:, 1] - y0, color=COLORS[k % len(COLORS)], linewidth=lw)


def crop_origin(masks):
    """Top-left corner of the region centred on the largest on-disc instance."""
    best, area = None, 0
    for m in masks:
        ys, xs = np.nonzero(m)
        if len(xs) > area and np.hypot(xs.mean() - CX, ys.mean() - CY) / R < 0.8:
            best, area = (xs.mean(), ys.mean()), len(xs)
    x0 = int(np.clip(best[0] - CROP / 2, 0, 2048 - CROP))
    y0 = int(np.clip(best[1] - CROP / 2, 0, 2048 - CROP))
    return x0, y0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default=str(TEST), help="folder with the competition test images")
    test = Path(ap.parse_args().test_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    sub = pd.read_csv(SUB)
    sub["stem"] = sub["filament_id"].str.rsplit("_", n=1).str[0]
    counts = sub.groupby("stem").size().sort_values(ascending=False)
    stems = [counts.index[0]]
    for s in counts.index[1:]:
        if len(stems) == 3:
            break
        if s[:4] not in {t[:4] for t in stems}:
            stems.append(s)

    fig = plt.figure(figsize=(7.0, 1.86))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 1.0], wspace=0.03)
    for j, stem in enumerate([stems[0]] + stems):
        img = cv2.imread(str(test / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        masks = masks_of(sub, stem)
        ax = fig.add_subplot(gs[0, j])
        if j == 0:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            outline(ax, masks, 0.45)
            x0, y0 = crop_origin(masks)
            ax.add_patch(plt.Rectangle((x0, y0), CROP, CROP, fill=False, edgecolor="white", linewidth=0.7))
            ax.set_title(f"{stem}: {len(masks)} instances", fontsize=6.5, pad=2)
        else:
            x0, y0 = crop_origin(masks)
            ax.imshow(img[y0:y0 + CROP, x0:x0 + CROP], cmap="gray", vmin=0, vmax=255)
            outline(ax, masks, 0.9, x0, y0)
            ax.set_xlim(0, CROP)
            ax.set_ylim(CROP, 0)
            ax.set_title(f"{stem}, enlarged", fontsize=6.5, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.subplots_adjust(left=0.004, right=0.996, top=0.925, bottom=0.008)
    fig.savefig(OUT / "examples.png", dpi=300)
    print("wrote", OUT / "examples.png", "frames:", stems)


if __name__ == "__main__":
    main()
