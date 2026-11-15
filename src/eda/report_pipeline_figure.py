"""Draw the data-pipeline graphic of the technical report.

The report template asks for one graphic of the entire pipeline. This draws it from a list of
stages, left to right: the two candidate lanes, fusion, featurisation, the selector, the keep
rule and the submission file, with the parameters that define each stage written in the box.

Output: report/overleaf/figures/pipeline.png (300 dpi)

Usage: python src/eda/report_pipeline_figure.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report" / "overleaf" / "figures"

LANE, STAGE, DATA = "#dbe9f6", "#eeeeee", "#fdf3d7"

# (x, y, width, height, title, parameter lines, colour), in axes units
BOXES = [
    (0.000, 0.29, 0.115, 0.42, "Input", ["GONG H-alpha", "2048 x 2048", "used as delivered"], DATA),
    (0.150, 0.56, 0.225, 0.42, "Detector lane",
     ["YOLO11m-seg, 1792 px", "5 fold models, pooled", "instance masks, conf >= 0.05"], LANE),
    (0.150, 0.02, 0.225, 0.42, "Semantic lane",
     ["U-Net (EfficientNet-B3), 1536 px", "12 members, flip averaging", "components of field >= 0.2"], LANE),
    (0.410, 0.29, 0.140, 0.42, "Fusion", ["cluster at IoU > 0.5", "weighted vote 0.5", "pixel-disjoint masks"], STAGE),
    (0.578, 0.29, 0.140, 0.42, "Features", ["boundary ring", "removed, 42 features", "per candidate"], STAGE),
    (0.746, 0.29, 0.150, 0.42, "Selector", ["p(match) x E[IoU | match]", "gradient boosting,", "mean of 5 bags"], STAGE),
    (0.924, 0.29, 0.116, 0.42, "Output", ["conf >= 0.40 and", "p_sel >= 0.60;", "RLE submission"], DATA),
]
ARROWS = [(0.115, 0.57, 0.150, 0.74), (0.115, 0.43, 0.150, 0.26), (0.375, 0.74, 0.410, 0.57),
          (0.375, 0.26, 0.410, 0.43), (0.550, 0.50, 0.578, 0.50), (0.718, 0.50, 0.746, 0.50),
          (0.896, 0.50, 0.924, 0.50)]


def box(ax, x, y, w, h, title, lines, color):
    """One stage: a rounded box with a bold title and smaller parameter lines."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                linewidth=0.8, edgecolor="#333333", facecolor=color))
    ax.text(x + w / 2, y + h - 0.04, title, ha="center", va="top", fontsize=7.2, fontweight="bold")
    ax.text(x + w / 2, y + h - 0.135, chr(10).join(lines), ha="center", va="top", fontsize=5.9, linespacing=1.25)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 1.55))
    ax.set_xlim(-0.005, 1.045)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for b in BOXES:
        box(ax, *b)
    for x0, y0, x1, y1 in ARROWS:
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                     linewidth=0.8, color="#333333", shrinkA=0, shrinkB=0))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    fig.savefig(OUT / "pipeline.png", dpi=300)
    print("wrote", OUT / "pipeline.png")


if __name__ == "__main__":
    main()
