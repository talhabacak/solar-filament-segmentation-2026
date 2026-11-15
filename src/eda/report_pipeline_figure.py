"""Draw the data-pipeline graphic of the technical report.

The report template asks for one graphic of the entire pipeline. This draws it from a list of
stages, left to right: the two candidate lanes, fusion, featurisation, the selector, the keep
rule and the submission file, with the parameters that define each stage written in the box.

Nothing is placed by hand. Every box is as wide and as high as its own text, measured with the
renderer, plus a fixed padding, and the font size is the largest at which the whole chain fits
the page width; a label can therefore never run out of its box.

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

WIDTH = 7.0                      # inches: the text width of the template
PAD_X, PAD_Y = 0.085, 0.07       # padding inside a box, inches
GAP, LANE_GAP = 0.20, 0.16       # between columns, and between the two stacked lanes
EDGE = "#2f3b4a"
LANE, STAGE, DATA = "#dce8f5", "#f1f1ef", "#fbefcf"
PSEL = r"$p_{\mathrm{sel}}$"

# Columns of the chain; the second column holds the two lanes stacked.
COLUMNS = [
    [("Input", ["GONG H-alpha", "full-disc frame", "2048 × 2048 px"], DATA)],
    [("Detector lane", ["YOLO11m-seg at 1792 px", "5 fold models, pooled", "masks with conf ≥ 0.05"], LANE),
     ("Semantic lane", ["U-Net (EfficientNet-B3)", "at 1536 px, 12 members", "components with p ≥ 0.2"], LANE)],
    [("Fusion", ["cluster at IoU > 0.5", "weighted vote 0.5", "disjoint masks"], STAGE)],
    [("Features", ["boundary ring", "removed;", "42 features"], STAGE)],
    [("Selector", ["p(match) ×", "E[IoU | match];", "boosted trees,", "mean of 5 bags"], STAGE)],
    [("Output", ["keep if conf ≥ 0.40", "and " + PSEL + " ≥ 0.60;", "RLE submission"], DATA)],
]


def measure(fig, ax, text, size, weight="normal"):
    """Width and height of a text block in inches at the given font size."""
    t = ax.text(0, 0, text, fontsize=size, fontweight=weight, linespacing=1.3)
    box = t.get_window_extent(fig.canvas.get_renderer())
    t.remove()
    return box.width / fig.dpi, box.height / fig.dpi


def layout(fig, ax, size):
    """Box sizes for one font size: a list of columns, each a list of (w, h, title_h)."""
    cols = []
    for column in COLUMNS:
        boxes = []
        for title, lines, _ in column:
            tw, th = measure(fig, ax, title, size + 1.2, "bold")
            bw, bh = measure(fig, ax, chr(10).join(lines), size)
            boxes.append((max(tw, bw) + 2 * PAD_X, th + bh + 3 * PAD_Y, th))
        width = max(b[0] for b in boxes)
        cols.append([(width, h, th) for _, h, th in boxes])
    return cols


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(WIDTH, 2.0), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    size = 8.0
    while True:                                     # the largest font at which the chain fits
        cols = layout(fig, ax, size)
        total = sum(c[0][0] for c in cols) + GAP * (len(cols) - 1)
        if total <= WIDTH - 0.04 or size <= 4.0:
            break
        size -= 0.1
    height = max(sum(b[1] for b in c) + LANE_GAP * (len(c) - 1) for c in cols) + 0.06
    fig.set_size_inches(WIDTH, height)
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, height)

    x = (WIDTH - total) / 2
    centres = []                                    # per column: list of (left, right, y_centre)
    for column, boxes in zip(COLUMNS, cols):
        stack = sum(b[1] for b in boxes) + LANE_GAP * (len(boxes) - 1)
        top = height / 2 + stack / 2
        here = []
        for (title, lines, color), (w, h, th) in zip(column, boxes):
            y = top - h
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                        linewidth=0.7, edgecolor=EDGE, facecolor=color))
            ax.text(x + w / 2, top - PAD_Y, title, ha="center", va="top", fontsize=size + 1.2,
                    fontweight="bold", color="#111111")
            ax.text(x + w / 2, top - 2 * PAD_Y - th, chr(10).join(lines), ha="center", va="top",
                    fontsize=size, linespacing=1.3, color="#111111")
            here.append((x, x + w, y + h / 2))
            top = y - LANE_GAP
        centres.append(here)
        x += boxes[0][0] + GAP

    for a, b in zip(centres[:-1], centres[1:]):     # every box of a column feeds every box of the next
        for _, right, ya in a:
            for left, _, yb in b:
                ax.add_patch(FancyArrowPatch((right + 0.015, ya), (left - 0.015, yb), arrowstyle="-|>",
                                             mutation_scale=6.5, linewidth=0.8, color=EDGE,
                                             shrinkA=0, shrinkB=0))
    fig.savefig(OUT / "pipeline.png", dpi=300)
    print(f"wrote {OUT / 'pipeline.png'}  font {size:.1f} pt, {WIDTH} x {height:.2f} in")


if __name__ == "__main__":
    main()
