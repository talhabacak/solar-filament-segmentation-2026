"""Instance morphology EDA for MAGFiLO Kaggle 2026 train annotations.

Computes per-annotation morphology features (area, bbox, spine length, width
proxy, PCA orientation/eccentricity, centroid, part/point counts), estimates
the solar disk center/radius per image (downsample -> threshold -> min
enclosing circle), derives normalized radial positions, and produces
distribution / per-category / spatial / per-image analyses.

Outputs (all under outputs/eda/):
  - morphology_instances.parquet   one row per annotation (8199 rows)
  - morphology_stats.json          headline statistics
  - morph_distributions.png        area / spine length / width proxy / aspect
  - morph_per_category.png         per-category comparisons
  - morph_radial.png               radial histogram + count/mean-area vs r
  - morph_centroid_heatmap.png     2D density of centroids (disk coords)
  - morph_per_image_counts.png     per-entry counts + annotator disagreement

Re-runnable; paths are resolved relative to the repo root (two levels above
this file).
"""

import json
import math
import os
from collections import defaultdict
from itertools import combinations

import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.validation import make_valid
from tqdm import tqdm

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data", "MAGFiLO_1.0_Kaggle_2026")
ANN_JSON = os.path.join(DATA_DIR, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train", "train_images")
OUT_DIR = os.path.join(REPO_ROOT, "outputs", "eda")
os.makedirs(OUT_DIR, exist_ok=True)

CAT_NAMES = {1: "Left", 2: "Right", 3: "Unidentifiable", 4: "Ambiguous"}
# Okabe-Ito derived, CVD-validated categorical palette (fixed order by cat id)
CAT_COLORS = {1: "#0072B2", 2: "#E69F00", 3: "#009E73", 4: "#CC79A7"}

PCTS = [1, 5, 25, 50, 75, 95, 99]


def sanitize(obj):
    """Recursively replace NaN/inf floats with None so the JSON is strict-valid."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, (np.floating, np.integer)):
        v = obj.item()
        return sanitize(v) if isinstance(v, float) else v
    return obj


def pct_dict(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {}
    q = np.percentile(v, PCTS)
    return {f"p{p}": round(float(x), 4) for p, x in zip(PCTS, q)}


# ----------------------------------------------------------------------------
# Per-annotation morphology
# ----------------------------------------------------------------------------

def polyline_length(flat):
    pts = np.asarray(flat, dtype=float).reshape(-1, 2)
    if len(pts) < 2:
        return 0.0, len(pts)
    return float(np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(axis=1)).sum()), len(pts)


def polygon_features(seg):
    """Return (shapely_area, centroid_x, centroid_y, n_parts, n_points, pts_all)."""
    total_area = 0.0
    cx_acc, cy_acc, w_acc = 0.0, 0.0, 0.0
    n_parts = 0
    n_points = 0
    all_pts = []
    for part in seg:
        pts = np.asarray(part, dtype=float).reshape(-1, 2)
        n_points += len(pts)
        all_pts.append(pts)
        if len(pts) < 3:
            continue
        n_parts += 1
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = make_valid(poly)
        a = poly.area
        total_area += a
        if a > 0:
            c = poly.centroid
            cx_acc += c.x * a
            cy_acc += c.y * a
            w_acc += a
    all_pts = np.concatenate(all_pts, axis=0) if all_pts else np.zeros((0, 2))
    if w_acc > 0:
        cx, cy = cx_acc / w_acc, cy_acc / w_acc
    elif len(all_pts):
        cx, cy = float(all_pts[:, 0].mean()), float(all_pts[:, 1].mean())
    else:
        cx, cy = np.nan, np.nan
    return total_area, cx, cy, n_parts, n_points, all_pts


def pca_orientation(pts):
    """PCA of polygon vertex cloud -> (angle_deg in [-90, 90), eccentricity)."""
    if len(pts) < 3:
        return np.nan, np.nan
    p = pts - pts.mean(axis=0)
    cov = np.cov(p.T)
    if not np.all(np.isfinite(cov)):
        return np.nan, np.nan
    evals, evecs = np.linalg.eigh(cov)  # ascending
    lam2, lam1 = max(evals[0], 0.0), max(evals[1], 1e-12)
    v = evecs[:, 1]  # major axis
    ang = math.degrees(math.atan2(v[1], v[0]))
    if ang >= 90.0:
        ang -= 180.0
    elif ang < -90.0:
        ang += 180.0
    ecc = math.sqrt(max(0.0, 1.0 - lam2 / lam1))
    return ang, ecc


def build_instance_table(coco):
    img_by_id = {im["id"]: im for im in coco["images"]}
    rows = []
    for ann in tqdm(coco["annotations"], desc="annotations"):
        seg = ann.get("segmentation") or []
        poly_area, cx, cy, n_parts, n_pts, all_pts = polygon_features(seg)
        spine_len, spine_npts = polyline_length(ann.get("spine") or [])
        bx, by, bw, bh = ann["bbox"]
        long_side = max(bw, bh)
        short_side = max(min(bw, bh), 1e-9)
        ang, ecc = pca_orientation(all_pts)
        width_proxy = poly_area / spine_len if spine_len > 0 else np.nan
        im = img_by_id[ann["image_id"]]
        rows.append(
            dict(
                ann_id=ann["id"],
                image_id=ann["image_id"],
                file_name=im["file_name"],
                category_id=ann["category_id"],
                category=CAT_NAMES.get(ann["category_id"], str(ann["category_id"])),
                poly_area=poly_area,
                coco_area=float(ann.get("area", np.nan)),
                bbox_x=bx,
                bbox_y=by,
                bbox_w=bw,
                bbox_h=bh,
                bbox_area=bw * bh,
                aspect_ratio=long_side / short_side,
                spine_len=spine_len,
                spine_npts=spine_npts,
                width_proxy=width_proxy,
                orientation_deg=ang,
                eccentricity=ecc,
                centroid_x=cx,
                centroid_y=cy,
                n_parts=n_parts,
                n_poly_points=n_pts,
            )
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Disk detection (fast: downsample, threshold, min enclosing circle)
# ----------------------------------------------------------------------------

def detect_disk(path, ds=512):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.nan, np.nan, np.nan
    h, w = img.shape
    small = cv2.resize(img, (ds, ds), interpolation=cv2.INTER_AREA)
    _, mask = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.nan, np.nan, np.nan
    cnt = max(cnts, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(cnt)
    sx, sy = w / ds, h / ds
    return cx * sx, cy * sy, r * (sx + sy) / 2.0


def detect_all_disks(file_names):
    cache_path = os.path.join(OUT_DIR, "disk_geometry_cache.csv")
    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path)
        if set(file_names) <= set(cached["file_name"]):
            return cached
    recs = []
    for fn in tqdm(sorted(set(file_names)), desc="disk detection"):
        cx, cy, r = detect_disk(os.path.join(TRAIN_IMG_DIR, fn))
        recs.append(dict(file_name=fn, disk_cx=cx, disk_cy=cy, disk_r=r))
    disks = pd.DataFrame(recs)
    disks.to_csv(cache_path, index=False)
    return disks


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)
HIST_COLOR = "#0072B2"


def plot_distributions(df):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    specs = [
        ("poly_area", "Polygon area (px$^2$)", True),
        ("spine_len", "Spine length (px)", True),
        ("width_proxy", "Width proxy = area / spine length (px)", True),
        ("aspect_ratio", "Bbox aspect ratio (long/short)", True),
    ]
    for ax, (col, label, logx) in zip(axes.ravel(), specs):
        v = df[col].dropna()
        v = v[v > 0]
        bins = np.logspace(np.log10(v.min()), np.log10(v.max()), 60) if logx else 60
        ax.hist(v, bins=bins, color=HIST_COLOR, edgecolor="white", linewidth=0.3)
        if logx:
            ax.set_xscale("log")
        med = v.median()
        ax.axvline(med, color="#333333", lw=1, ls="--")
        ax.text(med, ax.get_ylim()[1] * 0.95, f" median {med:.1f}", fontsize=8, va="top")
        ax.set_xlabel(label)
        ax.set_ylabel("Instances")
    fig.suptitle("Instance morphology distributions (n=%d annotations)" % len(df))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "morph_distributions.png"), dpi=150)
    plt.close(fig)


def plot_per_category(df):
    cats = [1, 2, 3, 4]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, (col, label) in zip(
        axes,
        [
            ("poly_area", "Polygon area (px$^2$)"),
            ("spine_len", "Spine length (px)"),
            ("width_proxy", "Width proxy (px)"),
        ],
    ):
        data = [df.loc[(df.category_id == c) & (df[col] > 0), col].dropna() for c in cats]
        bp = ax.boxplot(
            data,
            tick_labels=[f"{CAT_NAMES[c]}\n(n={len(d)})" for c, d in zip(cats, data)],
            showfliers=False,
            patch_artist=True,
            medianprops=dict(color="#222222"),
        )
        for patch, c in zip(bp["boxes"], cats):
            patch.set_facecolor(CAT_COLORS[c])
            patch.set_alpha(0.75)
        ax.set_yscale("log")
        ax.set_ylabel(label)
    fig.suptitle("Morphology by chirality category")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "morph_per_category.png"), dpi=150)
    plt.close(fig)


def plot_radial(df, rbin_stats):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    r = df["r_norm"].dropna()
    axes[0].hist(r, bins=np.arange(0, 1.15, 0.025), color=HIST_COLOR, edgecolor="white", lw=0.3)
    axes[0].axvline(0.94, color="#333333", ls="--", lw=1)
    axes[0].text(0.94, axes[0].get_ylim()[1] * 0.95, " r=0.94 (~70 deg)", fontsize=8, va="top")
    axes[0].set_xlabel("Normalized radial position r")
    axes[0].set_ylabel("Instances")
    axes[0].set_title("Centroid radial distribution")

    centers = rbin_stats["r_center"]
    axes[1].bar(centers, rbin_stats["count"], width=0.09, color=HIST_COLOR, edgecolor="white")
    axes[1].set_xlabel("r bin center")
    axes[1].set_ylabel("Instance count")
    axes[1].set_title("Count vs r")

    axes[2].plot(centers, rbin_stats["mean_area"], marker="o", ms=5, color=HIST_COLOR, lw=2)
    axes[2].set_xlabel("r bin center")
    axes[2].set_ylabel("Mean polygon area (px$^2$)")
    axes[2].set_title("Mean area vs r")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "morph_radial.png"), dpi=150)
    plt.close(fig)


def plot_heatmap(df):
    d = df.dropna(subset=["dx_norm", "dy_norm"])
    fig, ax = plt.subplots(figsize=(7, 6.3))
    h = ax.hist2d(
        d["dx_norm"],
        d["dy_norm"],
        bins=np.linspace(-1.1, 1.1, 89),
        cmap="Blues",
        cmin=None,
    )
    fig.colorbar(h[3], ax=ax, label="Centroids per bin")
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color="#333333", lw=1, ls="--")
    ax.set_aspect("equal")
    ax.invert_yaxis()  # image coordinates: y grows downward
    ax.set_xlabel("(x - disk_cx) / R")
    ax.set_ylabel("(y - disk_cy) / R  (image coords, N up after invert)")
    ax.set_title("Filament centroid density in disk coordinates (n=%d)" % len(d))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "morph_centroid_heatmap.png"), dpi=150)
    plt.close(fig)


def plot_per_image(entry_counts, pair_diffs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(
        entry_counts,
        bins=np.arange(-0.5, entry_counts.max() + 1.5, 1),
        color=HIST_COLOR,
        edgecolor="white",
        lw=0.3,
    )
    axes[0].set_xlabel("Instances per image entry (annotator copy)")
    axes[0].set_ylabel("Image entries")
    axes[0].set_title("Per-entry instance count (n=%d entries)" % len(entry_counts))

    if len(pair_diffs):
        axes[1].hist(
            pair_diffs,
            bins=np.arange(-0.5, max(pair_diffs) + 1.5, 1),
            color="#E69F00",
            edgecolor="white",
            lw=0.3,
        )
    axes[1].set_xlabel("|n_i - n_j| between annotator copies")
    axes[1].set_ylabel("Copy pairs")
    axes[1].set_title("Annotator instance-count disagreement")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "morph_per_image_counts.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    with open(ANN_JSON, "r") as f:
        coco = json.load(f)

    df = build_instance_table(coco)

    # --- disk geometry -------------------------------------------------------
    disks = detect_all_disks(df["file_name"].unique())
    df = df.merge(disks, on="file_name", how="left")
    df["dx_norm"] = (df["centroid_x"] - df["disk_cx"]) / df["disk_r"]
    df["dy_norm"] = (df["centroid_y"] - df["disk_cy"]) / df["disk_r"]
    df["r_norm"] = np.hypot(df["dx_norm"], df["dy_norm"])

    # --- per-entry counts & annotator agreement ------------------------------
    img_entries = pd.DataFrame(
        [(im["id"], im["file_name"]) for im in coco["images"]],
        columns=["image_id", "file_name"],
    )
    ann_counts = df.groupby("image_id").size().rename("n_instances")
    img_entries = img_entries.merge(ann_counts, on="image_id", how="left")
    img_entries["n_instances"] = img_entries["n_instances"].fillna(0).astype(int)

    copies = img_entries.groupby("file_name")["n_instances"].agg(list)
    copies_per_file = copies.apply(len)
    pair_diffs_all = []
    per_file_disagree = []
    for fn, ns in copies.items():
        if len(ns) < 2:
            continue
        diffs = [abs(a - b) for a, b in combinations(ns, 2)]
        pair_diffs_all.extend(diffs)
        per_file_disagree.append(
            dict(
                file_name=fn,
                n_copies=len(ns),
                counts=sorted(ns),
                mean_abs_diff=float(np.mean(diffs)),
                max_abs_diff=int(max(diffs)),
            )
        )
    dis_df = pd.DataFrame(per_file_disagree).sort_values(
        ["max_abs_diff", "mean_abs_diff"], ascending=False
    )
    worst10 = dis_df.head(10).to_dict("records")
    for w in worst10:
        w["counts"] = [int(x) for x in w["counts"]]

    # --- radial bin stats ----------------------------------------------------
    edges = np.arange(0, 1.101, 0.1)
    rb = pd.cut(df["r_norm"], edges)
    grp = df.groupby(rb, observed=False)
    rbin_stats = pd.DataFrame(
        {
            "r_center": [(a.left + a.right) / 2 for a in grp.size().index],
            "count": grp.size().values,
            "mean_area": grp["poly_area"].mean().values,
        }
    )

    # --- plots ---------------------------------------------------------------
    plot_distributions(df)
    plot_per_category(df)
    plot_radial(df, rbin_stats)
    plot_heatmap(df)
    plot_per_image(img_entries["n_instances"], pair_diffs_all)

    # --- stats json ----------------------------------------------------------
    valid_r = df["r_norm"].dropna()
    ratio = (df["coco_area"] / df["poly_area"].replace(0, np.nan)).dropna()
    per_cat = {}
    for c in [1, 2, 3, 4]:
        sub = df[df.category_id == c]
        per_cat[CAT_NAMES[c]] = dict(
            n=int(len(sub)),
            frac=round(len(sub) / len(df), 4),
            median_area=round(float(sub["poly_area"].median()), 1),
            median_spine_len=round(float(sub["spine_len"].median()), 1),
            median_width_proxy=round(float(sub["width_proxy"].median()), 2),
            median_aspect=round(float(sub["aspect_ratio"].median()), 2),
            median_eccentricity=round(float(sub["eccentricity"].median()), 3),
            median_r_norm=round(float(sub["r_norm"].median()), 3),
            mean_n_parts=round(float(sub["n_parts"].mean()), 3),
        )

    stats = {
        "n_annotations": int(len(df)),
        "n_image_entries": int(len(img_entries)),
        "n_unique_files": int(img_entries["file_name"].nunique()),
        "copies_per_file_distribution": copies_per_file.value_counts().sort_index().to_dict(),
        "percentiles": {
            "poly_area": pct_dict(df["poly_area"]),
            "coco_area": pct_dict(df["coco_area"]),
            "bbox_area": pct_dict(df["bbox_area"]),
            "bbox_w": pct_dict(df["bbox_w"]),
            "bbox_h": pct_dict(df["bbox_h"]),
            "aspect_ratio": pct_dict(df["aspect_ratio"]),
            "spine_len": pct_dict(df["spine_len"]),
            "spine_npts": pct_dict(df["spine_npts"]),
            "width_proxy": pct_dict(df["width_proxy"]),
            "eccentricity": pct_dict(df["eccentricity"]),
            "n_poly_points": pct_dict(df["n_poly_points"]),
            "r_norm": pct_dict(valid_r),
        },
        "thin_instances": {
            "width_proxy_lt_4px": int((df["width_proxy"] < 4).sum()),
            "width_proxy_lt_8px": int((df["width_proxy"] < 8).sum()),
            "frac_lt_4px": round(float((df["width_proxy"] < 4).mean()), 4),
            "frac_lt_8px": round(float((df["width_proxy"] < 8).mean()), 4),
        },
        "coco_vs_shapely_area": {
            "median_ratio_coco_over_shapely": round(float(ratio.median()), 4),
            "p5_ratio": round(float(ratio.quantile(0.05)), 4),
            "p95_ratio": round(float(ratio.quantile(0.95)), 4),
            "corr_log": round(
                float(
                    np.corrcoef(
                        np.log10(df["coco_area"].clip(lower=1)),
                        np.log10(df["poly_area"].clip(lower=1)),
                    )[0, 1]
                ),
                4,
            ),
        },
        "multi_part": {
            "n_multi_part": int((df["n_parts"] > 1).sum()),
            "frac_multi_part": round(float((df["n_parts"] > 1).mean()), 4),
            "max_parts": int(df["n_parts"].max()),
        },
        "per_category": per_cat,
        "disk_geometry": {
            "radius_px": {
                "min": round(float(disks["disk_r"].min()), 1),
                "median": round(float(disks["disk_r"].median()), 1),
                "max": round(float(disks["disk_r"].max()), 1),
            },
            "n_failed": int(disks["disk_r"].isna().sum()),
        },
        "radial": {
            "max_r": round(float(valid_r.max()), 4),
            "p99_r": round(float(valid_r.quantile(0.99)), 4),
            "p999_r": round(float(valid_r.quantile(0.999)), 4),
            "n_r_gt_0.94": int((valid_r > 0.94).sum()),
            "n_r_gt_1.0": int((valid_r > 1.0).sum()),
            "bins": rbin_stats.round(2).to_dict("records"),
        },
        "per_entry_instance_count": {
            **pct_dict(img_entries["n_instances"]),
            "mean": round(float(img_entries["n_instances"].mean()), 3),
            "max": int(img_entries["n_instances"].max()),
            "n_zero": int((img_entries["n_instances"] == 0).sum()),
        },
        "annotator_disagreement": {
            "n_files_multi_copy": int((copies_per_file >= 2).sum()),
            "n_copy_pairs": len(pair_diffs_all),
            "mean_abs_count_diff": round(float(np.mean(pair_diffs_all)), 3) if pair_diffs_all else None,
            "median_abs_count_diff": round(float(np.median(pair_diffs_all)), 1) if pair_diffs_all else None,
            "frac_pairs_equal": round(float(np.mean(np.array(pair_diffs_all) == 0)), 4) if pair_diffs_all else None,
            "worst10": worst10,
        },
    }

    stats = sanitize(stats)
    with open(os.path.join(OUT_DIR, "morphology_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, default=str, allow_nan=False)

    df.to_parquet(os.path.join(OUT_DIR, "morphology_instances.parquet"), index=False)
    print(json.dumps(stats, indent=2, default=str, allow_nan=False))


if __name__ == "__main__":
    main()
