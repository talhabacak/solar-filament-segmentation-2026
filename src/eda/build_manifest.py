"""Build the canonical dataset manifest for the MAGFiLO Kaggle 2026 competition data.

Outputs (all under outputs/eda/):
  - manifest_entries.parquet   : one row per COCO image entry (1154 rows)
  - manifest_physical.parquet  : one row per physical image file, train+test, with split column
  - manifest_summary.md        : human-readable summary tables

Re-runnable; all paths are resolved relative to the repo root (parent of src/eda/).
"""

import json
import os
import re
import sys
from collections import Counter

import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA_ROOT = os.path.join(REPO_ROOT, "data", "MAGFiLO_1.0_Kaggle_2026")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train", "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test", "test_images")
COCO_JSON = os.path.join(DATA_ROOT, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
OUT_DIR = os.path.join(REPO_ROOT, "outputs", "eda")
os.makedirs(OUT_DIR, exist_ok=True)

SITE_NAMES = {
    "B": "Big Bear",
    "C": "Cerro Tololo",
    "L": "Learmonth",
    "M": "Mauna Loa",
    "T": "Teide",
    "U": "Udaipur",
}

CAT_NAMES = {1: "Left", 2: "Right", 3: "Unidentifiable", 4: "Ambiguous"}

STEM_RE = re.compile(r"^(\d{14})([A-Z])h$")


def parse_stem(stem):
    """Parse a physical stem like '20140609195854Bh' -> dict of timestamp parts + site."""
    m = STEM_RE.match(stem)
    if not m:
        return {
            "timestamp": None, "site": None, "site_name": None,
            "year": None, "month": None, "day": None, "hour": None,
            "parse_ok": False,
        }
    ts, site = m.group(1), m.group(2)
    return {
        "timestamp": pd.Timestamp(
            year=int(ts[0:4]), month=int(ts[4:6]), day=int(ts[6:8]),
            hour=int(ts[8:10]), minute=int(ts[10:12]), second=int(ts[12:14]),
        ),
        "site": site,
        "site_name": SITE_NAMES.get(site, "UNKNOWN"),
        "year": int(ts[0:4]),
        "month": int(ts[4:6]),
        "day": int(ts[6:8]),
        "hour": int(ts[8:10]),
        "parse_ok": True,
    }


def disk_and_intensity_stats(img_path, work_size=512):
    """Estimate solar disk center/radius (in original coords) and intensity stats.

    Loads grayscale, downsamples to work_size, thresholds, takes largest contour,
    fits min enclosing circle, scales back to original resolution.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"read_ok": False}
    h, w = img.shape
    small = cv2.resize(img, (work_size, work_size), interpolation=cv2.INTER_AREA)
    scale = w / float(work_size)  # images are square 2048x2048

    stats = {
        "read_ok": True,
        "img_h": h,
        "img_w": w,
        "int_mean": float(small.mean()),
        "int_std": float(small.std()),
        "int_p1": float(np.percentile(small, 1)),
        "int_p99": float(np.percentile(small, 99)),
    }

    _, mask = cv2.threshold(small, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        stats.update({"disk_ok": False, "cx": np.nan, "cy": np.nan, "R": np.nan})
        return stats
    largest = max(contours, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(largest)
    stats.update({
        "disk_ok": True,
        "cx": float(cx * scale),
        "cy": float(cy * scale),
        "R": float(r * scale),
        # fraction of the small image above threshold; sanity signal
        "frac_bright": float((mask > 0).mean()),
    })
    return stats


def main():
    # ------------------------------------------------------------ 1. entries
    with open(COCO_JSON, "r", encoding="utf-8") as f:
        coco = json.load(f)

    ann_by_image = {}
    for a in coco["annotations"]:
        ann_by_image.setdefault(a["image_id"], []).append(a)

    entry_rows = []
    for im in coco["images"]:
        entry_id = im["id"]
        file_name = im["file_name"]
        stem = os.path.splitext(file_name)[0]
        parsed = parse_stem(stem)
        anns = ann_by_image.get(entry_id, [])
        cat_counts = Counter(a["category_id"] for a in anns)
        annotator_prefix = entry_id.split("-")[0] if "-" in entry_id else None
        row = {
            "entry_id": entry_id,
            "annotator_prefix": annotator_prefix,
            "file_name": file_name,
            "stem": stem,
            "license": im.get("license"),
            "height": im.get("height"),
            "width": im.get("width"),
            "date_captured": im.get("date_captured"),
            "num_annotations": len(anns),
            **{f"n_cat_{cid}_{name}": cat_counts.get(cid, 0) for cid, name in CAT_NAMES.items()},
            **parsed,
        }
        entry_rows.append(row)

    df_entries = pd.DataFrame(entry_rows)

    # cross-check: date_captured should match parsed timestamp
    dc = pd.to_datetime(df_entries["date_captured"])
    ts_mismatch = int((dc != df_entries["timestamp"]).sum())

    # ------------------------------------------------------------ 2. duplicates & disk coverage
    dup_counts = df_entries.groupby("file_name").size()
    dup_dist = dup_counts.value_counts().sort_index()  # copies -> n physical images

    disk_train = set(os.listdir(TRAIN_IMG_DIR))
    json_files = set(df_entries["file_name"])
    missing_on_disk = sorted(json_files - disk_train)
    extra_on_disk = sorted(disk_train - json_files)

    # ------------------------------------------------------------ 3. test filenames
    test_files = sorted(os.listdir(TEST_IMG_DIR))
    test_rows = []
    for fn in test_files:
        stem = os.path.splitext(fn)[0]
        test_rows.append({"file_name": fn, "stem": stem, **parse_stem(stem)})
    df_test = pd.DataFrame(test_rows)

    # ------------------------------------------------------------ physical-level train frame
    agg = df_entries.groupby("file_name").agg(
        n_entries=("entry_id", "size"),
        total_annotations=("num_annotations", "sum"),
        mean_annotations=("num_annotations", "mean"),
        **{f"total_cat_{cid}_{name}": (f"n_cat_{cid}_{name}", "sum") for cid, name in CAT_NAMES.items()},
    ).reset_index()
    meta = df_entries.drop_duplicates("file_name")[
        ["file_name", "stem", "timestamp", "site", "site_name", "year", "month", "day", "hour"]
    ]
    df_phys_train = meta.merge(agg, on="file_name", how="left")
    df_phys_train["split"] = "train"

    df_phys_test = df_test.copy()
    df_phys_test["split"] = "test"
    for c in ["n_entries", "total_annotations", "mean_annotations"] + [
        f"total_cat_{cid}_{name}" for cid, name in CAT_NAMES.items()
    ]:
        df_phys_test[c] = np.nan
    df_phys_test = df_phys_test.drop(columns=["parse_ok"], errors="ignore")
    df_phys_train = df_phys_train.drop(columns=["parse_ok"], errors="ignore")

    df_phys = pd.concat([df_phys_train, df_phys_test], ignore_index=True, sort=False)

    # ------------------------------------------------------------ 5. disk geometry + intensity
    print(f"Processing {len(df_phys)} images for disk geometry / intensity ...", flush=True)
    stat_rows = []
    for i, row in df_phys.iterrows():
        img_dir = TRAIN_IMG_DIR if row["split"] == "train" else TEST_IMG_DIR
        path = os.path.join(img_dir, row["file_name"])
        s = disk_and_intensity_stats(path)
        s["file_name"] = row["file_name"]
        stat_rows.append(s)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(df_phys)}", flush=True)
    df_stats = pd.DataFrame(stat_rows)
    df_phys = df_phys.merge(df_stats, on="file_name", how="left")

    # derived geometry diagnostics
    cx0 = df_phys["img_w"] / 2.0
    cy0 = df_phys["img_h"] / 2.0
    df_phys["center_offset"] = np.sqrt((df_phys["cx"] - cx0) ** 2 + (df_phys["cy"] - cy0) ** 2)

    # ------------------------------------------------------------ save parquet
    entries_path = os.path.join(OUT_DIR, "manifest_entries.parquet")
    phys_path = os.path.join(OUT_DIR, "manifest_physical.parquet")
    df_entries.to_parquet(entries_path, index=False)
    df_phys.to_parquet(phys_path, index=False)

    # ------------------------------------------------------------ 4+6. summary markdown
    def md_table(df, index_name=None):
        return df.to_markdown()

    lines = []
    lines.append("# MAGFiLO 1.0 Kaggle 2026 — Dataset Manifest Summary\n")
    lines.append(f"Generated by `src/eda/build_manifest.py`.\n")

    lines.append("## Counts & integrity\n")
    lines.append(f"- COCO image entries: **{len(df_entries)}** (expected 1154)")
    lines.append(f"- COCO annotations: **{len(coco['annotations'])}** (expected 8199)")
    lines.append(f"- Unique physical train files in JSON: **{dup_counts.shape[0]}** (expected 707)")
    lines.append(f"- Train files on disk: **{len(disk_train)}**")
    lines.append(f"- Test files on disk: **{len(test_files)}**")
    lines.append(f"- JSON files missing on disk: **{len(missing_on_disk)}** {missing_on_disk if missing_on_disk else ''}")
    lines.append(f"- Disk files not in JSON: **{len(extra_on_disk)}** {extra_on_disk if extra_on_disk else ''}")
    lines.append(f"- Entries whose `date_captured` mismatches filename timestamp: **{ts_mismatch}**")
    n_badstem = int((~df_entries["parse_ok"]).sum()) + int((~df_test["parse_ok"]).sum())
    lines.append(f"- Stems failing the `YYYYMMDDHHMMSS + site + 'h'` pattern (train+test): **{n_badstem}**")
    zero_ann = df_entries[df_entries["num_annotations"] == 0]
    lines.append(f"- Image entries with zero annotations: **{len(zero_ann)}**"
                 + (f" -> {zero_ann['entry_id'].tolist()}" if 0 < len(zero_ann) <= 20 else ""))
    lines.append("")

    lines.append("## Duplicate (annotator copy) distribution\n")
    dd = dup_dist.rename_axis("copies_per_physical_image").rename("n_physical_images").to_frame()
    dd["n_entries"] = dd.index * dd["n_physical_images"]
    lines.append(md_table(dd))
    lines.append(f"\nTotal entries check: {int(dd['n_entries'].sum())} (should be {len(df_entries)})\n")

    lines.append("## Annotation category totals (entry level)\n")
    cat_tot = pd.Series(
        {name: int(df_entries[f"n_cat_{cid}_{name}"].sum()) for cid, name in CAT_NAMES.items()},
        name="total_annotations",
    ).to_frame()
    lines.append(md_table(cat_tot))
    lines.append("")

    lines.append("## Site x split (physical images)\n")
    site_split = pd.crosstab(df_phys["site_name"], df_phys["split"])
    site_split["train_pct"] = (site_split.get("train", 0) / site_split.get("train", 0).sum() * 100).round(1)
    site_split["test_pct"] = (site_split.get("test", 0) / site_split.get("test", 0).sum() * 100).round(1)
    lines.append(md_table(site_split))
    lines.append("")

    lines.append("## Year x split (physical images)\n")
    year_split = pd.crosstab(df_phys["year"], df_phys["split"])
    lines.append(md_table(year_split))
    lines.append("")

    lines.append("## Month x split (physical images)\n")
    month_split = pd.crosstab(df_phys["month"], df_phys["split"])
    lines.append(md_table(month_split))
    lines.append("")

    lines.append("## Hour-of-day x site (physical, train+test)\n")
    hour_site = pd.crosstab(df_phys["hour"], df_phys["site"])
    lines.append(md_table(hour_site))
    lines.append("")

    lines.append("## Intensity stats per site (from 512px downsample, train+test)\n")
    int_site = df_phys.groupby("site_name")[["int_mean", "int_std", "int_p1", "int_p99"]].agg(["mean", "std"]).round(2)
    int_site.columns = ["_".join(c) for c in int_site.columns]
    lines.append(md_table(int_site))
    lines.append("")

    lines.append("## Disk geometry (original 2048x2048 coords)\n")
    geo = df_phys[["cx", "cy", "R", "center_offset"]].describe().round(2)
    lines.append(md_table(geo))
    lines.append("")
    lines.append("### Disk geometry per split\n")
    geo_split = df_phys.groupby("split")[["cx", "cy", "R", "center_offset"]].agg(["mean", "std", "min", "max"]).round(2)
    geo_split.columns = ["_".join(c) for c in geo_split.columns]
    lines.append(md_table(geo_split))
    lines.append("")

    n_fail = int((~df_phys["disk_ok"].fillna(False)).sum()) if "disk_ok" in df_phys else 0
    lines.append(f"Images where disk fit failed: **{n_fail}**\n")

    lines.append("### Top 10 most off-center disks\n")
    worst_off = df_phys.nlargest(10, "center_offset")[
        ["file_name", "split", "site", "cx", "cy", "R", "center_offset"]
    ].round(1)
    lines.append(md_table(worst_off.set_index("file_name")))
    lines.append("")

    lines.append("### Top 10 radius outliers (|R - median|)\n")
    med_R = df_phys["R"].median()
    df_phys["_r_dev"] = (df_phys["R"] - med_R).abs()
    worst_R = df_phys.nlargest(10, "_r_dev")[["file_name", "split", "site", "R", "_r_dev"]].round(1)
    lines.append(md_table(worst_R.set_index("file_name")))
    lines.append("")

    lines.append("### Top 10 intensity outliers (|z| of int_mean within site)\n")
    grp = df_phys.groupby("site_name")["int_mean"]
    df_phys["_int_z"] = (df_phys["int_mean"] - grp.transform("mean")) / grp.transform("std")
    worst_int = df_phys.reindex(df_phys["_int_z"].abs().nlargest(10).index)[
        ["file_name", "split", "site", "int_mean", "int_std", "int_p1", "int_p99", "_int_z"]
    ].round(2)
    lines.append(md_table(worst_int.set_index("file_name")))
    lines.append("")

    summary_path = os.path.join(OUT_DIR, "manifest_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    df_phys = df_phys.drop(columns=["_r_dev", "_int_z"])
    df_phys.to_parquet(phys_path, index=False)  # re-save without helper cols

    # ------------------------------------------------------------ console recap
    print("=" * 70)
    print("ENTRIES:", len(df_entries), "| unique physical:", dup_counts.shape[0],
          "| on disk train:", len(disk_train), "| test:", len(test_files))
    print("Duplicate distribution (copies -> n physical):", dict(dup_dist))
    print("Missing on disk:", missing_on_disk, "| Extra on disk:", extra_on_disk)
    print("date_captured mismatches:", ts_mismatch, "| bad stems:", n_badstem,
          "| zero-annotation entries:", len(zero_ann))
    print("\nSite x split:\n", site_split)
    print("\nYear x split:\n", year_split)
    print("\nMonth x split:\n", month_split)
    print("\nDisk R: min %.1f  median %.1f  max %.1f" % (df_phys['R'].min(), med_R, df_phys['R'].max()))
    print("Center offset: median %.1f  p99 %.1f  max %.1f" % (
        df_phys['center_offset'].median(), df_phys['center_offset'].quantile(0.99), df_phys['center_offset'].max()))
    print("\nWorst off-center:\n", worst_off.to_string())
    print("\nWorst radius:\n", worst_R.to_string())
    print("\nWorst intensity:\n", worst_int.to_string())
    print("\nSaved:", entries_path)
    print("Saved:", phys_path)
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
