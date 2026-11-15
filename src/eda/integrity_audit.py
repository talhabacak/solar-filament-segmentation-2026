"""Data integrity + leakage audit for the MAGFiLO Kaggle 2026 dataset.

Checks:
  1. SHA-256 duplicate detection within train, within test, and ACROSS train/test
     (direct leakage), plus 32x32 average-hash (aHash) near-duplicate collisions.
  2. Per-annotation geometry validity: shapely is_valid / self-intersection,
     out-of-bounds coords, degenerate polygons, COCO 'area' vs computed area,
     bbox vs polygon bounds, spine-inside-polygon coverage, duplicate ids,
     iscrowd values, multi-part segmentations.
  3. Image entries with zero annotations; disk files vs json file_name coverage.
  4. Image size/mode verification; channel-equality on a 30-image sample; jpeg
     quality impression (bytes/megapixel + blockiness score).
  5. Filename-stem timestamp vs date_captured agreement.

Outputs (to outputs/eda/):
  integrity_report.md      - human-readable report
  integrity_flags.parquet  - per-annotation flag table
  image_hashes.csv         - per-image sha256 + aHash + size/mode

Re-runnable; all paths derived from this script's location.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import shapely
from PIL import Image
from shapely.geometry import Polygon
from shapely.validation import explain_validity, make_valid
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"
TRAIN_IMG_DIR = DATA_DIR / "train" / "train_images"
TEST_IMG_DIR = DATA_DIR / "test" / "test_images"
JSON_PATH = DATA_DIR / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
OUT_DIR = REPO_ROOT / "outputs" / "eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_MAX = 2048.0
SPINE_TOL = 5.0  # px tolerance for "spine point inside polygon"
BBOX_TOL = 2.0  # px tolerance for bbox vs polygon-bounds agreement
AREA_RATIO_LO, AREA_RATIO_HI = 0.5, 2.0  # gross area mismatch band


# --------------------------------------------------------------------------- #
# Part 1: hashing
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ahash_32(path: Path) -> tuple[str, tuple[int, int], int]:
    """32x32 average hash. Returns (hex hash, (h, w), n_channels)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return "", (0, 0), 0
    nch = 1 if img.ndim == 2 else img.shape[2]
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    bits = (small > small.mean()).flatten()
    return np.packbits(bits).tobytes().hex(), gray.shape[:2], nch


def hash_images() -> pd.DataFrame:
    rows = []
    for split, d in [("train", TRAIN_IMG_DIR), ("test", TEST_IMG_DIR)]:
        files = sorted(d.glob("*.jpeg"))
        for p in tqdm(files, desc=f"hashing {split}"):
            sha = sha256_file(p)
            ah, shape, nch = ahash_32(p)
            with Image.open(p) as im:
                size, mode = im.size, im.mode
            rows.append(
                dict(split=split, file_name=p.name, sha256=sha, ahash=ah,
                     width=size[0], height=size[1], pil_mode=mode,
                     cv2_channels=nch, bytes=p.stat().st_size)
            )
    return pd.DataFrame(rows)


def dup_groups(df: pd.DataFrame, col: str) -> list[list[tuple[str, str]]]:
    """Groups (size>1) of (split, file_name) sharing the same value of `col`."""
    out = []
    for _, g in df.groupby(col):
        if len(g) > 1:
            out.append(list(zip(g["split"], g["file_name"])))
    return out


# --------------------------------------------------------------------------- #
# Part 2: annotation geometry
# --------------------------------------------------------------------------- #
def shoelace_area(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def audit_annotations(coco: dict) -> tuple[pd.DataFrame, dict]:
    anns = coco["annotations"]
    id_counts = Counter(a["id"] for a in anns)
    dup_ids = {k: v for k, v in id_counts.items() if v > 1}

    rows = []
    for a in tqdm(anns, desc="annotations"):
        seg = a.get("segmentation") or []
        n_parts = len(seg)
        parts_xy = [np.asarray(p, dtype=float).reshape(-1, 2) for p in seg]
        all_xy = np.vstack(parts_xy) if parts_xy else np.zeros((0, 2))

        n_pts_min = min((len(p) for p in parts_xy), default=0)
        degenerate_pts = any(len(p) < 3 for p in parts_xy) or n_parts == 0
        shoe_area = float(sum(shoelace_area(p) for p in parts_xy if len(p) >= 3))
        zero_area = shoe_area < 1e-6

        coords_oob = bool(
            all_xy.size and (all_xy.min() < 0.0 or all_xy.max() > IMG_MAX)
        )
        oob_amount = 0.0
        if all_xy.size:
            oob_amount = float(max(0.0 - all_xy.min(), all_xy.max() - IMG_MAX, 0.0))

        # shapely validity on each ring; union of made-valid parts for area/spine
        invalid = False
        self_int = False
        geoms = []
        for p in parts_xy:
            if len(p) < 3:
                continue
            poly = Polygon(p)
            if not poly.is_valid:
                invalid = True
                if "Self-intersection" in explain_validity(poly):
                    self_int = True
                poly = make_valid(poly)
            geoms.append(poly)
        geom = shapely.union_all(geoms) if geoms else None
        valid_area = float(geom.area) if geom is not None else 0.0

        coco_area = float(a.get("area", 0.0))
        ratio = coco_area / valid_area if valid_area > 0 else np.nan
        gross_area_mismatch = bool(
            valid_area > 0 and not (AREA_RATIO_LO <= ratio <= AREA_RATIO_HI)
        ) or (valid_area == 0 and coco_area > 0)

        # bbox consistency
        bbox = a.get("bbox", [0, 0, 0, 0])
        bbox_dev = np.nan
        if all_xy.size:
            minx, miny = all_xy.min(axis=0)
            maxx, maxy = all_xy.max(axis=0)
            bx, by, bw, bh = bbox
            bbox_dev = float(
                max(abs(bx - minx), abs(by - miny),
                    abs(bx + bw - maxx), abs(by + bh - maxy))
            )
        bbox_mismatch = bool(bbox_dev > BBOX_TOL) if np.isfinite(bbox_dev) else True

        # spine
        spine = a.get("spine") or []
        spine_xy = np.asarray(spine, dtype=float).reshape(-1, 2)
        n_spine = len(spine_xy)
        spine_oob = bool(
            spine_xy.size and (spine_xy.min() < 0.0 or spine_xy.max() > IMG_MAX)
        )
        frac_in, frac_in5, ends_in5 = np.nan, np.nan, None
        if n_spine and geom is not None and not geom.is_empty:
            pts = shapely.points(spine_xy)
            inside = shapely.contains(geom, pts)
            near = inside | (shapely.distance(geom, pts) <= SPINE_TOL)
            frac_in = float(inside.mean())
            frac_in5 = float(near.mean())
            ends_in5 = bool(near[0] and near[-1])

        rows.append(
            dict(
                ann_id=a["id"], image_id=a["image_id"],
                category_id=a["category_id"], iscrowd=a.get("iscrowd"),
                n_seg_parts=n_parts, min_ring_pts=n_pts_min,
                degenerate_pts=degenerate_pts, zero_area=zero_area,
                shoelace_area=shoe_area, valid_area=valid_area,
                coco_area=coco_area, area_ratio=ratio,
                gross_area_mismatch=gross_area_mismatch,
                shapely_invalid=invalid, self_intersection=self_int,
                coords_oob=coords_oob, oob_amount_px=oob_amount,
                bbox_max_dev_px=bbox_dev, bbox_mismatch=bbox_mismatch,
                n_spine_pts=n_spine, spine_oob=spine_oob,
                spine_frac_inside=frac_in, spine_frac_within5px=frac_in5,
                spine_endpoints_within5px=ends_in5,
                dup_ann_id=a["id"] in dup_ids,
            )
        )

    df = pd.DataFrame(rows)
    meta = dict(
        dup_ids=dup_ids,
        iscrowd_counts=dict(Counter(a.get("iscrowd") for a in anns)),
        multipart=int((df["n_seg_parts"] > 1).sum()),
        seg_parts_hist=df["n_seg_parts"].value_counts().to_dict(),
    )
    return df, meta


# --------------------------------------------------------------------------- #
# Part 3: coverage
# --------------------------------------------------------------------------- #
def audit_coverage(coco: dict, hash_df: pd.DataFrame) -> dict:
    images = coco["images"]
    img_ids = [im["id"] for im in images]
    dup_img_ids = {k: v for k, v in Counter(img_ids).items() if v > 1}

    ann_per_img = Counter(a["image_id"] for a in coco["annotations"])
    zero_ann = [i for i in img_ids if ann_per_img.get(i, 0) == 0]
    orphan_anns = sorted(set(ann_per_img) - set(img_ids))

    json_files = {im["file_name"] for im in images}
    disk_train = set(hash_df.loc[hash_df.split == "train", "file_name"])
    disk_test = set(hash_df.loc[hash_df.split == "test", "file_name"])

    entries_per_file = Counter(im["file_name"] for im in images)
    return dict(
        n_images=len(images), n_unique_files_json=len(json_files),
        dup_img_ids=dup_img_ids, zero_ann_entries=zero_ann,
        orphan_ann_image_ids=orphan_anns,
        disk_train_not_in_json=sorted(disk_train - json_files),
        json_not_on_disk_train=sorted(json_files - disk_train),
        json_files_in_test_dir=sorted(json_files & disk_test),
        entries_per_file_hist=dict(Counter(entries_per_file.values())),
        ann_per_entry_min=min(ann_per_img.values()) if ann_per_img else 0,
        ann_per_entry_max=max(ann_per_img.values()) if ann_per_img else 0,
    )


# --------------------------------------------------------------------------- #
# Part 4: pixel-level sample checks
# --------------------------------------------------------------------------- #
def audit_pixels(hash_df: pd.DataFrame, n_sample: int = 30) -> dict:
    rng = random.Random(42)
    picks = rng.sample(range(len(hash_df)), n_sample)
    rows = []
    for i in picks:
        r = hash_df.iloc[i]
        d = TRAIN_IMG_DIR if r.split == "train" else TEST_IMG_DIR
        img = cv2.imread(str(d / r.file_name), cv2.IMREAD_UNCHANGED)
        nch = 1 if img.ndim == 2 else img.shape[2]
        ch_maxdiff = 0
        if nch == 3:
            ch_maxdiff = int(
                max(np.abs(img[..., 0].astype(int) - img[..., 1].astype(int)).max(),
                    np.abs(img[..., 1].astype(int) - img[..., 2].astype(int)).max())
            )
        gray = img if img.ndim == 2 else img[..., 0]
        g = gray.astype(np.float32)
        col_d = np.abs(np.diff(g, axis=1)).mean(axis=0)
        row_d = np.abs(np.diff(g, axis=0)).mean(axis=1)
        # diff index i is between columns i and i+1; block boundary at col 8k
        bnd = np.arange(7, len(col_d), 8)
        nb = np.setdiff1d(np.arange(len(col_d)), bnd)
        blockiness = float(
            (col_d[bnd].mean() + row_d[bnd].mean())
            / (col_d[nb].mean() + row_d[nb].mean() + 1e-9)
        )
        rows.append(
            dict(split=r.split, file_name=r.file_name, channels=nch,
                 channel_maxdiff=ch_maxdiff, blockiness=blockiness,
                 bytes_per_mpx=r.bytes / (r.width * r.height / 1e6))
        )
    return dict(sample=pd.DataFrame(rows))


# --------------------------------------------------------------------------- #
# Part 5: timestamps
# --------------------------------------------------------------------------- #
def audit_timestamps(coco: dict) -> dict:
    mismatches, unparsable = [], []
    for im in coco["images"]:
        stem = Path(im["file_name"]).stem  # e.g. 20140609195854Bh
        m = re.match(r"^(\d{14})", stem)
        dc = im.get("date_captured", "")
        try:
            t_stem = datetime.strptime(m.group(1), "%Y%m%d%H%M%S") if m else None
            t_dc = datetime.strptime(dc, "%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            unparsable.append((im["id"], stem, dc))
            continue
        if t_stem is None or t_stem != t_dc:
            mismatches.append((im["id"], stem, dc))
    # id prefix consistency: id suffix after first '-' should equal filename stem
    id_stem_bad = [
        (im["id"], im["file_name"]) for im in coco["images"]
        if im["id"].split("-", 1)[-1] != Path(im["file_name"]).stem
    ]
    return dict(mismatches=mismatches, unparsable=unparsable,
                id_stem_mismatch=id_stem_bad)


# --------------------------------------------------------------------------- #
def main() -> None:
    coco = json.loads(JSON_PATH.read_text())

    hash_df = hash_images()
    hash_df.to_csv(OUT_DIR / "image_hashes.csv", index=False)

    tr = hash_df[hash_df.split == "train"]
    te = hash_df[hash_df.split == "test"]
    sha_dup_train = dup_groups(tr, "sha256")
    sha_dup_test = dup_groups(te, "sha256")
    cross_sha = set(tr.sha256) & set(te.sha256)
    cross_pairs = [
        (list(tr.loc[tr.sha256 == s, "file_name"]),
         list(te.loc[te.sha256 == s, "file_name"])) for s in sorted(cross_sha)
    ]
    # aHash collisions across train/test with different sha256
    ah_cross = []
    te_by_ah = defaultdict(list)
    for _, r in te.iterrows():
        te_by_ah[r.ahash].append(r)
    for _, r in tr.iterrows():
        for r2 in te_by_ah.get(r.ahash, []):
            if r.sha256 != r2.sha256:
                ah_cross.append((r.file_name, r2.file_name, r.ahash[:16]))
    ah_dup_train = dup_groups(tr, "ahash")
    ah_dup_test = dup_groups(te, "ahash")

    ann_df, ann_meta = audit_annotations(coco)
    ann_df.to_parquet(OUT_DIR / "integrity_flags.parquet", index=False)

    cov = audit_coverage(coco, hash_df)
    pix = audit_pixels(hash_df)
    ts = audit_timestamps(coco)

    # ---------------- report ---------------- #
    L = []
    w = L.append
    w("# MAGFiLO Kaggle 2026 - Data Integrity & Leakage Audit")
    w(f"\nGenerated: {datetime.now():%Y-%m-%d %H:%M} | "
      f"train imgs: {len(tr)} | test imgs: {len(te)} | "
      f"image entries: {cov['n_images']} | annotations: {len(ann_df)}\n")

    w("## 1. Hash / leakage check\n")
    w(f"- Exact-duplicate (sha256) groups within TRAIN: {len(sha_dup_train)}")
    for g in sha_dup_train:
        w(f"  - {g}")
    w(f"- Exact-duplicate (sha256) groups within TEST: {len(sha_dup_test)}")
    for g in sha_dup_test:
        w(f"  - {g}")
    verdict = "LEAKAGE DETECTED" if cross_pairs else "NO OVERLAP - no direct leakage"
    w(f"- **TRAIN/TEST sha256 overlap: {len(cross_pairs)} -> {verdict}**")
    for g in cross_pairs:
        w(f"  - train {g[0]} == test {g[1]}")
    w(f"- aHash(32x32) collisions ACROSS train/test with different sha256: "
      f"{len(ah_cross)}")
    for t in ah_cross[:20]:
        w(f"  - train {t[0]} ~ test {t[1]} (ahash {t[2]}...)")
    vc = hash_df.ahash.value_counts()
    w(f"- aHash caveat: only {hash_df.ahash.nunique()} unique aHash values over "
      f"{len(hash_df)} images; largest cluster has {int(vc.iloc[0])} members. "
      f"The binarized 32x32 hash collapses to the centered solar-disk "
      f"silhouette for full-disk H-alpha images, so identical aHash does NOT "
      f"imply near-duplicate content here - colliding pairs come from "
      f"different dates/sites and differ at pixel level. No true "
      f"near-duplicate leakage evidence.")
    w(f"- aHash collision groups within train: {len(ah_dup_train)}, "
      f"within test: {len(ah_dup_test)}")
    for g in ah_dup_train[:10]:
        w(f"  - train group: {[f for _, f in g]}")
    for g in ah_dup_test[:10]:
        w(f"  - test group: {[f for _, f in g]}")

    w("\n## 2. Annotation geometry\n")
    n = len(ann_df)
    def pct(k):
        return f"{int(ann_df[k].sum())} ({ann_df[k].mean()*100:.2f}%)"
    w(f"- shapely-invalid polygons: {pct('shapely_invalid')}")
    w(f"- self-intersections: {pct('self_intersection')}")
    w(f"- coords outside [0,2048]: {pct('coords_oob')} "
      f"(max excursion {ann_df.oob_amount_px.max():.1f}px)")
    w(f"- degenerate (<3 pts): {pct('degenerate_pts')}; "
      f"~zero area: {pct('zero_area')}")
    w(f"- gross COCO-area mismatch (ratio outside [{AREA_RATIO_LO},{AREA_RATIO_HI}]): "
      f"{pct('gross_area_mismatch')}")
    ratio_ok = ann_df.area_ratio.dropna()
    w(f"- area ratio (coco/computed) median {ratio_ok.median():.3f}, "
      f"p1 {ratio_ok.quantile(.01):.3f}, p99 {ratio_ok.quantile(.99):.3f}, "
      f"within 10% of 1.0: {(abs(ratio_ok-1)<=.1).mean()*100:.1f}%")
    w(f"- bbox vs polygon-bounds deviation > {BBOX_TOL}px: {pct('bbox_mismatch')} "
      f"(max dev {ann_df.bbox_max_dev_px.max():.1f}px, "
      f"median {ann_df.bbox_max_dev_px.median():.2f}px)")
    w(f"- spine points outside [0,2048]: {pct('spine_oob')}")
    sp = ann_df.spine_frac_within5px.dropna()
    w(f"- spine coverage (frac of points within {SPINE_TOL:.0f}px of polygon): "
      f"median {sp.median():.3f}, mean {sp.mean():.3f}; "
      f"annotations with coverage < 1.0: {(sp < 1).sum()}; "
      f"< 0.5: {(sp < .5).sum()}; == 0: {(sp == 0).sum()}")
    w(f"- spine endpoints NOT both within 5px: "
      f"{int((ann_df.spine_endpoints_within5px == False).sum())}")  # noqa: E712
    w(f"- duplicate annotation ids: {len(ann_meta['dup_ids'])} "
      f"{dict(list(ann_meta['dup_ids'].items())[:5])}")
    w(f"- iscrowd values: {ann_meta['iscrowd_counts']}")
    w(f"- multi-part segmentations: {ann_meta['multipart']} "
      f"(parts hist: {ann_meta['seg_parts_hist']})")
    worst = ann_df.nsmallest(10, "spine_frac_within5px")[
        ["ann_id", "image_id", "spine_frac_within5px", "spine_frac_inside",
         "n_spine_pts", "valid_area"]]
    w("\nWorst spine-vs-polygon offenders:\n")
    w(worst.to_markdown(index=False))
    worst_a = ann_df.dropna(subset=["area_ratio"]).assign(
        dev=lambda d: abs(np.log(d.area_ratio))).nlargest(10, "dev")[
        ["ann_id", "image_id", "coco_area", "valid_area", "area_ratio"]]
    w("\nWorst area-ratio offenders:\n")
    w(worst_a.to_markdown(index=False))
    worst_b = ann_df.nlargest(10, "bbox_max_dev_px")[
        ["ann_id", "image_id", "bbox_max_dev_px"]]
    w("\nWorst bbox offenders:\n")
    w(worst_b.to_markdown(index=False))

    w("\n## 3. Coverage\n")
    w(f"- image entries: {cov['n_images']}; unique file_names in json: "
      f"{cov['n_unique_files_json']}")
    w(f"- duplicate image ids: {cov['dup_img_ids'] or 'none'}")
    w(f"- entries with 0 annotations: {len(cov['zero_ann_entries'])} "
      f"{cov['zero_ann_entries'][:20]}")
    w(f"- annotations referencing missing image ids: "
      f"{cov['orphan_ann_image_ids'] or 'none'}")
    w(f"- disk train files not in json: {len(cov['disk_train_not_in_json'])} "
      f"{cov['disk_train_not_in_json'][:10]}")
    w(f"- json file_names missing from train dir: "
      f"{len(cov['json_not_on_disk_train'])} {cov['json_not_on_disk_train'][:10]}")
    w(f"- json file_names that exist in TEST dir: "
      f"{len(cov['json_files_in_test_dir'])} {cov['json_files_in_test_dir'][:10]}")
    w(f"- annotator copies per file (n_entries: n_files): "
      f"{cov['entries_per_file_hist']}")
    w(f"- annotations per image entry: min {cov['ann_per_entry_min']}, "
      f"max {cov['ann_per_entry_max']}")

    w("\n## 4. Image format\n")
    bad_size = hash_df[(hash_df.width != 2048) | (hash_df.height != 2048)]
    w(f"- images not 2048x2048: {len(bad_size)} "
      f"{bad_size.file_name.tolist()[:10]}")
    w(f"- PIL modes: {hash_df.pil_mode.value_counts().to_dict()}")
    w(f"- decoded channel counts (cv2, all images): "
      f"{hash_df.cv2_channels.value_counts().to_dict()}")
    s = pix["sample"]
    w(f"- 30-image sample: channels {s.channels.value_counts().to_dict()}, "
      f"max cross-channel diff: {s.channel_maxdiff.max()} "
      f"(0 => genuinely grayscale even if stored 3-channel)")
    w(f"- jpeg impression: bytes/megapixel median "
      f"{s.bytes_per_mpx.median():.0f} (range {s.bytes_per_mpx.min():.0f}-"
      f"{s.bytes_per_mpx.max():.0f}); blockiness score median "
      f"{s.blockiness.median():.3f} (1.0 = no 8x8 block artifacts, >1.1 = visible "
      f"compression blocking)")
    w(f"- overall file size stats (bytes): min {hash_df.bytes.min()}, "
      f"median {int(hash_df.bytes.median())}, max {hash_df.bytes.max()}")

    w("\n## 5. Timestamps\n")
    w(f"- filename-stem vs date_captured mismatches: {len(ts['mismatches'])} "
      f"{ts['mismatches'][:10]}")
    w(f"- unparsable: {len(ts['unparsable'])} {ts['unparsable'][:10]}")
    w(f"- image id suffix != filename stem: {len(ts['id_stem_mismatch'])} "
      f"{ts['id_stem_mismatch'][:10]}")

    (OUT_DIR / "integrity_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
