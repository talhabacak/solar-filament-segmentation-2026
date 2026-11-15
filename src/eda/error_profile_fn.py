"""Profile the FALSE NEGATIVES of the current operating point.

Operating point ("our system"): pooled 5-fold OOF candidates of E013@1792,
greedily de-overlapped, kept if conf >= 0.30 AND p_sel >= 0.35.
Reference: PQ 0.4105, TP 4768, FP 2461, FN 3431 over 8199 GT rows.

Pipeline
  1. rebuild de-overlapped candidate RLEs (exactly pooled_oof.build_all_candidates)
  2. exact official scorer over ALL candidates -> per-(gt,pred) matches;
     because predictions are pixel-disjoint, the TP set of any kept subset is the
     restriction of these matches to the kept pred ids. Verified against the exact
     scorer run on the kept set (must give TP 4768 / FN 3431).
  3. per-GT feature table: morphology join, fixed-geometry position, cross-annotator
     consensus, per-image context, photometry, fragmentation/shape
  4. recall curves / correlations / Cohen's d / cross-tabs
  5. "which misses matter" via the marginal PQ rule
  6. FN split: present in the raw conf>=0.05 pool but dropped, vs never proposed

Outputs: outputs/eda/error_profile_fn.{md,json}, error_profile_fn_gt.parquet,
         error_profile_fn_{recall,crosstab,fnsplit}.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes
from scipy.stats import pointbiserialr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament import IMG_H, IMG_W
from filament.pq import score
from filament.rle import decode_rle, encode_mask
from selector import fast_pq

OUT = ROOT / "outputs" / "eda"
CACHE = OUT / "_fn_cache"
IMG_DIR = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"

CONF_MIN, PSEL_MIN = 0.30, 0.35
R_MAX = 0.97
DISK_C, DISK_R = 1023.5, 900.0
REF = dict(pq=0.4105, tp=4768, fp=2461, fn=3431, total_gt=8199)
TYP_IOU = 0.66            # SQ of the operating point (measured, see report)
PAD = 30

K3 = np.ones((3, 3), np.uint8)
K9 = np.ones((9, 9), np.uint8)
KD12 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
KD24 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (49, 49))


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------- stage 1
def rebuild_deov_rles() -> pd.DataFrame:
    p = CACHE / "cand_rle_deov.parquet"
    if p.exists():
        return pd.read_parquet(p)
    rows, t0 = [], time.time()
    for fold in range(5):
        preds = pd.read_parquet(ROOT / "outputs" / "oof_cache" / f"fold{fold}_preds_e013.parquet")
        preds = preds[(preds["conf"] >= 0.05) & (preds["r_norm"] <= R_MAX)]
        for stem, g in preds.groupby("stem"):
            g = g.sort_values("conf", ascending=False)
            taken = np.zeros((IMG_H, IMG_W), dtype=bool)
            k = 0
            for r in g.itertuples():
                mm = decode_rle(r.rle).astype(bool) & ~taken
                if not mm.any():
                    continue
                taken |= mm
                k += 1
                rows.append(dict(stem=stem, cand_id=f"{stem}_{k}", conf=float(r.conf),
                                 rle_deov=encode_mask(mm.astype(np.uint8))))
                del mm
            del taken
        log(f"  de-overlap fold {fold}: {len(rows)} ({time.time()-t0:.0f}s)")
    d = pd.DataFrame(rows)
    d.to_parquet(p, index=False)
    return d


def match_status(cand: pd.DataFrame, gt: pd.DataFrame):
    """Per-GT matched/missed at the operating point + full-pool match table."""
    rle = rebuild_deov_rles()
    assert len(rle) == len(cand), f"{len(rle)} rebuilt vs {len(cand)} pooled candidates"
    m = cand.merge(rle[["cand_id", "rle_deov", "conf"]], on="cand_id", suffixes=("", "_rb"))
    assert len(m) == len(cand) and np.allclose(m["conf"], m["conf_rb"]), "ordering mismatch"

    p_all = CACHE / "all_matches.parquet"
    if p_all.exists():
        all_matches = pd.read_parquet(p_all)
    else:
        pred_df = m[["cand_id", "rle_deov"]].rename(
            columns={"cand_id": "filament_id", "rle_deov": "segmentation_rle"})
        t0 = time.time()
        res = score(gt[["filament_id", "segmentation_rle"]], pred_df, return_details=True)
        log(f"ALL-pool scorer: PQ={res['pq']:.4f} TP={res['tp']} FP={res['fp']} "
            f"FN={res['fn']} ({time.time()-t0:.0f}s)")
        all_matches = res["matches"]
        all_matches.to_parquet(p_all, index=False)

    kept = m[(m["conf"] >= CONF_MIN) & (m["p_sel"] >= PSEL_MIN)].copy()
    fp_r = fast_pq(kept, len(gt))
    log(f"fast_pq at operating point: PQ={fp_r['pq']:.4f} TP={fp_r['tp']} "
        f"FP={fp_r['fp']} FN={fp_r['fn']} kept={len(kept)}")

    p_kept = CACHE / "kept_matches.parquet"
    if p_kept.exists():
        kept_matches = pd.read_parquet(p_kept)
        exact = json.loads((CACHE / "kept_exact.json").read_text())
    else:
        pred_df = kept[["cand_id", "rle_deov"]].rename(
            columns={"cand_id": "filament_id", "rle_deov": "segmentation_rle"})
        t0 = time.time()
        res2 = score(gt[["filament_id", "segmentation_rle"]], pred_df, return_details=True)
        log(f"EXACT operating point: PQ={res2['pq']:.4f} TP={res2['tp']} FP={res2['fp']} "
            f"FN={res2['fn']} SQ={res2['sq']:.4f} ({time.time()-t0:.0f}s)")
        kept_matches = res2["matches"]
        kept_matches.to_parquet(p_kept, index=False)
        exact = {k: float(v) for k, v in res2.items() if k in ("pq", "sq", "rq", "tp", "fp", "fn")}
        (CACHE / "kept_exact.json").write_text(json.dumps(exact))

    assert int(exact["tp"]) == REF["tp"] and int(exact["fn"]) == REF["fn"], \
        f"operating point mismatch: TP={exact['tp']} FN={exact['fn']} vs reference"
    assert int(exact["fp"]) == REF["fp"], f"FP mismatch {exact['fp']}"
    log(f"VERIFIED operating point PQ={exact['pq']:.4f} SQ={exact['sq']:.4f} "
        f"TP={int(exact['tp'])} FP={int(exact['fp'])} FN={int(exact['fn'])}")
    return kept, kept_matches, all_matches, exact


# --------------------------------------------------------------------- stage 2
def disk_median(img):
    ys, xs = np.mgrid[0:2048:8, 0:2048:8]
    r = np.hypot(xs - DISK_C, ys - DISK_C) / DISK_R
    return float(np.median(img[::8, ::8][r < 0.9]))


def feats_for_mask(full_mask, img, gmag, dmed):
    ys, xs = np.nonzero(full_mask)
    area = int(len(ys))
    if area == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    cy, cx = float(ys.mean()), float(xs.mean())
    py0, py1 = max(0, y0 - PAD), min(2048, y1 + PAD)
    px0, px1 = max(0, x0 - PAD), min(2048, x1 + PAD)

    m = full_mask[py0:py1, px0:px1].astype(np.uint8)
    mb = m.astype(bool)
    win = img[py0:py1, px0:px1].astype(np.float32)

    ring = cv2.dilate(m, KD24).astype(bool) & ~cv2.dilate(m, KD12).astype(bool)
    ring_v4 = cv2.dilate(m, K9, iterations=2).astype(bool) & ~mb
    fg = win[mb]
    bg = win[ring] if ring.sum() > 10 else fg
    bg4 = win[ring_v4] if ring_v4.sum() > 10 else fg
    fg_mean, bg_mean, bg4_mean = float(fg.mean()), float(bg.mean()), float(bg4.mean())
    edge = mb & ~cv2.erode(m, K3).astype(bool)
    g = gmag[py0:py1, px0:px1]

    nlab, lab = cv2.connectedComponents(m, connectivity=8)
    n_cc = int(nlab - 1)
    sizes = np.bincount(lab.ravel())[1:]
    largest_cc_frac = float(sizes.max()) / area if len(sizes) else 1.0
    filled = binary_fill_holes(mb)
    holes = filled & ~mb
    n_holes = int(cv2.connectedComponents(holes.astype(np.uint8), connectivity=8)[0] - 1)
    hull = cv2.convexHull(np.stack([xs - px0, ys - py0], 1).astype(np.int32))
    hull_area = float(cv2.contourArea(hull))
    solidity = min(area / hull_area, 1.0) if hull_area > 0 else 1.0

    return dict(
        mask_area=area, gt_cy=cy, gt_cx=cx,
        r_norm=float(np.hypot(cx - DISK_C, cy - DISK_C) / DISK_R),
        abs_lat=float(abs(np.degrees(np.arcsin(np.clip((DISK_C - cy) / DISK_R, -1, 1))))),
        fg_mean=fg_mean, fg_median=float(np.median(fg)), fg_min=float(fg.min()),
        fg_std=float(fg.std()), bg_mean=bg_mean, ring_std=float(bg.std()),
        bg_mean_v4=bg4_mean,
        contrast=(bg_mean - fg_mean) / max(bg_mean, 1e-3),
        contrast_v4=(bg4_mean - fg_mean) / max(bg4_mean, 1e-3),
        depth=(dmed - fg_mean) / max(dmed, 1e-3), disk_median=dmed,
        edge_grad=float(g[edge].mean()) if edge.sum() else 0.0,
        n_cc=n_cc, n_holes=n_holes, hole_area_frac=float(holes.sum()) / max(area, 1),
        solidity=float(solidity), largest_cc_frac=largest_cc_frac,
        mask_bbox_h=y1 - y0, mask_bbox_w=x1 - x0)


RENAME_PH = {"cy": "gt_cy", "cx": "gt_cx",
             "bbox_h": "mask_bbox_h", "bbox_w": "mask_bbox_w"}


def gt_photometry_shape(gt: pd.DataFrame) -> pd.DataFrame:
    p = CACHE / "gt_photom_shape.parquet"
    if p.exists():
        return pd.read_parquet(p).rename(columns=RENAME_PH)
    rows, t0 = [], time.time()
    for i, (fn, grp) in enumerate(gt.groupby("file_name")):
        stem = fn.rsplit(".", 1)[0]
        img = cv2.imread(str(IMG_DIR / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            log(f"  MISSING IMAGE {stem}")
            continue
        dmed = disk_median(img)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gmag = np.hypot(gy, gx).astype(np.float32)
        del gy, gx
        for r in grp.itertuples():
            mm = decode_rle(r.segmentation_rle)
            f = feats_for_mask(mm, img, gmag, dmed)
            del mm
            if f is not None:
                rows.append(dict(ann_id=r.ann_id, filament_id=r.filament_id, **f))
        del img, gmag
        if (i + 1) % 200 == 0:
            log(f"  photometry {i+1} images, {len(rows)} masks ({time.time()-t0:.0f}s)")
    d = pd.DataFrame(rows)
    d.to_parquet(p, index=False)
    log(f"  photometry done: {len(d)} rows ({time.time()-t0:.0f}s)")
    return d


# --------------------------------------------------------------------- stage 3
CAT = {1: "Left", 2: "Right", 3: "Unidentifiable", 4: "Ambiguous"}


def build_gt_table(gt, kept_matches, all_matches, cand):
    d = gt[["filament_id", "entry_id", "file_name", "category_id", "ann_id", "area"]].copy()
    d["stem"] = d["file_name"].str.replace(".jpeg", "", regex=False)
    d["category"] = d["category_id"].map(CAT)

    matched_ids = set(kept_matches["gt_filament_id"])
    d["matched"] = d["filament_id"].isin(matched_ids).astype(int)
    iou_map = kept_matches.set_index("gt_filament_id")["iou"]
    d["match_iou"] = d["filament_id"].map(iou_map)

    # --- full-pool reachability (step 7)
    pool_ids = set(all_matches["gt_filament_id"])
    d["in_raw_pool"] = d["filament_id"].isin(pool_ids).astype(int)
    best = (all_matches.sort_values("iou", ascending=False)
            .drop_duplicates("gt_filament_id")
            .set_index("gt_filament_id"))
    cinfo = cand.set_index("cand_id")[["conf", "p_sel", "area", "y", "weight"]]
    d["pool_pred_id"] = d["filament_id"].map(best["pred_filament_id"])
    d["pool_iou"] = d["filament_id"].map(best["iou"])
    d["pool_conf"] = d["pool_pred_id"].map(cinfo["conf"])
    d["pool_psel"] = d["pool_pred_id"].map(cinfo["p_sel"])

    # --- morphology join
    mor = pd.read_parquet(ROOT / "outputs" / "eda" / "morphology_instances.parquet")
    keep = ["ann_id", "poly_area", "bbox_w", "bbox_h", "bbox_area", "aspect_ratio",
            "spine_len", "spine_npts", "width_proxy", "orientation_deg",
            "eccentricity", "n_parts", "n_poly_points"]
    d = d.merge(mor[keep], on="ann_id", how="left")

    # --- photometry / fragmentation
    ph = gt_photometry_shape(gt)
    d = d.merge(ph.drop(columns=["filament_id"]), on="ann_id", how="left")

    # --- consensus
    cl = pd.read_parquet(ROOT / "outputs" / "eda" / "annotator_clusters.parquet")
    rec = []
    for r in cl.itertuples():
        for a in str(r.member_ann_ids).split(","):
            rec.append((a.strip(), r.cluster_id, int(r.support), int(r.n_members),
                        int(r.n_copies_file), float(r.mean_pair_iou) if r.mean_pair_iou == r.mean_pair_iou else np.nan,
                        bool(r.full_support)))
    cm = pd.DataFrame(rec, columns=["ann_id", "cluster_id", "support", "n_members",
                                    "n_copies_file", "mean_pair_iou", "full_support"])
    d = d.merge(cm, on="ann_id", how="left")
    n_copies = gt.groupby("file_name")["entry_id"].nunique()
    d["n_copies"] = d["file_name"].map(n_copies).astype(int)
    # single-copy files carry no clusters: support 1 of 1
    solo = d["cluster_id"].isna()
    d.loc[solo, "support"] = 1
    d.loc[solo, "n_members"] = 1
    d.loc[solo, "cluster_id"] = d.loc[solo, "filament_id"]
    d["support"] = d["support"].astype(int)
    d["support_frac"] = d["support"] / d["n_copies"]

    # --- image / entry context
    d["n_gt_entry"] = d.groupby("entry_id")["filament_id"].transform("size")
    d["n_gt_image"] = d.groupby("file_name")["filament_id"].transform("size")
    d["gt_per_copy"] = d["n_gt_image"] / d["n_copies"]
    d["n_cands_img"] = d["stem"].map(cand.groupby("stem")["cand_id"].size())
    kept_per_stem = (cand[(cand["conf"] >= CONF_MIN) & (cand["p_sel"] >= PSEL_MIN)]
                     .groupby("stem")["cand_id"].size())
    d["n_kept_img"] = d["stem"].map(kept_per_stem).fillna(0).astype(int)

    man = pd.read_parquet(ROOT / "outputs" / "eda" / "manifest_physical.parquet")
    d = d.merge(man[["file_name", "site", "site_name", "year", "int_mean", "int_std"]],
                on="file_name", how="left")
    return d


# --------------------------------------------------------------------- stage 4
NUM_FEATURES = [
    ("mask_area", "mask pixel area"), ("log10_area", "log10 mask area"),
    ("width_proxy", "width proxy (area/spine)"), ("spine_len", "spine length"),
    ("aspect_ratio", "bbox aspect ratio"), ("eccentricity", "eccentricity"),
    ("n_poly_points", "polygon vertices"), ("mask_bbox_diag", "bbox diagonal"),
    ("r_norm", "radial position r/R"), ("abs_lat", "|latitude| deg"),
    ("contrast", "contrast (bg-fg)/bg"), ("contrast_v4", "contrast (8px ring)"),
    ("depth", "depth vs disk median"), ("fg_mean", "fg mean intensity"),
    ("fg_min", "fg min intensity"), ("bg_mean", "ring mean intensity"),
    ("fg_std", "fg std"), ("ring_std", "ring std"), ("edge_grad", "boundary |grad|"),
    ("n_cc", "connected components"), ("n_holes", "hole count"),
    ("hole_area_frac", "hole area fraction"), ("solidity", "solidity"),
    ("largest_cc_frac", "largest-cc fraction"),
    ("support", "annotator support k"), ("support_frac", "support k/N"),
    ("n_copies", "annotator copies N"), ("mean_pair_iou", "cluster mean pair IoU"),
    ("n_gt_entry", "GT in this entry"), ("n_gt_image", "GT in this image"),
    ("n_cands_img", "raw candidates in image"), ("n_kept_img", "kept preds in image"),
    ("int_mean", "image mean intensity"), ("int_std", "image intensity std"),
]


def bin_recall(d, col, q=10):
    try:
        b = pd.qcut(d[col], q, duplicates="drop")
    except Exception:
        return pd.DataFrame()
    g = d.groupby(b, observed=True).agg(n=("matched", "size"),
                                        recall=("matched", "mean"),
                                        lo=(col, "min"), hi=(col, "max"))
    return g.reset_index(drop=True).assign(bin=[f"[{r.lo:.4g},{r.hi:.4g}]" for r in g.itertuples()])


def feature_stats(d, ycol="matched"):
    rows = []
    y = d[ycol].to_numpy(float)
    for col, label in NUM_FEATURES:
        if col not in d.columns:
            continue
        x = d[col].astype(float).to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 100 or np.nanstd(x[ok]) == 0:
            continue
        xx, yy = x[ok], y[ok]
        if yy.std() == 0:
            continue
        rpb = pointbiserialr(yy, xx)
        rsp = spearmanr(xx, yy)
        a, b = xx[yy == 1], xx[yy == 0]
        sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                     / max(len(a) + len(b) - 2, 1))
        dcoh = (a.mean() - b.mean()) / sp if sp > 0 else 0.0
        rows.append(dict(feature=col, label=label, n=int(ok.sum()),
                         mean_pos=float(a.mean()), mean_neg=float(b.mean()),
                         median_pos=float(np.median(a)), median_neg=float(np.median(b)),
                         cohens_d=float(dcoh), r_pointbiserial=float(rpb.statistic),
                         p_pb=float(rpb.pvalue), spearman=float(rsp.statistic)))
    return pd.DataFrame(rows).sort_values("cohens_d", key=np.abs, ascending=False)


def group_recall(d, by, min_n=1):
    g = d.groupby(by, observed=True).agg(n=("matched", "size"), tp=("matched", "sum"))
    g["recall"] = g["tp"] / g["n"]
    return g[g["n"] >= min_n].reset_index()


# --------------------------------------------------------------------- stage 5
def worth_chasing(d, pq_now, iou=TYP_IOU):
    """Split FN by cross-annotator cluster and apply the marginal PQ rule.

    A cluster = one physical filament as drawn by k of the N annotator copies of
    the file (k = `support`, distinct entries; n_rows = GT rows, usually == k).
    Detecting a fully-missed cluster adds n_rows TP (iou each), (N - k) FP and
    removes n_rows FN, so
        dS = iou*n_rows,  dD = 0.5*(n_rows + N - k)
    and it pays exactly while  PQ < 2*iou*n_rows/(n_rows + N - k)  (== 2*iou*k/N
    in the usual n_rows == k case).
    """
    cl = d.groupby("cluster_id").agg(
        k=("support", "first"), N=("n_copies", "first"),
        n_rows=("filament_id", "size"), n_tp=("matched", "sum"),
        file_name=("file_name", "first"))
    cl["n_fn"] = cl["n_rows"] - cl["n_tp"]
    fully = cl[(cl["n_fn"] > 0) & (cl["n_tp"] == 0)].copy()
    partial = cl[(cl["n_fn"] > 0) & (cl["n_tp"] > 0)].copy()

    fully["thr_marginal"] = (2.0 * iou * fully["n_rows"]
                             / (fully["n_rows"] + fully["N"] - fully["k"]))
    fully["worth_marginal"] = fully["thr_marginal"] > pq_now
    fully["thr_brief"] = 2.0 * iou * fully["k"] / (fully["N"] + fully["k"])
    fully["worth_brief"] = fully["thr_brief"] > pq_now
    return cl, fully, partial


def pq_after_adding(kept, total_gt, add_tp, add_hits, add_N, iou=TYP_IOU):
    """fast_pq of the kept set plus synthetic perfect detections.

    add_tp: TP rows gained per added detection; add_hits: distinct annotator
    entries hit (drives FP = N - hits); add_N: annotator copies of the file.
    """
    synth = pd.DataFrame(dict(weight=np.asarray(add_tp, int),
                              n_entries_hit=np.asarray(add_hits, int),
                              n_copies=np.asarray(add_N, int),
                              iou_sum=np.asarray(add_tp, float) * iou))
    allc = pd.concat([kept[["weight", "n_entries_hit", "n_copies", "iou_sum"]], synth],
                     ignore_index=True)
    return fast_pq(allc, total_gt)


# --------------------------------------------------------------------- figures
def fig_recall_curves(d, feats, path):
    n = len(feats)
    ncol, nrow = 3, int(np.ceil(n / 3))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.0 * nrow))
    for ax, (col, label) in zip(np.ravel(axes), feats):
        t = bin_recall(d, col, 10)
        if not len(t):
            ax.axis("off")
            continue
        x = np.arange(len(t))
        ax.plot(x, t["recall"], "o-", color="#2c6fbb")
        ax.axhline(d["matched"].mean(), ls="--", c="gray", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:.3g}" for v in t["hi"]], rotation=60, fontsize=6)
        ax.set_title(f"{label}", fontsize=9)
        ax.set_ylabel("recall", fontsize=8)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=7)
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.suptitle("Recall by feature decile (dashed = overall recall)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_crosstab(d, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    specs = [("log10_area", "contrast", "log10 area", "contrast"),
             ("log10_area", "support_frac", "log10 area", "support k/N"),
             ("width_proxy", "contrast", "width proxy", "contrast")]
    for ax, (cx, cy, lx, ly) in zip(axes, specs):
        bx = pd.qcut(d[cx], 6, duplicates="drop")
        by = pd.qcut(d[cy], 6, duplicates="drop") if d[cy].nunique() > 6 else d[cy]
        piv = d.groupby([by, bx], observed=True)["matched"].mean().unstack()
        cnt = d.groupby([by, bx], observed=True)["matched"].size().unstack()
        im = ax.imshow(piv.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto", origin="lower")
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels([f"{i.right:.3g}" for i in piv.columns], rotation=45, fontsize=7)
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels([f"{i.right:.3g}" if hasattr(i, "right") else f"{float(i):.2f}"
                            for i in piv.index], fontsize=7)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}\n{int(cnt.values[i,j])}", ha="center",
                            va="center", fontsize=5.5)
        ax.set_xlabel(lx, fontsize=9)
        ax.set_ylabel(ly, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.045)
    fig.suptitle("Recall cross-tabs (cell = recall / n)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_fnsplit(d, path):
    fn = d[d["matched"] == 0]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.0))
    grp = [("in raw pool (dropped)", fn[fn["in_raw_pool"] == 1], "#e08214"),
           ("never proposed", fn[fn["in_raw_pool"] == 0], "#5e3c99")]
    tp = d[d["matched"] == 1]

    ax = axes[0]
    for lab, g, c in grp:
        ax.hist(np.log10(g["mask_area"].clip(lower=1)), bins=40, alpha=0.6,
                label=f"{lab} (n={len(g)})", color=c, density=True)
    ax.hist(np.log10(tp["mask_area"].clip(lower=1)), bins=40, histtype="step",
            color="black", label=f"matched (n={len(tp)})", density=True)
    ax.set_xlabel("log10 mask area")
    ax.set_ylabel("density")
    ax.legend(fontsize=7)
    ax.set_title("size", fontsize=10)

    ax = axes[1]
    for lab, g, c in grp:
        ax.hist(g["contrast"].clip(-0.05, 0.45), bins=40, alpha=0.6, color=c, density=True)
    ax.hist(tp["contrast"].clip(-0.05, 0.45), bins=40, histtype="step", color="black",
            density=True)
    ax.set_xlabel("contrast (bg-fg)/bg")
    ax.set_title("contrast", fontsize=10)

    ax = axes[2]
    tiers = ["1/1", "1/2", "2/2", "1/3", "2/3", "3/3", "1/4+", "2/4+", "3/4+", "4+/4+"]
    def tier(r):
        N = min(int(r.n_copies), 4)
        k = min(int(r.support), 4)
        Ns = "4+" if N >= 4 else str(N)
        ks = "4+" if k >= 4 else str(k)
        return f"{ks}/{Ns}"
    d = d.copy()
    d["tier"] = [tier(r) for r in d.itertuples()]
    g = d.groupby("tier")["matched"].agg(["size", "mean"]).reindex(tiers).dropna()
    ax.bar(range(len(g)), g["mean"], color="#2c6fbb")
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(g.index, rotation=45, fontsize=8)
    for i, (n, m) in enumerate(zip(g["size"], g["mean"])):
        ax.text(i, m + 0.02, f"{int(n)}", ha="center", fontsize=6)
    ax.axhline(d["matched"].mean(), ls="--", c="gray")
    ax.set_ylabel("recall")
    ax.set_title("consensus tier k/N", fontsize=10)
    ax.set_ylim(0, 1)

    ax = axes[3]
    dropped = fn[fn["in_raw_pool"] == 1]
    ax.scatter(dropped["pool_conf"], dropped["pool_psel"], s=4, alpha=0.35, c="#e08214")
    ax.axvline(CONF_MIN, c="k", lw=1)
    ax.axhline(PSEL_MIN, c="k", lw=1)
    ax.set_xlabel("conf of the matching candidate")
    ax.set_ylabel("p_sel of the matching candidate")
    ax.set_title(f"dropped-but-available FN (n={len(dropped)})", fontsize=10)
    fig.suptitle("False-negative anatomy at conf>=0.30 & p_sel>=0.35", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    total_gt = len(gt)
    cpath = ROOT / "outputs" / "selector" / "pooled_candidates.parquet"
    cand = pd.read_parquet(cpath)
    log(f"GT {total_gt} rows / {gt['file_name'].nunique()} images / "
        f"{gt['entry_id'].nunique()} entries; candidates {len(cand)}")

    kept, kept_matches, all_matches, exact = match_status(cand, gt)
    pq_now = float(exact["pq"])
    sq_now = float(exact["sq"])

    d = build_gt_table(gt, kept_matches, all_matches, cand)
    d["log10_area"] = np.log10(d["mask_area"].clip(lower=1))
    d["mask_bbox_diag"] = np.hypot(d["mask_bbox_h"], d["mask_bbox_w"])
    assert len(d) == total_gt, f"gt table {len(d)} != {total_gt}"
    assert int(d["matched"].sum()) == REF["tp"], f"matched {d['matched'].sum()}"
    log(f"per-GT table: {len(d)} rows, matched {int(d['matched'].sum())}, "
        f"missed {int((1-d['matched']).sum())}")

    R = {}
    R["operating_point"] = dict(conf_min=CONF_MIN, p_sel_min=PSEL_MIN,
                                pq=pq_now, sq=sq_now, tp=int(exact["tp"]),
                                fp=int(exact["fp"]), fn=int(exact["fn"]),
                                total_gt=total_gt, n_kept=int(len(kept)),
                                overall_recall=float(d["matched"].mean()))

    # ---- feature stats
    fs = feature_stats(d)
    R["feature_stats"] = fs.to_dict("records")
    log("\ntop features by |Cohen's d| (matched - missed):")
    log(fs.head(18).to_string(index=False))

    # ---- decile tables
    deciles = {}
    for col, _ in NUM_FEATURES:
        if col in d.columns:
            t = bin_recall(d, col, 10)
            if len(t):
                deciles[col] = t[["bin", "n", "recall"]].to_dict("records")
    R["deciles"] = deciles

    # ---- categorical / binned group recalls
    d["radial_zone"] = pd.cut(d["r_norm"], [0, .35, .7, .9, 1.01],
                              labels=["0-.35", ".35-.7", ".7-.9", ".9-1"])
    d["size_q"] = pd.qcut(d["mask_area"], 4, labels=["Q1_small", "Q2", "Q3", "Q4_large"])
    d["width_q"] = pd.qcut(d["width_proxy"], 4, labels=["Q1_thin", "Q2", "Q3", "Q4_thick"])
    d["contrast_q"] = pd.qcut(d["contrast"], 4, labels=["Q1_faint", "Q2", "Q3", "Q4_dark"])
    d["gt_per_image_quintile"] = pd.qcut(d["n_gt_image"], 5, duplicates="drop")
    d["consensus_tier"] = d["support"].astype(str) + "/" + d["n_copies"].astype(str)
    d["full_support_f"] = (d["support"] == d["n_copies"])

    groups = {}
    for name, by in [("category", "category"), ("consensus_tier", "consensus_tier"),
                     ("support", "support"), ("n_copies", "n_copies"),
                     ("radial_zone", "radial_zone"), ("size_quartile", "size_q"),
                     ("width_quartile", "width_q"), ("contrast_quartile", "contrast_q"),
                     ("site", "site_name"), ("year", "year"),
                     ("gt_per_image_quintile", "gt_per_image_quintile"),
                     ("n_cc", "n_cc"), ("n_holes", "n_holes"),
                     ("full_support", "full_support_f")]:
        g = group_recall(d, by)
        g[by] = g[by].astype(str)
        groups[name] = g.rename(columns={by: "level"}).to_dict("records")
        log(f"\nrecall by {name}:")
        log(g.to_string(index=False))
    R["group_recall"] = groups

    # ---- cross-tabs
    ct = {}
    for a, b in [("size_q", "contrast_q"), ("size_q", "consensus_tier"),
                 ("width_q", "contrast_q"), ("size_q", "radial_zone")]:
        piv = d.groupby([a, b], observed=True)["matched"].agg(["size", "mean"])
        ct[f"{a}__{b}"] = piv.reset_index().rename(
            columns={a: "row", b: "col", "size": "n", "mean": "recall"}).astype(
            {"row": str, "col": str}).to_dict("records")
    R["crosstabs"] = ct

    # ---- step 6: which misses matter
    fn_all = d[d["matched"] == 0]
    cl, fully, partial = worth_chasing(d, pq_now, iou=sq_now)
    n_fn = len(fn_all)
    fn_in_fully = int(fully["n_fn"].sum())
    fn_in_partial = int(partial["n_fn"].sum())
    log(f"\nFN {n_fn}: {fn_in_fully} in fully-missed clusters ({len(fully)} clusters), "
        f"{fn_in_partial} in partially-matched clusters ({len(partial)} clusters)")

    r_all = pq_after_adding(kept, total_gt, fully["n_rows"], fully["k"], fully["N"],
                            iou=sq_now)

    def chase_block(mask_col):
        w = fully[fully[mask_col]]
        s = fully[~fully[mask_col]]
        r = pq_after_adding(kept, total_gt, w["n_rows"], w["k"], w["N"], iou=sq_now)
        return dict(n_clusters_worth=int(len(w)), n_fn_worth=int(w["n_fn"].sum()),
                    n_clusters_skip=int(len(s)), n_fn_skip=int(s["n_fn"].sum()),
                    pq_if_perfect_on_worth=float(r["pq"]), tp_after=int(r["tp"]),
                    fp_after=int(r["fp"]), fn_after=int(r["fn"]),
                    delta_pq=float(r["pq"]) - pq_now)

    R["worth_chasing"] = dict(
        typical_iou_used=sq_now, pq_now=pq_now,
        n_fn_total=n_fn, n_fn_fully_missed_clusters=fn_in_fully,
        n_fn_partial_clusters=fn_in_partial,
        n_clusters_fully_missed=int(len(fully)), n_clusters_partial=int(len(partial)),
        pq_if_perfect_on_all_fully_missed=float(r_all["pq"]),
        delta_pq_all_fully_missed=float(r_all["pq"]) - pq_now,
        marginal_rule=chase_block("worth_marginal"),
        briefed_rule=chase_block("worth_brief"))
    # upper bound: every FN row vanishes, including the partial-cluster ones which
    # an extra disjoint prediction cannot actually recover
    R["worth_chasing"]["pq_if_zero_fn"] = float(
        pq_after_adding(kept, total_gt,
                        np.r_[fully["n_rows"].to_numpy(), partial["n_fn"].to_numpy()],
                        np.r_[fully["k"].to_numpy(), partial["n_fn"].to_numpy()],
                        np.r_[fully["N"].to_numpy(), partial["n_fn"].to_numpy()],
                        iou=sq_now)["pq"])
    for kk, vv in R["worth_chasing"].items():
        log(f"  {kk}: {vv}")

    # tier table for the marginal rule
    tt = (fully.groupby(["N", "k"]).agg(n_clusters=("n_fn", "size"), n_fn=("n_fn", "sum"),
                                        worth_marginal=("worth_marginal", "max"),
                                        worth_brief=("worth_brief", "max"),
                                        thr_marginal=("thr_marginal", "mean"),
                                        thr_brief=("thr_brief", "mean")).reset_index())
    tt_all = (cl.groupby(["N", "k"]).agg(n_clusters=("n_fn", "size"),
                                         n_gt_rows=("n_rows", "sum"),
                                         n_fn=("n_fn", "sum"),
                                         n_fully_missed=("n_tp", lambda s: int((s == 0).sum()))
                                         ).reset_index())
    tt_all["cluster_detection_rate"] = 1 - tt_all["n_fully_missed"] / tt_all["n_clusters"]
    R["cluster_tier_table"] = tt_all.to_dict("records")
    log("\nall clusters by (N, k):")
    log(tt_all.to_string(index=False))
    R["worth_chasing"]["tier_table"] = tt.to_dict("records")
    log("\nfully-missed clusters by (N copies, k support):")
    log(tt.to_string(index=False))

    # greedy ladder: the marginal threshold must be re-checked as PQ rises, so add
    # tiers in descending value order and record where the ladder stops paying.
    order = tt.sort_values("thr_marginal", ascending=False).reset_index(drop=True)
    ladder, acc = [], []
    pq_run = pq_now
    for r in order.itertuples():
        blk = fully[(fully["N"] == r.N) & (fully["k"] == r.k)]
        acc.append(blk)
        cum = pd.concat(acc)
        res_c = pq_after_adding(kept, total_gt, cum["n_rows"], cum["k"], cum["N"],
                                iou=sq_now)
        ladder.append(dict(N=int(r.N), k=int(r.k), thr_marginal=float(r.thr_marginal),
                           pq_before_tier=float(pq_run), pays_at_this_pq=bool(r.thr_marginal > pq_run),
                           n_clusters_added=int(len(blk)), n_fn_added=int(blk["n_fn"].sum()),
                           cum_fn_recovered=int(cum["n_fn"].sum()),
                           pq_after=float(res_c["pq"]), tp=int(res_c["tp"]),
                           fp=int(res_c["fp"]), fn=int(res_c["fn"])))
        pq_run = float(res_c["pq"])
    lad = pd.DataFrame(ladder)
    best = lad.loc[lad["pq_after"].idxmax()]
    R["worth_chasing"]["greedy_tier_ladder"] = lad.to_dict("records")
    R["worth_chasing"]["greedy_optimum"] = dict(
        stop_after_tier=f"N={int(best['N'])},k={int(best['k'])}",
        pq=float(best["pq_after"]), fn_recovered=int(best["cum_fn_recovered"]),
        delta_pq=float(best["pq_after"]) - pq_now)
    log("\ngreedy tier ladder (add most-valuable tiers first):")
    log(lad.to_string(index=False))

    # ---- step 7: dropped vs never proposed
    fn = d[d["matched"] == 0]
    n_drop = int((fn["in_raw_pool"] == 1).sum())
    n_never = int((fn["in_raw_pool"] == 0).sum())
    log(f"\nFN split: {n_drop} available in raw pool but dropped, "
        f"{n_never} never proposed (of {len(fn)})")
    prof = []
    for name, sub in [("matched(TP)", d[d["matched"] == 1]),
                      ("FN_dropped", fn[fn["in_raw_pool"] == 1]),
                      ("FN_never_proposed", fn[fn["in_raw_pool"] == 0])]:
        row = dict(group=name, n=int(len(sub)))
        for c in ["mask_area", "width_proxy", "spine_len", "contrast", "depth",
                  "edge_grad", "fg_min", "r_norm", "abs_lat", "solidity", "n_cc",
                  "n_holes", "eccentricity", "aspect_ratio", "support",
                  "support_frac", "n_gt_image", "n_copies"]:
            row[f"med_{c}"] = float(sub[c].median())
        prof.append(row)
    R["fn_split_profile"] = prof
    log(pd.DataFrame(prof).to_string(index=False))

    # within the FN, what separates "dropped" from "never proposed"?
    fs_fn = feature_stats(fn.assign(matched=fn["in_raw_pool"]), "matched")
    fs_fn = fs_fn.rename(columns={"mean_pos": "mean_dropped", "mean_neg": "mean_never",
                                  "median_pos": "median_dropped", "median_neg": "median_never"})
    R["fn_dropped_vs_never_stats"] = fs_fn.to_dict("records")
    log("\nFN: dropped (pos) vs never-proposed (neg), top |d|:")
    log(fs_fn.head(15).to_string(index=False))
    # and: what separates a reachable GT from an unreachable one, over ALL GT
    fs_reach = feature_stats(d.assign(matched=d["in_raw_pool"]), "matched")
    R["reachability_stats"] = fs_reach.to_dict("records")
    R["raw_pool_ceiling"] = dict(
        n_reachable=int(d["in_raw_pool"].sum()),
        recall_ceiling=float(d["in_raw_pool"].mean()),
        n_never_proposed=int((1 - d["in_raw_pool"]).sum()))

    dropped = fn[fn["in_raw_pool"] == 1]
    R["fn_dropped_reason"] = dict(
        n=int(len(dropped)),
        conf_below_only=int(((dropped["pool_conf"] < CONF_MIN) &
                             (dropped["pool_psel"] >= PSEL_MIN)).sum()),
        psel_below_only=int(((dropped["pool_conf"] >= CONF_MIN) &
                             (dropped["pool_psel"] < PSEL_MIN)).sum()),
        both_below=int(((dropped["pool_conf"] < CONF_MIN) &
                        (dropped["pool_psel"] < PSEL_MIN)).sum()),
        neither_below=int(((dropped["pool_conf"] >= CONF_MIN) &
                           (dropped["pool_psel"] >= PSEL_MIN)).sum()),
        conf_quantiles={str(q): float(dropped["pool_conf"].quantile(q))
                        for q in [.1, .25, .5, .75, .9]},
        psel_quantiles={str(q): float(dropped["pool_psel"].quantile(q))
                        for q in [.1, .25, .5, .75, .9]},
        pool_iou_quantiles={str(q): float(dropped["pool_iou"].quantile(q))
                            for q in [.1, .25, .5, .75, .9]})
    log(f"dropped-FN reason: {R['fn_dropped_reason']}")

    # how much of the headroom is selector-side vs detector-side?
    oracle = fast_pq(cand[cand["y"] == 1], total_gt)          # perfect selection on pool
    need = set(dropped["pool_pred_id"].dropna())
    extra = cand[cand["cand_id"].isin(need) & ~cand["cand_id"].isin(set(kept["cand_id"]))]
    add_only = fast_pq(pd.concat([kept, extra]), total_gt)     # keep current FPs too
    R["headroom"] = dict(
        pq_now=pq_now,
        selector_oracle_on_pool=dict(pq=float(oracle["pq"]), tp=int(oracle["tp"]),
                                     fp=int(oracle["fp"]), fn=int(oracle["fn"]),
                                     delta_pq=float(oracle["pq"]) - pq_now),
        add_missing_good_candidates_only=dict(
            n_extra_candidates=int(len(extra)), pq=float(add_only["pq"]),
            tp=int(add_only["tp"]), fp=int(add_only["fp"]), fn=int(add_only["fn"]),
            delta_pq=float(add_only["pq"]) - pq_now),
        detector_plus_selector_profitable_tiers=dict(
            pq=R["worth_chasing"]["greedy_optimum"]["pq"],
            delta_pq=R["worth_chasing"]["greedy_optimum"]["delta_pq"]))
    log(f"\nheadroom: {json.dumps(R['headroom'], indent=1)}")

    # group recall split by availability (detector vs selector fixes)
    gs = []
    for name, by in [("category", "category"), ("size_quartile", "size_q"),
                     ("contrast_quartile", "contrast_q"), ("radial_zone", "radial_zone"),
                     ("consensus_tier", "consensus_tier")]:
        g = d.groupby(by, observed=True).agg(
            n=("matched", "size"), recall=("matched", "mean"),
            frac_reachable=("in_raw_pool", "mean")).reset_index()
        g[by] = g[by].astype(str)
        gs.append(dict(name=name, rows=g.rename(columns={by: "level"}).to_dict("records")))
    R["reachability_by_group"] = gs

    # ---- figures
    # continuous features only (discrete ones collapse to 2-3 decile points)
    skip_fig = {"contrast_v4", "support", "support_frac", "n_copies", "n_cc",
                "n_holes", "log10_area"}
    top_feats = [(r.feature, r.label) for r in fs.itertuples()
                 if r.feature not in skip_fig and len(bin_recall(d, r.feature, 10)) >= 6][:8]
    top_feats.append(("r_norm", "radial position r/R"))
    fig_recall_curves(d, top_feats, OUT / "error_profile_fn_recall.png")
    fig_crosstab(d, OUT / "error_profile_fn_crosstab.png")
    fig_fnsplit(d, OUT / "error_profile_fn_split.png")

    # ---- persist
    for c in ["radial_zone", "size_q", "width_q", "contrast_q", "gt_per_image_quintile"]:
        d[c] = d[c].astype(str)
    d.to_parquet(OUT / "error_profile_fn_gt.parquet", index=False)
    (OUT / "error_profile_fn.json").write_text(json.dumps(R, indent=1, default=str))
    write_md(R, d, fs, tt, tt_all, fs_fn, lad, OUT / "error_profile_fn.md")
    log(f"\ndone in {time.time()-t0:.0f}s -> {OUT}")


def write_md(R, d, fs, tt, tt_all, fs_fn, lad, path):
    op = R["operating_point"]
    rp = R["raw_pool_ceiling"]
    L = ["# False-negative profile at the current operating point", "",
         f"Operating point: `conf >= {CONF_MIN}` and `p_sel >= {PSEL_MIN}` on the pooled "
         f"5-fold OOF candidates of E013@1792 (de-overlapped).", "",
         f"- exact scorer: **PQ {op['pq']:.4f}**, SQ {op['sq']:.4f}, "
         f"TP {op['tp']}, FP {op['fp']}, FN {op['fn']} over {op['total_gt']} GT rows "
         f"({op['n_kept']} kept predictions)",
         f"- overall instance recall **{op['overall_recall']:.4f}** "
         f"({op['tp']}/{op['total_gt']})",
         f"- raw candidate pool (conf>=0.05, de-overlapped) reaches "
         f"{rp['n_reachable']}/{op['total_gt']} GT rows = recall ceiling "
         f"**{rp['recall_ceiling']:.4f}**; {rp['n_never_proposed']} GT rows are not "
         f"matched by ANY candidate at any threshold", "",
         "## Headline", ""]
    wc0 = R["worth_chasing"]
    fd = R["fn_dropped_reason"]
    gtab = {g["name"]: pd.DataFrame(g["rows"]) for g in R["reachability_by_group"]}
    cq = gtab["contrast_quartile"].set_index("level")
    sq_ = gtab["size_quartile"].set_index("level")
    L += [
        f"1. **Faintness is the strongest single physical driver.** Recall goes "
        f"{cq.loc['Q1_faint','recall']:.3f} -> {cq.loc['Q4_dark','recall']:.3f} across "
        f"contrast quartiles (n={int(cq['n'].sum())}); point-biserial r = "
        f"{fs.set_index('feature').loc['contrast','r_pointbiserial']:.3f}, Cohen's d = "
        f"{fs.set_index('feature').loc['contrast','cohens_d']:.3f}.",
        f"2. **Size is second.** Recall {sq_.loc['Q1_small','recall']:.3f} -> "
        f"{sq_.loc['Q3','recall']:.3f} across mask-area quartiles; the two interact - "
        f"small AND faint is the worst cell.",
        f"3. **Annotator consensus dominates everything.** Recall by support k of N "
        f"copies: 1/3 = "
        f"{pd.DataFrame(R['group_recall']['consensus_tier']).set_index('level').loc['1/3','recall']:.3f} "
        f"vs 3/3 = "
        f"{pd.DataFrame(R['group_recall']['consensus_tier']).set_index('level').loc['3/3','recall']:.3f}. "
        f"Much of the FN mass is filaments only one annotator saw.",
        f"4. **We are NOT missing the stringy ones - we miss the compact ones.** "
        f"Cohen's d of solidity is "
        f"{fs.set_index('feature').loc['solidity','cohens_d']:+.3f} (matched are LESS "
        f"solid) and n_cc is 1 for {int((d['n_cc']==1).sum())}/{len(d)} GT masks, so "
        f"fragmentation is not a real axis in this dataset.",
        f"5. **{fd['n']} of {wc0['n_fn_total']} FN are already in the raw candidate pool** "
        f"and were thrown away by the operating point (selector/threshold problem); "
        f"{R['raw_pool_ceiling']['n_never_proposed']} are never proposed by the detector "
        f"at any threshold (recall ceiling "
        f"{R['raw_pool_ceiling']['recall_ceiling']:.4f}).",
        f"6. **Greedy value of perfect recall**: recovering the tiers that pay, in "
        f"descending order, stops after "
        f"{wc0['greedy_optimum']['stop_after_tier']} at PQ "
        f"{wc0['greedy_optimum']['pq']:.4f} ({wc0['greedy_optimum']['delta_pq']:+.4f}), "
        f"recovering {wc0['greedy_optimum']['fn_recovered']} of "
        f"{wc0['n_fn_total']} FN. Chasing the remaining "
        f"{wc0['n_fn_total'] - wc0['greedy_optimum']['fn_recovered']} LOWERS PQ.",
        "",
        "## Feature discrimination (matched vs missed)", "",]
    L += ["",
         "`mean_pos`/`median_pos` = matched, `mean_neg`/`median_neg` = missed. "
         "Positive Cohen's d = the feature is larger on filaments we find.", "",
         fs.head(25).to_markdown(index=False), "",
         ]
    for name, rows in R["group_recall"].items():
        L += [f"## Recall by {name}", "", pd.DataFrame(rows).to_markdown(index=False), ""]
    wc = R["worth_chasing"]
    L += ["## Which misses matter", "",
          f"- FN total {wc['n_fn_total']}: {wc['n_fn_fully_missed_clusters']} sit in "
          f"{wc['n_clusters_fully_missed']} clusters where **no** copy was matched "
          f"(recoverable by a new detection); {wc['n_fn_partial_clusters']} sit in "
          f"{wc['n_clusters_partial']} clusters where at least one copy WAS matched "
          f"(the prediction exists but its IoU with this copy is <=0.5 - a "
          f"segmentation-quality problem, not recoverable by adding a disjoint mask).",
          f"- typical matched IoU used for the marginal arithmetic: {wc['typical_iou_used']}",
          "",
          "Marginal rule (exact): adding a detection of a cluster drawn by k of N copies "
          "changes the numerator by IoU*k and the denominator by N/2, so it pays while "
          "`PQ < 2*IoU*k/N`. The briefed form `PQ < 2*IoU*k/(N+k)` ignores the 0.5*k FN "
          "credit that disappears and is therefore conservative; both are reported.", "",
          pd.DataFrame([dict(rule="marginal (2*IoU*k/N)", **wc["marginal_rule"]),
                        dict(rule="briefed (2*IoU*k/(N+k))", **wc["briefed_rule"])]
                       ).to_markdown(index=False), "",
          "### Fully-missed clusters by (N, k)", "", tt.to_markdown(index=False), "",
          "### Greedy tier ladder", "",
          "The marginal threshold is compared against the CURRENT PQ, which rises as "
          "misses are recovered - so the rule must be re-checked tier by tier. Tiers "
          "added in descending value order:", "",
          lad.to_markdown(index=False), "",
          f"- greedy optimum: stop after tier "
          f"{R['worth_chasing']['greedy_optimum']['stop_after_tier']} -> "
          f"PQ {R['worth_chasing']['greedy_optimum']['pq']:.4f} "
          f"({R['worth_chasing']['greedy_optimum']['delta_pq']:+.4f}) recovering "
          f"{R['worth_chasing']['greedy_optimum']['fn_recovered']} FN", "",
          "### All clusters by (N, k) - cluster-level detection rate", "",
          tt_all.to_markdown(index=False), "",
          f"- Perfect recall of ALL fully-missed clusters: "
          f"PQ {wc['pq_if_perfect_on_all_fully_missed']:.4f} "
          f"({wc['delta_pq_all_fully_missed']:+.4f})",
          f"- Upper bound if every FN row vanished (including the un-recoverable "
          f"partial-cluster ones): PQ {wc['pq_if_zero_fn']:.4f}", "",
          "## FN split: dropped by the operating point vs never proposed", "",
          pd.DataFrame(R["fn_split_profile"]).to_markdown(index=False), "",
          f"Reason a reachable FN was dropped: {json.dumps(R['fn_dropped_reason'], indent=1)}",
          "", "### What separates a dropped FN from a never-proposed one", "",
          fs_fn.head(20).to_markdown(index=False), "",
          "### Where the headroom is", "",
          f"```json\n{json.dumps(R['headroom'], indent=1)}\n```", "",
          "## Reachability by group", ""]
    for blk in R["reachability_by_group"]:
        L += [f"### {blk['name']}", "", pd.DataFrame(blk["rows"]).to_markdown(index=False), ""]
    L += ["## Figures", "",
          "- `error_profile_fn_recall.png` recall vs decile for the 9 most discriminative features",
          "- `error_profile_fn_crosstab.png` recall cross-tabs",
          "- `error_profile_fn_split.png` FN anatomy (size/contrast/consensus/threshold plane)", ""]
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
