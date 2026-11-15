"""Error profile of the pooled-OOF operating point: FP anatomy + strength/weakness scorecard.

Operating point ("our system"): keep de-overlapped E013@1792 pooled-OOF candidates with
    conf >= 0.30 AND p_sel >= 0.35
which scores pooled PQ 0.4105 / TP 4768 / FP 2461 / FN 3431 over all 8199 GT rows.

Sections
  1. FP identification   -- pure hallucination / partial consensus / clean TP, exact FP split
  2. FP features         -- per-class medians + Cohen's d (pure hallucination vs clean TP)
  3. Spatial / context   -- FP per image vs radius, |lat|, site, year, GT-count, cand density
  4. "Is it darkness?"   -- ROC-AUC of contrast / depth / fg_min, + tiny logistic (GroupKFold)
  5. Scorecard           -- per-stratum TP/FP/FN/SQ/PQ and absolute share of the lost
                            denominator mass 0.5*FP + 0.5*FN
  6. SQ losses           -- matched-IoU p10/50/90 by stratum, +0.05 IoU counterfactual,
                            near-miss FN (best IoU just under the 0.5 threshold)

Inputs (all pre-computed, nothing is retrained):
  outputs/selector/pooled_candidates_v4.parquet   (falls back to pooled_candidates.parquet)
  outputs/eda/cand_rle_deov.parquet               (built by src/eda/build_cand_rle_cache.py)
  outputs/gt_cache/gt_full.parquet
  outputs/eda/morphology_instances.parquet
  outputs/eda/annotator_clusters.parquet
  outputs/eda/manifest_physical.parquet

Outputs: outputs/eda/error_profile_fp.{md,json}, error_profile_fp_*.parquet, 3 PNGs.

Run: python src/eda/error_profile_fp.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament import IMG_H, IMG_W                      # noqa: E402
from filament.pq import _entry_iou_matrix, score       # noqa: E402
from filament.rle import rles_to_layers                # noqa: E402
from selector import fast_pq                           # noqa: E402

OUT = ROOT / "outputs" / "eda"
CONF_T, PSEL_T = 0.30, 0.35
DISK_C, DISK_R = 1023.5, 900.0
TOTAL_GT = 8199

# validated dataviz categorical slots (light surface #fcfcfb)
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SURF = "#fcfcfb"

FEATS = ["conf", "p_sel", "area", "width_proxy", "boundary_complexity",
         "eccentricity", "r_norm", "abs_lat", "n_cands_img", "rank_conf",
         "contrast", "depth", "fg_min", "edge_grad"]


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------- load
def load_all():
    v4 = ROOT / "outputs" / "selector" / "pooled_candidates_v4.parquet"
    v1 = ROOT / "outputs" / "selector" / "pooled_candidates.parquet"
    src = v4 if v4.exists() else v1
    cand = pd.read_parquet(src)
    log(f"candidates from {src.name}: {cand.shape}")
    has_photo = "contrast" in cand.columns
    if not has_photo:
        raise SystemExit("pooled_candidates_v4.parquet with photometry not available")

    rle = pd.read_parquet(OUT / "cand_rle_deov.parquet")[["cand_id", "rle_deov"]]
    cand = cand.merge(rle, on="cand_id", how="left")
    assert cand["rle_deov"].notna().all(), "missing de-overlapped RLEs"

    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    morph = pd.read_parquet(OUT / "morphology_instances.parquet")
    clus = pd.read_parquet(OUT / "annotator_clusters.parquet")
    man = pd.read_parquet(OUT / "manifest_physical.parquet")
    return cand, gt, morph, clus, man


def enrich_gt(gt, morph, clus, man):
    g = gt.copy()
    g["stem"] = g["file_name"].str.replace(".jpeg", "", regex=False)
    g["annotator_image"] = g["filament_id"].str.split("_", n=1).str[0]

    mcols = ["ann_id", "width_proxy", "spine_len", "aspect_ratio", "eccentricity",
             "centroid_x", "centroid_y", "n_poly_points"]
    g = g.merge(morph[mcols], on="ann_id", how="left")

    # FIXED disk geometry -- recomputed, ignoring morphology's stale per-image fits
    dx = g["centroid_x"] - DISK_C
    dy = g["centroid_y"] - DISK_C
    g["r_norm"] = np.hypot(dx, dy) / DISK_R
    g["abs_lat"] = np.abs(np.degrees(np.arcsin(
        np.clip((DISK_C - g["centroid_y"]) / DISK_R, -1.0, 1.0))))

    # per-file context
    ncop = gt.groupby("file_name")["entry_id"].nunique().rename("n_copies")
    nrows = gt.groupby("file_name").size().rename("n_gt_rows")
    ctx = pd.concat([ncop, nrows], axis=1).reset_index()
    ctx["gt_per_entry"] = ctx["n_gt_rows"] / ctx["n_copies"]
    g = g.merge(ctx, on="file_name", how="left")

    # cross-annotator support tier (only defined on the 296 multi-copy files)
    sup_rows = []
    for r in clus.itertuples():
        for a in str(r.member_ann_ids).split(","):
            sup_rows.append((a, int(r.n_members), int(r.n_copies_file),
                             bool(r.full_support), float(r.mean_pair_iou)
                             if pd.notna(r.mean_pair_iou) else np.nan))
    sup = pd.DataFrame(sup_rows, columns=["ann_id", "n_members", "n_copies_file",
                                          "full_support", "mean_pair_iou"])
    g = g.merge(sup, on="ann_id", how="left")

    m = man[["file_name", "site", "site_name", "year"]]
    g = g.merge(m, on="file_name", how="left")
    return g, ctx


# ---------------------------------------------------- exact + detailed scoring
def detailed_score(gt_df, pred_df):
    """Official PQ semantics (bit-for-bit) plus per-GT / per-FP best-IoU diagnostics."""
    gt_entry = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_img = pred_df["filament_id"].str.split("_", n=1).str[0]
    gt_by = {k: v for k, v in gt_df.groupby(gt_entry)}
    pred_by = {k: v for k, v in pred_df.groupby(pred_img)}

    matches, gt_rows, fp_rows = [], [], []
    tp_ious, fp_count, fn_count = [], 0, 0
    t0 = time.time()
    for i, ai in enumerate(gt_entry.unique()):
        image_id = ai.split("-", maxsplit=1)[1]
        gs = gt_by[ai]
        ps = pred_by.get(image_id)
        if ps is None or len(ps) == 0:
            fn_count += len(gs)
            for fid in gs["filament_id"]:
                gt_rows.append(dict(filament_id=fid, annotator_image=ai, matched=0,
                                    best_iou=0.0, best_pred=None))
            continue
        gl = rles_to_layers(gs["segmentation_rle"].tolist(), IMG_H, IMG_W)
        pl = rles_to_layers(ps["segmentation_rle"].tolist(), IMG_H, IMG_W)
        iou = _entry_iou_matrix(gl, pl)
        del gl, pl
        hit = iou > 0.5
        tp_ious.extend(iou[hit].tolist())
        fp_mask = hit.sum(axis=0) == 0
        fn_mask = hit.sum(axis=1) == 0
        fp_count += int(fp_mask.sum())
        fn_count += int(fn_mask.sum())

        gi, pj = np.nonzero(hit)
        gid = gs["filament_id"].to_numpy()
        pid = ps["filament_id"].to_numpy()
        for a, b in zip(gi, pj):
            matches.append(dict(annotator_image=ai, gt_filament_id=gid[a],
                                pred_filament_id=pid[b], iou=float(iou[a, b])))
        gbest = iou.max(axis=1)
        gbest_j = iou.argmax(axis=1)
        for a in range(len(gid)):
            gt_rows.append(dict(filament_id=gid[a], annotator_image=ai,
                                matched=int(not fn_mask[a]),
                                best_iou=float(gbest[a]),
                                best_pred=pid[int(gbest_j[a])]))
        pbest = iou.max(axis=0)
        pbest_i = iou.argmax(axis=0)
        for b in np.nonzero(fp_mask)[0]:
            fp_rows.append(dict(annotator_image=ai, pred_filament_id=pid[b],
                                best_iou=float(pbest[b]),
                                best_gt=gid[int(pbest_i[b])]))
        del iou, hit
        if (i + 1) % 200 == 0:
            log(f"  entries {i+1}/{gt_entry.nunique()} ({time.time()-t0:.0f}s)")

    tp = len(tp_ious)
    den = tp + 0.5 * fp_count + 0.5 * fn_count
    agg = dict(pq=float(sum(tp_ious) / den) if den else 0.0,
               sq=float(sum(tp_ious) / tp) if tp else 0.0,
               rq=float(tp / den) if den else 0.0,
               tp=tp, fp=int(fp_count), fn=int(fn_count),
               iou_sum=float(sum(tp_ious)))
    return agg, pd.DataFrame(matches), pd.DataFrame(gt_rows), pd.DataFrame(fp_rows)


# ------------------------------------------------------------------ utilities
def cohens_d(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    na, nb = len(a), len(b)
    s = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / s) if s > 0 else np.nan


def qbin(x, edges, labels):
    return pd.cut(x, bins=edges, labels=labels, include_lowest=True,
                  right=False).astype(object)


RZ_LAB = ["r<0.30", "r0.30-0.60", "r0.60-0.80", "r0.80-0.90", "r>=0.90"]
LB_LAB = ["|lat|<10", "|lat|10-20", "|lat|20-30", "|lat|30-45", "|lat|>=45"]
LEVEL_ORDER = {v: i for i, v in enumerate(RZ_LAB + LB_LAB)}


def radial_zone(r):
    return qbin(r, [0, 0.30, 0.60, 0.80, 0.90, 10.0], RZ_LAB)


def lat_band(a):
    return qbin(a, [0, 10, 20, 30, 45, 91], LB_LAB)


def order_levels(df, col="level"):
    df = df.copy()
    df["_k"] = [LEVEL_ORDER.get(v, 10_000) for v in df[col]]
    return df.sort_values(["_k", col]).drop(columns="_k").reset_index(drop=True)


# ==============================================================================
def main():
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    cand, gt, morph, clus, man = load_all()
    gte, ctx = enrich_gt(gt, morph, clus, man)

    cand["n_entries_hit"] = cand["n_entries_hit"].astype(int)
    cand["n_copies"] = cand["n_copies"].astype(int)
    kept = cand[(cand["conf"] >= CONF_T) & (cand["p_sel"] >= PSEL_T)].copy()
    kept["fp_copies"] = kept["n_copies"] - kept["n_entries_hit"]
    fastr = fast_pq(kept, TOTAL_GT)
    log(f"operating point: kept={len(kept)} fast_pq={fastr}")
    assert (fastr["tp"], fastr["fp"], fastr["fn"]) == (4768, 2461, 3431), fastr

    # ---------------- exact scorer pass (authoritative) -----------------------
    pred_df = kept[["cand_id", "rle_deov"]].rename(
        columns={"cand_id": "filament_id", "rle_deov": "segmentation_rle"})
    cdir = OUT / "error_profile_fp_cache"
    if (cdir / "agg.json").exists():
        log(f"reusing exact-scorer cache {cdir}")
        agg = json.loads((cdir / "agg.json").read_text())
        matches = pd.read_parquet(cdir / "matches.parquet")
        gtstat = pd.read_parquet(cdir / "gtstat.parquet")
        fpstat = pd.read_parquet(cdir / "fpstat.parquet")
    else:
        log("running exact scorer pass ...")
        agg, matches, gtstat, fpstat = detailed_score(
            gte[["filament_id", "segmentation_rle"]], pred_df)
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "agg.json").write_text(json.dumps(agg))
        matches.to_parquet(cdir / "matches.parquet", index=False)
        gtstat.to_parquet(cdir / "gtstat.parquet", index=False)
        fpstat.to_parquet(cdir / "fpstat.parquet", index=False)
    log(f"exact: PQ={agg['pq']:.4f} TP={agg['tp']} FP={agg['fp']} FN={agg['fn']} "
        f"SQ={agg['sq']:.4f}")
    assert (agg["tp"], agg["fp"], agg["fn"]) == (4768, 2461, 3431)
    assert abs(agg["pq"] - fastr["pq"]) < 1e-9
    IOU_SUM, DEN = agg["iou_sum"], agg["tp"] + 0.5 * agg["fp"] + 0.5 * agg["fn"]

    res = {"operating_point": dict(conf_t=CONF_T, p_sel_t=PSEL_T, n_candidates=len(cand),
                                   n_kept=len(kept), **{k: agg[k] for k in
                                   ["pq", "sq", "rq", "tp", "fp", "fn", "iou_sum"]},
                                   denominator=DEN, total_gt=TOTAL_GT)}

    # =============================== 1. FP identification ====================
    def klass(r):
        if r.n_entries_hit == 0:
            return "pure_hallucination"
        if r.n_entries_hit < r.n_copies:
            return "partial_consensus"
        return "clean_tp"
    kept["fp_class"] = [klass(r) for r in kept.itertuples()]
    ORDER = ["pure_hallucination", "partial_consensus", "clean_tp"]

    c1 = kept.groupby("fp_class").agg(
        n_kept=("cand_id", "size"),
        fp_copies=("fp_copies", "sum"),
        tp_copies=("weight", "sum"),
        gt_slots=("n_copies", "sum"),
        iou_sum=("iou_sum", "sum")).reindex(ORDER).fillna(0)
    c1["share_of_kept"] = c1["n_kept"] / len(kept)
    c1["share_of_FP"] = c1["fp_copies"] / agg["fp"]
    c1["precision_copies"] = c1["tp_copies"] / c1["gt_slots"]
    c1 = c1.reset_index()
    assert int(c1["fp_copies"].sum()) == agg["fp"], c1
    log("\n-- FP classes --\n" + c1.to_string(index=False))
    res["fp_classes"] = c1.to_dict("records")

    # multiplicity of the hallucinations: how many are on 1/2/3-copy images
    halluc = kept[kept.fp_class == "pure_hallucination"]
    res["pure_halluc_by_n_copies"] = (halluc.groupby("n_copies")
                                      .agg(n=("cand_id", "size"),
                                           fp=("fp_copies", "sum"))
                                      .reset_index().to_dict("records"))
    res["partial_by_hit_pattern"] = (kept[kept.fp_class == "partial_consensus"]
                                     .groupby(["n_copies", "n_entries_hit"])
                                     .agg(n=("cand_id", "size"),
                                          fp=("fp_copies", "sum"))
                                     .reset_index().to_dict("records"))

    # how close were the pure hallucinations to a GT? (best IoU in the entry)
    fps_h = fpstat.merge(kept[["cand_id", "fp_class"]],
                         left_on="pred_filament_id", right_on="cand_id", how="left")
    nm = fps_h.groupby("fp_class")["best_iou"].describe(
        percentiles=[.1, .5, .9]).reset_index()
    res["fp_best_iou_by_class"] = nm.to_dict("records")
    for lo, hi in [(0.0, 0.05), (0.05, 0.25), (0.25, 0.45), (0.45, 0.50)]:
        sel = fps_h[(fps_h.best_iou >= lo) & (fps_h.best_iou < hi)]
        res.setdefault("fp_best_iou_bins", []).append(
            dict(lo=lo, hi=hi, n_fp=int(len(sel)),
                 n_pure=int((sel.fp_class == "pure_hallucination").sum())))

    # =============================== 2. FP features ==========================
    med = kept.groupby("fp_class")[FEATS].median().reindex(ORDER).T
    med.columns = [f"median_{c}" for c in med.columns]
    A = kept[kept.fp_class == "pure_hallucination"]
    B = kept[kept.fp_class == "clean_tp"]
    P = kept[kept.fp_class == "partial_consensus"]
    med["d_pure_vs_cleanTP"] = [cohens_d(A[f], B[f]) for f in FEATS]
    med["d_partial_vs_cleanTP"] = [cohens_d(P[f], B[f]) for f in FEATS]
    med["abs_d"] = med["d_pure_vs_cleanTP"].abs()
    feat_tbl = med.sort_values("abs_d", ascending=False).reset_index().rename(
        columns={"index": "feature"})
    log("\n-- feature separation --\n" + feat_tbl.round(4).to_string(index=False))
    res["fp_features"] = feat_tbl.to_dict("records")

    # =============================== 3. spatial / context ====================
    kept["radial_zone"] = radial_zone(kept["r_norm"])
    kept["lat_band"] = lat_band(kept["abs_lat"])
    kept = kept.merge(ctx[["file_name", "n_copies", "n_gt_rows", "gt_per_entry"]]
                      .assign(stem=lambda d: d.file_name.str.replace(".jpeg", "", regex=False))
                      .drop(columns=["file_name", "n_copies"]), on="stem", how="left")
    kept = kept.merge(man[["stem", "site", "site_name", "year"]], on="stem", how="left")

    gpe_edges = np.unique(np.quantile(ctx["gt_per_entry"], [0, .2, .4, .6, .8, 1.0]))
    gpe_lab = [f"Q{i+1}" for i in range(len(gpe_edges) - 1)]
    ctx["gt_quintile"] = pd.cut(ctx["gt_per_entry"], gpe_edges, labels=gpe_lab,
                                include_lowest=True).astype(object)
    stem2q = dict(zip(ctx["file_name"].str.replace(".jpeg", "", regex=False),
                      ctx["gt_quintile"]))
    kept["gt_quintile"] = kept["stem"].map(stem2q)
    gte["gt_quintile"] = gte["file_name"].map(dict(zip(ctx.file_name, ctx.gt_quintile)))

    dens_edges = np.unique(np.quantile(cand.groupby("stem")["n_cands_img"].first(),
                                       [0, .2, .4, .6, .8, 1.0]))
    kept["dens_quintile"] = pd.cut(kept["n_cands_img"], dens_edges,
                                   labels=[f"D{i+1}" for i in range(len(dens_edges) - 1)],
                                   include_lowest=True).astype(object)

    n_img_total = kept["stem"].nunique()

    def ctx_table(col):
        g = kept.groupby(col).agg(n_kept=("cand_id", "size"),
                                  n_images=("stem", "nunique"),
                                  n_pure=("fp_class", lambda s: int((s == "pure_hallucination").sum())),
                                  fp_copies=("fp_copies", "sum"),
                                  tp_copies=("weight", "sum"),
                                  gt_slots=("n_copies", "sum"),
                                  iou_sum=("iou_sum", "sum")).reset_index()
        g["pure_share_of_kept"] = g["n_pure"] / g["n_kept"]
        g["fp_per_image"] = g["fp_copies"] / g["n_images"]
        g["precision_copies"] = g["tp_copies"] / g["gt_slots"]
        g["share_of_all_FP"] = g["fp_copies"] / agg["fp"]
        g = g.rename(columns={col: "level"})
        g["level"] = g["level"].astype(str)
        g = order_levels(g)
        g.insert(0, "dimension", col)
        return g

    ctx_tables = pd.concat([ctx_table(c) for c in
                            ["radial_zone", "lat_band", "site", "year",
                             "gt_quintile", "dens_quintile", "n_copies"]],
                           ignore_index=True)
    log("\n-- context --\n" + ctx_tables.round(4).to_string(index=False))
    res["context"] = ctx_tables.to_dict("records")

    # regime check: is the near-limb / low-activity effect confounded by n_copies?
    reg = (kept.groupby(["n_copies", "radial_zone"])
           .agg(n_kept=("cand_id", "size"),
                n_pure=("fp_class", lambda s: int((s == "pure_hallucination").sum())))
           .reset_index())
    reg = order_levels(reg, "radial_zone").sort_values(
        ["n_copies"], kind="stable").reset_index(drop=True)
    reg["pure_share"] = reg["n_pure"] / reg["n_kept"]
    reg["n_copies"] = reg["n_copies"].astype(int)
    res["regime_ncopies_x_radial"] = reg.to_dict("records")
    reg2 = (kept.groupby(["n_copies", "gt_quintile"])
            .agg(n_kept=("cand_id", "size"),
                 n_pure=("fp_class", lambda s: int((s == "pure_hallucination").sum())))
            .reset_index())
    reg2["pure_share"] = reg2["n_pure"] / reg2["n_kept"]
    reg2["n_copies"] = reg2["n_copies"].astype(int)
    res["regime_ncopies_x_gtquintile"] = reg2.to_dict("records")
    log("\n-- regime n_copies x radial --\n" + reg.round(4).to_string(index=False))

    # correlation between per-image candidate density and per-image FP
    per_img = kept.groupby("stem").agg(fp=("fp_copies", "sum"), tp=("weight", "sum"),
                                       n_cands=("n_cands_img", "first"),
                                       n_kept=("cand_id", "size"),
                                       gt_per_entry=("gt_per_entry", "first"),
                                       n_pure=("fp_class", lambda s: int((s == "pure_hallucination").sum())))
    from scipy.stats import pearsonr, spearmanr
    res["per_image_corr"] = {}
    for a, b in [("n_cands", "fp"), ("n_cands", "n_pure"), ("gt_per_entry", "fp"),
                 ("gt_per_entry", "n_pure"), ("n_kept", "fp")]:
        m = per_img[[a, b]].dropna()
        res["per_image_corr"][f"{a}~{b}"] = dict(
            pearson=float(pearsonr(m[a], m[b])[0]),
            spearman=float(spearmanr(m[a], m[b])[0]), n=int(len(m)))

    # =============================== 4. is it darkness? ======================
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    dk = {}
    for name, sub in [("pure_vs_cleanTP", kept[kept.fp_class != "partial_consensus"]),
                      ("pure_vs_anyTP", kept)]:
        y = (sub["fp_class"] == "pure_hallucination").astype(int).to_numpy()
        entry = dict(n=int(len(sub)), n_pos=int(y.sum()))
        for f in ["contrast", "depth", "fg_min", "edge_grad", "fg_mean", "conf", "p_sel"]:
            a = roc_auc_score(y, sub[f].to_numpy())
            entry[f"auc_{f}"] = float(a)
            entry[f"auc_{f}_oriented"] = float(max(a, 1 - a))
        X = sub[["contrast", "depth", "fg_min"]].to_numpy(float)
        groups = sub["stem"].to_numpy()
        oof = np.full(len(sub), np.nan)
        for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
            mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            mdl.fit(X[tr], y[tr])
            oof[va] = mdl.predict_proba(X[va])[:, 1]
        entry["auc_logit_contrast_depth_fgmin_oof"] = float(roc_auc_score(y, oof))
        X2 = sub[["contrast", "depth", "fg_min", "edge_grad", "fg_std", "ring_std"]].to_numpy(float)
        oof2 = np.full(len(sub), np.nan)
        for tr, va in GroupKFold(n_splits=5).split(X2, y, groups):
            mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            mdl.fit(X2[tr], y[tr])
            oof2[va] = mdl.predict_proba(X2[va])[:, 1]
        entry["auc_logit_all_photometric_oof"] = float(roc_auc_score(y, oof2))
        full = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X, y)
        entry["logit_coefs_contrast_depth_fgmin"] = [
            float(c) for c in full[-1].coef_[0]]

        # incremental value of photometry ON TOP of what the selector already knows
        def oof_auc(cols):
            Z = sub[cols].to_numpy(float)
            o = np.full(len(sub), np.nan)
            for tr, va in GroupKFold(n_splits=5).split(Z, y, groups):
                mm = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
                mm.fit(Z[tr], y[tr])
                o[va] = mm.predict_proba(Z[va])[:, 1]
            return float(roc_auc_score(y, o))
        base_cols = ["conf", "p_sel"]
        photo_cols = ["contrast", "depth", "fg_min", "edge_grad", "fg_std", "ring_std"]
        a_base = oof_auc(base_cols)
        a_both = oof_auc(base_cols + photo_cols)
        entry["auc_logit_conf_psel_oof"] = a_base
        entry["auc_logit_conf_psel_plus_photometric_oof"] = a_both
        entry["auc_gain_from_photometry"] = a_both - a_base
        dk[name] = entry
    # decile lift of contrast on the kept set
    kq = pd.qcut(kept["contrast"], 10, labels=False, duplicates="drop")
    dk["contrast_decile_pure_rate"] = (
        kept.assign(dec=kq).groupby("dec")
        .agg(n=("cand_id", "size"),
             pure_rate=("fp_class", lambda s: float((s == "pure_hallucination").mean())),
             contrast_lo=("contrast", "min"), contrast_hi=("contrast", "max"))
        .reset_index().to_dict("records"))
    res["darkness"] = dk
    log("\n-- darkness --\n" + json.dumps(
        {k: v for k, v in dk.items() if k != "contrast_decile_pure_rate"}, indent=1))

    # =============================== 5. scorecard ============================
    gt_ok = gtstat.merge(gte.drop(columns=["annotator_image"]),
                         on="filament_id", how="left")
    assert len(gt_ok) == TOTAL_GT
    m2 = matches.merge(gte[["filament_id", "area", "width_proxy", "category_id",
                            "r_norm", "abs_lat", "n_copies", "site", "gt_quintile",
                            "full_support", "file_name"]],
                       left_on="gt_filament_id", right_on="filament_id", how="left")

    # bin edges from the GT distribution, applied to GT rows AND to candidates
    a_edges = list(np.quantile(gte["area"], [0, .25, .5, .75, 1.0]))
    a_edges[0], a_edges[-1] = 0, 1e12
    a_lab = ["A1_smallest", "A2", "A3", "A4_largest"]
    w_edges = list(np.nanquantile(gte["width_proxy"], [0, .25, .5, .75, 1.0]))
    w_edges[0], w_edges[-1] = 0, 1e12
    w_lab = ["W1_thinnest", "W2", "W3", "W4_thickest"]
    CATN = {1: "cat1_Left", 2: "cat2_Right", 3: "cat3_Unidentifiable"}

    def assign(df, area_col, width_col, r_col):
        return pd.DataFrame({
            "size_quartile": qbin(df[area_col], a_edges, a_lab),
            "width_quartile": qbin(df[width_col], w_edges, w_lab),
            "radial_zone": radial_zone(df[r_col]),
            "consensus_tier": df["n_copies"].map(
                {1: "T1_single_annotator", 2: "T2_two_annotators", 3: "T3_three_annotators"}),
            "site": df["site"],
            "gt_quintile": df["gt_quintile"]})

    gt_str = assign(gt_ok, "area", "width_proxy", "r_norm")
    gt_str["category"] = gt_ok["category_id"].map(CATN)
    gt_str["filament_id"] = gt_ok["filament_id"].values
    gt_str["matched"] = gt_ok["matched"].values

    kp_str = assign(kept, "area", "width_proxy", "r_norm")
    kp_str["category"] = "unassigned_pred"
    kp_str["cand_id"] = kept["cand_id"].values
    kp_str["fp_copies"] = kept["fp_copies"].values

    mt_str = gt_str.set_index("filament_id").loc[m2["gt_filament_id"].values].reset_index()
    mt_str["iou"] = m2["iou"].values

    DIMS = ["size_quartile", "width_quartile", "radial_zone", "category",
            "consensus_tier", "site", "gt_quintile"]
    rows = []
    for d in DIMS:
        tp = mt_str.groupby(d).agg(tp=("iou", "size"), iou_sum=("iou", "sum"))
        fn = gt_str[gt_str.matched == 0].groupby(d).size().rename("fn")
        fp = kp_str.groupby(d)["fp_copies"].sum().rename("fp")
        gtn = gt_str.groupby(d).size().rename("n_gt")
        t = pd.concat([tp, fn, fp, gtn], axis=1).fillna(0).reset_index()
        t = t.rename(columns={d: "level"})
        t.insert(0, "dimension", d)
        rows.append(t)
    sc = pd.concat(rows, ignore_index=True)
    for c in ["tp", "fn", "fp", "n_gt"]:
        sc[c] = sc[c].astype(int)
    sc["sq"] = np.where(sc.tp > 0, sc.iou_sum / sc.tp.replace(0, np.nan), np.nan)
    sc["den"] = sc.tp + 0.5 * sc.fp + 0.5 * sc.fn
    sc["pq_stratum"] = np.where(sc.den > 0, sc.iou_sum / sc.den.replace(0, np.nan), 0.0)
    sc["rq_stratum"] = np.where(sc.den > 0, sc.tp / sc.den.replace(0, np.nan), 0.0)
    sc["lost_mass"] = 0.5 * sc.fp + 0.5 * sc.fn
    sc["lost_mass_share"] = sc["lost_mass"] / (0.5 * agg["fp"] + 0.5 * agg["fn"])
    sc["recall_copies"] = sc.tp / (sc.tp + sc.fn).replace(0, np.nan)
    # global PQ if this stratum became perfect (all its FN recovered at IoU 1, its FP dropped)
    sc["pq_if_stratum_perfect"] = ((IOU_SUM - sc.iou_sum + sc.tp + sc.fn) /
                                   (agg["tp"] + sc.fn +
                                    0.5 * (agg["fp"] - sc.fp) + 0.5 * (agg["fn"] - sc.fn)))
    sc["delta_pq_if_perfect"] = sc["pq_if_stratum_perfect"] - agg["pq"]

    # sanity: each dimension must reconstruct the global totals
    for d in DIMS:
        s = sc[sc.dimension == d]
        assert int(s.tp.sum()) == agg["tp"], (d, s.tp.sum())
        assert int(s.fn.sum()) == agg["fn"], (d, s.fn.sum())
        assert int(s.fp.sum()) == agg["fp"], (d, s.fp.sum())
    sc = sc.sort_values("lost_mass", ascending=False).reset_index(drop=True)
    log("\n-- scorecard (top 20 by lost mass) --\n" +
        sc.head(20)[["dimension", "level", "tp", "fp", "fn", "sq", "pq_stratum",
                     "lost_mass", "lost_mass_share", "delta_pq_if_perfect"]]
        .round(4).to_string(index=False))
    res["scorecard"] = sc.to_dict("records")

    # supplementary: attribute each FP (pred, entry) pair to the category of the GT
    # instance it overlaps most inside that entry (only where the overlap is real)
    fp_cat = fpstat.merge(gte[["filament_id", "category_id"]],
                          left_on="best_gt", right_on="filament_id", how="left")
    fp_cat["bucket"] = np.where(fp_cat.best_iou >= 0.10,
                                fp_cat.category_id.map(CATN), "no_gt_overlap_iou<0.10")
    fpc = fp_cat.groupby("bucket").size().rename("fp_copies").reset_index()
    fpc["share_of_FP"] = fpc["fp_copies"] / agg["fp"]
    res["fp_by_best_overlap_gt_category"] = fpc.to_dict("records")
    log("\n-- FP by best-overlap GT category --\n" + fpc.round(4).to_string(index=False))

    # mask-thickness / size bias: GT vs kept-candidate marginal distributions
    bias = []
    for nm, gcol, kcol in [("area", "area", "area"),
                           ("width_proxy", "width_proxy", "width_proxy"),
                           ("r_norm", "r_norm", "r_norm")]:
        qs = [0, .10, .25, .50, .75, .90, 1.0]
        bias.append(dict(quantity=nm,
                         gt=[float(v) for v in np.nanquantile(gte[gcol], qs)],
                         kept_cand=[float(v) for v in np.nanquantile(kept[kcol], qs)],
                         quantiles=qs))
    res["gt_vs_pred_marginals"] = bias
    res["width_proxy_caveat"] = (
        "GT width_proxy = polygon_area / polygon_spine_length (morphology_instances); "
        "candidate width_proxy = deoverlapped_mask_area / skeletonize(mask).sum(). "
        "The two estimators are not identical, so the W1_thinnest FP count is a lower "
        "bound; the GT-side TP/FN split in that quartile is unaffected.")

    # GT-side support tier (multi-copy files only) -- TP/FN only, no FP assignment
    mc = gt_ok[gt_ok.n_copies > 1].copy()
    mc["support_tier"] = np.where(mc.full_support.astype("boolean").fillna(False),
                                  "full_support", "partial_support")
    mtp = mt_str.copy()
    sup_map = dict(zip(mc.filament_id, mc.support_tier))
    mtp["support_tier"] = m2["gt_filament_id"].map(sup_map).values
    st = pd.concat([
        mtp.dropna(subset=["support_tier"]).groupby("support_tier")
           .agg(tp=("iou", "size"), iou_sum=("iou", "sum")),
        mc[mc.matched == 0].groupby("support_tier").size().rename("fn"),
        mc.groupby("support_tier").size().rename("n_gt")], axis=1).fillna(0).reset_index()
    st["sq"] = st.iou_sum / st.tp
    st["recall"] = st.tp / st.n_gt
    res["gt_support_tier"] = st.to_dict("records")
    log("\n-- GT support tier --\n" + st.round(4).to_string(index=False))

    # =============================== 6. SQ losses ============================
    def iou_dist(col):
        g = mt_str.groupby(col)["iou"].agg(
            n="size", p10=lambda s: s.quantile(.10), p50="median",
            p90=lambda s: s.quantile(.90), mean="mean").reset_index()
        g = g.rename(columns={col: "level"})
        g = order_levels(g)
        g.insert(0, "dimension", col)
        return g
    sq_tbl = pd.concat([iou_dist(c) for c in
                        ["size_quartile", "width_quartile", "category",
                         "radial_zone", "consensus_tier"]], ignore_index=True)
    log("\n-- matched IoU distribution --\n" + sq_tbl.round(4).to_string(index=False))
    res["sq_by_stratum"] = sq_tbl.to_dict("records")

    ious = m2["iou"].to_numpy()
    scen = {}
    scen["baseline"] = dict(pq=agg["pq"], iou_sum=IOU_SUM, tp=agg["tp"],
                            fp=agg["fp"], fn=agg["fn"])
    # global bounds: what each error type is worth in absolute PQ
    scen["all_FP_removed"] = dict(
        pq=float(IOU_SUM / (agg["tp"] + 0.5 * agg["fn"])),
        delta_pq=float(IOU_SUM / (agg["tp"] + 0.5 * agg["fn"]) - agg["pq"]))
    scen["only_pure_halluc_FP_removed"] = dict(
        n_fp_removed=int(c1.loc[c1.fp_class == "pure_hallucination", "fp_copies"].iloc[0]),
        pq=float(IOU_SUM / (agg["tp"] + 0.5 * (agg["fp"] - 1599) + 0.5 * agg["fn"])),
        delta_pq=float(IOU_SUM / (agg["tp"] + 0.5 * (agg["fp"] - 1599) + 0.5 * agg["fn"])
                       - agg["pq"]))
    scen["all_FN_recovered_at_iou1"] = dict(
        pq=float((IOU_SUM + agg["fn"]) /
                 (agg["tp"] + agg["fn"] + 0.5 * agg["fp"])),
        delta_pq=float((IOU_SUM + agg["fn"]) /
                       (agg["tp"] + agg["fn"] + 0.5 * agg["fp"]) - agg["pq"]))
    scen["all_FN_recovered_at_current_SQ"] = dict(
        pq=float((IOU_SUM + agg["fn"] * agg["sq"]) /
                 (agg["tp"] + agg["fn"] + 0.5 * agg["fp"])),
        delta_pq=float((IOU_SUM + agg["fn"] * agg["sq"]) /
                       (agg["tp"] + agg["fn"] + 0.5 * agg["fp"]) - agg["pq"]))
    for bump in [0.02, 0.05, 0.10]:
        new = np.minimum(ious + bump, 1.0)
        scen[f"iou_plus_{bump:g}"] = dict(
            pq=float(new.sum() / DEN), delta_pq=float((new.sum() - IOU_SUM) / DEN),
            iou_sum=float(new.sum()), n_capped=int((ious + bump > 1.0).sum()))
    # near-miss FN: unmatched GT whose best pred IoU is just under 0.5
    unm = gtstat[gtstat.matched == 0]
    for lo in [0.40, 0.45, 0.48]:
        nmiss = unm[unm.best_iou > lo]
        # those GT flip to TP at their (bumped) IoU; the pred stops being an FP in that entry
        gained_iou = float(np.minimum(nmiss.best_iou + 0.05, 1.0).sum())
        n_new = int(len(nmiss))
        # each newly matched GT removes one FN; its pred may already be a TP elsewhere,
        # so the FP reduction is bounded by the number of distinct (pred, entry) FP pairs
        fp_keys = set(zip(fpstat.pred_filament_id, fpstat.annotator_image))
        n_fp_removed = int(sum(1 for r in nmiss.itertuples()
                               if (r.best_pred, r.annotator_image) in fp_keys))
        tp2 = agg["tp"] + n_new
        fp2 = agg["fp"] - n_fp_removed
        fn2 = agg["fn"] - n_new
        den2 = tp2 + 0.5 * fp2 + 0.5 * fn2
        scen[f"nearmiss_flip_bestiou_gt_{lo:g}"] = dict(
            n_near_miss_fn=n_new, n_fp_removed=n_fp_removed,
            pq=float((IOU_SUM + gained_iou) / den2),
            delta_pq=float((IOU_SUM + gained_iou) / den2 - agg["pq"]))
    unm_desc = unm["best_iou"].describe(percentiles=[.5, .75, .9, .95]).to_dict()
    res["fn_best_iou"] = {k: float(v) for k, v in unm_desc.items()}
    res["fn_best_iou_zero_share"] = float((unm.best_iou == 0).mean())
    res["sq_scenarios"] = scen
    log("\n-- SQ scenarios --\n" + json.dumps(scen, indent=1))

    # =============================== outputs =================================
    sc.to_parquet(OUT / "error_profile_fp_scorecard.parquet", index=False)
    kept[["cand_id", "stem", "fold", "fp_class", "fp_copies", "n_copies",
          "n_entries_hit", "weight", "iou_sum", "conf", "p_sel", "area",
          "width_proxy", "r_norm", "abs_lat", "contrast", "depth", "fg_min",
          "edge_grad", "radial_zone", "lat_band", "site", "year", "gt_quintile",
          "dens_quintile"]].to_parquet(
        OUT / "error_profile_fp_kept.parquet", index=False)
    gt_ok_out = gt_ok[["filament_id", "annotator_image", "file_name", "matched",
                       "best_iou", "category_id", "area", "width_proxy", "r_norm",
                       "abs_lat", "n_copies", "site", "gt_quintile"]]
    gt_ok_out.to_parquet(OUT / "error_profile_fp_gt.parquet", index=False)
    matches.to_parquet(OUT / "error_profile_fp_matches.parquet", index=False)
    ctx_tables.to_parquet(OUT / "error_profile_fp_context.parquet", index=False)

    make_plots(kept, c1, feat_tbl, ctx_tables, sc, mt_str, dk)
    write_md(res, c1, feat_tbl, ctx_tables, sc, sq_tbl, st, agg)
    with open(OUT / "error_profile_fp.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=float)
    log(f"\nDONE in {time.time()-t_start:.0f}s -> outputs/eda/error_profile_fp.*")


# ------------------------------------------------------------------- plotting
def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def make_plots(kept, c1, feat_tbl, ctx_tables, sc, mt_str, dk):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- PNG 1: FP anatomy
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURF)
    ax = axes[0]
    _style(ax)
    lbl = ["pure\nhallucination", "partial\nconsensus", "clean TP"]
    ax.barh(lbl, c1["fp_copies"], color=[C2, C4, C3], height=.55)
    for i, (v, n) in enumerate(zip(c1["fp_copies"], c1["n_kept"])):
        ax.text(v + 30, i, f"{int(v)} FP  ({int(n)} cands)", va="center",
                fontsize=9, color=INK2)
    ax.set_xlim(0, c1["fp_copies"].max() * 1.42)
    ax.set_xlabel("false-positive annotator-copies", color=INK2, fontsize=9)
    ax.set_title("Where the 2461 FP come from", color=INK, fontsize=11, loc="left")

    ax = axes[1]
    _style(ax)
    ft = feat_tbl.dropna(subset=["d_pure_vs_cleanTP"]).sort_values("d_pure_vs_cleanTP")
    cols = [C2 if v > 0 else C1 for v in ft["d_pure_vs_cleanTP"]]
    ax.barh(ft["feature"], ft["d_pure_vs_cleanTP"], color=cols, height=.6)
    for i, v in enumerate(ft["d_pure_vs_cleanTP"]):
        ax.text(v + (0.03 if v > 0 else -0.03), i, f"{v:+.2f}", va="center",
                ha="left" if v > 0 else "right", fontsize=8, color=INK2)
    ax.axvline(0, color="#c3c2b7", lw=1)
    lim = max(abs(ft["d_pure_vs_cleanTP"])) * 1.35
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Cohen's d   (pure hallucination  -  clean TP)", color=INK2, fontsize=9)
    ax.set_title("Feature separation", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "error_profile_fp_anatomy.png", dpi=150, facecolor=SURF)
    plt.close(fig)

    # ---- PNG 2: context
    dims = [("radial_zone", "radial zone"), ("lat_band", "|latitude| band"),
            ("dens_quintile", "candidate-density quintile"), ("site", "site")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), facecolor=SURF)
    for axi, (d, nice) in zip(axes.ravel(), dims):
        _style(axi)
        t = ctx_tables[ctx_tables.dimension == d].copy()
        x = np.arange(len(t))
        axi.bar(x - .2, t["pure_share_of_kept"], width=.38, color=C2,
                label="pure-hallucination share of kept")
        axi.bar(x + .2, 1 - t["precision_copies"], width=.38, color=C1,
                label="FP share of GT-copy slots")
        for xi, (a, b) in enumerate(zip(t["pure_share_of_kept"], 1 - t["precision_copies"])):
            axi.text(xi - .2, a + .008, f"{a:.2f}", ha="center", fontsize=7, color=INK2)
            axi.text(xi + .2, b + .008, f"{b:.2f}", ha="center", fontsize=7, color=INK2)
        axi.set_xticks(x)
        axi.set_xticklabels([str(v) for v in t["level"]], fontsize=8, rotation=0)
        axi.grid(axis="y", color=GRID, lw=.8)
        axi.grid(axis="x", visible=False)
        axi.set_ylim(0, max(.6, float(max(t["pure_share_of_kept"].max(),
                                          (1 - t["precision_copies"]).max())) * 1.25))
        axi.set_title(nice, color=INK, fontsize=10, loc="left")
    axes[0, 0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Hallucination rate by context (kept candidates, conf>=0.30 & p_sel>=0.35)",
                 color=INK, fontsize=11, x=.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(OUT / "error_profile_fp_context.png", dpi=150, facecolor=SURF)
    plt.close(fig)

    # ---- PNG 3: scorecard + IoU
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), facecolor=SURF)
    ax = axes[0]
    _style(ax)
    top = sc[sc.level != "unassigned_pred"].head(14).iloc[::-1]
    lab = [f"{d.replace('_',' ')}: {l}" for d, l in zip(top.dimension, top.level)]
    ax.barh(lab, 0.5 * top.fn, color=C1, height=.6, label="0.5 x FN (missed GT)")
    ax.barh(lab, 0.5 * top.fp, left=0.5 * top.fn, color=C2, height=.6,
            label="0.5 x FP (spurious)")
    for i, (v, s) in enumerate(zip(top.lost_mass, top.lost_mass_share)):
        ax.text(v + 12, i, f"{v:.0f}  ({s*100:.0f}%)", va="center", fontsize=8, color=INK2)
    ax.set_xlim(0, top.lost_mass.max() * 1.33)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_xlabel("lost PQ denominator mass  (0.5*FP + 0.5*FN);  total = 2946",
                  color=INK2, fontsize=9)
    ax.set_title("Where PQ is lost, in absolute mass", color=INK, fontsize=11, loc="left")

    ax = axes[1]
    _style(ax)
    ax.grid(axis="y", color=GRID, lw=.8)
    ax.grid(axis="x", visible=False)
    groups = ["size_quartile", "width_quartile"]
    pos, ticks, colors = [], [], []
    data = []
    for gi, g in enumerate(groups):
        for li, lv in enumerate(sorted(mt_str[g].dropna().unique())):
            data.append(mt_str[mt_str[g] == lv]["iou"].to_numpy())
            pos.append(gi * 5 + li)
            ticks.append(str(lv))
            colors.append(C1 if gi == 0 else C3)
    bp = ax.boxplot(data, positions=pos, widths=.62, patch_artist=True,
                    showfliers=False, medianprops=dict(color=INK, lw=1.4),
                    whiskerprops=dict(color="#c3c2b7"), capprops=dict(color="#c3c2b7"),
                    boxprops=dict(lw=0))
    for p, c in zip(bp["boxes"], colors):
        p.set_facecolor(c)
        p.set_alpha(.85)
        p.set_edgecolor(SURF)
        p.set_linewidth(2)
    for p, d in zip(pos, data):
        ax.text(p, np.median(d) + .012, f"{np.median(d):.2f}", ha="center",
                fontsize=7.5, color=INK2)
    ax.set_xticks(pos)
    ax.set_xticklabels(ticks, fontsize=7.5, rotation=30, ha="right")
    ax.axhline(0.5, color="#d03b3b", lw=1, ls="--")
    ax.text(pos[-1], .505, "IoU>0.5 match threshold", fontsize=7.5, color="#d03b3b",
            ha="right")
    ax.set_ylabel("matched-pair IoU (SQ)", color=INK2, fontsize=9)
    ax.set_title("Matched-pair mask quality: size (blue) vs width (green)",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "error_profile_fp_scorecard.png", dpi=150, facecolor=SURF)
    plt.close(fig)
    log("wrote 3 PNGs")


# ---------------------------------------------------------------- markdown
def md_table(df, cols=None, floatfmt=4):
    d = df[cols] if cols else df
    d = d.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].round(floatfmt)
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    def cell(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            return "[" + ", ".join(str(x) for x in v) + "]"
        return "" if pd.isna(v) else str(v)
    body = ["| " + " | ".join(cell(v) for v in r) + " |"
            for r in d.itertuples(index=False)]
    return "\n".join([head, sep] + body)


def write_md(res, c1, feat_tbl, ctx_tables, sc, sq_tbl, st, agg):
    L = []
    A = L.append
    op = res["operating_point"]
    A("# Error profile at the current operating point\n")
    A(f"Operating point: **conf >= {CONF_T} AND p_sel >= {PSEL_T}** on the 10411 pooled "
      f"5-fold OOF de-overlapped candidates of E013@1792.\n")
    A(f"Exact official scorer (verified against `fast_pq`): **PQ {op['pq']:.4f}**, "
      f"SQ {op['sq']:.4f}, RQ {op['rq']:.4f}, TP {op['tp']}, FP {op['fp']}, "
      f"FN {op['fn']}, matched-IoU sum {op['iou_sum']:.2f}, denominator "
      f"{op['denominator']:.1f}, over {op['total_gt']} GT rows / 1154 annotator entries "
      f"/ 707 images. Kept candidates: {op['n_kept']}.\n")
    A("Every candidate is scored once per annotator copy of its image, so one kept "
      "candidate on a 3-annotator image occupies 3 GT-copy slots and can contribute up "
      "to 3 FP.\n")

    A("\n## 1. Anatomy of the 2461 false positives\n")
    A(md_table(c1, ["fp_class", "n_kept", "share_of_kept", "fp_copies", "share_of_FP",
                    "tp_copies", "gt_slots", "precision_copies"]))
    A("\n`fp_copies = n_copies - n_entries_hit` summed per class; the three classes "
      "partition the 4441 kept candidates and their FP sum is exactly 2461.\n")
    A("\nHow close the FP predictions came to any GT in the same entry "
      "(best IoU with any GT of that annotator):\n")
    A(md_table(pd.DataFrame(res["fp_best_iou_bins"])))

    A("\n## 2. Feature separation (medians per class, Cohen's d ranked)\n")
    A(md_table(feat_tbl, ["feature", "median_pure_hallucination",
                          "median_partial_consensus", "median_clean_tp",
                          "d_pure_vs_cleanTP", "d_partial_vs_cleanTP"]))
    A("\nPositive d = larger in pure hallucinations than in clean TPs.\n")

    A("\n## 3. Spatial / context\n")
    A(md_table(ctx_tables, ["dimension", "level", "n_kept", "n_images", "n_pure",
                            "pure_share_of_kept", "fp_copies", "fp_per_image",
                            "precision_copies", "share_of_all_FP"]))
    A("\n**Interpretation warning on `pure_share_of_kept` vs n_copies.** "
      "\"Pure hallucination\" is defined against the *union* of the annotators on that "
      "image, so it is mechanically easier to be a pure hallucination on a 1-annotator "
      "image (one chance to be confirmed) than on a 3-annotator image (three chances). "
      "The annotator-count-neutral metric is `precision_copies` = matched copies / "
      "GT-copy slots, which is essentially flat and slightly *worse* on more-annotated "
      "images: 0.6824 (1 copy) / 0.6557 (2) / 0.6412 (3). Use `precision_copies` for "
      "cross-n_copies comparisons and `pure_share_of_kept` only within a fixed n_copies "
      "(see the cross-tab below, where the near-limb effect survives the control).\n")
    A("\n`n_images` is the number of images that own at least one kept candidate in that "
      "level, so `fp_per_image` is FP-copies per *contributing* image (denominators "
      "differ across levels of the same dimension for candidate-level splits such as "
      "radial_zone and lat_band).\n")
    A("\nRegime cross-tab -- pure-hallucination share by n_copies x radial zone "
      "(checks whether the near-limb effect is just the 1-annotator effect):\n")
    A(md_table(pd.DataFrame(res["regime_ncopies_x_radial"])))
    A("\n...and by n_copies x GT-activity quintile:\n")
    A(md_table(pd.DataFrame(res["regime_ncopies_x_gtquintile"])))
    A("\nPer-image correlations (686 images with >=1 kept candidate):\n")
    A(md_table(pd.DataFrame([dict(pair=k, **v) for k, v in
                             res["per_image_corr"].items()])))

    A("\n## 4. Is it darkness? Photometric separability\n")
    for k in ["pure_vs_cleanTP", "pure_vs_anyTP"]:
        d = res["darkness"][k]
        A(f"\n**{k}** (n={d['n']}, positives={d['n_pos']}):\n")
        rowsd = [dict(score=f, auc=d[f"auc_{f}"], auc_oriented=d[f"auc_{f}_oriented"])
                 for f in ["contrast", "depth", "fg_min", "edge_grad", "fg_mean",
                           "conf", "p_sel"]]
        rowsd.append(dict(score="logit(contrast,depth,fg_min) OOF",
                          auc=d["auc_logit_contrast_depth_fgmin_oof"],
                          auc_oriented=d["auc_logit_contrast_depth_fgmin_oof"]))
        rowsd.append(dict(score="logit(all 6 photometric) OOF",
                          auc=d["auc_logit_all_photometric_oof"],
                          auc_oriented=d["auc_logit_all_photometric_oof"]))
        rowsd.append(dict(score="logit(conf,p_sel) OOF",
                          auc=d["auc_logit_conf_psel_oof"],
                          auc_oriented=d["auc_logit_conf_psel_oof"]))
        rowsd.append(dict(score="logit(conf,p_sel + 6 photometric) OOF",
                          auc=d["auc_logit_conf_psel_plus_photometric_oof"],
                          auc_oriented=d["auc_logit_conf_psel_plus_photometric_oof"]))
        A(md_table(pd.DataFrame(rowsd)))
        A(f"\nIncremental AUC from photometry on top of (conf, p_sel): "
          f"**{d['auc_gain_from_photometry']:+.4f}** "
          f"({d['auc_logit_conf_psel_oof']:.4f} -> "
          f"{d['auc_logit_conf_psel_plus_photometric_oof']:.4f}, "
          f"5-fold GroupKFold by image).\n")
    A("\nPure-hallucination rate by contrast decile (kept set):\n")
    A(md_table(pd.DataFrame(res["darkness"]["contrast_decile_pure_rate"])))

    A("\n## 5. Scorecard -- absolute PQ loss by stratum\n")
    A("Bin edges come from the GT distribution and are applied to GT rows (TP/FN) and "
      "to kept candidates (FP) alike; `category` is a GT-only property so all FP land "
      "in `unassigned_pred`. Every dimension reconstructs TP=4768/FP=2461/FN=3431.\n")
    A(md_table(sc, ["dimension", "level", "n_gt", "tp", "fp", "fn", "sq", "rq_stratum",
                    "pq_stratum", "recall_copies", "lost_mass", "lost_mass_share",
                    "delta_pq_if_perfect"]))
    A("\n`lost_mass = 0.5*FP + 0.5*FN` (total 2946.0). `delta_pq_if_perfect` = global "
      "PQ change if that stratum were solved perfectly (all its FN recovered at IoU 1, "
      "all its FP removed).\n")
    A("\nSupplementary -- the same 2461 FP attributed to the category of the GT instance "
      "they overlap most inside their own entry (`no_gt_overlap` = best IoU < 0.10):\n")
    A(md_table(pd.DataFrame(res["fp_by_best_overlap_gt_category"])))
    A("\nGT vs kept-candidate marginals (quantiles 0/10/25/50/75/90/100):\n")
    A(md_table(pd.DataFrame([dict(quantity=b["quantity"],
                                  gt=[round(v, 3) for v in b["gt"]],
                                  kept_cand=[round(v, 3) for v in b["kept_cand"]])
                             for b in res["gt_vs_pred_marginals"]])))
    A(f"\nCaveat: {res['width_proxy_caveat']}\n")
    A("\nGT-side cross-annotator support (multi-copy files only, TP/FN only):\n")
    A(md_table(st))

    A("\n## 6. SQ losses\n")
    A(md_table(sq_tbl))
    A("\nCounterfactuals:\n")
    rows = []
    for k, v in res["sq_scenarios"].items():
        rows.append(dict(scenario=k, pq=v.get("pq"), delta_pq=v.get("delta_pq"),
                         extra=json.dumps({kk: vv for kk, vv in v.items()
                                           if kk not in ("pq", "delta_pq")})))
    A(md_table(pd.DataFrame(rows)))
    A(f"\nUnmatched GT (FN) best-IoU with any kept prediction: "
      f"{json.dumps({k: round(v, 4) for k, v in res['fn_best_iou'].items()})}; "
      f"share of FN with zero overlap = {res['fn_best_iou_zero_share']:.4f}.\n")

    A("\n## Figures\n")
    A("- `error_profile_fp_anatomy.png` -- FP decomposition + Cohen's d ranking")
    A("- `error_profile_fp_context.png` -- hallucination rate by radius/latitude/density/site")
    A("- `error_profile_fp_scorecard.png` -- lost denominator mass + matched-IoU boxes")
    (OUT / "error_profile_fp.md").write_text("\n".join(L), encoding="utf-8")
    log("wrote error_profile_fp.md")


if __name__ == "__main__":
    main()
