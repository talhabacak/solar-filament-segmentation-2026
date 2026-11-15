"""Component-level candidate selector on fold-0 OOF candidates of E013.

Pipeline:
  1. Reproduce the postprocess baseline de-overlap: per stem sort by conf desc,
     greedy pixel de-overlap, keep conf>=0.05 & r_norm<=0.97 (no area/conf cut).
  2. Label each surviving de-overlapped candidate with the official matcher:
     weight = #GT annotator copies matched with IoU>0.5, y = (weight >= 1).
  3. Shape/position/context features per candidate.
  4. HistGradientBoostingClassifier, 5-fold GroupKFold by file_name (honest
     internal OOF within fold 0). AUC + permutation importances on held-out.
  5. PQ impact with the exact scorer: baseline vs selector sweep vs hybrid
     sweep vs oracle ceiling.

Run:  python src/models/selector.py
Outputs under outputs/selector/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament import IMG_H, IMG_W
from filament.pq import score
from filament.rle import decode_rle, encode_mask

DISK_C = 1023.5
DISK_R = 900.0
R_MAX = 0.97
OUT_DIR = ROOT / "outputs" / "selector"
CACHE = ROOT / "outputs" / "oof_cache" / "fold0_preds_e013.parquet"

FEATURES = ["conf", "log10_area", "r_norm", "abs_lat", "bbox_aspect",
            "eccentricity", "boundary_complexity", "skel_len", "width_proxy",
            "n_cands_img", "rank_conf"]


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- stage 1+2
def mask_features(mm: np.ndarray) -> dict:
    """Features from a de-overlapped boolean mask (computed on the bbox crop)."""
    from maskops import nonzero_boxed
    ys, xs = nonzero_boxed(mm)          # np.nonzero(mm), element for element, without the full scan
    area = int(len(ys))
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    h = int(y1 - y0 + 1)
    w = int(x1 - x0 + 1)
    crop = mm[y0:y1 + 1, x0:x1 + 1]

    cy, cx = float(ys.mean()), float(xs.mean())
    r_norm = float(np.hypot(cx - DISK_C, cy - DISK_C) / DISK_R)
    sin_lat = np.clip((DISK_C - cy) / DISK_R, -1.0, 1.0)
    abs_lat = float(abs(np.degrees(np.arcsin(sin_lat))))

    # eccentricity: PCA of pixel coords (subsample if huge)
    if area > 20000:
        idx = np.random.default_rng(0).choice(area, 20000, replace=False)
        pts = np.stack([ys[idx], xs[idx]], axis=1).astype(np.float32)
    else:
        pts = np.stack([ys, xs], axis=1).astype(np.float32)
    pts -= pts.mean(axis=0)
    if area >= 3:
        cov = (pts.T @ pts) / len(pts)
        ev = np.linalg.eigvalsh(cov)          # ascending
        ecc = float(np.sqrt(max(0.0, 1.0 - ev[0] / ev[1]))) if ev[1] > 0 else 0.0
    else:
        ecc = 0.0

    # boundary complexity: perimeter of the largest contour, ^2 / (4 pi A)
    cnts, _ = cv2.findContours(crop.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        perim = float(cv2.arcLength(c, True))
    else:                                     # degenerate (single pixel line)
        perim = 2.0 * (h + w)
    complexity = perim * perim / (4.0 * np.pi * max(area, 1))

    # skeleton length on the bbox crop
    from skimage.morphology import skeletonize
    skel_len = int(skeletonize(crop).sum())
    width_proxy = area / max(skel_len, 1)

    return dict(area=area, r_norm=r_norm, abs_lat=abs_lat,
                bbox_aspect=max(h, w) / max(min(h, w), 1),
                eccentricity=ecc, boundary_complexity=complexity,
                skel_len=skel_len, width_proxy=width_proxy,
                cy=cy, cx=cx)


def build_candidates() -> pd.DataFrame:
    """Greedy de-overlap of all conf>=0.05 & r_norm<=0.97 candidates + features."""
    preds = pd.read_parquet(CACHE)
    n0 = len(preds)
    preds = preds[(preds["conf"] >= 0.05) & (preds["r_norm"] <= R_MAX)]
    log(f"candidates: {n0} raw -> {len(preds)} after conf>=0.05 & r_norm<={R_MAX}")

    rows = []
    for stem, g in preds.groupby("stem"):
        g = g.sort_values("conf", ascending=False)
        taken = np.zeros((IMG_H, IMG_W), dtype=bool)
        k = 0
        for r in g.itertuples():
            m = decode_rle(r.rle).astype(bool)
            mm = m & ~taken
            a = int(mm.sum())
            if a == 0:
                continue
            taken |= mm
            k += 1
            f = mask_features(mm)
            rows.append(dict(stem=stem, cand_id=f"{stem}_{k}", conf=float(r.conf),
                             raw_area=int(r.area), rle_deov=encode_mask(mm.astype(np.uint8)),
                             meta_r_norm=float(r.r_norm), **f))
        # per-image context
        for i in range(len(rows) - k, len(rows)):
            rows[i]["n_cands_img"] = k
    df = pd.DataFrame(rows)
    df["rank_conf"] = df.groupby("stem")["conf"].rank(ascending=False, method="first")
    df["log10_area"] = np.log10(df["area"])
    log(f"de-overlapped survivors: {len(df)} over {df['stem'].nunique()} stems "
        f"(dropped {len(preds) - len(df)} fully-covered)")
    return df


def load_gt_val():
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    val_files = sorted(folds.loc[folds["fold"] == 0, "file_name"])
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt_val = gt[gt["file_name"].isin(val_files)][
        ["filament_id", "segmentation_rle", "entry_id", "file_name"]].reset_index(drop=True)
    log(f"fold0 val: {len(val_files)} files, {len(gt_val)} GT rows, "
        f"{gt_val['entry_id'].nunique()} annotator entries")
    return gt_val, val_files


def label_candidates(cand: pd.DataFrame, gt_val: pd.DataFrame):
    """Official matcher on ALL de-overlapped candidates -> per-candidate labels."""
    pred_df = cand[["cand_id", "rle_deov"]].rename(
        columns={"cand_id": "filament_id", "rle_deov": "segmentation_rle"})
    t0 = time.time()
    res = score(gt_val[["filament_id", "segmentation_rle"]], pred_df,
                return_details=True)
    log(f"all-candidates score: PQ={res['pq']:.4f} TP={res['tp']} FP={res['fp']} "
        f"FN={res['fn']}  ({time.time() - t0:.0f}s)")
    m = res["matches"]
    if len(m):
        agg = m.groupby("pred_filament_id").agg(
            weight=("gt_filament_id", "size"),
            n_entries_hit=("annotator_image", "nunique"),
            iou_sum=("iou", "sum"),
            iou_max=("iou", "max"))
    else:
        agg = pd.DataFrame(columns=["weight", "n_entries_hit", "iou_sum", "iou_max"])
    cand = cand.merge(agg, left_on="cand_id", right_index=True, how="left")
    for c in ["weight", "n_entries_hit", "iou_sum", "iou_max"]:
        cand[c] = cand[c].fillna(0)
    cand["weight"] = cand["weight"].astype(int)
    cand["n_entries_hit"] = cand["n_entries_hit"].astype(int)
    cand["y"] = (cand["weight"] >= 1).astype(int)

    n_copies = gt_val.groupby("file_name")["entry_id"].nunique()
    stem2copies = {fn.rsplit(".", 1)[0]: int(v) for fn, v in n_copies.items()}
    cand["n_copies"] = cand["stem"].map(stem2copies)
    log(f"labels: {cand['y'].sum()}/{len(cand)} positive "
        f"({cand['y'].mean():.3f}); weight dist {cand['weight'].value_counts().sort_index().to_dict()}")
    return cand, res


# ---------------------------------------------------------------- fast PQ
def fast_pq(kept: pd.DataFrame, total_gt: int):
    """Exact PQ from per-candidate match stats (valid because preds are
    pixel-disjoint: each candidate's matches are independent of the kept set,
    and one GT row can match at most one disjoint pred at IoU>0.5)."""
    tp = int(kept["weight"].sum())
    iou_sum = float(kept["iou_sum"].sum())
    fp = int((kept["n_copies"] - kept["n_entries_hit"]).sum())
    fn = total_gt - tp
    den = tp + 0.5 * fp + 0.5 * fn
    return dict(pq=iou_sum / den if den else 0.0,
                sq=iou_sum / tp if tp else 0.0,
                rq=tp / den if den else 0.0, tp=tp, fp=fp, fn=fn,
                n_kept=len(kept))


def exact_pq(kept: pd.DataFrame, gt_val: pd.DataFrame):
    pred_df = kept[["cand_id", "rle_deov"]].rename(
        columns={"cand_id": "filament_id", "rle_deov": "segmentation_rle"})
    return score(gt_val[["filament_id", "segmentation_rle"]], pred_df)


# ---------------------------------------------------------------- stage 3
def train_selector(cand: pd.DataFrame):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    X = cand[FEATURES].to_numpy(dtype=np.float64)
    y = cand["y"].to_numpy()
    groups = cand["stem"].to_numpy()

    def make_model():
        return HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=25, l2_regularization=1.0,
            early_stopping=False, random_state=42)

    gkf = GroupKFold(n_splits=5)
    p_oof = np.full(len(cand), np.nan)
    fold_aucs, perms = [], []
    for i, (tr, va) in enumerate(gkf.split(X, y, groups)):
        mdl = make_model().fit(X[tr], y[tr])
        p_oof[va] = mdl.predict_proba(X[va])[:, 1]
        auc = roc_auc_score(y[va], p_oof[va])
        fold_aucs.append(auc)
        pi = permutation_importance(mdl, X[va], y[va], scoring="roc_auc",
                                    n_repeats=8, random_state=0)
        perms.append(pi.importances_mean)
        log(f"  internal fold {i}: n_val={len(va)} AUC={auc:.4f}")

    pooled_auc = roc_auc_score(y, p_oof)
    imp = pd.Series(np.mean(perms, axis=0), index=FEATURES).sort_values(ascending=False)
    log(f"pooled internal-OOF AUC={pooled_auc:.4f}  "
        f"(per-fold {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f})")
    log("permutation importance (held-out, mean over folds):")
    for k, v in imp.items():
        log(f"  {k:>20s}  {v:+.4f}")

    final_model = make_model().fit(X, y)
    return p_oof, pooled_auc, fold_aucs, imp, final_model


# ---------------------------------------------------------------- stage 4
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(0)

    gt_val, _ = load_gt_val()
    total_gt = len(gt_val)

    cand = build_candidates()
    cand, _allres = label_candidates(cand, gt_val)

    # ---- baseline reproduction check
    base_kept = cand[(cand["conf"] >= 0.40) & (cand["area"] >= 128)]
    base_exact = exact_pq(base_kept, gt_val)
    base_fast = fast_pq(base_kept, total_gt)
    log(f"baseline conf>=0.40 area>=128: exact PQ={base_exact['pq']:.4f} "
        f"(fast {base_fast['pq']:.4f}) TP={base_exact['tp']} FP={base_exact['fp']} FN={base_exact['fn']}")
    assert abs(base_exact["pq"] - base_fast["pq"]) < 1e-6, "fast formula mismatch"
    ref = 0.4178
    if abs(base_exact["pq"] - ref) > 0.002:
        log(f"!! baseline off reference {ref} by {base_exact['pq'] - ref:+.4f} — STOP, "
            "check de-overlap ordering vs sweep_oof.sweep_fold")
        raise SystemExit(1)

    # ---- selector
    p_oof, pooled_auc, fold_aucs, imp, final_model = train_selector(cand)
    cand["p_selector"] = p_oof

    # ---- PQ sweeps (fast formula; exact-verified at the best points)
    rows = []
    ts = np.round(np.arange(0.05, 0.951, 0.05), 2)
    for t in ts:
        r = fast_pq(cand[cand["p_selector"] >= t], total_gt)
        rows.append(dict(policy="selector", t=float(t), **r))
    for t in ts:
        r = fast_pq(cand[(cand["conf"] >= 0.25) & (cand["p_selector"] >= t)], total_gt)
        rows.append(dict(policy="hybrid_c25", t=float(t), **r))
    oracle_kept = cand[cand["y"] == 1]
    rows.append(dict(policy="oracle", t=np.nan, **fast_pq(oracle_kept, total_gt)))
    rows.append(dict(policy="baseline", t=np.nan, **base_fast))
    sweep = pd.DataFrame(rows)

    best_sel = sweep[sweep["policy"] == "selector"].sort_values("pq").iloc[-1]
    best_hyb = sweep[sweep["policy"] == "hybrid_c25"].sort_values("pq").iloc[-1]

    # exact verification of the headline numbers
    ver_sel = exact_pq(cand[cand["p_selector"] >= best_sel["t"]], gt_val)
    ver_hyb = exact_pq(cand[(cand["conf"] >= 0.25) &
                            (cand["p_selector"] >= best_hyb["t"])], gt_val)
    ver_ora = exact_pq(oracle_kept, gt_val)
    log(f"exact verify: selector t={best_sel['t']:.2f} PQ={ver_sel['pq']:.4f} "
        f"(fast {best_sel['pq']:.4f}); hybrid t={best_hyb['t']:.2f} PQ={ver_hyb['pq']:.4f} "
        f"(fast {best_hyb['pq']:.4f}); oracle PQ={ver_ora['pq']:.4f}")
    for fast_v, ex_v in [(best_sel["pq"], ver_sel["pq"]), (best_hyb["pq"], ver_hyb["pq"])]:
        assert abs(fast_v - ex_v) < 1e-6, "fast/exact disagree"

    log("\nfull PQ table:")
    log(sweep.to_string(index=False))

    # ---- persist
    import joblib
    cand.to_parquet(OUT_DIR / "fold0_candidates_e013.parquet", index=False)
    sweep.to_csv(OUT_DIR / "pq_sweep.csv", index=False)
    joblib.dump(dict(model=final_model, features=FEATURES), OUT_DIR / "selector_hgb.joblib")

    results = dict(
        baseline=dict(policy="conf>=0.40 & area>=128", **{k: v for k, v in base_exact.items()}),
        best_selector=dict(t=float(best_sel["t"]), **{k: v for k, v in ver_sel.items()}),
        best_hybrid=dict(conf_min=0.25, t=float(best_hyb["t"]), **{k: v for k, v in ver_hyb.items()}),
        oracle=dict(**{k: v for k, v in ver_ora.items()}),
        internal_oof_auc=float(pooled_auc),
        fold_aucs=[float(a) for a in fold_aucs],
        permutation_importance=imp.to_dict(),
        n_candidates=int(len(cand)),
        n_positive=int(cand["y"].sum()),
        features=FEATURES,
    )
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))

    gain_sel = ver_sel["pq"] - base_exact["pq"]
    gain_hyb = ver_hyb["pq"] - base_exact["pq"]
    md = [
        "# Fold-0 component selector (E013 OOF candidates)",
        "",
        f"- candidates after de-overlap: {len(cand)} ({cand['y'].sum()} positive, {cand['y'].mean():.1%})",
        f"- internal 5-fold GroupKFold OOF AUC: **{pooled_auc:.4f}**",
        f"- baseline (conf>=0.40 & area>=128): **PQ {base_exact['pq']:.4f}** "
        f"(TP {base_exact['tp']} FP {base_exact['fp']} FN {base_exact['fn']})",
        f"- best selector-only (p>= {best_sel['t']:.2f}): **PQ {ver_sel['pq']:.4f}** ({gain_sel:+.4f})",
        f"- best hybrid (conf>=0.25 & p>= {best_hyb['t']:.2f}): **PQ {ver_hyb['pq']:.4f}** ({gain_hyb:+.4f})",
        f"- oracle ceiling (keep exactly y=1): **PQ {ver_ora['pq']:.4f}**",
        "",
        "## Permutation importance (held-out AUC drop)",
        imp.to_frame("importance").to_markdown(),
        "",
        "## Full sweep",
        sweep.to_markdown(index=False),
        "",
    ]
    (OUT_DIR / "results.md").write_text("\n".join(md))
    log(f"\nsaved -> {OUT_DIR}")
    log(f"HEADLINE: baseline {base_exact['pq']:.4f} | best selector "
        f"{max(ver_sel['pq'], ver_hyb['pq']):.4f} | oracle {ver_ora['pq']:.4f}")


if __name__ == "__main__":
    main()
