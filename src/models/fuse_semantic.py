"""Fuse detector + semantic candidate pools and measure the real PQ gain.

Measured motivation (fold 0): detector reachability 0.7093, semantic 0.6563,
UNION 0.7755 — the semantic pool reaches ~100 GT rows the detector never
proposes at any confidence, and its masks are slightly better (SQ 0.679 vs
0.671). This script clusters both pools together, fuses each cluster into one
mask, trains a source-aware selector on the fused pool (honest GroupKFold by
image), and sweeps the keep policy with the exact PQ scorer.

Usage:
  python src/models/fuse_semantic.py --fold 0 --sem-thr 0.3 --sem-min-area 256
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament import IMG_H, IMG_W
from filament.pq import score, _entry_iou_matrix
from filament.rle import decode_rle, encode_mask
from selector import mask_features, fast_pq
from semantic_candidates import prob_to_instances

CACHE = ROOT / "outputs" / "oof_cache"
OUT = ROOT / "outputs" / "semantic"
R_MAX = 0.97

BASE_FEATS = ["conf", "log10_area", "r_norm", "abs_lat", "bbox_aspect",
              "eccentricity", "boundary_complexity", "skel_len", "width_proxy",
              "n_cands_img", "rank_conf"]
SRC_FEATS = ["det_conf", "sem_conf", "from_det", "from_sem", "n_sources"]
FEATS = BASE_FEATS + SRC_FEATS


def log(*a):
    print(*a, flush=True)


def build_fused(fold, sem_thr, sem_min_area, det_tag="_e013comp"):
    probs = np.load(OUT / f"fold{fold}_probs.npz")
    det_path = CACHE / f"fold{fold}_preds{det_tag}.parquet"
    det = pd.read_parquet(det_path if det_path.exists()
                          else CACHE / f"fold{fold}_preds_e013.parquet")
    det = det[(det["conf"] >= 0.05) & (det["r_norm"] <= R_MAX)]
    if "view" in det.columns:
        det = det[det["view"] == "o"]

    rows = []
    t0 = time.time()
    stems = sorted(probs.files)
    for n, stem in enumerate(stems):
        items = []                                    # (source, conf, mask)
        g = det[det["stem"] == stem].sort_values("conf", ascending=False)
        for r in g.itertuples():
            items.append(("det", float(r.conf), decode_rle(r.rle).astype(bool)))
        prob = probs[stem].astype(np.float32) / 255.0
        for c in prob_to_instances(prob, sem_thr, sem_min_area):
            items.append(("sem", float(c["conf"]), c["mask"].astype(bool)))
        if not items:
            continue

        # cluster across sources by IoU > 0.5 (union-find)
        masks = np.stack([m for _, _, m in items]).astype(np.uint8)
        iou = _entry_iou_matrix(masks, masks)
        parent = list(range(len(items)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        ii, jj = np.nonzero(iou > 0.5)
        for i, j in zip(ii, jj):
            if i < j:
                ri, rj = find(int(i)), find(int(j))
                if ri != rj:
                    parent[rj] = ri
        clusters = {}
        for i in range(len(items)):
            clusters.setdefault(find(i), []).append(i)

        fused = []
        for members in clusters.values():
            srcs = {items[i][0] for i in members}
            confs = np.array([items[i][1] for i in members], dtype=np.float32)
            w = confs / confs.sum()
            acc = np.tensordot(w, masks[members].astype(np.float32), axes=1)
            m = acc >= 0.5
            if not m.any():
                m = masks[members[int(np.argmax(confs))]].astype(bool)
            det_c = max([items[i][1] for i in members if items[i][0] == "det"], default=0.0)
            sem_c = max([items[i][1] for i in members if items[i][0] == "sem"], default=0.0)
            fused.append(dict(mask=m, det_conf=det_c, sem_conf=sem_c,
                              from_det=int("det" in srcs), from_sem=int("sem" in srcs),
                              n_sources=len(srcs),
                              conf=max(det_c, sem_c)))
        fused.sort(key=lambda d: -d["conf"])

        taken = np.zeros((IMG_H, IMG_W), dtype=bool)
        kept = []
        for d in fused:                                # greedy de-overlap
            mm = d["mask"] & ~taken
            if not mm.any():
                continue
            taken |= mm
            d["mask"] = mm
            kept.append(d)
        for k, d in enumerate(kept, start=1):
            f = mask_features(d["mask"])
            if f["r_norm"] > R_MAX:
                continue
            rows.append(dict(stem=stem, cand_id=f"{stem}_{k}",
                             rle=encode_mask(d["mask"].astype(np.uint8)),
                             conf=d["conf"], det_conf=d["det_conf"], sem_conf=d["sem_conf"],
                             from_det=d["from_det"], from_sem=d["from_sem"],
                             n_sources=d["n_sources"], n_cands_img=len(kept),
                             rank_conf=float(k), **f))
        if (n + 1) % 50 == 0:
            log(f"  fused {n+1}/{len(stems)} images ({time.time()-t0:.0f}s)")
    cand = pd.DataFrame(rows)
    cand["log10_area"] = np.log10(cand["area"])
    return cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--sem-thr", type=float, default=0.3)
    ap.add_argument("--sem-min-area", type=int, default=256)
    ap.add_argument("--det-tag", default="_e013comp", help="detector cache tag")
    ap.add_argument("--out-tag", default="", help="suffix for result files")
    a = ap.parse_args()

    cand = build_fused(a.fold, a.sem_thr, a.sem_min_area, a.det_tag)
    log(f"fused candidates: {len(cand)} "
        f"(det-only {int(((cand.from_det==1)&(cand.from_sem==0)).sum())}, "
        f"sem-only {int(((cand.from_det==0)&(cand.from_sem==1)).sum())}, "
        f"both {int((cand.n_sources==2).sum())})")

    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    val_files = sorted(folds.loc[folds["fold"] == a.fold, "file_name"])
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt_val = gt[gt["file_name"].isin(val_files)]
    total_gt = len(gt_val)

    pred_df = cand[["cand_id", "rle"]].rename(
        columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
    res = score(gt_val[["filament_id", "segmentation_rle"]], pred_df, return_details=True)
    log(f"ALL fused: PQ={res['pq']:.4f} TP={res['tp']} FP={res['fp']} FN={res['fn']}")
    m = res["matches"]
    agg = m.groupby("pred_filament_id").agg(weight=("gt_filament_id", "size"),
                                            n_entries_hit=("annotator_image", "nunique"),
                                            iou_sum=("iou", "sum"))
    cand = cand.merge(agg, left_on="cand_id", right_index=True, how="left")
    for c in ["weight", "n_entries_hit", "iou_sum"]:
        cand[c] = cand[c].fillna(0)
    cand["y"] = (cand["weight"] >= 1).astype(int)
    n_copies = gt_val.groupby("file_name")["entry_id"].nunique()
    cand["n_copies"] = cand["stem"].map(
        {fn.rsplit(".", 1)[0]: int(v) for fn, v in n_copies.items()})
    log(f"labels: {int(cand['y'].sum())}/{len(cand)} positive; "
        f"sem-only positives {int(cand[(cand.from_sem==1)&(cand.from_det==0)]['y'].sum())}")

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    X = cand[FEATS].to_numpy(np.float64)
    y = cand["y"].to_numpy()
    groups = cand["stem"].to_numpy()

    def mk():
        return HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=25, l2_regularization=1.0,
            early_stopping=False, random_state=42)

    p = np.full(len(cand), np.nan)
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
        p[va] = mk().fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    cand["p_sel"] = p
    log(f"fused selector OOF AUC {roc_auc_score(y, p):.4f}")

    best, rowsout = None, []
    for conf_t in [0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4]:
        for p_t in np.arange(0.05, 0.65, 0.05):
            k = cand[(cand["conf"] >= conf_t) & (cand["p_sel"] >= p_t)]
            if k.empty:
                continue
            r = fast_pq(k, total_gt)
            rowsout.append(dict(conf=conf_t, p_sel=round(float(p_t), 2), **r))
            if best is None or r["pq"] > best[1]["pq"]:
                best = ((conf_t, round(float(p_t), 2)), r)
    df = pd.DataFrame(rowsout).sort_values("pq", ascending=False)
    df.to_csv(OUT / f"fold{a.fold}_fused_sweep.csv", index=False)
    log(f"\nbest fused policy conf>={best[0][0]} p_sel>={best[0][1]}: "
        f"PQ={best[1]['pq']:.4f} SQ={best[1]['sq']:.4f} TP={best[1]['tp']} "
        f"FP={best[1]['fp']} FN={best[1]['fn']}")
    log(df.head(8).to_string(index=False))

    # verify the best policy with the exact scorer (fast_pq is exact but verify anyway)
    k = cand[(cand["conf"] >= best[0][0]) & (cand["p_sel"] >= best[0][1])]
    pred_df = k[["cand_id", "rle"]].rename(
        columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
    r = score(gt_val[["filament_id", "segmentation_rle"]], pred_df)
    log(f"exact scorer check: PQ={r['pq']:.6f} (fast {best[1]['pq']:.6f})")
    kept_sem_only = int(k[(k.from_sem == 1) & (k.from_det == 0)].shape[0])
    log(f"kept sem-only candidates: {kept_sem_only} of {len(k)}")

    cand.drop(columns=["rle"]).to_parquet(OUT / f"fold{a.fold}_fused_candidates.parquet",
                                          index=False)
    with open(OUT / f"fold{a.fold}_fused_result.json", "w") as f:
        json.dump({"best_policy": {"conf": best[0][0], "p_sel": best[0][1]},
                   "best": best[1], "exact_check": r,
                   "n_candidates": int(len(cand)),
                   "kept_sem_only": kept_sem_only}, f, indent=1, default=float)


if __name__ == "__main__":
    main()
