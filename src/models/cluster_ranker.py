"""Pick the best member of a cluster instead of always emitting the consensus.

Measured headroom (fold 0, oracle selection on both sides so the selector is not
what is compared):

    consensus (what we ship)   0.5476
    largest member             0.5261
    highest-confidence member  0.5466
    best member (oracle)       0.5701

The cheap heuristics both lose, so the consensus is a good default and beating
it needs a learned choice — but the oracle says that choice is worth +0.0225,
which is the largest single lever still standing. With the leaderboard's real
shape (field tops out at 0.40, we are at 0.38) that is no longer a rounding
error; it is most of what separates us from the top of the legitimate field.

The discriminating evidence should be the semantic probability map: for two
candidate boundaries around the same filament, the one that agrees with the
dense-supervision lane is the better cut. So each option carries its own shape
features plus the semantic statistics under and around it, plus how it sits
inside its cluster.

Everything is out-of-fold by image, and the final number comes from the exact
scorer after de-overlap — fast_pq cannot be used because de-overlap changes the
masks after their labels were computed.

Usage: python src/models/cluster_ranker.py --folds 0,1,2,3,4
"""

import argparse
import json
import sys
import time
from collections import defaultdict
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
from selector import mask_features
from semantic_candidates import prob_to_instances, R_MAX

OUT = ROOT / "outputs" / "selector"
SEM_THR, SEM_MIN_AREA = 0.3, 256
RING_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))

FEATS = ["conf", "log10_area", "r_norm", "abs_lat", "bbox_aspect", "eccentricity",
         "boundary_complexity", "skel_len", "width_proxy",
         "is_det", "is_sem", "is_consensus",
         "n_members", "n_det_members", "n_sem_members",
         "cluster_conf_max", "cluster_conf_mean",
         "iou_with_consensus", "area_ratio_consensus", "area_rank", "conf_rank",
         "sem_prob_mean", "sem_prob_p90", "sem_frac_over_50", "sem_ring_mean",
         "sem_contrast"]


def log(*a):
    print(*a, flush=True)


def opt_id(stem, ci, key):
    """pq.py splits the image stem at the FIRST underscore, so the rest of the
    id must contain none or the prediction is attributed to no image."""
    return f"{stem}_{ci}x{key}"


def sem_stats(m, prob):
    ys, xs = np.nonzero(m)
    y0, y1 = max(0, ys.min() - 12), min(IMG_H, ys.max() + 13)
    x0, x1 = max(0, xs.min() - 12), min(IMG_W, xs.max() + 13)
    sub_m = m[y0:y1, x0:x1]
    sub_p = prob[y0:y1, x0:x1]
    ring = (cv2.dilate(sub_m.astype(np.uint8), RING_K) > 0) & ~sub_m
    pv = sub_p[sub_m]
    rv = sub_p[ring] if ring.sum() > 10 else pv
    return dict(sem_prob_mean=float(pv.mean()), sem_prob_p90=float(np.percentile(pv, 90)),
                sem_frac_over_50=float((pv >= 0.5).mean()),
                sem_ring_mean=float(rv.mean()),
                sem_contrast=float(pv.mean() - rv.mean()))


def build_fold(fold):
    """Every cluster's options (members + consensus) with features."""
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    stems = [f.rsplit(".", 1)[0]
             for f in sorted(folds.loc[folds["fold"] == fold, "file_name"])]
    det = pd.read_parquet(ROOT / "outputs" / "oof_cache" / f"fold{fold}_preds_e013.parquet")
    det = det[(det["conf"] >= 0.05) & (det["r_norm"] <= R_MAX)]
    if "view" in det.columns:
        det = det[det["view"] == "o"]
    z = np.load(ROOT / "outputs" / "semantic" / f"fold{fold}_probs_cldice.npz")

    rows, t0 = [], time.time()
    for n, stem in enumerate(stems):
        g = det[det["stem"] == stem].sort_values("conf", ascending=False)
        prob = z[stem].astype(np.float32) / 255.0
        items = [("det", float(c), decode_rle(r).astype(bool))
                 for c, r in zip(g["conf"], g["rle"])]
        items += [("sem", float(c["conf"]), c["mask"].astype(bool))
                  for c in prob_to_instances(prob, SEM_THR, SEM_MIN_AREA)]
        if not items:
            continue
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
        groups = defaultdict(list)
        for i in range(len(items)):
            groups[find(i)].append(i)

        for ci, members in enumerate(groups.values()):
            confs = np.array([items[i][1] for i in members], np.float32)
            acc = np.tensordot(confs / confs.sum(),
                               masks[members].astype(np.float32), axes=1)
            cons = acc >= 0.5
            if not cons.any():
                cons = masks[members[int(np.argmax(confs))]].astype(bool)
            n_det = sum(1 for i in members if items[i][0] == "det")
            opts = [("cons", cons, float(confs.max()), 0, 0, 1)]
            for k, i in enumerate(members):
                s, c, m = items[i]
                opts.append((f"m{k}", m, c, int(s == "det"), int(s == "sem"), 0))
            areas = np.array([o[1].sum() for o in opts], np.float64)
            oconf = np.array([o[2] for o in opts], np.float64)
            a_cons = max(float(cons.sum()), 1.0)
            for oi, (key, m, c, isd, iss, isc) in enumerate(opts):
                if m.sum() < 32:
                    continue
                f = mask_features(m)
                if f["r_norm"] > R_MAX:
                    continue
                inter = float(np.logical_and(m, cons).sum())
                union = float(np.logical_or(m, cons).sum())
                rows.append(dict(
                    opt_id=opt_id(stem, ci, key), stem=stem, fold=fold,
                    cluster=f"{stem}_{ci}", rle=encode_mask(m.astype(np.uint8)),
                    conf=c, is_det=isd, is_sem=iss, is_consensus=isc,
                    n_members=len(members), n_det_members=n_det,
                    n_sem_members=len(members) - n_det,
                    cluster_conf_max=float(confs.max()),
                    cluster_conf_mean=float(confs.mean()),
                    iou_with_consensus=inter / union if union else 0.0,
                    area_ratio_consensus=float(m.sum()) / a_cons,
                    area_rank=float((areas > m.sum()).sum()),
                    conf_rank=float((oconf > c).sum()),
                    **f, **sem_stats(m, prob)))
        if (n + 1) % 40 == 0:
            log(f"  fold {fold}: {n+1}/{len(stems)} images, {len(rows)} options "
                f"({time.time()-t0:.0f}s)")
    df = pd.DataFrame(rows)
    df["log10_area"] = np.log10(np.maximum(df["area"], 1))
    return df


def label(df, gt_cache):
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt_sub = gt[gt["file_name"].isin(set(df["stem"] + ".jpeg"))]
    pred = df[["opt_id", "rle"]].rename(
        columns={"opt_id": "filament_id", "rle": "segmentation_rle"})
    r = score(gt_sub[["filament_id", "segmentation_rle"]], pred,
              gt_cache=gt_cache, return_details=True)
    iou_sum = r["matches"].groupby("pred_filament_id")["iou"].sum()
    ncop = gt_sub.groupby("file_name")["entry_id"].nunique()
    df["iou_sum"] = df["opt_id"].map(iou_sum).fillna(0.0)
    df["n_copies"] = df["stem"].map(
        {f.rsplit(".", 1)[0]: int(v) for f, v in ncop.items()}).fillna(1)
    df["value"] = df["iou_sum"] / df["n_copies"].clip(lower=1)
    return df, gt_sub


def oof_rank(df):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    X = df[FEATS].to_numpy(np.float64)
    y = df["value"].to_numpy(np.float64)
    g = df["stem"].to_numpy()
    p = np.full(len(df), np.nan)
    for tr, va in GroupKFold(n_splits=5).split(X, y, g):
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                          max_leaf_nodes=31, min_samples_leaf=25,
                                          l2_regularization=1.0,
                                          early_stopping=False, random_state=42)
        p[va] = m.fit(X[tr], y[tr]).predict(X[va])
    return p


def evaluate(picked, gt_sub, gt_cache, col, q0=0.53):
    """Select clusters by value>PQ/2 on `col`, de-overlap, score exactly."""
    q = q0
    out = None
    for _ in range(4):
        keep = picked[picked[col] > q / 2.0].sort_values(col, ascending=False)
        rows = []
        for stem, g in keep.groupby("stem", sort=False):
            taken = np.zeros((IMG_H, IMG_W), bool)
            k = 0
            for r in g.itertuples():
                m = decode_rle(r.rle).astype(bool) & ~taken
                if m.sum() < 64:
                    continue
                taken |= m
                k += 1
                rows.append((f"{stem}_{k}", encode_mask(m.astype(np.uint8))))
        df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
        r = score(gt_sub[["filament_id", "segmentation_rle"]], df,
                  gt_cache=gt_cache, return_details=True)
        out = (r, len(df))
        if abs(r["pq"] - q) < 1e-3:
            break
        q = r["pq"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    a = ap.parse_args()

    cache = OUT / "cluster_options.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        log(f"reusing {len(df)} options from {cache.name}")
    else:
        parts = [build_fold(int(f)) for f in a.folds.split(",")]
        df = pd.concat(parts, ignore_index=True)
        df.to_parquet(cache, index=False)
        log(f"built {len(df)} options over {df['cluster'].nunique()} clusters")

    gt_cache = {}
    t0 = time.time()
    df, gt_sub = label(df, gt_cache)
    log(f"labelled ({time.time()-t0:.0f}s); {int((df['value']>0).sum())}/{len(df)} "
        f"options have value>0")

    df["p"] = oof_rank(df)
    from scipy.stats import spearmanr
    log(f"option ranker Spearman {spearmanr(df['p'], df['value']).statistic:.4f}")

    res = {}
    for name, pick_col in (("consensus (current)", None),
                           ("learned pick", "p"),
                           ("oracle pick", "value")):
        if pick_col is None:
            picked = df[df["is_consensus"] == 1].copy()
            picked["score"] = picked["p"]
        else:
            idx = df.groupby("cluster")[pick_col].idxmax()
            picked = df.loc[idx].copy()
            picked["score"] = picked["p"] if pick_col == "p" else picked["value"]
        r, n = evaluate(picked, gt_sub, gt_cache, "score")
        log(f"{name:20s} clusters {len(picked):5d} kept {n:5d}  PQ {r['pq']:.4f} "
            f"SQ {r['sq']:.4f} TP {r['tp']:5d} FP {r['fp']:5d} FN {r['fn']:5d}")
        res[name] = dict(clusters=len(picked), kept=n, pq=r["pq"], sq=r["sq"],
                         tp=r["tp"], fp=r["fp"], fn=r["fn"])

    with open(OUT / "cluster_ranker.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    log(f"saved -> {OUT/'cluster_ranker.json'}")


if __name__ == "__main__":
    main()
