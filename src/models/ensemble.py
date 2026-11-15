"""5-fold instance ensemble: cross-model mask clustering + consensus fusion.

Per image: pool all 5 models' raw candidates -> union-find clusters on mask
IoU>0.5 -> per cluster: support (#models), score = mean_conf * sqrt(support/5),
mask = conf-weighted pixel mean >= 0.5 -> greedy de-overlap by score -> selector
features + p_sel -> policy filter.

Calibration on the train split is mildly OPTIMISTIC (each model saw 4/5 of these
images in training) — mitigated by choosing mid-plateau policies; the honest
reference remains the pooled single-model OOF (0.4003 base / 0.4105 selector).

Usage:
  python src/models/ensemble.py calibrate     # train split: label + policy grid
  python src/models/ensemble.py submit --out outputs/submissions/subN.csv \
      [--score-min S] [--p-min P] [--support-min K]
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament import IMG_H, IMG_W
from filament.pq import score, _entry_iou_matrix
from filament.rle import decode_rle, encode_mask, write_submission
from selector import FEATURES, mask_features, fast_pq

R_MAX = 0.97
CACHE = ROOT / "outputs" / "oof_cache"
OUT = ROOT / "outputs" / "selector"


def log(*a):
    print(*a, flush=True)


def load_split(split, models=None):
    """models: list of cache ids (default: the 5 fold models '0'..'4')."""
    ids = [str(m) for m in (models if models is not None else range(5))]
    dfs = []
    for mid in ids:
        d = pd.read_parquet(CACHE / f"cross_m{mid}_{split}.parquet")
        d["model"] = mid
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    return df[(df["conf"] >= 0.05) & (df["r_norm"] <= R_MAX)]


def ensemble_image(g, n_models=5):
    """g: candidates of one image (all models). Returns list of cluster dicts."""
    g = g.reset_index(drop=True)
    masks = np.stack([decode_rle(r) for r in g["rle"]]).astype(np.uint8)
    iou = _entry_iou_matrix(masks, masks)
    parent = list(range(len(g)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    idx_i, idx_j = np.nonzero(iou > 0.5)
    for i, j in zip(idx_i, idx_j):
        if i < j:
            ri, rj = find(int(i)), find(int(j))
            if ri != rj:
                parent[rj] = ri
    clusters = defaultdict(list)
    for i in range(len(g)):
        clusters[find(i)].append(i)

    out = []
    for members in clusters.values():
        sub = g.iloc[members]
        support = int(sub["model"].nunique())
        confs = sub["conf"].to_numpy()
        w = confs / confs.sum()
        acc = np.tensordot(w, masks[members].astype(np.float32), axes=1)
        mask = acc >= 0.5
        if not mask.any():
            mask = masks[members[int(np.argmax(confs))]].astype(bool)
        sc = float(confs.mean()) * float(np.sqrt(support / n_models))
        out.append(dict(support=support, mean_conf=float(confs.mean()),
                        max_conf=float(confs.max()), score=sc, mask=mask))
    out.sort(key=lambda d: -d["score"])
    # greedy de-overlap by score
    taken = np.zeros((IMG_H, IMG_W), dtype=bool)
    final = []
    for d in out:
        mm = d["mask"] & ~taken
        if not mm.any():
            continue
        taken |= mm
        d["mask"] = mm
        final.append(d)
    return final


def build_candidates(split, models=None):
    df = load_split(split, models)
    n_models = df["model"].nunique()
    sel = joblib.load(OUT / "selector_v2.joblib")
    model, feats = sel["model"], sel["features"]
    rows = []
    t0 = time.time()
    for n, (stem, g) in enumerate(df.groupby("stem")):
        clusters = ensemble_image(g, n_models=n_models)
        feat_rows = []
        for k, d in enumerate(clusters, start=1):
            f = mask_features(d["mask"])
            feat_rows.append(dict(stem=stem, cand_id=f"{stem}_{k}",
                                  conf=d["mean_conf"], score=d["score"],
                                  support=d["support"], max_conf=d["max_conf"],
                                  rle_deov=encode_mask(d["mask"].astype(np.uint8)), **f))
        for i, fr in enumerate(feat_rows):
            fr["n_cands_img"] = len(feat_rows)
            fr["rank_conf"] = i + 1.0
        rows.extend(feat_rows)
        if (n + 1) % 100 == 0:
            log(f"  {split} {n+1} images ({time.time()-t0:.0f}s)")
    cand = pd.DataFrame(rows)
    cand["log10_area"] = np.log10(cand["area"])
    X = cand[list(feats)].to_numpy(np.float64)
    cand["p_sel"] = model.predict_proba(X)[:, 1]
    return cand


def calibrate(models=None, tag=""):
    cand = build_candidates("train", models)
    cand.drop(columns=["rle_deov"]).to_parquet(OUT / f"ensemble_train_candidates_meta{tag}.parquet")
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    pred_df = cand[["cand_id", "rle_deov"]].rename(
        columns={"cand_id": "filament_id", "rle_deov": "segmentation_rle"})
    res = score(gt[["filament_id", "segmentation_rle"]], pred_df, return_details=True)
    log(f"ensemble ALL: PQ={res['pq']:.4f} TP={res['tp']} FP={res['fp']} FN={res['fn']}")
    m = res["matches"]
    agg = m.groupby("pred_filament_id").agg(weight=("gt_filament_id", "size"),
                                            n_entries_hit=("annotator_image", "nunique"),
                                            iou_sum=("iou", "sum"))
    cand = cand.merge(agg, left_on="cand_id", right_index=True, how="left")
    for c in ["weight", "n_entries_hit", "iou_sum"]:
        cand[c] = cand[c].fillna(0)
    n_copies = gt.groupby("file_name")["entry_id"].nunique()
    stem2copies = {fn.rsplit(".", 1)[0]: int(v) for fn, v in n_copies.items()}
    cand["n_copies"] = cand["stem"].map(stem2copies)
    total_gt = len(gt)

    results = {}
    for smin in [0.10, 0.15, 0.20, 0.25, 0.30]:
        for pmin in [0.0, 0.2, 0.3, 0.4]:
            for kmin in [1, 2, 3]:
                k = cand[(cand["score"] >= smin) & (cand["p_sel"] >= pmin)
                         & (cand["support"] >= kmin)]
                r = fast_pq(k, total_gt)
                results[f"s{smin}_p{pmin}_k{kmin}"] = r
    top = sorted(results.items(), key=lambda kv: -kv[1]["pq"])[:10]
    for name, r in top:
        log(f"{name}: PQ={r['pq']:.4f} TP={r['tp']} FP={r['fp']} FN={r['fn']} kept={r['n_kept']}")
    with open(OUT / f"ensemble_calibration{tag}.json", "w") as f:
        json.dump({k: v for k, v in results.items()}, f, indent=1)


def submit(out_csv, score_min, p_min, support_min, models=None):
    cand = build_candidates("test", models)
    keep = cand[(cand["score"] >= score_min) & (cand["p_sel"] >= p_min)
                & (cand["support"] >= support_min)]
    log(f"test: {len(cand)} clusters -> {len(keep)} kept over {keep['stem'].nunique()} images")
    preds = defaultdict(list)
    for r in keep.itertuples():
        preds[r.stem].append(decode_rle(r.rle_deov))
    df = write_submission(dict(preds), ROOT / out_csv)
    log(f"wrote {len(df)} rows -> {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["calibrate", "submit"])
    ap.add_argument("--out", default="outputs/submissions/sub_ensemble.csv")
    ap.add_argument("--score-min", type=float, default=0.15)
    ap.add_argument("--p-min", type=float, default=0.30)
    ap.add_argument("--support-min", type=int, default=2)
    ap.add_argument("--models", default=None, help="comma list of cross-cache ids")
    a = ap.parse_args()
    models = a.models.split(",") if a.models else None
    if a.mode == "calibrate":
        calibrate(models, tag=("_8m" if models else ""))
    else:
        submit(a.out, a.score_min, a.p_min, a.support_min, models)
