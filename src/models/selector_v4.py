"""Selector v4: add IMAGE-APPEARANCE features (physics the model never saw).

Filaments are H-alpha absorption structures — genuinely darker than the
surrounding chromosphere. The v2/v3 selectors used only shape/position/conf, so
a bright or flat candidate looked identical to a real dark filament. This adds
per-candidate photometric contrast features computed from the original image:

    fg_mean/median/min   intensity inside the mask
    bg_mean              intensity in a dilated ring around it (local background)
    contrast             (bg_mean - fg_mean) / bg_mean       (positive = darker)
    depth                (disk_median - fg_mean) / disk_median
    fg_std, ring_std     texture terms
    edge_gradient        mean |grad| on the mask boundary

Evaluated exactly like v2/v3: GroupKFold-by-image OOF AUC + cross-fitted policy
gain against the conf/area baseline, using the verified fast-PQ decomposition.
"""

import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament.rle import decode_rle
from selector import FEATURES as BASE_FEATURES, fast_pq

OUT = ROOT / "outputs" / "selector"
IMG_DIR = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
CACHE_DIR = ROOT / "outputs" / "oof_cache"
R_MAX = 0.97
DISK_C, DISK_R = 1023.5, 900.0

APPEARANCE = ["fg_mean", "fg_median", "fg_min", "bg_mean", "contrast", "depth",
              "fg_std", "ring_std", "edge_grad"]
FEATURES_V4 = BASE_FEATURES + APPEARANCE


def log(*a):
    print(*a, flush=True)


def disk_median(img):
    ys, xs = np.mgrid[0:2048:8, 0:2048:8]
    r = np.hypot(xs - DISK_C, ys - DISK_C) / DISK_R
    vals = img[::8, ::8][r < 0.9]
    return float(np.median(vals))


def appearance_feats(img, gy, gx, mask_bbox, bbox):
    """mask_bbox: bool crop; bbox: (y0,y1,x0,x1) in full-image coords."""
    y0, y1, x0, x1 = bbox
    pad = 12
    py0, py1 = max(0, y0 - pad), min(2048, y1 + pad)
    px0, px1 = max(0, x0 - pad), min(2048, x1 + pad)
    win = img[py0:py1, px0:px1].astype(np.float32)
    m = np.zeros(win.shape, dtype=bool)
    m[y0 - py0:y1 - py0, x0 - px0:x1 - px0] = mask_bbox
    if m.sum() == 0:
        return None
    ring = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=2).astype(bool) & ~m
    fg = win[m]
    bg = win[ring] if ring.sum() > 10 else fg
    grad = np.hypot(gy[py0:py1, px0:px1], gx[py0:py1, px0:px1]).astype(np.float32)
    edge = m & ~cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    bg_mean = float(bg.mean())
    fg_mean = float(fg.mean())
    return dict(fg_mean=fg_mean, fg_median=float(np.median(fg)), fg_min=float(fg.min()),
                bg_mean=bg_mean,
                contrast=(bg_mean - fg_mean) / max(bg_mean, 1e-3),
                fg_std=float(fg.std()), ring_std=float(bg.std()),
                edge_grad=float(grad[edge].mean()) if edge.sum() else 0.0)


def build():
    cand = pd.read_parquet(OUT / "pooled_candidates.parquet")
    # rebuild the de-overlapped masks exactly as pooled_oof did
    log(f"{len(cand)} pooled candidates; recomputing appearance features")
    rows = []
    t0 = time.time()
    for fold in range(5):
        preds = pd.read_parquet(CACHE_DIR / f"fold{fold}_preds_e013.parquet")
        preds = preds[(preds["conf"] >= 0.05) & (preds["r_norm"] <= R_MAX)]
        for stem, g in preds.groupby("stem"):
            img = cv2.imread(str(IMG_DIR / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
            dmed = disk_median(img)
            gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
            gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
            g = g.sort_values("conf", ascending=False)
            taken = np.zeros((2048, 2048), dtype=bool)
            k = 0
            for r in g.itertuples():
                mm = decode_rle(r.rle).astype(bool) & ~taken
                if not mm.any():
                    continue
                taken |= mm
                k += 1
                ys, xs = np.nonzero(mm)
                bbox = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
                f = appearance_feats(img, gy, gx, mm[bbox[0]:bbox[1], bbox[2]:bbox[3]], bbox)
                if f is None:
                    continue
                f["depth"] = (dmed - f["fg_mean"]) / max(dmed, 1e-3)
                rows.append(dict(cand_id=f"{stem}_{k}", **f))
        log(f"  fold {fold}: {len(rows)} rows ({time.time()-t0:.0f}s)")
    app = pd.DataFrame(rows)
    out = cand.merge(app, on="cand_id", how="left")
    for c in APPEARANCE:
        out[c] = out[c].fillna(out[c].median())
    out.to_parquet(OUT / "pooled_candidates_v4.parquet", index=False)
    return out


def evaluate(cand):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    gt_n = len(pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet"))
    groups = cand["stem"].to_numpy()
    y = cand["y"].to_numpy()

    def mk():
        return HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=25, l2_regularization=1.0,
            early_stopping=False, random_state=42)

    for name, feats in [("v2 (shape only)", BASE_FEATURES), ("v4 (+appearance)", FEATURES_V4)]:
        X = cand[feats].to_numpy(np.float64)
        p = np.full(len(cand), np.nan)
        for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
            p[va] = mk().fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
        cand[f"p_{name[:2]}"] = p
        auc = roc_auc_score(y, p)
        log(f"{name}: OOF AUC {auc:.4f}")

    # cross-fitted policy gain for each
    files = np.array(sorted(cand["stem"].unique()))
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    stems_gt = gt["file_name"].str.replace(".jpeg", "", regex=False)
    rng = np.random.default_rng(11)
    grid = [(c, p) for c in [0.20, 0.25, 0.30, 0.35] for p in np.arange(0.05, 0.60, 0.05)]
    for col, label in [("p_v2", "v2 (shape only)"), ("p_v4", "v4 (+appearance)")]:
        gains, abs_pqs = [], []
        for rep in range(10):
            perm = rng.permutation(len(files))
            half = set(files[perm[:len(files) // 2]])
            for tr_h, ev_h in [(half, set(files) - half), (set(files) - half, half)]:
                ctr, cev = cand[cand["stem"].isin(tr_h)], cand[cand["stem"].isin(ev_h)]
                gt_tr = int(stems_gt.isin(tr_h).sum())
                gt_ev = gt_n - gt_tr
                best, best_pq = None, -1
                for c, p in grid:
                    k = ctr[(ctr["conf"] >= c) & (ctr[col] >= p)]
                    pq = fast_pq(k, gt_tr)["pq"]
                    if pq > best_pq:
                        best, best_pq = (c, p), pq
                k_ev = cev[(cev["conf"] >= best[0]) & (cev[col] >= best[1])]
                base = cev[(cev["conf"] >= 0.40) & (cev["area"] >= 128)]
                pq_ev = fast_pq(k_ev, gt_ev)["pq"]
                gains.append(pq_ev - fast_pq(base, gt_ev)["pq"])
                abs_pqs.append(pq_ev)
        log(f"{label}: cross-fitted gain {np.mean(gains):+.4f} +/- {np.std(gains):.4f} "
            f"| abs PQ {np.mean(abs_pqs):.4f}")

    # full-data best policy for each (reporting)
    for col in ["p_v2", "p_v4"]:
        best, best_pq = None, -1
        for c, p in grid:
            k = cand[(cand["conf"] >= c) & (cand[col] >= p)]
            pq = fast_pq(k, gt_n)["pq"]
            if pq > best_pq:
                best, best_pq = (c, p), pq
        log(f"{col}: pooled best policy conf>={best[0]} p>={best[1]:.2f} -> PQ {best_pq:.4f}")

    X = cand[FEATURES_V4].to_numpy(np.float64)
    joblib.dump({"model": mk().fit(X, y), "features": FEATURES_V4},
                OUT / "selector_v4.joblib")
    cand.drop(columns=[c for c in cand.columns if c.startswith("rle")], errors="ignore") \
        .to_parquet(OUT / "pooled_candidates_v4.parquet", index=False)


if __name__ == "__main__":
    path = OUT / "pooled_candidates_v4.parquet"
    if path.exists() and "fg_mean" in pd.read_parquet(path, columns=None).columns:
        cand = pd.read_parquet(path)
        log(f"reusing {path.name}")
    else:
        cand = build()
    evaluate(cand)
