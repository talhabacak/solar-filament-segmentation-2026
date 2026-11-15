"""Failure gallery: what we never see, and what we see badly.

Splits every GT instance (all 5 folds, detector lane at the standard operating
point) into three review groups and renders 20 examples of each as OUTLINE-ONLY
overlays (no fill) so the underlying H-alpha structure stays visible:

  NEVER      best IoU over the whole raw candidate pool < 0.10
             -> the model never proposes anything there at any confidence
  NEAR_MISS  0.10 <= best IoU <= 0.50
             -> proposed but never crosses the PQ matching threshold
  WORST_TP   matched (IoU > 0.50) but in the lowest IoU decile
             -> found, delineated badly; this is the SQ loss

Colours: GT = green, our prediction = red. Amber = other GT instances of the same
annotator entry (context).

Usage: python src/eda/failure_gallery.py [--n 20]
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
from filament.pq import _entry_iou_matrix
from filament.rle import decode_rle, rles_to_layers

CACHE = ROOT / "outputs" / "oof_cache"
IMG_DIR = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
OUT = ROOT / "outputs" / "gallery"
R_MAX = 0.97
CONF_OP, MIN_AREA_OP = 0.30, 128
GT_COLOR, PRED_COLOR, CTX_COLOR = (60, 220, 60), (60, 60, 240), (0, 170, 255)


def log(*a):
    print(*a, flush=True)


def deoverlap(g):
    """Greedy conf-desc de-overlap of one image's raw candidates."""
    taken = np.zeros((IMG_H, IMG_W), dtype=bool)
    out = []
    for r in g.sort_values("conf", ascending=False).itertuples():
        m = decode_rle(r.rle).astype(bool) & ~taken
        if not m.any():
            continue
        taken |= m
        out.append((float(r.conf), m))
    return out


def analyse():
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"], folds["fold"]))

    recs = []
    t0 = time.time()
    for fold in range(5):
        det = pd.read_parquet(CACHE / f"fold{fold}_preds_e013.parquet")
        det = det[(det["conf"] >= 0.05) & (det["r_norm"] <= R_MAX)]
        if "view" in det.columns:
            det = det[det["view"] == "o"]
        val_files = [f for f, k in fold_of.items() if k == fold]
        gt_f = gt[gt["file_name"].isin(val_files)]
        for stem, g in det.groupby("stem"):
            cands = deoverlap(g)
            if not cands:
                continue
            pool = np.stack([m for _, m in cands]).astype(np.uint8)
            op_idx = [i for i, (c, m) in enumerate(cands)
                      if c >= CONF_OP and m.sum() >= MIN_AREA_OP]
            sub = gt_f[gt_f["file_name"] == f"{stem}.jpeg"]
            for entry_id, e in sub.groupby("entry_id"):
                gl = rles_to_layers(e["segmentation_rle"].tolist())
                iou = _entry_iou_matrix(gl, pool)            # (n_gt, n_pool)
                for gi in range(len(e)):
                    row = e.iloc[gi]
                    best_all = float(iou[gi].max()) if iou.shape[1] else 0.0
                    j_all = int(iou[gi].argmax()) if iou.shape[1] else -1
                    if op_idx:
                        sub_iou = iou[gi, op_idx]
                        best_op = float(sub_iou.max())
                        j_op = op_idx[int(sub_iou.argmax())]
                    else:
                        best_op, j_op = 0.0, -1
                    recs.append(dict(ann_id=row["ann_id"], entry_id=entry_id,
                                     file_name=row["file_name"], stem=stem, fold=fold,
                                     gt_area=int(gl[gi].sum()),
                                     best_iou_pool=best_all, best_j_pool=j_all,
                                     best_iou_op=best_op, best_j_op=j_op))
        log(f"  fold {fold} done ({time.time()-t0:.0f}s, {len(recs)} GT rows)")
    return pd.DataFrame(recs)


def render(df, group, n, gt, morph):
    """Outline-only overlay crops for n examples of a group."""
    dest = OUT / group
    dest.mkdir(parents=True, exist_ok=True)
    sel = df.head(n)
    tiles = []
    for r in sel.itertuples():
        img = cv2.imread(str(IMG_DIR / f"{r.stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        e = gt[(gt["entry_id"] == r.entry_id)]
        gl = rles_to_layers(e["segmentation_rle"].tolist())
        gi = int(np.nonzero(e["ann_id"].values == r.ann_id)[0][0])

        for k in range(len(e)):                    # context GT in amber
            if k == gi:
                continue
            c, _ = cv2.findContours(gl[k], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, c, -1, CTX_COLOR, 2)
        c, _ = cv2.findContours(gl[gi], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, c, -1, GT_COLOR, 3)

        det = pd.read_parquet(CACHE / f"fold{r.fold}_preds_e013.parquet")
        det = det[(det["stem"] == r.stem) & (det["conf"] >= 0.05) & (det["r_norm"] <= R_MAX)]
        if "view" in det.columns:
            det = det[det["view"] == "o"]
        cands = deoverlap(det)
        j = int(r.best_j_op) if r.best_j_op >= 0 else int(r.best_j_pool)
        if j >= 0 and j < len(cands):
            pc, _ = cv2.findContours(cands[j][1].astype(np.uint8),
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, pc, -1, PRED_COLOR, 2)

        ys, xs = np.nonzero(gl[gi])
        pad = max(60, int(0.35 * max(np.ptp(ys), np.ptp(xs))))
        y0, y1 = max(0, ys.min() - pad), min(IMG_H, ys.max() + pad)
        x0, x1 = max(0, xs.min() - pad), min(IMG_W, xs.max() + pad)
        crop = vis[y0:y1, x0:x1]
        side = max(crop.shape[:2])
        canvas = np.zeros((side, side, 3), np.uint8)
        canvas[:crop.shape[0], :crop.shape[1]] = crop
        tile = cv2.resize(canvas, (384, 384), interpolation=cv2.INTER_CUBIC)
        w = morph.get(r.ann_id, {})
        cv2.putText(tile, f"A{r.gt_area} W{w.get('width_proxy', 0):.0f} "
                          f"IoU{r.best_iou_pool:.2f}", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(dest / f"{r.stem}_{str(r.ann_id)[:8]}.png"), tile)
        tiles.append(tile)

    if tiles:
        cols = 5
        rows_n = int(np.ceil(len(tiles) / cols))
        sheet = np.zeros((rows_n * 384, cols * 384, 3), np.uint8)
        for i, t in enumerate(tiles):
            rr, cc = divmod(i, cols)
            sheet[rr * 384:(rr + 1) * 384, cc * 384:(cc + 1) * 384] = t
        cv2.imwrite(str(OUT / f"montage_{group}.png"), sheet)
    log(f"  rendered {len(tiles)} tiles -> {dest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    cachef = OUT / "gt_match_status.parquet"
    if cachef.exists():
        df = pd.read_parquet(cachef)
        log(f"reusing {cachef.name} ({len(df)} rows)")
    else:
        df = analyse()
        df.to_parquet(cachef, index=False)

    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    m = pd.read_parquet(ROOT / "outputs" / "eda" / "morphology_instances.parquet")
    df = df.merge(m[["ann_id", "poly_area", "width_proxy", "spine_len", "category",
                     "eccentricity"]], on="ann_id", how="left")
    morph = m.set_index("ann_id")[["width_proxy"]].to_dict("index")

    never = df[df["best_iou_pool"] < 0.10]
    near = df[(df["best_iou_pool"] >= 0.10) & (df["best_iou_pool"] <= 0.50)]
    tp = df[df["best_iou_op"] > 0.50]
    worst_tp = tp.nsmallest(max(len(tp) // 10, a.n), "best_iou_op")

    stats = {}
    for name, sub in [("ALL", df), ("NEVER", never), ("NEAR_MISS", near),
                      ("TP", tp), ("WORST_TP", worst_tp)]:
        stats[name] = dict(
            n=int(len(sub)), share=round(len(sub) / len(df), 4),
            area_p10=float(sub["poly_area"].quantile(.10)),
            area_med=float(sub["poly_area"].median()),
            area_p90=float(sub["poly_area"].quantile(.90)),
            width_p10=float(sub["width_proxy"].quantile(.10)),
            width_med=float(sub["width_proxy"].median()),
            width_p90=float(sub["width_proxy"].quantile(.90)),
            spine_med=float(sub["spine_len"].median()),
            ecc_med=float(sub["eccentricity"].median()),
            pct_unidentifiable=round(float((sub["category"] == "Unidentifiable").mean()), 4)
            if "category" in sub else None)
        s = stats[name]
        log(f"{name:10s} n={s['n']:5d} ({s['share']:.1%})  area p10/med/p90 "
            f"{s['area_p10']:.0f}/{s['area_med']:.0f}/{s['area_p90']:.0f}  "
            f"width {s['width_p10']:.1f}/{s['width_med']:.1f}/{s['width_p90']:.1f}  "
            f"spine {s['spine_med']:.0f}  U%={s['pct_unidentifiable']}")
    with open(OUT / "failure_stats.json", "w") as f:
        json.dump(stats, f, indent=1)

    render(never.sort_values("poly_area", ascending=False), "never_found", a.n, gt, morph)
    render(near.sort_values("best_iou_pool", ascending=False), "near_miss", a.n, gt, morph)
    render(worst_tp.sort_values("best_iou_op"), "worst_tp", a.n, gt, morph)
    log(f"\nmontages -> {OUT}")


if __name__ == "__main__":
    main()
