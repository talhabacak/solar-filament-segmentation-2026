"""E123 -- orientation and context features in the selector (Talha's "yon bilgisi").

Physics (saf_bilim ch. 03/04): filaments lie along the PIL with threads at
20-35 degrees to the spine; the chromospheric fibril texture around a true
filament is coherent and roughly aligned with it, while a hallucinated dark
patch sits in texture with no preferred direction or a direction unrelated to
its own axis. The selector has shape and probability features but nothing
about the local texture ORIENTATION, and only weak candidate CONTEXT
(density, rank). Two feature families, each a single-variable E120 test:

  orientation (6)  structure tensor of the raw image (gradient sigma 1.5,
                   integration sigma 4) in the candidate's bounding box:
                   coh_in       mean coherence (l1-l2)/(l1+l2) inside the mask
                   coh_ring     mean coherence in a 12-px ring around it
                   tex_align    |angle between the mask's principal axis and
                                the dominant texture orientation inside| (deg)
                   ring_align   the same against the ring's texture
                   grad_norm    mean gradient magnitude across the boundary
                   aniso_ratio  coh_in / (coh_ring + 0.05)
  context (4)      n_strong_300  kept-strong candidates (conf >= 0.7) within
                                 300 px of the centroid (this image)
                   mass_share    this candidate's area / sum of candidate
                                 areas in the image
                   nn_strong     distance to the nearest strong candidate
                   align_nn      |angle between own axis and the nearest
                                 strong neighbour's axis|

Protocol: E115b honest OOF refit on the production labelled pool with the 32
production features vs +orientation vs +context vs both; exact PQ at
(0.40, 0.60), paired image bootstrap; bands pays >= +0.003 / hurts <= -0.003 /
TIED inside +-0.005.

Usage (on the box): python src/eda/orientation_selector_features.py
"""

import json
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "eda"))
from filament.pq import score  # noqa: E402
from filament.rle import decode_rle  # noqa: E402
from e115c_threshold_ci import per_image, pq, oof_scores  # noqa: E402

OUT = ROOT / "outputs" / "eda"
IMG = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
BUNDLE = ROOT / "outputs" / "selector" / "selector_v5_shrunkblob5prodts.joblib"
A_POOL = OUT / "e098d_fused_pool_all_blob5_rebuild_labelled.parquet"
POLICY, REPS, FLOOR, EQUIV = (0.40, 0.60), 4000, 0.003, 0.005
ORI = ["coh_in", "coh_ring", "tex_align", "ring_align", "grad_norm", "aniso_ratio"]
CTX = ["n_strong_300", "mass_share", "nn_strong", "align_nn"]
K12 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))


def log(*a):
    print(*a, flush=True)


def principal_angle(m):
    ys, xs = np.nonzero(m)
    if len(xs) < 5:
        return 0.0
    c = np.cov(np.vstack([xs, ys]).astype(np.float64))
    w, v = np.linalg.eigh(c)
    return float(np.degrees(np.arctan2(v[1, -1], v[0, -1])) % 180.0)


def angle_diff(a, b):
    d = abs(a - b) % 180.0
    return float(min(d, 180.0 - d))


def structure_tensor(gray):
    g = cv2.GaussianBlur(gray, (0, 0), 1.5)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 4.0)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 4.0)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 4.0)
    tr = jxx + jyy
    det = jxx * jyy - jxy * jxy
    disc = np.sqrt(np.maximum(tr * tr / 4 - det, 0))
    l1, l2 = tr / 2 + disc, tr / 2 - disc
    coh = (l1 - l2) / (l1 + l2 + 1e-6)
    # dominant orientation (perpendicular to the gradient direction): the eigenvector of l1
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy)          # gradient-dominant direction
    ori = (np.degrees(theta) + 90.0) % 180.0                # texture (edge) orientation
    return coh, ori, np.sqrt(gx * gx + gy * gy)


def circ_mean_deg(angles_deg, weights=None):
    a = np.radians(angles_deg * 2.0)
    w = np.ones_like(a) if weights is None else weights
    s, c = (w * np.sin(a)).sum(), (w * np.cos(a)).sum()
    return float((np.degrees(np.arctan2(s, c)) / 2.0) % 180.0)


def main():
    feats = list(joblib.load(BUNDLE)["features"])
    a = pd.read_parquet(A_POOL)
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    a["fold"] = a["stem"].map(fold_of).astype(int)
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt["fold"] = gt["stem"].map(fold_of)
    gt = gt[gt["fold"].notna()].copy()
    gt["fold"] = gt["fold"].astype(int)
    cache = OUT / "orientation_context_features.parquet"
    if cache.exists():
        ext = pd.read_parquet(cache)
        log(f"features from cache: {len(ext)} rows")
    else:
        rows, t0 = [], time.time()
        for n, (stem, g) in enumerate(a.groupby("stem")):
            img = cv2.imread(str(IMG / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            coh, ori, gm = structure_tensor(img)
            masks = {cid: decode_rle(r).astype(bool) for cid, r in zip(g["cand_id"], g["rle"])}
            axes = {cid: principal_angle(m) for cid, m in masks.items()}
            cents = {cid: (r.cy, r.cx) for cid, r in zip(g["cand_id"], g.itertuples())}
            strong = g[g["conf"] >= 0.7]
            total_area = float(g["area"].sum())
            for r in g.itertuples():
                m = masks[r.cand_id]
                ys, xs = np.nonzero(m)
                y0, y1, x0, x1 = max(ys.min() - 14, 0), min(ys.max() + 15, 2048), max(xs.min() - 14, 0), min(xs.max() + 15, 2048)
                mc = m[y0:y1, x0:x1]
                ring = cv2.dilate(mc.astype(np.uint8), K12).astype(bool) & ~mc
                cc, oc, gc = coh[y0:y1, x0:x1], ori[y0:y1, x0:x1], gm[y0:y1, x0:x1]
                bnd = mc & ~cv2.erode(mc.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                d = dict(cand_id=r.cand_id,
                         coh_in=float(cc[mc].mean()), coh_ring=float(cc[ring].mean()) if ring.any() else 0.0,
                         tex_align=angle_diff(axes[r.cand_id], circ_mean_deg(oc[mc], cc[mc])),
                         ring_align=angle_diff(axes[r.cand_id], circ_mean_deg(oc[ring], cc[ring])) if ring.any() else 90.0,
                         grad_norm=float(gc[bnd].mean()) if bnd.any() else 0.0)
                d["aniso_ratio"] = d["coh_in"] / (d["coh_ring"] + 0.05)
                cy, cx = cents[r.cand_id]
                others = strong[strong["cand_id"] != r.cand_id]
                if len(others):
                    dist = np.sqrt((others["cy"].to_numpy() - cy) ** 2 + (others["cx"].to_numpy() - cx) ** 2)
                    j = int(np.argmin(dist))
                    d.update(n_strong_300=int((dist <= 300).sum()), nn_strong=float(dist[j]),
                             align_nn=angle_diff(axes[r.cand_id], axes[others["cand_id"].iloc[j]]))
                else:
                    d.update(n_strong_300=0, nn_strong=2048.0, align_nn=90.0)
                d["mass_share"] = float(r.area) / max(total_area, 1.0)
                rows.append(d)
            if n % 50 == 0:
                log(f"  {n}/{a['stem'].nunique()} images ({time.time()-t0:.0f}s)")
        ext = pd.DataFrame(rows)
        ext.to_parquet(cache, index=False)
        log(f"features computed for {len(ext)} candidates ({time.time()-t0:.0f}s)")
    a = a.merge(ext, on="cand_id", how="left")
    stems = np.array(sorted(set(a["stem"])))
    gts = gt[gt["stem"].isin(stems)]
    ncopy = gts.groupby("stem")["entry_id"].nunique().reindex(stems).fillna(1).to_numpy()

    def exact(col, fold=None):
        sub = a[(a["conf"] >= POLICY[0]) & (a[col] >= POLICY[1])]
        g = gts
        if fold is not None:
            sub, g = sub[sub["fold"] == fold], gts[gts["fold"] == fold]
        pred = sub[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
        return score(g[["filament_id", "segmentation_rle"]], pred)

    arms = {"base": feats, "orientation": feats + ORI, "context": feats + CTX, "both": feats + ORI + CTX}
    res, parts = {}, {}
    from sklearn.metrics import roc_auc_score
    for name, fl in arms.items():
        col = f"p_{name}"
        a[col] = oof_scores(a, fl, a["fold"].to_numpy())
        per = [exact(col, f)["pq"] for f in range(5)]
        r = exact(col)
        sub = a[(a["conf"] >= POLICY[0]) & (a[col] >= POLICY[1])]
        st, _ = per_image(sub, gts, stems, ncopy)
        parts[name] = st
        res[name] = dict(pq=r["pq"], tp=r["tp"], fp=r["fp"], fn=r["fn"], per_fold=per, auc=float(roc_auc_score(a["y"], a[col])))
        log(f"{name:12s} PQ {r['pq']:.4f} TP {r['tp']} FP {r['fp']} FN {r['fn']} AUC(q) {res[name]['auc']:.4f} | per fold " + " ".join(f"{x:.4f}" for x in per))
    rng = np.random.default_rng(0)
    n = len(stems)
    for name in ("orientation", "context", "both"):
        boot = np.empty(REPS)
        for i in range(REPS):
            idx = rng.integers(0, n, n)
            v = [pq(s["tp"][idx].sum(), s["fp"][idx].sum(), s["fn"][idx].sum(), s["iou"][idx].sum()) for s in (parts["base"], parts[name])]
            boot[i] = v[1] - v[0]
        d = res[name]["pq"] - res["base"]["pq"]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        dpf = [res[name]["per_fold"][f] - res["base"]["per_fold"][f] for f in range(5)]
        verdict = "PAYS" if d >= FLOOR else "HURTS" if d <= -FLOOR else ("TIED" if lo > -EQUIV and hi < EQUIV else "UNDERPOWERED")
        res[name].update(delta=float(d), ci=[float(lo), float(hi)], p_gt0=float((boot > 0).mean()), delta_folds=dpf, verdict=verdict)
        log(f"  {name:12s} {d:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  P(>0) {(boot > 0).mean():.3f}  folds+ {sum(x > 0 for x in dpf)}/5  -> {verdict}")
    json.dump(res, open(OUT / "orientation_selector_features.json", "w"), indent=1, default=float)
    log(f"\nwrote {OUT / 'orientation_selector_features.json'}")


if __name__ == "__main__":
    main()
