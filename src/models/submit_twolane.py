"""Two-lane test submission: 5 detector folds + 5 semantic folds, fused.

Everything below is the configuration that measured best on fold 0 (0.4383):
detector candidates at NMS 0.05, semantic candidates from the clDice U-Nets at
prob>=0.3 / area>=256, clustered together at IoU>0.5, fused by confidence-weighted
mean, de-overlapped greedily, then filtered by a source-aware selector.

The selector is trained on the fold-0 fused pool (the only pool with honest
labels available at submission time) and applied to the test pool.

Usage:
  python src/models/submit_twolane.py --out outputs/submissions/sub006.csv \
      [--conf 0.4] [--p-min 0.45]
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament import IMG_H, IMG_W
from filament.pq import _entry_iou_matrix
from filament.rle import decode_rle, encode_mask, write_submission
from selector import mask_features
from semantic_candidates import load_model, predict_prob, prob_to_instances
from fuse_semantic import FEATS

CACHE = ROOT / "outputs" / "oof_cache"
OUT = ROOT / "outputs" / "semantic"
TEST = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "test" / "test_images"
R_MAX = 0.97


def log(*a):
    print(*a, flush=True)


def semantic_test_probs(sem_thr, sem_min_area):
    """Average the 5 fold U-Nets' probability maps per test image."""
    cache = OUT / "test_probs_mean.npz"
    if cache.exists():
        log("reusing cached test probability maps")
        z = np.load(cache)
        return {s: z[s].astype(np.float32) / 255.0 for s in z.files}
    stems = sorted(p.stem for p in TEST.glob("*.jpeg"))
    acc = {s: np.zeros((IMG_H, IMG_W), dtype=np.float32) for s in stems}
    for fold in range(5):
        w = ROOT / "outputs" / "runs_semantic" / f"sem_b3_f{fold}_1536_cldice" / "best.pt"
        model, imgsz, channels = load_model(str(w))
        t0 = time.time()
        for i, s in enumerate(stems):
            acc[s] += predict_prob(model, imgsz, s, tta=True, img_dir=TEST,
                                   channels=channels) / 5.0
        del model
        torch.cuda.empty_cache()
        log(f"  semantic fold {fold} done ({time.time()-t0:.0f}s)")
    np.savez_compressed(cache, **{s: (v * 255).astype(np.uint8) for s, v in acc.items()})
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--p-min", type=float, default=0.45)
    ap.add_argument("--sem-thr", type=float, default=0.3)
    ap.add_argument("--sem-min-area", type=int, default=256)
    ap.add_argument("--det-models", default="0n,1n,2n,3n,4n")
    a = ap.parse_args()

    # ---- detector candidates (already cached at NMS 0.05 for the test split)
    det_dfs = []
    for mid in a.det_models.split(","):
        d = pd.read_parquet(CACHE / f"cross_m{mid}_test.parquet")
        d["model"] = mid
        det_dfs.append(d)
    det = pd.concat(det_dfs, ignore_index=True)
    det = det[(det["conf"] >= 0.05) & (det["r_norm"] <= R_MAX)]
    log(f"detector candidates: {len(det)} over {det['stem'].nunique()} images")

    probs = semantic_test_probs(a.sem_thr, a.sem_min_area)

    # ---- selector trained on the fold-0 fused pool
    fused0 = pd.read_parquet(OUT / "fold0_fused_candidates.parquet")
    from sklearn.ensemble import HistGradientBoostingClassifier
    sel = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=25,
        l2_regularization=1.0, early_stopping=False, random_state=42)
    sel.fit(fused0[FEATS].to_numpy(np.float64), fused0["y"].to_numpy())
    log(f"selector trained on {len(fused0)} fold-0 fused candidates")

    preds = {}
    t0 = time.time()
    stems = sorted(probs)
    for n, stem in enumerate(stems):
        items = []
        g = det[det["stem"] == stem].sort_values("conf", ascending=False)
        for r in g.itertuples():
            items.append(("det", float(r.conf), decode_rle(r.rle).astype(bool)))
        for c in prob_to_instances(probs[stem], a.sem_thr, a.sem_min_area):
            items.append(("sem", float(c["conf"]), c["mask"].astype(bool)))
        if not items:
            preds[stem] = []
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
        clusters = defaultdict(list)
        for i in range(len(items)):
            clusters[find(i)].append(i)

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
                              n_sources=len(srcs), conf=max(det_c, sem_c)))
        fused.sort(key=lambda d: -d["conf"])

        taken = np.zeros((IMG_H, IMG_W), dtype=bool)
        kept = []
        for d in fused:
            mm = d["mask"] & ~taken
            if not mm.any():
                continue
            taken |= mm
            d["mask"] = mm
            kept.append(d)

        rows, masks_out = [], []
        for k, d in enumerate(kept, start=1):
            f = mask_features(d["mask"])
            if f["r_norm"] > R_MAX:
                continue
            rows.append(dict(conf=d["conf"], log10_area=np.log10(f["area"]),
                             det_conf=d["det_conf"], sem_conf=d["sem_conf"],
                             from_det=d["from_det"], from_sem=d["from_sem"],
                             n_sources=d["n_sources"], n_cands_img=len(kept),
                             rank_conf=float(k),
                             **{kk: f[kk] for kk in ("r_norm", "abs_lat", "bbox_aspect",
                                                     "eccentricity", "boundary_complexity",
                                                     "skel_len", "width_proxy")}))
            masks_out.append(d["mask"])
        if not rows:
            preds[stem] = []
            continue
        X = pd.DataFrame(rows)[FEATS].to_numpy(np.float64)
        p = sel.predict_proba(X)[:, 1]
        preds[stem] = [m.astype(np.uint8) for m, c, pp in zip(masks_out, [r["conf"] for r in rows], p)
                       if c >= a.conf and pp >= a.p_min]
        if (n + 1) % 40 == 0:
            log(f"  {n+1}/{len(stems)} images ({time.time()-t0:.0f}s)")

    n_inst = sum(len(v) for v in preds.values())
    n_empty = sum(1 for v in preds.values() if not v)
    log(f"test: {n_inst} instances over {len(preds)-n_empty} images ({n_empty} empty)")
    df = write_submission({k: v for k, v in preds.items() if v}, ROOT / a.out)
    log(f"wrote {len(df)} rows -> {a.out}")


if __name__ == "__main__":
    main()
