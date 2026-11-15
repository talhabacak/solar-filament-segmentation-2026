"""E186 -- augmentation for the SELECTOR, which this project never tried.

Talha, 2026-09-13: "are the augmentations only for segmentation, do we do them for the selector
too, and are they in both YOLO and the U-Net?" The answer for the two networks is yes (the
U-Net with a recipe calibrated to the GONG stations' measured spread, the detector with the
same policy minus two flags this session is now testing). For the selector, what exists is
RESAMPLING of rows -- seed bagging (byte-identical, HGB is deterministic) and real
image-bootstrap bagging (+0.0010, tied). Nothing has ever perturbed the INPUT.

The selector's inputs are 42 numbers computed from a candidate mask. A one-pixel difference in
that mask moves every geometric and photometric feature, and the mask it will meet at test time
is drawn by a different model realisation than the one that produced the training rows. So the
honest augmentation is: perturb the mask the way the pipeline's own noise perturbs it, recompute
the features, and keep the label. That teaches the selector which features are stable and which
are mask noise -- the tabular analogue of what flip and gamma do for the U-Net.

Jitter: a morphological open/close by one pixel, or a +/-1 px dilation/erosion, chosen at
random per row. Calibrated to the pipeline's own scale rather than guessed: the production
submission shrinks every mask by 5%, and two annotators differ at IoU 0.629, so a one-pixel
boundary move (mask IoU ~0.95 for a 9 px-wide ribbon) is an order of magnitude smaller than
either -- a regulariser, not a relabelling.

Labels are untouched (y and iou_sum come from the original mask), which is deliberate: the
point is that a candidate's WORTH does not change when its boundary wobbles by a pixel.

Read with the standard deployment-faithful protocol: the selector is fitted out of fold on
augmented rows, then applied to the UNaugmented test fold at the deployed gate.

Usage: python src/eda/selector_feature_augmentation.py [--copies 1]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/eda", "src/models"):
    sys.path.insert(0, str(ROOT / p))
from filament.pq import score  # noqa: E402
from filament.rle import decode_rle  # noqa: E402
from selector import mask_features  # noqa: E402
from extra_features import EXTRA, extra_features, structure_tensor  # noqa: E402
from e115c_threshold_ci import make_cls, make_reg, per_image, pq  # noqa: E402

OUT = ROOT / "outputs" / "eda"
IMG = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
SEM = ROOT / "outputs" / "semantic"
RING = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))


def log(*a):
    print(*a, flush=True)


def jitter(m, rng):
    """A one-pixel boundary wobble: dilate, erode, open or close."""
    k = np.ones((3, 3), np.uint8)
    op = rng.integers(0, 4)
    u = m.astype(np.uint8)
    if op == 0:
        u = cv2.dilate(u, k)
    elif op == 1:
        u = cv2.erode(u, k)
    elif op == 2:
        u = cv2.morphologyEx(u, cv2.MORPH_OPEN, k)
    else:
        u = cv2.morphologyEx(u, cv2.MORPH_CLOSE, k)
    return u.astype(bool)


def recompute(m, img, T, prob):
    """The mask-dependent features, exactly as the pool builder computes them."""
    if m.sum() < 16:
        return None
    f = mask_features(m)
    ys, xs = np.nonzero(m)
    y0, y1 = max(0, ys.min() - 12), min(2048, ys.max() + 13)
    x0, x1 = max(0, xs.min() - 12), min(2048, xs.max() + 13)
    sub_m, sub_p = m[y0:y1, x0:x1], prob[y0:y1, x0:x1]
    ring = (cv2.dilate(sub_m.astype(np.uint8), RING) > 0) & ~sub_m
    pv = sub_p[sub_m]
    rv = sub_p[ring] if ring.sum() > 10 else pv
    f.update(sem_prob_mean=float(pv.mean()), sem_prob_max=float(pv.max()),
             sem_prob_p90=float(np.percentile(pv, 90)),
             sem_frac_over_50=float((pv >= 0.5).mean()),
             sem_ring_mean=float(rv.mean()),
             sem_contrast=float(pv.mean() - rv.mean()))
    f.update(extra_features(m, img, T))
    f["log10_area"] = float(np.log10(max(f["area"], 1)))   # the pool builder derives it too
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", type=int, default=1, help="jittered copies per training row")
    a = ap.parse_args()
    feats = list(joblib.load(ROOT / "outputs/selector/selector_v5_shrunkblob5prodts.joblib")["features"]) + list(EXTRA)
    folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    pool = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    pool = pool.merge(pd.read_parquet(OUT / "extra_features_t020.parquet")[["cand_id"] + list(EXTRA)],
                      on="cand_id", how="left")
    pool["fold"] = pool["stem"].map(fold_of).astype(int)
    stems_all = np.array(sorted(fold_of))
    ncopy = gt.groupby("stem")["entry_id"].nunique().reindex(stems_all).fillna(1).to_numpy()

    cache = OUT / f"selector_aug_rows_{a.copies}.parquet"
    if cache.exists():
        aug = pd.read_parquet(cache)
        log(f"reusing {len(aug)} jittered rows")
    else:
        rng = np.random.default_rng(0)
        probs = {k: np.load(SEM / f"fold{k}_probs_blob5ens.npz") for k in range(5)}
        rows = []
        for n, (stem, g) in enumerate(pool.groupby("stem")):
            img = cv2.imread(str(IMG / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            imf = img.astype(np.float32)
            T = structure_tensor(imf)
            f = fold_of[stem]
            pr = cv2.resize(probs[f][stem], (2048, 2048), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            for r in g.itertuples():
                m0 = decode_rle(r.rle).astype(bool)
                for c in range(a.copies):
                    d = recompute(jitter(m0, rng), imf, T, pr)
                    if d is None:
                        continue
                    d["cand_id"] = f"{r.cand_id}_j{c}"
                    d["src_id"] = r.cand_id
                    rows.append(d)
            if (n + 1) % 100 == 0:
                log(f"  {n + 1} images, {len(rows)} jittered rows")
        aug = pd.DataFrame(rows)
        aug.to_parquet(cache, index=False)
        log(f"wrote {len(aug)} jittered rows")
    keep = ["conf", "det_conf", "sem_conf", "from_det", "from_sem", "n_sources", "n_det_models",
            "det_conf_mean", "det_conf_std", "det_conf_max", "member_iou_mean", "member_iou_min",
            "nn_dist", "n_within_200px", "n_cands_img", "rank_conf", "img_cand_density",
            "img_sem_area_frac", "y", "iou_sum", "fold", "stem"]
    keep = [c for c in keep if c in pool.columns]
    aug = aug.merge(pool[["cand_id"] + keep].rename(columns={"cand_id": "src_id"}), on="src_id", how="left")
    aug = aug.dropna(subset=["y"])
    log(f"{len(aug)} augmented rows usable; base pool {len(pool)}")

    def run(train_aug):
        sel = []
        for F in range(5):
            trn = pool[pool["fold"] != F]
            if train_aug:
                extra = aug[aug["fold"] != F]
                miss = [c for c in feats if c not in extra.columns]
                # never silently fall back to the baseline: a no-op that reports +0.0000 with a
                # zero-width CI is indistinguishable from a real null, and that is exactly what
                # the first version of this script did (2026-09-13, log10_area was missing)
                assert not miss, f"jittered rows lack {len(miss)} features: {miss[:6]}"
                trn = pd.concat([trn[feats + ["y", "iou_sum", "fold"]],
                                 extra[feats + ["y", "iou_sum", "fold"]]], ignore_index=True)
            tst = pool[pool["fold"] == F].copy()
            X, y, v = trn[feats].to_numpy(np.float64), trn["y"].to_numpy(), trn["iou_sum"].to_numpy(np.float64)
            inner = np.zeros(len(trn))
            for g in [k for k in range(5) if k != F]:
                itr, msk = (trn["fold"] != g).to_numpy(), (trn["fold"] == g).to_numpy()
                if msk.sum() == 0:
                    continue
                c = make_cls().fit(X[itr], y[itr])
                r = make_reg().fit(X[itr][y[itr] == 1], v[itr][y[itr] == 1])
                inner[msk] = c.predict_proba(X[msk])[:, 1] * np.clip(r.predict(X[msk]), 0, None)
            ref = np.sort(inner[inner > 0]) if (inner > 0).any() else np.sort(inner)
            cls = make_cls().fit(X, y)
            reg = make_reg().fit(X[y == 1], v[y == 1])
            Z = tst[feats].to_numpy(np.float64)
            tst["p"] = np.searchsorted(ref, cls.predict_proba(Z)[:, 1] * np.clip(reg.predict(Z), 0, None),
                                       side="right") / len(ref)
            sel.append(tst[(tst["conf"] >= 0.40) & (tst["p"] >= 0.60)])
        sub = pd.concat(sel)
        r = score(gt[["filament_id", "segmentation_rle"]],
                  sub[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"}))
        return r, per_image(sub, gt, stems_all, ncopy)[0]

    rb, pb = run(False)
    log(f"baseline selector      PQ {rb['pq']:.4f} TP {rb['tp']} FP {rb['fp']} FN {rb['fn']}")
    ra, pa = run(True)
    log(f"+ jittered training rows PQ {ra['pq']:.4f} TP {ra['tp']} FP {ra['fp']} FN {ra['fn']}")
    rng = np.random.default_rng(0)
    n = len(stems_all)
    boot = np.empty(4000)
    for i in range(4000):
        idx = rng.integers(0, n, n)
        boot[i] = (pq(pa["tp"][idx].sum(), pa["fp"][idx].sum(), pa["fn"][idx].sum(), pa["iou"][idx].sum())
                   - pq(pb["tp"][idx].sum(), pb["fp"][idx].sum(), pb["fn"][idx].sum(), pb["iou"][idx].sum()))
    log(f"delta {ra['pq'] - rb['pq']:+.4f}  95% CI [{np.quantile(boot, .025):+.4f}, "
        f"{np.quantile(boot, .975):+.4f}]  P(delta>0) {(boot > 0).mean():.3f}")
    json.dump(dict(base_pq=float(rb["pq"]), aug_pq=float(ra["pq"]), n_aug=int(len(aug)),
                   delta=float(ra["pq"] - rb["pq"]), lo=float(np.quantile(boot, .025)),
                   hi=float(np.quantile(boot, .975))),
              open(OUT / "selector_feature_augmentation.json", "w"), indent=1)
    log("SELECTOR_AUG_DONE")


if __name__ == "__main__":
    main()
