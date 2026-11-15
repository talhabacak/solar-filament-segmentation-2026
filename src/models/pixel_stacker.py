"""E135 -- a per-pixel gradient-boosted stacker over the two lanes.

Talha's question: would a LightGBM-style pixel model help? Not as a segmenter --
a tree on hand-crafted pixel features will not beat a 1536-input U-Net, and the
project has killed ten "new candidate lane" attempts. But there is a place where
a pixel-level tree is the natural tool and has never been used: the two lanes are
combined by a FIXED linear blend, P = 0.5*blob + 0.5*clDice, and the detector's
own masks never enter the probability field at all. A tree can learn, per pixel,
how much to trust each lane given where the pixel is and what its neighbourhood
looks like -- which is stacking, the standard use of a tree over model outputs.

Features per pixel (all from artefacts we already have, no new inference):
    blob, cldice                 the two out-of-fold probability fields
    diff, agree                  |blob - cldice|, min(blob, cldice)
    blob_5, cldice_5             5x5 mean of each (local support)
    blob_max9, cldice_max9       9x9 max of each (is a strong response nearby)
    ring_blob, ring_cldice       31x31 mean minus the 9x9 mean (local contrast of
                                 the field itself, i.e. is this a ridge or a blob)
    det_cov, det_conf            detector-mask coverage at that pixel and the
                                 confidence of the covering detection (0 if none)
    img, img_lc                  raw intensity and its 31x31 local contrast
    r_norm                       radial position
Target: the annotator agreement fraction k/N at that pixel (the quantity BCE on
all copies converges to, and the quantity whose 0.40 level the metric wants).

Training is per fold and out of fold: the model for fold k is fitted on the other
four folds' pixels only. Sampling: every foreground pixel is a candidate but
capped, plus an equal number of background pixels drawn from the 31x31 dilation
of the foreground (the decision boundary) and a tenth as many from far away.

Output: fold{k}_probs_stack.npz, a drop-in replacement for the blended field, so
the rest of the pipeline (pool, selector, policy) is unchanged and the comparison
is single-variable.

Usage:
  python src/models/pixel_stacker.py --folds 0,1,2,3,4 --sample 400000
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from filament.rle import decode_rle  # noqa: E402

SEM = ROOT / "outputs" / "semantic"
CACHE = ROOT / "outputs" / "oof_cache"
IMG = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
FEATS = ["blob", "cldice", "diff", "agree", "blob_5", "cldice_5", "blob_max9", "cldice_max9",
         "ring_blob", "ring_cldice", "det_cov", "det_conf", "img", "img_lc", "r_norm"]
# E161b: the same stacker with the registered magnetogram as extra inputs, so the
# pixel-level +0.027 AUC of E161 can be asked the only question that matters --
# does it survive the pool, the selector and the exact scorer -- against a matched
# stacker without them. Both fields are bad replacements for the U-Net (E135); the
# PAIR isolates what the magnetic inputs add.
MAG_FEATS = ["mag_b", "mag_abs", "mag_bs", "mag_abs_s", "mag_grad", "mag_logdist", "mag_cos2", "mag_sin2"]
K5 = (5, 5)
K9 = np.ones((9, 9), np.uint8)


def log(*a):
    print(*a, flush=True)


def magnetic_features(stem):
    """Registered half-scale magnetogram (E154c) and its PIL maps, upsampled to 2048."""
    import sys as _s
    _s.path.insert(0, str(ROOT / "src" / "models"))
    from magnetogram_features import load_registered, pil_maps
    B = load_registered(stem)
    if B is None:
        return {k: np.full((2048, 2048), np.nan, np.float32) for k in MAG_FEATS}
    m = pil_maps(B)
    up = lambda a: cv2.resize(np.nan_to_num(a.astype(np.float32)), (2048, 2048), interpolation=cv2.INTER_LINEAR)
    Bf = up(B)
    return dict(mag_b=Bf, mag_abs=np.abs(Bf), mag_bs=up(m["Bs"]), mag_abs_s=np.abs(up(m["Bs"])),
                mag_grad=up(m["grad"]), mag_logdist=up(np.log1p(m["dist"])),
                mag_cos2=up(m["cos2"]), mag_sin2=up(m["sin2"]))


def image_features(stem, fold, det_by_stem):
    b = np.load(SEM / f"fold{fold}_probs_blob1536.npz")[stem].astype(np.float32) / 255.0
    c = np.load(SEM / f"fold{fold}_probs_cldice.npz")[stem].astype(np.float32) / 255.0
    img = cv2.imread(str(IMG / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    det_cov = np.zeros_like(b)
    det_conf = np.zeros_like(b)
    for rle, cf in det_by_stem.get(stem, []):
        m = decode_rle(rle).astype(bool)
        det_cov[m] = 1.0
        det_conf[m] = np.maximum(det_conf[m], cf)
    f = {}
    f["blob"], f["cldice"] = b, c
    f["diff"] = np.abs(b - c)
    f["agree"] = np.minimum(b, c)
    f["blob_5"] = cv2.blur(b, K5)
    f["cldice_5"] = cv2.blur(c, K5)
    f["blob_max9"] = cv2.dilate(b, K9)
    f["cldice_max9"] = cv2.dilate(c, K9)
    f["ring_blob"] = cv2.blur(b, (31, 31)) - f["blob_5"]
    f["ring_cldice"] = cv2.blur(c, (31, 31)) - f["cldice_5"]
    f["det_cov"], f["det_conf"] = det_cov, det_conf
    f["img"] = img / 255.0
    f["img_lc"] = (cv2.blur(img, (31, 31)) - img) / 255.0
    yy, xx = np.mgrid[0:2048, 0:2048].astype(np.float32)
    f["r_norm"] = np.sqrt((xx - 1023.5) ** 2 + (yy - 1023.5) ** 2) / 900.0
    return f


def agreement_map(gt_g):
    entries = sorted(gt_g["entry_id"].unique())
    acc = np.zeros((2048, 2048), np.float32)
    for e in entries:
        m = np.zeros((2048, 2048), bool)
        for r in gt_g.loc[gt_g["entry_id"] == e, "segmentation_rle"]:
            m |= decode_rle(r).astype(bool)
        acc += m
    return acc / max(len(entries), 1)


def sample_pixels(f, agree, n_fg, rng):
    fg = agree > 0
    band = cv2.dilate(fg.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool) & ~fg
    far = ~fg & ~band
    idx = []
    for mask, k in ((fg, n_fg), (band, n_fg), (far, max(n_fg // 10, 1))):
        p = np.flatnonzero(mask.ravel())
        if len(p) == 0:
            continue
        idx.append(rng.choice(p, size=min(k, len(p)), replace=False))
    sel = np.concatenate(idx)
    X = np.stack([f[k].ravel()[sel] for k in FEATS + (MAG_FEATS if "mag_b" in f else [])], axis=1)
    return X, agree.ravel()[sel]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--sample", type=int, default=3000, help="foreground pixels sampled per image")
    ap.add_argument("--tag", default="stack")
    ap.add_argument("--magnetic", action="store_true")
    a = ap.parse_args()
    folds_l = [int(x) for x in a.folds.split(",")]
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    rng = np.random.default_rng(0)

    det_by_stem = {}
    for f in range(5):
        d = pd.read_parquet(CACHE / f"fold{f}_preds_e013.parquet")
        d = d[d["conf"] >= 0.05]
        for stem, g in d.groupby("stem"):
            det_by_stem[stem] = list(zip(g["rle"], g["conf"]))

    # per fold: train on the OTHER folds' pixels, then write the field for this fold
    from sklearn.ensemble import HistGradientBoostingRegressor
    for fold in folds_l:
        t0 = time.time()
        Xs, ys = [], []
        for stem, f_ in fold_of.items():
            if f_ == fold:
                continue
            feats = image_features(stem, f_, det_by_stem)
            if a.magnetic:
                feats.update(magnetic_features(stem))
            X, y = sample_pixels(feats, agreement_map(gt[gt["stem"] == stem]), a.sample, rng)
            Xs.append(X.astype(np.float32)); ys.append(y.astype(np.float32))
            if len(Xs) % 100 == 0:
                log(f"  fold {fold}: {len(Xs)} training images ({time.time()-t0:.0f}s)")
        X = np.concatenate(Xs); y = np.concatenate(ys)
        log(f"fold {fold}: {X.shape[0]} pixels, {X.shape[1]} features ({time.time()-t0:.0f}s)")
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                          min_samples_leaf=200, l2_regularization=1.0, random_state=0)
        m.fit(X, y)
        del X, y, Xs, ys
        out = {}
        stems = [s for s, f_ in fold_of.items() if f_ == fold]
        for i, stem in enumerate(sorted(stems)):
            feats = image_features(stem, fold, det_by_stem)
            if a.magnetic:
                feats.update(magnetic_features(stem))
            Z = np.stack([feats[k].ravel() for k in FEATS + (MAG_FEATS if a.magnetic else [])], axis=1)
            p = m.predict(Z).reshape(2048, 2048)
            out[stem] = np.clip(np.rint(p * 255), 0, 255).astype(np.uint8)
            if i % 40 == 0:
                log(f"  fold {fold}: field {i}/{len(stems)} ({time.time()-t0:.0f}s)")
        np.savez_compressed(SEM / f"fold{fold}_probs_{a.tag}.npz", **out)
        log(f"fold {fold}: wrote fold{fold}_probs_{a.tag}.npz ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
