"""E161 -- does the magnetic field, at the pixel level, know where filaments are?

Three uses of the magnetogram have now been measured and two are closed:

  selector feature      E160   nine PIL summaries, 78% coverage   -0.0011  TIED
  generation prior      E159   never-proposed U rows sit no closer to a PIL than
                               the found ones (53.0 vs 51.4 px)
  U-Net input channel   open   ~3 A100 hours plus data plumbing

Before spending the third, a cheaper question decides whether it can pay at all.
Hand-made summaries per mask may simply be the wrong reduction; a network sees the
field as an image. The honest proxy for "what a network could extract" without
training one is a per-pixel model on magnetic inputs ONLY -- the registered B, its
smoothed version, |grad B|, the distance to the nearest PIL, the PIL tangent
director -- predicting the annotator agreement fraction k/N at that pixel, and its
AUC for "any annotator marked this pixel".

Read against two anchors:
    ~0.50   the field carries no pixel-level information about filaments; the
            channel is dead and the GPU is not spent
    the H-alpha U-Net's own pixel AUC on the same pixels (computed here from the
            production probability field) -- the bar a channel would have to ADD to

A magnetic-only pixel AUC well above 0.5 does not prove the channel helps (the
U-Net may already know everything the field knows), but one near 0.5 proves it
cannot. Sampling: foreground pixels plus an equal number from the 31 px band around
them (the decision boundary) plus a tenth as many far away, per image, out of fold.

Usage: python src/eda/magnetic_pixel_test.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/eda", "src/models"):
    sys.path.insert(0, str(ROOT / p))
from filament.rle import decode_rle  # noqa: E402
from magnetogram_features import load_registered, pil_maps  # noqa: E402

OUT = ROOT / "outputs" / "eda"
SEM = ROOT / "outputs" / "semantic"
MAG = ROOT / "outputs" / "magnetograms"
N_FG = 1500


def log(*a):
    print(*a, flush=True)


def agreement_half(gt_g):
    entries = sorted(gt_g["entry_id"].unique())
    acc = np.zeros((2048, 2048), np.float32)
    for e in entries:
        m = np.zeros((2048, 2048), bool)
        for r in gt_g.loc[gt_g["entry_id"] == e, "segmentation_rle"]:
            m |= decode_rle(r).astype(bool)
        acc += m
    acc /= max(len(entries), 1)
    return cv2.resize(acc, (1024, 1024), interpolation=cv2.INTER_AREA)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    # E164: the same test for another co-temporal product. "aia304" reads reg304_*.npy
    # (He II 304 A intensity, filaments dark, prominences bright) instead of the field;
    # its inputs are the log intensity, a smoothed version, the local contrast and the
    # gradient magnitude, i.e. what a network could read off the image.
    ap.add_argument("--source", default="hmi", choices=["hmi", "aia304"])
    a_ = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    pref = "reg_" if a_.source == "hmi" else "rega304_"
    reg = sorted({p.stem[len(pref):] for p in MAG.glob(pref + "*.npy")} & set(gt["stem"]))
    log(f"{len(reg)} registered training images")
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:1024, 0:1024].astype(np.float32)
    rnorm = np.hypot(xx - 511.5, yy - 511.5) / 452.0

    X, y, f, h = [], [], [], []
    fields = {k: np.load(SEM / f"fold{k}_probs_blob5ens.npz") for k in range(5)}
    for i, stem in enumerate(reg):
        if a_.source == "hmi":
            B = load_registered(stem)
            maps = pil_maps(B)
        else:
            A = np.load(MAG / f"rega304_{stem}.npy").astype(np.float32)
            on = np.isfinite(A) & (A > 0)
            la = np.where(on, np.log1p(np.clip(A, 0, None)), 0.0)
            sm = ndimage.gaussian_filter(la, 4.0)
            gy, gx = np.gradient(sm)
            maps = dict(on=on, Bs=sm, grad=np.hypot(gx, gy), dist=np.zeros_like(sm),
                        cos2=la, sin2=(la - ndimage.gaussian_filter(la, 16.0)))   # log I, local contrast
            B = A
        agree = agreement_half(gt[gt["stem"] == stem])
        ha = cv2.resize(fields[fold_of[stem]][stem], (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        fg = agree > 0
        band = ndimage.binary_dilation(fg, iterations=15) & ~fg & maps["on"]
        far = ~fg & ~band & maps["on"] & (rnorm < 0.95)
        idx = []
        for m, k in ((fg & maps["on"], N_FG), (band, N_FG), (far, N_FG // 10)):
            p = np.flatnonzero(m.ravel())
            if len(p):
                idx.append(rng.choice(p, size=min(k, len(p)), replace=False))
        sel = np.concatenate(idx)
        Braw = np.nan_to_num(B.ravel()[sel])
        if a_.source == "hmi":
            feats = np.stack([Braw, np.abs(Braw), maps["Bs"].ravel()[sel], np.abs(maps["Bs"].ravel()[sel]),
                              maps["grad"].ravel()[sel], np.log1p(maps["dist"].ravel()[sel]),
                              maps["cos2"].ravel()[sel], maps["sin2"].ravel()[sel], rnorm.ravel()[sel]], axis=1)
        else:
            feats = np.stack([maps["cos2"].ravel()[sel], maps["Bs"].ravel()[sel], maps["sin2"].ravel()[sel],
                              maps["grad"].ravel()[sel], rnorm.ravel()[sel]], axis=1)
        X.append(feats.astype(np.float32)); y.append((agree.ravel()[sel] > 0).astype(np.int8))
        f.append(np.full(len(sel), fold_of[stem], np.int8)); h.append(ha.ravel()[sel])
        if (i + 1) % 50 == 0:
            log(f"  {i + 1} images, {sum(len(a) for a in y)} pixels")
    X, y, f, h = map(np.concatenate, (X, y, f, h))
    log(f"{len(y)} pixels, {y.mean():.1%} filament\n")

    oof = np.zeros(len(y))
    for k in range(5):
        tr, te = f != k, f == k
        if te.sum() == 0 or tr.sum() == 0:
            continue
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                           min_samples_leaf=200, l2_regularization=1.0, random_state=0)
        m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    auc_mag = roc_auc_score(y, oof)
    auc_ha = roc_auc_score(y, h)
    # and the combination: does the field add to what the U-Net already says?
    oof2 = np.zeros(len(y))
    X2 = np.concatenate([X, h[:, None]], axis=1)
    for k in range(5):
        tr, te = f != k, f == k
        if te.sum() == 0 or tr.sum() == 0:
            continue
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                           min_samples_leaf=200, l2_regularization=1.0, random_state=0)
        m.fit(X2[tr], y[tr]); oof2[te] = m.predict_proba(X2[te])[:, 1]
    auc_both = roc_auc_score(y, oof2)
    # WA-02 (audit 2026-09-10): the comparison above pits a raw score against a fitted
    # learner. The capacity-matched control fits the SAME learner on the H-alpha score
    # and position only; and a negative control shuffles the magnetograms across
    # images so any gain that survives is not local physical signal.
    def oof_auc(Xm):
        o = np.zeros(len(y))
        for k in range(5):
            tr, te = f != k, f == k
            if te.sum() == 0 or tr.sum() == 0:
                continue
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                               min_samples_leaf=200, l2_regularization=1.0, random_state=0)
            m.fit(Xm[tr], y[tr]); o[te] = m.predict_proba(Xm[te])[:, 1]
        return roc_auc_score(y, o)
    auc_ctl = oof_auc(np.stack([h, X[:, -1]], axis=1))          # H-alpha score + r_norm, same learner
    perm = rng.permutation(len(y))
    auc_shuf = oof_auc(np.concatenate([X[perm, :-1], X[:, -1:], h[:, None]], axis=1))  # magnetics from other pixels
    log(f"  capacity-matched control, HGB(field, r_norm)   {auc_ctl:.4f}")
    log(f"  same + magnetic inputs                         {auc_both:.4f}   ({auc_both - auc_ctl:+.4f} over the matched control)")
    log(f"  negative control, magnetics SHUFFLED           {auc_shuf:.4f}   ({auc_shuf - auc_ctl:+.4f})")
    log("pixel-level AUC for 'some annotator marked this pixel' (fg + boundary band + far):")
    log(f"  magnetic inputs only            {auc_mag:.4f}   (0.5 = the field knows nothing)")
    log(f"  the H-alpha U-Net field alone   {auc_ha:.4f}")
    log(f"  U-Net field + magnetic inputs   {auc_both:.4f}   ({auc_both - auc_ha:+.4f} over the field)")
    # the population that matters: the boundary band, where the U-Net is unsure
    band_mask = np.abs(h - 0.5) < 0.3
    if band_mask.sum() > 1000 and 0 < y[band_mask].mean() < 1:
        log(f"\n  on the {band_mask.sum()} uncertain pixels (U-Net p in 0.2..0.8, {y[band_mask].mean():.1%} filament):")
        log(f"    field alone {roc_auc_score(y[band_mask], h[band_mask]):.4f}   "
            f"magnetic only {roc_auc_score(y[band_mask], oof[band_mask]):.4f}   "
            f"both {roc_auc_score(y[band_mask], oof2[band_mask]):.4f}")
    json.dump(dict(n_images=len(reg), n_pixels=int(len(y)), auc_magnetic=float(auc_mag),
                   auc_halpha=float(auc_ha), auc_both=float(auc_both), auc_matched_control=float(auc_ctl),
                   auc_shuffled=float(auc_shuf)),
              open(OUT / f"magnetic_pixel_test_{a_.source}.json", "w"), indent=1)
    log(f"\nwrote {OUT / 'magnetic_pixel_test.json'}")


if __name__ == "__main__":
    main()
