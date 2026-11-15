"""Does image quality explain the honest selector's residual errors? (saf_bilim ch. 07; roadmap S007 precondition)

Physics: GONG is a ground-based network; seeing, transparency, sharpening and
JPEG processing differ by site and hour, and the PSF's wide wings fill in dark
thin structures. If the production pipeline's per-image miss rate or FP rate
depends on measurable image quality BEYOND what the selector already sees
(image-level candidate density, semantic area fraction, radius), a quality
vector is a legitimate new selector feature family. If not, S007 is closed
before any model is built.

Per image (703 folds_v1 stems, honest E115b pool, production policy):
  quality   lap_var   variance of the Laplacian inside the disc (r < 0.95)
            hf_energy high-pass energy fraction (sharpness / sharpening halo)
            jpeg8     8x8 block-boundary gradient excess (compression artefacts)
            lowfreq   std of the non-radial low-frequency residual inside the disc
                      (clouds / transparency gradients / flat-field)
            limb_noise std of the off-limb background (r > 1.05)
            mean_int  disc mean intensity
  errors    tp, fp, fn per image (per_image of e115c), miss = fn/(tp+fn),
            fpr = fp/(tp+fp), n_gt
  known     img_cand_density, img_sem_area_frac (means of the pool's image
            features), fraction of GT rows at r > 0.8

Reports Spearman correlations of miss and fpr with each quality feature, the
partial correlation after regressing out the known image features, and a
grouped-OOF gradient-boosting R^2 for miss/fpr from known vs known+quality.
Gate: partial |rho| >= 0.20 (or OOF R^2 gain >= 0.03) for miss or fpr on
>= 4/5 folds opens a selector-feature test; otherwise S007 closes.

Usage: python src/eda/image_quality_residual.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "eda"))
from e115c_threshold_ci import per_image  # noqa: E402

OUT = ROOT / "outputs" / "eda"
IMG = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
CONF, PCUT = 0.40, 0.60
CX = CY = 1024.0
R = 900.0


def log(*a):
    print(*a, flush=True)


def quality(img):
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    rr = np.sqrt((xx - CX) ** 2 + (yy - CY) ** 2) / R
    disc = rr < 0.95
    off = rr > 1.05
    f = img.astype(np.float32)
    lap = cv2.Laplacian(f, cv2.CV_32F)
    blur = cv2.GaussianBlur(f, (0, 0), 2.0)
    hp = f - blur
    low = cv2.GaussianBlur(f, (0, 0), 60.0)
    # radial profile of the low-frequency image; residual = non-radial part
    bins = np.clip((rr * 100).astype(int), 0, 199)
    prof = np.bincount(bins[disc], weights=low[disc], minlength=200) / np.maximum(np.bincount(bins[disc], minlength=200), 1)
    radial = prof[bins]
    nonrad = (low - radial)[disc]
    # jpeg 8x8 block excess: mean |gradient| on block boundaries vs elsewhere
    gx = np.abs(np.diff(f, axis=1))
    col = np.arange(gx.shape[1])
    on_b = (col % 8 == 7)
    jpeg8 = float(gx[:, on_b][disc[:, :-1][:, on_b]].mean() / max(gx[:, ~on_b][disc[:, :-1][:, ~on_b]].mean(), 1e-6))
    return dict(lap_var=float(lap[disc].var()), hf_energy=float((hp[disc] ** 2).mean() / max(f[disc].var(), 1e-6)),
                jpeg8=jpeg8, lowfreq=float(nonrad.std()), limb_noise=float(f[off].std()), mean_int=float(f[disc].mean()))


def main():
    cand = pd.read_parquet(OUT / "e115b_labelled_pool.parquet").merge(pd.read_parquet(OUT / "e115b_p_oof.parquet"), on="cand_id")
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    stems = np.array(sorted(fold_of))
    gts = gt[gt["stem"].isin(stems)]
    ncopy = gts.groupby("stem")["entry_id"].nunique().reindex(stems).fillna(1).to_numpy()
    sub = cand[(cand["conf"] >= CONF) & (cand["p_oof"] >= PCUT)]
    st, r = per_image(sub, gts, stems, ncopy)
    log(f"honest production policy over {len(stems)} images: exact PQ {r['pq']:.4f}")
    df = pd.DataFrame({"stem": stems, "fold": [fold_of[s] for s in stems], "tp": st["tp"], "fp": st["fp"], "fn": st["fn"], "iou": st["iou"]})
    df["miss"] = df["fn"] / np.maximum(df["tp"] + df["fn"], 1)
    df["fpr"] = df["fp"] / np.maximum(df["tp"] + df["fp"], 1)
    known = cand.groupby("stem").agg(img_cand_density=("img_cand_density", "mean"), img_sem_area_frac=("img_sem_area_frac", "mean"),
                                     n_cand=("cand_id", "size"), mean_r=("r_norm", "mean")).reindex(stems)
    limb_frac = gts.assign(r=np.nan)
    df = df.merge(known, left_on="stem", right_index=True, how="left")
    q = []
    for i, s in enumerate(stems):
        img = cv2.imread(str(IMG / f"{s}.jpeg"), cv2.IMREAD_GRAYSCALE)
        q.append(quality(img))
        if i % 100 == 0:
            log(f"  quality {i}/{len(stems)}")
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(q)], axis=1)
    df.to_parquet(OUT / "image_quality_residual.parquet", index=False)

    from scipy.stats import spearmanr
    qcols = ["lap_var", "hf_energy", "jpeg8", "lowfreq", "limb_noise", "mean_int"]
    kcols = ["img_cand_density", "img_sem_area_frac", "n_cand", "mean_r"]
    res = {}
    log("\nSpearman with per-image error rates (raw, and partial after regressing out known image features):")
    log(f"{'feature':14s} {'miss rho':>9s} {'partial':>9s} {'fpr rho':>9s} {'partial':>9s}")
    from sklearn.linear_model import LinearRegression
    Xk = df[kcols].fillna(0).to_numpy()
    for target in ("miss", "fpr"):
        y = df[target].to_numpy()
        y_res = y - LinearRegression().fit(Xk, y).predict(Xk)
        res[target] = {}
        for c in qcols:
            x = df[c].to_numpy()
            x_res = x - LinearRegression().fit(Xk, x).predict(Xk)
            res[target][c] = dict(rho=float(spearmanr(x, y).statistic), partial=float(spearmanr(x_res, y_res).statistic))
    for c in qcols:
        log(f"{c:14s} {res['miss'][c]['rho']:+9.3f} {res['miss'][c]['partial']:+9.3f} {res['fpr'][c]['rho']:+9.3f} {res['fpr'][c]['partial']:+9.3f}")
    # grouped-OOF R^2: known vs known + quality
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    log("\ngrouped-OOF R^2 (HGB) per fold, known vs known+quality:")
    res["r2"] = {}
    for target in ("miss", "fpr"):
        y = df[target].to_numpy()
        out = {}
        for name, cols in (("known", kcols), ("known+quality", kcols + qcols)):
            X = df[cols].fillna(0).to_numpy()
            pred = np.zeros(len(y))
            for tr, te in GroupKFold(n_splits=5).split(X, y, df["fold"].to_numpy()):
                m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, random_state=0)
                m.fit(X[tr], y[tr])
                pred[te] = m.predict(X[te])
            r2f = []
            for f in range(5):
                sel = df["fold"].to_numpy() == f
                ss = ((y[sel] - pred[sel]) ** 2).sum()
                st_ = ((y[sel] - y[sel].mean()) ** 2).sum()
                r2f.append(1 - ss / max(st_, 1e-9))
            out[name] = r2f
            log(f"  {target:5s} {name:14s} " + " ".join(f"{x:+.3f}" for x in r2f) + f"  mean {np.mean(r2f):+.3f}")
        gain = [out["known+quality"][f] - out["known"][f] for f in range(5)]
        res["r2"][target] = dict(known=out["known"], with_quality=out["known+quality"], gain=gain)
        log(f"  {target:5s} gain from quality: " + " ".join(f"{x:+.3f}" for x in gain) + f"  mean {np.mean(gain):+.3f}, folds+ {sum(x > 0 for x in gain)}/5")
    best_partial = max(abs(v["partial"]) for t in ("miss", "fpr") for v in res[t].values())
    gain_ok = any(np.mean(res["r2"][t]["gain"]) >= 0.03 and sum(x > 0 for x in res["r2"][t]["gain"]) >= 4 for t in ("miss", "fpr"))
    verdict = "OPEN: quality carries residual information -> selector feature test" if (best_partial >= 0.20 or gain_ok) else "CLOSED: image quality explains no residual beyond known image features"
    log(f"\nmax |partial rho| {best_partial:.3f}; VERDICT: {verdict}")
    res["verdict"] = verdict
    json.dump(res, open(OUT / "image_quality_residual.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
