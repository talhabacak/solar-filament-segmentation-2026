"""E201 -- a local read that tries to PREDICT the board instead of contradicting it.

Ten arms now have a board rank and none of our local numbers order them: Spearman(local delta,
board rank) = +0.53 (the wrong sign), the footprint-change rules fail (ledger 40c), and the
partition noise floor is 0.0015 (ledger 38). The one thing we understand about the gap is the
POPULATION: src/eda/test_set_inference.py measured that test candidates carry ~20% less evidence
in the same score band, i.e. the two sets differ in covariates, not (necessarily) in the mapping.

That is a covariate shift, and covariate shift has a standard correction: weight each local image
by the density ratio p(test)/p(local) estimated from observable image-level features, then read
PQ as a weighted pooled ratio

    PQ_w = sum_i w_i * iou_i / sum_i w_i * (tp_i + fp_i/2 + fn_i/2)

The weights are estimated from the FEATURE distributions only -- never from board ranks, because
fitting ten points to the thing we want to predict is the selection-on-noise mistake this project
has made before. The board ranking is held out and used once, as validation: if the reweighted
read orders the arms better than the unweighted one, we have an instrument; if not, we have
proof that local validation cannot see this and should stop pretending.

Usage: python src/eda/transfer_instrument.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/eda", "src/models"):
    sys.path.insert(0, str(ROOT / p))
from e115c_threshold_ci import per_image  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

OUT = ROOT / "outputs" / "eda"
DROP = {"cand_id", "stem", "fold", "rle", "y", "iou_sum", "n_copies", "n_entries_hit", "weight"}

# arm -> (score file, conf, p) ; board rank is HELD OUT, used only at the end
ARMS = {
    "sub036 bag5 (0.40,0.60)":  ("bag5_oof_scores.parquet",    0.40, 0.60),
    "sub033 tek fit (0.40,0.60)": ("t020pox_oof_scores.parquet", 0.40, 0.60),
    "sub040 conf 0.45":         ("t020pox_oof_scores.parquet", 0.45, 0.60),
    "sub034 conf .45 p .575":   ("t020pox_oof_scores.parquet", 0.45, 0.575),
    "sub039 p 0.55":            ("t020pox_oof_scores.parquet", 0.40, 0.55),
}
BOARD = {"sub036 bag5 (0.40,0.60)": 1, "sub033 tek fit (0.40,0.60)": 6, "sub040 conf 0.45": 8,
         "sub034 conf .45 p .575": 11, "sub039 p 0.55": 19}


def image_features(pool):
    num = [c for c in pool.columns if c not in DROP and pd.api.types.is_numeric_dtype(pool[c])]
    g = pool.groupby("stem")[num].mean()
    g["n_cand"] = pool.groupby("stem").size()
    return g


def density_ratio_weights(local, test):
    """Cross-fitted p(test|x)/p(local|x) per local image. Cross-fitted so no image is weighted by
    a model that saw it, and clipped at the 1st/99th percentile so one image cannot dominate."""
    cols = [c for c in local.columns if c in test.columns]
    X = np.vstack([local[cols].to_numpy(float), test[cols].to_numpy(float)])
    y = np.r_[np.zeros(len(local)), np.ones(len(test))]
    p = np.zeros(len(X))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=15,
                                           min_samples_leaf=20, random_state=0).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    q = np.clip(p[:len(local)], 1e-3, 1 - 1e-3)
    w = q / (1 - q)
    lo, hi = np.percentile(w, [1, 99])
    return pd.Series(np.clip(w, lo, hi), index=local.index), float(np.mean(p[len(local):] > 0.5)), \
        float(np.mean(p[:len(local)] > 0.5))


def main():
    pool = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    test = pd.read_parquet(OUT / "test_pool_t020pox.parquet")
    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt = gt[gt["stem"].isin(set(pool["stem"]))]
    stems = np.array(sorted(set(pool["stem"])))
    ncopy = gt.groupby("stem")["entry_id"].nunique().reindex(stems).fillna(1).to_numpy()

    L, T = image_features(pool), image_features(test)
    w, sep_test, sep_local = density_ratio_weights(L, T)
    print(f"yogunluk orani: {len(L)} yerel / {len(T)} test goruntusu; ayirt edilebilirlik "
          f"test {sep_test:.2f} yerel {sep_local:.2f}")
    print(f"agirliklar: medyan {w.median():.3f}, [%5 {w.quantile(.05):.3f}, %95 {w.quantile(.95):.3f}], "
          f"etkin ornek {w.sum()**2/ (w**2).sum():.0f} / {len(w)}\n")
    wv = w.reindex(stems).to_numpy()

    rows = []
    for name, (sf, conf, pc) in ARMS.items():
        sc = pd.read_parquet(OUT / sf)
        col = "p" if "p" in sc.columns else sc.columns[-1]
        d = pool.merge(sc[["cand_id", col]], on="cand_id")
        keep = (d["conf"] >= conf) & (d[col] >= pc)
        st = per_image(d[keep][["cand_id", "rle", "stem"]], gt, stems, ncopy)[0]
        den = st["tp"] + st["fp"] / 2 + st["fn"] / 2
        plain = st["iou"].sum() / den.sum()
        wtd = (wv * st["iou"]).sum() / (wv * den).sum()
        rows.append(dict(kol=name, kept=int(keep.sum()), PQ=plain, PQ_agirlikli=wtd,
                         fark=wtd - plain, tahta=BOARD[name]))
        print(f"{name:26s} tutulan {int(keep.sum()):5d}  PQ {plain:.4f}  agirlikli {wtd:.4f}  "
              f"({wtd-plain:+.4f})", flush=True)
    r = pd.DataFrame(rows)
    for col in ("PQ", "PQ_agirlikli"):
        rho = spearmanr(r[col], r["tahta"]).statistic
        print(f"\nSpearman({col}, tahta sirasi) = {rho:+.3f}   "
              f"(-1.00 mukemmel: yuksek PQ = kucuk sira numarasi)")
    r = r.sort_values("PQ_agirlikli", ascending=False)
    print("\nagirlikli siralamaya gore:")
    print(r[["kol", "PQ", "PQ_agirlikli", "tahta"]].to_string(index=False))
    r.to_json(OUT / "transfer_instrument.json", orient="records", indent=1)


if __name__ == "__main__":
    main()
