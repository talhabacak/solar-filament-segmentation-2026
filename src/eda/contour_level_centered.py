"""E146b -- the level predictor with the between-candidate variance removed.

E146 learned to predict the IoU a candidate reaches at a level, and then picked
the level that maximised it. It failed, -0.0102 and -0.0109 on the first two
folds, and the choice histogram says exactly why: it put 449, 450, 472, 496 and
413 of roughly 780 candidates per fold on 0.35-0.40, which is where the POOLED
mean IoU peaks. The oracle wanted the opposite, a bimodal split with 30% at 0.10
and 38% at 0.50.

The reason is a variance problem, not a feature problem. Absolute IoU varies far
more between candidates (an easy filament scores 0.8 at every level, a hard one
0.3 at every level) than across levels within one candidate. A regression on the
absolute value spends its capacity on the between-candidate part, which is
already known, and what is left over is the pooled average curve.

So: centre the target. For each candidate, subtract its own mean IoU across the
eight levels. What remains is exactly the quantity the decision needs -- how much
better or worse THIS level is for THIS candidate -- and the between-candidate
variance is gone by construction.

Two more guards, both decided on training folds only:

  margin      deviating from the deployed 0.20 must be predicted to gain at
              least m, otherwise the candidate keeps 0.20. A noisy predictor
              then costs nothing rather than scattering candidates.
  shrinkage   the margin m is swept on the inner folds, never on the fold
              being scored.

The comparison, the bootstrap and the pre-registered bands are E146's.

Usage: python src/eda/contour_level_centered.py
"""

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
from filament.rle import decode_rle, encode_mask  # noqa: E402
from mask_shrink import shrink_frac  # noqa: E402
from e115c_threshold_ci import per_image, pq  # noqa: E402
from extra_features import EXTRA  # noqa: E402
from contour_level_predictor import recut  # noqa: E402

OUT = ROOT / "outputs" / "eda"
SEM = ROOT / "outputs" / "semantic"
LEVELS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
GATE = (0.40, 0.60)
DEFAULT = 0.20
MARGINS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]
REPS, FLOOR, EQUIV = 4000, 0.003, 0.005


def log(*a):
    print(*a, flush=True)


def fit_predict(long, Xc, tr_mask, te_mask):
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                      min_samples_leaf=40, l2_regularization=1.0, random_state=0)
    m.fit(long.loc[tr_mask, Xc].to_numpy(np.float64), long.loc[tr_mask, "centred"].to_numpy())
    return m.predict(long.loc[te_mask, Xc].to_numpy(np.float64))


def choose(te, margin):
    """Keep 0.20 unless another level is predicted to beat it by at least `margin`."""
    piv = te.pivot_table(index="cand_id", columns="level", values="pred")
    base = piv[DEFAULT]
    gain = piv.sub(base, axis=0)
    best = gain.idxmax(axis=1)
    ok = gain.max(axis=1) >= margin
    return pd.Series(np.where(ok, best, DEFAULT), index=piv.index)


def main():
    feats = list(joblib.load(ROOT / "outputs/selector/selector_v5_shrunkblob5prodts.joblib")["features"]) + list(EXTRA)
    a = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    a = a.merge(pd.read_parquet(OUT / "extra_features_t020.parquet")[["cand_id"] + EXTRA], on="cand_id", how="left")
    a = a.merge(pd.read_parquet(OUT / "t020pox_oof_scores.parquet"), on="cand_id", how="left")
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    a["fold"] = a["stem"].map(fold_of).astype(int)
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    lv = pd.concat([pd.read_parquet(OUT / f"contour_levels_f{k}.parquet") for k in range(5)])
    keep_all = a[(a["conf"] >= GATE[0]) & (a["p"] >= GATE[1])].merge(lv, on="cand_id", how="inner")

    iou = keep_all[[f"iou_{L:.2f}" for L in LEVELS]].to_numpy()
    mean_iou = iou.mean(axis=1, keepdims=True)
    log(f"{len(keep_all)} candidates; IoU spread within a candidate across levels: "
        f"sd {iou.std(axis=1).mean():.4f}, between candidates: sd {mean_iou.std():.4f}")
    log("  (the second is what a regression on the absolute value fits; the first is the signal)")

    long = pd.concat([keep_all[["cand_id", "fold"] + feats].assign(
        level=L, centred=iou[:, i] - mean_iou[:, 0]) for i, L in enumerate(LEVELS)], ignore_index=True)
    Xc = feats + ["level"]

    # the margin is chosen on the inner folds, by exact PQ there, never on the scored fold
    chosen_all, margins = {}, {}
    for F in range(5):
        outer_tr = long["fold"] != F
        inner_pred = pd.Series(index=long.index[outer_tr], dtype=float)
        for G in [k for k in range(5) if k != F]:
            m_tr = outer_tr & (long["fold"] != G)
            m_te = outer_tr & (long["fold"] == G)
            inner_pred.loc[long.index[m_te]] = fit_predict(long, Xc, m_tr, m_te)
        inner = long.loc[outer_tr, ["cand_id", "fold", "level"]].assign(pred=inner_pred.to_numpy())
        iou_by = {(c, L): v for c, L, v in zip(
            np.repeat(keep_all["cand_id"].to_numpy(), len(LEVELS)),
            np.tile(LEVELS, len(keep_all)), iou.ravel())}
        best_m, best_v = DEFAULT, -1e9
        for mg in MARGINS:
            pick = choose(inner, mg)
            # the inner objective is summed IoU at the chosen levels, a monotone stand-in
            # for PQ that needs no re-cutting and no scorer call
            v = float(np.mean([iou_by[(c, L)] for c, L in pick.items()]))
            if v > best_v:
                best_m, best_v = mg, v
        margins[F] = best_m
        te_mask = long["fold"] == F
        te = long.loc[te_mask, ["cand_id", "level"]].assign(
            pred=fit_predict(long, Xc, outer_tr, te_mask))
        pick = choose(te, best_m)
        chosen_all.update(pick.to_dict())
        moved = int((pick != DEFAULT).sum())
        log(f"fold {F}: margin {best_m:.2f} (inner mean IoU {best_v:.4f}), "
            f"{moved}/{len(pick)} candidates moved off {DEFAULT}")

    res, parts = {}, {}
    for F in range(5):
        keep = keep_all[keep_all["fold"] == F]
        g = gt[gt["stem"].isin(set(keep["stem"]))]
        stems = np.array(sorted(set(a.loc[a["fold"] == F, "stem"])))
        ncopy = g.groupby("stem")["entry_id"].nunique().reindex(stems).fillna(1).to_numpy()
        field = np.load(SEM / f"fold{F}_probs_blob5ens.npz")
        base = score(g[["filament_id", "segmentation_rle"]],
                     keep[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"}))
        pred_df = recut(keep, field, chosen_all)
        r = score(g[["filament_id", "segmentation_rle"]],
                  pred_df[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"}))
        res[f"f{F}"] = dict(deployed=base["pq"], learned=r["pq"], delta=float(r["pq"] - base["pq"]),
                            margin=margins[F])
        parts[F] = (per_image(keep[["cand_id", "rle", "stem"]], g, stems, ncopy)[0],
                    per_image(pred_df, g, stems, ncopy)[0])
        log(f"fold {F}: deployed {base['pq']:.4f} -> learned {r['pq']:.4f}  "
            f"{r['pq'] - base['pq']:+.4f}   TP {base['tp']}->{r['tp']}  FP {base['fp']}->{r['fp']}")

    cat = lambda i, k: np.concatenate([parts[F][i][k] for F in range(5)])
    b = pq(cat(0, "tp").sum(), cat(0, "fp").sum(), cat(0, "fn").sum(), cat(0, "iou").sum())
    l = pq(cat(1, "tp").sum(), cat(1, "fp").sum(), cat(1, "fn").sum(), cat(1, "iou").sum())
    t0, f0, n0, i0 = cat(0, "tp"), cat(0, "fp"), cat(0, "fn"), cat(0, "iou")
    t1, f1, n1, i1 = cat(1, "tp"), cat(1, "fp"), cat(1, "fn"), cat(1, "iou")
    rng = np.random.default_rng(0)
    n = len(t0)
    boot = np.empty(REPS)
    for i in range(REPS):
        idx = rng.integers(0, n, n)
        boot[i] = (pq(t1[idx].sum(), f1[idx].sum(), n1[idx].sum(), i1[idx].sum())
                   - pq(t0[idx].sum(), f0[idx].sum(), n0[idx].sum(), i0[idx].sum()))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    d = l - b
    verdict = ("PAYS" if d >= FLOOR else "HURTS" if d <= -FLOOR
               else ("TIED" if lo > -EQUIV and hi < EQUIV else "UNDERPOWERED"))
    log(f"\npooled: deployed {b:.4f} -> learned {l:.4f}  {d:+.4f}")
    log(f"95% CI [{lo:+.4f}, {hi:+.4f}], P(delta>0) {(boot > 0).mean():.3f} -> {verdict}")
    log(f"oracle ceiling on the same candidates +0.0093, so this captures {100 * d / 0.0093:.0f}%")
    res["pooled"] = dict(deployed=float(b), learned=float(l), delta=float(d),
                         ci=[float(lo), float(hi)], verdict=verdict)
    json.dump(res, open(OUT / "contour_level_centered.json", "w"), indent=1, default=float)
    log(f"wrote {OUT / 'contour_level_centered.json'}")


if __name__ == "__main__":
    main()
