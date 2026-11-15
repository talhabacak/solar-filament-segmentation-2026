"""E146 -- learn the per-candidate contour level, and see how much of the ceiling it takes.

E145 measured the ceiling on all five folds, with a control that reproduces the
deployed masks to within 0.0013 every time:

    fold      0        1        2        3        4      mean
    oracle  +0.0111  +0.0103  +0.0122  +0.0042  +0.0087  +0.0093

and the levels the oracle wants are bimodal: 30% of candidates want 0.10, 38%
want 0.50. That is not noise around one right answer, it is two populations, and
separating them is what the photometry features were added for.

The oracle uses the ground truth, so it is a bound. This is the method.

Rather than classifying the winning level, the model predicts the IoU a candidate
would reach AT a given level, with the level as an input feature. Every candidate
then contributes eight training rows instead of one, the model learns the SHAPE of
the level-versus-IoU curve rather than its argmax, and the decision is taken by
maximising the predicted curve. A classifier on the argmax throws away the fact
that 0.45 and 0.50 are nearly the same choice.

Honest throughout: the model for fold k is fitted only on the other four folds,
the chosen levels are applied to fold k, the masks are re-cut by the same routine
that produced the control, shrunk by the deployed 0.05, and scored by the exact
scorer. Reported against the deployed masks with the paired image bootstrap and
the pre-registered bands (pays at >= +0.003, hurts at <= -0.003).

Usage: python src/eda/contour_level_predictor.py
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

OUT = ROOT / "outputs" / "eda"
SEM = ROOT / "outputs" / "semantic"
LEVELS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
GATE = (0.40, 0.60)
PAD = 40
REPS, FLOOR, EQUIV = 4000, 0.003, 0.005


def log(*a):
    print(*a, flush=True)


def recut(keep, field, levels_by_cand):
    """Re-extract each candidate at its assigned level; the E145 routine exactly."""
    rows = []
    for stem, gg in keep.groupby("stem"):
        P = field[stem].astype(np.float32) / 255.0
        for cid, rle in zip(gg["cand_id"], gg["rle"]):
            m0 = decode_rle(rle).astype(bool)
            ys, xs = np.nonzero(m0)
            if len(ys) == 0:
                continue
            flat = int(np.argmax(np.where(m0, P, -1.0)))
            cy, cx = flat // P.shape[1], flat % P.shape[1]
            y0, y1 = max(ys.min() - PAD, 0), min(ys.max() + PAD + 1, P.shape[0])
            x0, x1 = max(xs.min() - PAD, 0), min(xs.max() + PAD + 1, P.shape[1])
            L = levels_by_cand[cid]
            _, lab = cv2.connectedComponents((P[y0:y1, x0:x1] >= L).astype(np.uint8))
            k = lab[cy - y0, cx - x0]
            mm = (lab == k) if k > 0 else np.zeros(lab.shape, bool)
            full = np.zeros(P.shape, bool)
            full[y0:y1, x0:x1] = mm
            rows.append((cid, encode_mask(shrink_frac(full, 0.05).astype(np.uint8)), stem))
    return pd.DataFrame(rows, columns=["cand_id", "rle", "stem"])


def main():
    from sklearn.ensemble import HistGradientBoostingRegressor
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
    log(f"level table: {len(lv)} candidates over five folds")
    keep_all = a[(a["conf"] >= GATE[0]) & (a["p"] >= GATE[1])].merge(lv, on="cand_id", how="inner")
    log(f"joined to {len(keep_all)} deployed candidates")

    # long form: one row per (candidate, level), target = the IoU reached there
    long = pd.concat([keep_all[["cand_id", "fold"] + feats].assign(
        level=L, target=keep_all[f"iou_{L:.2f}"].to_numpy()) for L in LEVELS], ignore_index=True)
    Xc = feats + ["level"]
    log(f"training rows {len(long)} ({len(feats)} candidate features + the level)")

    res, parts, chosen_all = {}, {}, {}
    for F in range(5):
        tr = long[long["fold"] != F]
        te = long[long["fold"] == F]
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                          min_samples_leaf=40, l2_regularization=1.0, random_state=0)
        m.fit(tr[Xc].to_numpy(np.float64), tr["target"].to_numpy())
        te = te.assign(pred=m.predict(te[Xc].to_numpy(np.float64)))
        pick = te.loc[te.groupby("cand_id")["pred"].idxmax(), ["cand_id", "level"]]
        chosen_all.update(dict(zip(pick["cand_id"], pick["level"])))
        log(f"fold {F}: chose " + ", ".join(
            f"{L:.2f}x{int((pick['level'] == L).sum())}" for L in LEVELS if (pick["level"] == L).any()))

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
        d = r["pq"] - base["pq"]
        res[f"f{F}"] = dict(deployed=base["pq"], learned=r["pq"], delta=float(d),
                            tp=r["tp"], fp=r["fp"], fn=r["fn"])
        pb = per_image(keep[["cand_id", "rle", "stem"]], g, stems, ncopy)[0]
        pp = per_image(pred_df, g, stems, ncopy)[0]
        parts[F] = (pb, pp)
        log(f"fold {F}: deployed {base['pq']:.4f} -> learned {r['pq']:.4f}  {d:+.4f}   "
            f"TP {base['tp']}->{r['tp']}  FP {base['fp']}->{r['fp']}")

    tp = np.concatenate([parts[F][0]["tp"] for F in range(5)])
    fp = np.concatenate([parts[F][0]["fp"] for F in range(5)])
    fn = np.concatenate([parts[F][0]["fn"] for F in range(5)])
    io = np.concatenate([parts[F][0]["iou"] for F in range(5)])
    tp2 = np.concatenate([parts[F][1]["tp"] for F in range(5)])
    fp2 = np.concatenate([parts[F][1]["fp"] for F in range(5)])
    fn2 = np.concatenate([parts[F][1]["fn"] for F in range(5)])
    io2 = np.concatenate([parts[F][1]["iou"] for F in range(5)])
    b = pq(tp.sum(), fp.sum(), fn.sum(), io.sum())
    l = pq(tp2.sum(), fp2.sum(), fn2.sum(), io2.sum())
    rng = np.random.default_rng(0)
    n = len(tp)
    boot = np.empty(REPS)
    for i in range(REPS):
        idx = rng.integers(0, n, n)
        boot[i] = (pq(tp2[idx].sum(), fp2[idx].sum(), fn2[idx].sum(), io2[idx].sum())
                   - pq(tp[idx].sum(), fp[idx].sum(), fn[idx].sum(), io[idx].sum()))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    d = l - b
    verdict = "PAYS" if d >= FLOOR else "HURTS" if d <= -FLOOR else ("TIED" if lo > -EQUIV and hi < EQUIV else "UNDERPOWERED")
    log(f"\npooled over all five folds: deployed {b:.4f} -> learned {l:.4f}  {d:+.4f}")
    log(f"95% CI [{lo:+.4f}, {hi:+.4f}], P(delta>0) {(boot > 0).mean():.3f} -> {verdict}")
    log(f"the oracle ceiling on the same candidates was +0.0093, so this captures "
        f"{100 * d / 0.0093:.0f}% of it")
    res["pooled"] = dict(deployed=float(b), learned=float(l), delta=float(d),
                         ci=[float(lo), float(hi)], verdict=verdict,
                         fraction_of_oracle=float(d / 0.0093))
    json.dump(res, open(OUT / "contour_level_predictor.json", "w"), indent=1, default=float)
    log(f"wrote {OUT / 'contour_level_predictor.json'}")


if __name__ == "__main__":
    main()
