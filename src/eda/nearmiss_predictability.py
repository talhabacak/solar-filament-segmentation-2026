"""Is the per-instance offset a signal, or is it the annotator's drawing noise?

`nearmiss_offsets.py` reported 455 threshold crossings under a per-instance
oracle offset. Two things have to be fixed before that number means anything,
and a third has to be tested before it is worth building on.

FIX 1 -- one mask per candidate, not one per annotator. The 455 was counted over
(candidate, entry) pairs, so it quietly let a candidate wear a different width
for each annotator scoring it. The submission carries one mask. The honest
oracle picks a single offset per candidate and takes whatever that does across
all the entries of its image.

FIX 2 -- the marginal rule, not the crossing count. A crossing is not free once
you also count what the same offset does to the candidate's OTHER entries: an
offset that gains a match against annotator A can lose one against annotator B.
Scored properly, a candidate should move from the deployed geometry to offset k
only when the numerator it gains beats what the denominator charges:

    d_num > PQ * (0.5 * d_tp + 0.5 * d_fp)

which is the same marginal rule the selector already uses, applied to geometry
instead of to keep/drop.

TEST -- predictability. The oracle sees the ground truth. A deployed predictor
sees the image. `sq_anatomy` already measured that 74% of our over-segmentation
and 82% of our under-segmentation sits within 3 px of the ground-truth boundary
and concluded the residual is annotator drawing noise; the offsets these
crossings need are 1-6 px, i.e. the same size as that noise. So the question is
not "is there headroom" but "is the headroom addressed by anything observable".
This fits a grouped-CV model on the candidate features we already have and
reports the crossings a realistic predictor reaches against the oracle's.

CAVEAT, recorded rather than resolved: positive offsets dilate, and the
submission requires pixel-disjoint masks. Some of the oracle's expansions would
collide with a neighbouring candidate and are simply not available. The oracle
below is therefore still an over-estimate, and the honest reading is an upper
bound on an upper bound.

Usage: python src/eda/nearmiss_predictability.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

OUT = ROOT / "outputs" / "eda"
OFFSETS = list(range(-6, 7))
THR = 0.5
# the signed-distance grid has no zero: interior pixels are <= -1 and exterior
# >= 1, so `sdf <= -1` and `sdf <= 0` both select the mask exactly. k = -1 and
# k = 0 are the same geometry, and the effective erosion grid starts at -2.
IDENTITY = 0


def log(*a):
    print(*a, flush=True)


def main():
    df = pd.read_parquet(OUT / "nearmiss_offsets.parquet")
    pool = pd.read_parquet(ROOT / "outputs" / "selector" / "fused_pool_all_scored.parquet")
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt = gt[gt["stem"].isin(set(pool["stem"]))]
    total_gt = len(gt)
    n_ann = gt.groupby("stem")["entry_id"].nunique()

    psel = "p_BASE+SRC+CROSS"
    keep = pool[(pool[psel] >= 0.35) & (pool["conf"] >= 0.40)]
    log(f"deployed policy keeps {len(keep)} of {len(pool)} candidates")
    df = df[df["cand_id"].isin(set(keep["cand_id"]))]
    log(f"overlapping (candidate, entry) pairs among kept: {len(df)}")

    # ---- per-candidate aggregation: tp and numerator at each offset ---------
    cols = [f"o{k}" for k in OFFSETS]
    per = {}
    for k in OFFSETS:
        c = df[f"o{k}"]
        hit = c > THR
        g = df.assign(_tp=hit.astype(int), _num=np.where(hit, c, 0.0)) \
              .groupby("cand_id")[["_tp", "_num"]].sum()
        per[k] = g
    cand_ids = per[IDENTITY].index
    tp = pd.DataFrame({k: per[k]["_tp"].reindex(cand_ids).fillna(0) for k in OFFSETS})
    num = pd.DataFrame({k: per[k]["_num"].reindex(cand_ids).fillna(0) for k in OFFSETS})

    stems = keep.set_index("cand_id")["stem"].reindex(cand_ids)
    n_e = stems.map(n_ann).astype(float)
    fp = n_e.values[:, None] - tp.values            # entries this candidate misses

    # ---- current operating point (all candidates at the deployed geometry) --
    tp0 = float(tp[IDENTITY].sum())
    num0 = float(num[IDENTITY].sum())
    fp0 = float(fp[:, OFFSETS.index(IDENTITY)].sum())
    den0 = tp0 + 0.5 * fp0 + 0.5 * (total_gt - tp0)
    pq0 = num0 / den0
    log(f"\ndeployed geometry: PQ {pq0:.4f}  TP {tp0:.0f}  FP {fp0:.0f}  "
        f"FN {total_gt-tp0:.0f}")

    # ---- FIX 1 + FIX 2: one offset per candidate, chosen by the marginal rule
    d_num = num.values - num[IDENTITY].values[:, None]
    d_tp = tp.values - tp[IDENTITY].values[:, None]
    d_fp = fp - fp[:, OFFSETS.index(IDENTITY)][:, None]
    value = d_num - pq0 * (0.5 * d_tp + 0.5 * d_fp)
    best_j = value.argmax(axis=1)
    best_k = np.array(OFFSETS)[best_j]
    gain = value[np.arange(len(best_j)), best_j]
    moved = gain > 1e-9

    tp1 = tp0 + d_tp[np.arange(len(best_j)), best_j].sum()
    num1 = num0 + d_num[np.arange(len(best_j)), best_j].sum()
    fp1 = fp0 + d_fp[np.arange(len(best_j)), best_j].sum()
    den1 = tp1 + 0.5 * fp1 + 0.5 * (total_gt - tp1)
    log(f"oracle, one offset per candidate, marginal rule:")
    log(f"  PQ {num1/den1:.4f}  TP {tp1:.0f}  FP {fp1:.0f}  "
        f"({moved.sum()} of {len(moved)} candidates move)")
    log(f"  headroom over the deployed geometry: {num1/den1 - pq0:+.4f}")
    log(f"  offsets it wants:")
    vc = pd.Series(best_k[moved]).value_counts().sort_index()
    log("   " + vc.to_string().replace("\n", "\n   "))

    # ---- TEST: can anything observable predict that offset? ----------------
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold

    FEATS = [c for c in ["conf", "det_conf", "sem_conf", "n_sources", "n_det_models",
                         "det_conf_mean", "det_conf_std", "member_iou_mean",
                         "sem_prob_mean", "sem_prob_max", "sem_prob_p90",
                         "sem_frac_over_50", "sem_ring_mean", "sem_contrast",
                         "nn_dist", "n_within_200px", "area", "log10_area",
                         "r_norm", "abs_lat", "bbox_aspect", "eccentricity",
                         "boundary_complexity", "skel_len", "width_proxy"]
             if c in pool.columns]
    X = keep.set_index("cand_id").reindex(cand_ids)[FEATS].to_numpy(np.float64)
    y = gain.copy()                     # value of moving, signed target below
    y_k = best_k.astype(float)
    groups = stems.to_numpy()

    pred_k = np.zeros(len(y_k))
    for tr, te in GroupKFold(n_splits=5).split(X, y_k, groups):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06)
        m.fit(X[tr], y_k[tr])
        pred_k[te] = m.predict(X[te])
    snapped = np.array([min(OFFSETS, key=lambda k: abs(k - p)) for p in pred_k])
    j = np.array([OFFSETS.index(k) for k in snapped])
    tp2 = tp0 + d_tp[np.arange(len(j)), j].sum()
    num2 = num0 + d_num[np.arange(len(j)), j].sum()
    fp2 = fp0 + d_fp[np.arange(len(j)), j].sum()
    den2 = tp2 + 0.5 * fp2 + 0.5 * (total_gt - tp2)
    corr = float(np.corrcoef(y_k, pred_k)[0, 1])
    log(f"\ngrouped-CV predictor of the oracle offset:")
    log(f"  correlation with the oracle offset: {corr:.4f}")
    log(f"  PQ if deployed: {num2/den2:.4f}  ({num2/den2 - pq0:+.4f} vs deployed)")
    log(f"  oracle captured: "
        f"{100*(num2/den2 - pq0)/max(num1/den1 - pq0, 1e-9):.1f}%")

    json.dump(dict(pq_deployed=pq0, pq_oracle=float(num1 / den1),
                   pq_predicted=float(num2 / den2), corr=corr,
                   n_moved=int(moved.sum()), n_cand=int(len(moved))),
              open(OUT / "nearmiss_predictability.json", "w"), indent=1)
    log(f"\nwrote {OUT/'nearmiss_predictability.json'}")


if __name__ == "__main__":
    main()
