"""E172 -- let each candidate see the others: the cheap, honest form of "attention in the selector".

Talha, 2026-09-12: "we could try adding simple attention, to the U-Net and to the selector too."

The selector scores every candidate independently from its own 42 features. Some image-level
context is already there (`n_cands_img`, `rank_conf`, `img_cand_density`, `img_sem_area_frac`,
`nn_dist`, `n_within_200px`), but nothing that depends on how GOOD the other candidates are.
A set-transformer over an image's candidates is the expensive version of that idea; this is the
cheapest version with the same content, and it has to move before the expensive one is funded:

run the deployed selector once (out of fold, exactly as in production), then describe every
candidate by its own score RELATIVE to its image and its neighbourhood:

  p_rank      rank of my first-pass score inside my image, scaled to 0..1
  p_minus_max my score minus the best score in my image
  p_share     my score divided by the image's total score
  n_above     how many candidates in my image score higher
  nn_p        the score of my nearest-centroid neighbour
  nn_p_delta  that score minus mine
  p_sum_200   the summed score of candidates within 200 px of me

Why these and not others: the metric's break-even is per candidate, so a purely relative
feature can only help through the population where absolute score is ambiguous -- and that is
exactly the 11.5 % of on-disc ground-truth copies that have a good candidate the selector
dropped. If the pool of an image is strong, a mid-score candidate is more likely a real
filament the model rendered poorly; if the pool is weak, the same score is more likely noise.

Leakage: the first-pass score for fold F's candidates is produced by a selector fitted on the
other four folds, so the context features of a test candidate never see its own label.

Writes outputs/eda/context_features_t020.parquet; read the result with
  python outputs/eda/_feature_add_test.py outputs/eda/context_features_t020.parquet \\
      p_rank p_minus_max p_share n_above nn_p nn_p_delta p_sum_200
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/eda", "src/models"):
    sys.path.insert(0, str(ROOT / p))
from e115c_threshold_ci import make_cls, make_reg  # noqa: E402
from extra_features import EXTRA  # noqa: E402

OUT = ROOT / "outputs" / "eda"
NEW = ["p_rank", "p_minus_max", "p_share", "n_above", "nn_p", "nn_p_delta", "p_sum_200"]


def main():
    feats = list(joblib.load(ROOT / "outputs/selector/selector_v5_shrunkblob5prodts.joblib")["features"]) + list(EXTRA)
    folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    pool = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    pool = pool.merge(pd.read_parquet(OUT / "extra_features_t020.parquet")[["cand_id"] + list(EXTRA)],
                      on="cand_id", how="left")
    pool["fold"] = pool["stem"].map(fold_of).astype(int)

    # first pass: the deployed score, out of fold
    p = np.zeros(len(pool))
    for F in range(5):
        trn = pool[pool["fold"] != F]
        te = (pool["fold"] == F).to_numpy()
        X, y, v = trn[feats].to_numpy(np.float64), trn["y"].to_numpy(), trn["iou_sum"].to_numpy(np.float64)
        inner = np.zeros(len(trn))
        for g in [k for k in range(5) if k != F]:
            itr, msk = (trn["fold"] != g).to_numpy(), (trn["fold"] == g).to_numpy()
            c = make_cls().fit(X[itr], y[itr])
            r = make_reg().fit(X[itr][y[itr] == 1], v[itr][y[itr] == 1])
            inner[msk] = c.predict_proba(X[msk])[:, 1] * np.clip(r.predict(X[msk]), 0, None)
        ref = np.sort(inner)
        cls = make_cls().fit(X, y)
        reg = make_reg().fit(X[y == 1], v[y == 1])
        Z = pool.loc[te, feats].to_numpy(np.float64)
        p[te] = np.searchsorted(ref, cls.predict_proba(Z)[:, 1] * np.clip(reg.predict(Z), 0, None),
                                side="right") / len(ref)
        print(f"  fold {F}: first-pass scores for {int(te.sum())} candidates", flush=True)
    pool["p1"] = p

    rows = []
    for stem, g in pool.groupby("stem"):
        pv = g["p1"].to_numpy()
        cx, cy = g["cx"].to_numpy(), g["cy"].to_numpy()
        order = pv.argsort()
        rank = np.empty(len(pv))
        rank[order] = np.arange(len(pv))
        rank = rank / max(len(pv) - 1, 1)
        tot = pv.sum()
        d = np.hypot(cx[:, None] - cx[None, :], cy[:, None] - cy[None, :])
        np.fill_diagonal(d, np.inf)
        nn = d.argmin(axis=1) if len(pv) > 1 else np.zeros(len(pv), int)
        near = (d < 200)
        for i, cid in enumerate(g["cand_id"].to_numpy()):
            rows.append(dict(cand_id=cid, p_rank=float(rank[i]),
                             p_minus_max=float(pv[i] - pv.max()),
                             p_share=float(pv[i] / tot) if tot > 0 else 0.0,
                             n_above=int((pv > pv[i]).sum()),
                             nn_p=float(pv[nn[i]]) if len(pv) > 1 else 0.0,
                             nn_p_delta=float(pv[nn[i]] - pv[i]) if len(pv) > 1 else 0.0,
                             p_sum_200=float(pv[near[i]].sum())))
    d = pd.DataFrame(rows)
    d.to_parquet(OUT / "context_features_t020.parquet", index=False)
    print(f"wrote {len(d)} rows -> context_features_t020.parquet")
    # sanity: do the new features separate positives at all?
    m = pool[["cand_id", "y"]].merge(d, on="cand_id")
    print("\nmean by label:")
    print(m.groupby("y")[NEW].mean().round(3).to_string())
    print("CONTEXT_FEATURES_DONE")


if __name__ == "__main__":
    main()
