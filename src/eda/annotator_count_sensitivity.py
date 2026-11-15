"""Does the number of annotators on an image move our optimal operating point?

The gap this is aimed at. We score 0.4290 cross-fitted on the training images
and 0.38 on the leaderboard, and the usual suspects are all ruled out:
`train_test_shift.py` measured train and test to be i.i.d. (0.03-0.15 sd over 31
shared days), the selector is cross-fitted, and the transductive experiment that
would have exploited a shift returned exactly nothing.

One structural difference has never been checked. PQ pools globally over
ANNOTATOR-IMAGE entries, so a prediction on an image with three annotators is
scored three times and a prediction on a singly-annotated image once. Our
training images are 411 single, 145 double, 151 triple -- 1.63 entries per image
on average. If the 180 test images were chosen to carry more annotators (a
natural choice for a fairer benchmark), then every marginal prediction costs
more there than it does here, and the threshold we tuned on the training mix is
too loose for the test set.

The value rule makes the direction explicit: predict iff k/N > PQ/(2*SQ), where
N is the entry count. On a singly-annotated image anything the annotator drew has
k/N = 1 and is worth predicting; at N = 3 an instance only one annotator drew has
k/N = 0.33, barely above the 0.30 bar. Higher N therefore wants a TIGHTER keep
threshold, and deploying the mixed-set threshold on a triple-annotated set would
overshoot in exactly the direction that loses PQ.

This measures the size of that effect without needing to know anything about the
test set: it sweeps the keep threshold separately within each annotator-count
stratum of our own training data and reports what a threshold tuned on the mix
costs when the deployment stratum is different.

Nothing here uses external data. The strata come from the competition's own
training annotations.

Usage: python src/eda/annotator_count_sensitivity.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from selector import fast_pq

OUT = ROOT / "outputs" / "eda"
PSEL = "p_BASE+SRC+CROSS"
GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def log(*a):
    print(*a, flush=True)


def sweep(cand, total_gt):
    """PQ over the keep-threshold grid; returns the grid table and the argmax."""
    rows = []
    for t in GRID:
        k = cand[cand[PSEL] >= t]
        r = fast_pq(k, total_gt)
        rows.append(dict(thr=float(t), **{q: r[q] for q in
                                          ("pq", "sq", "rq", "tp", "fp", "fn", "n_kept")}))
    df = pd.DataFrame(rows)
    return df, df.loc[df["pq"].idxmax()]


def main():
    cand = pd.read_parquet(ROOT / "outputs" / "selector" / "fused_pool_all_scored.parquet")
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt = gt[gt["stem"].isin(set(cand["stem"]))]

    # entries per image, straight from the annotations: this is the stratum
    n_ann = gt.groupby("stem")["entry_id"].nunique()
    cand["n_ann"] = cand["stem"].map(n_ann)
    gt["n_ann"] = gt["stem"].map(n_ann)
    log(f"pool {len(cand)} candidates, {len(gt)} ground-truth rows")
    log("images per stratum: " +
        ", ".join(f"{k}-annotator {v}" for k, v in
                  n_ann.value_counts().sort_index().items()))
    log("ground-truth rows per stratum: " +
        ", ".join(f"{k} -> {v}" for k, v in
                  gt.groupby("n_ann").size().items()))

    all_df, all_best = sweep(cand, len(gt))
    log(f"\nWHOLE TRAINING MIX (1.63 entries/image)")
    log(f"  best threshold {all_best['thr']:.2f}: PQ {all_best['pq']:.4f} "
        f"SQ {all_best['sq']:.4f} RQ {all_best['rq']:.4f} "
        f"TP {int(all_best['tp'])} FP {int(all_best['fp'])} FN {int(all_best['fn'])}")
    t_mix = float(all_best["thr"])

    log(f"\nPER STRATUM -- what the mixed threshold ({t_mix:.2f}) costs")
    log(f"{'stratum':>8s} {'n_gt':>6s} {'own thr':>8s} {'own PQ':>8s} "
        f"{'PQ @ mix':>9s} {'cost':>8s}")
    res = {}
    for k, g in cand.groupby("n_ann"):
        gt_k = int((gt["n_ann"] == k).sum())
        df, best = sweep(g, gt_k)
        at_mix = float(df.loc[df["thr"] == t_mix, "pq"].iloc[0])
        cost = at_mix - float(best["pq"])
        log(f"{int(k):8d} {gt_k:6d} {best['thr']:8.2f} {best['pq']:8.4f} "
            f"{at_mix:9.4f} {cost:+8.4f}")
        res[int(k)] = dict(n_gt=gt_k, own_thr=float(best["thr"]),
                           own_pq=float(best["pq"]), pq_at_mix=at_mix,
                           cost=cost, tp=int(best["tp"]), fp=int(best["fp"]),
                           fn=int(best["fn"]))
        df.to_csv(OUT / f"annctsens_stratum{int(k)}.csv", index=False)

    log("\nreading: 'own thr' is where each stratum peaks and 'cost' is what the")
    log("mixed-set threshold gives up there. A cost near zero means the operating")
    log("point does not care about annotator count and this cannot explain the")
    log("leaderboard gap; a large negative cost means our threshold is tuned for")
    log("the wrong entry-count mix and a hedge is worth submitting.")

    json.dump(dict(mix_thr=t_mix, mix_pq=float(all_best["pq"]), strata=res),
              open(OUT / "annotator_count_sensitivity.json", "w"), indent=1)
    all_df.to_csv(OUT / "annctsens_mix.csv", index=False)
    log(f"\nwrote {OUT/'annotator_count_sensitivity.json'}")


if __name__ == "__main__":
    main()
