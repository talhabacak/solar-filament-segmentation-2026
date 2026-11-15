"""Where does a single random validation split of n images put OUR pipeline?

Other teams report local PQ on one random image split (discussion 744765: 106 of 707 images,
pooled over every annotator entry, IoU >= 0.5). Such a number is not comparable with our pooled
five-fold OOF, so this script draws many random n-image subsets of sub036's out-of-fold per-image
statistics and reads off the distribution our own pipeline would show on a split of that size.

It also decomposes a reported (TP, FP, FN, SQ) into recall / precision / units per image, and
re-expresses our PQ as a function of recall at our own precision and SQ.

Inputs : outputs/eda/sub036_oof_per_image.parquet  (stem, tp, fp, fn, iou; written by
         src/eda/final_pair_selection.py)
Output : outputs/eda/random_split_reference_n{N}.json

Usage: python src/eda/random_split_reference.py [--n 106] [--reps 20000]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "eda"

# Reported in discussion 744765 (2026-10-01 and 2026-10-03): 106 images, 170 annotator entries.
REPORTED = {
    "reported_2026-10-01": dict(tp=855, fp=313, fn=403, sq=0.6870, n_img=106),
    "reported_2026-10-03": dict(tp=863, fp=293, fn=395, sq=0.6812, n_img=106),
}


def summary(tp, fp, fn, iou_sum=None, sq=None):
    sq = iou_sum / tp if sq is None else sq
    rq = tp / (tp + 0.5 * fp + 0.5 * fn)
    return dict(PQ=sq * rq, SQ=sq, RQ=rq, recall=tp / (tp + fn), precision=tp / (tp + fp),
                gt_units=tp + fn, pred_units=tp + fp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=106)
    ap.add_argument("--reps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    d = pd.read_parquet(OUT / "sub036_oof_per_image.parquet")
    A = d[["tp", "fp", "fn", "iou"]].to_numpy(np.float64)
    tot = A.sum(0)
    ours = summary(*tot[:3], iou_sum=tot[3])
    ours["gt_units_per_img"] = tot[0:3:2].sum() / len(d)
    print("ours, pooled OOF:", {k: round(float(v), 4) for k, v in ours.items()})

    rng = np.random.default_rng(a.seed)
    pq = np.empty(a.reps)
    rec = np.empty(a.reps)
    for r in range(a.reps):
        tp, fp, fn, iou = A[rng.choice(len(A), a.n, replace=False)].sum(0)
        pq[r] = iou / (tp + 0.5 * fp + 0.5 * fn)
        rec[r] = tp / (tp + fn)
    q = np.percentile(pq, [5, 50, 95, 99, 99.9])
    dist = dict(n=a.n, reps=a.reps, mean=pq.mean(), sd=pq.std(), p5=q[0], p50=q[1], p95=q[2],
                p99=q[3], p999=q[4], max=pq.max(), recall_mean=rec.mean(), recall_sd=rec.std(),
                recall_max=rec.max())
    print("our PQ on random %d-image subsets:" % a.n, {k: round(float(v), 4) for k, v in dist.items()})

    rep = {}
    for name, s in REPORTED.items():
        r = summary(s["tp"], s["fp"], s["fn"], sq=s["sq"])
        r["gt_units_per_img"] = (s["tp"] + s["fn"]) / s["n_img"]
        r["p_ours_at_least_this_pq"] = float((pq >= r["PQ"]).mean())
        r["p_ours_at_least_this_recall"] = float((rec >= r["recall"]).mean())
        rep[name] = r
        print(name, {k: round(float(v), 5) for k, v in r.items()})

    # PQ as a function of recall at our precision and SQ (per GT unit).
    curve = {}
    for rc in np.arange(0.58, 0.705, 0.02):
        tp, fn = rc, 1 - rc
        fp = tp * (1 - ours["precision"]) / ours["precision"]
        curve[f"{rc:.2f}"] = ours["SQ"] * tp / (tp + 0.5 * fp + 0.5 * fn)
    print("PQ vs recall at our precision/SQ:", {k: round(v, 4) for k, v in curve.items()})

    res = dict(ours=ours, random_subsets=dist, reported=rep, pq_vs_recall_at_our_precision=curve)
    (OUT / f"random_split_reference_n{a.n}.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")
    print("->", OUT / f"random_split_reference_n{a.n}.json")


if __name__ == "__main__":
    main()
