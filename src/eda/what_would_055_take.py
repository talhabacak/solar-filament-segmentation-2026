"""Four teams score 0.53-0.55. What would have to be true for us to?

Not a guess at their method -- a budget. PQ decomposes exactly:

    PQ = sum(IoU over TP) / (TP + FP/2 + FN/2) = SQ * RQ

so any route to 0.55 is some combination of raising SQ (the mean IoU of masks
that already matched) and raising RQ (matching more ground truth without paying
for it in false positives). This computes which combinations are even
arithmetically available from where we stand, using the measured per-GT table.

The multi-annotator structure is what makes this non-obvious. One prediction set
is scored against EVERY annotator copy of an image, so predicting a filament that
only 1 of 3 annotators drew buys one TP and pays two FPs -- and it is easy to get
the accounting wrong. The first version of this script charged the
average contribution, SQ*k / (k + (N-k)/2), and concluded the metric wants the
minority tiers SKIPPED. That is wrong: NOT predicting a cluster still costs 0.5k
in the denominator, because those k ground-truth rows become false negatives. The
marginal is the difference:

    dNum = SQ * k
    dDen = [k + 0.5(N-k)] - 0.5k = 0.5N
    marginal = 2 * SQ * k / N

So the rule is: predict a cluster iff 2*SQ*(k/N) > PQ, i.e. k/N > PQ / (2*SQ).
At SQ 0.706 and PQ 0.41 that threshold is 0.29 -- and the lowest tier, 1 of 3,
is 0.333. Every tier clears it. The metric does NOT want us to skip
minority-consensus filaments at our current score; it wants us to find them.

(This also confirms selector_v6's rule, iou_sum/n > PQ/2, was right: iou_sum/n is
SQ*k/N, so the two are the same inequality.)

Computed per consensus tier:
  * the exact marginal value of predicting that tier, given SQ
  * the metric-optimal policy (predict a tier only while its marginal exceeds the
    PQ it produces), solved as a fixed point since the threshold depends on the
    outcome
  * the PQ that policy reaches with perfect recall inside the predicted tiers
  * the SQ that would be needed to reach 0.55

Usage: python src/eda/what_would_055_take.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "outputs" / "eda"
TARGET = 0.55


def log(*a):
    print(*a, flush=True)


def main():
    # 2026-09-14: the tier structure below is a property of the labels and does not move with
    # the operating point, but SQ does, and this parquet was built at the August one (SQ 0.706,
    # PQ 0.41). --sq lets the tier policy be re-solved at the shipped system's SQ without
    # rebuilding the profile, so the report can quote a current ceiling.
    ap = argparse.ArgumentParser()
    ap.add_argument("--sq", type=float, default=None,
                    help="override the matched-mask quality; default reads it from the parquet")
    ap.add_argument("--pq", type=float, default=None, help="the PQ that SQ goes with, for the log")
    a = ap.parse_args()
    d = pd.read_parquet(OUT / "error_profile_fn_gt.parquet")
    log(f"{len(d)} GT rows over {d['entry_id'].nunique()} annotator entries, "
        f"{d['stem'].nunique()} images\n")

    if "n_copies" not in d:
        d["n_copies"] = d.groupby("stem")["entry_id"].transform("nunique")
    if "support" not in d:
        d["support"] = d.groupby(["stem", "ann_id"])["entry_id"].transform("nunique")
    d["tier"] = (d["support"].astype(int).astype(str) + "/"
                 + d["n_copies"].astype(int).astype(str))

    sq_now = float(d.loc[d["matched"], "match_iou"].mean())
    if a.sq is not None:
        log(f"parquet SQ {sq_now:.4f} (August operating point) overridden with {a.sq:.4f}")
        sq_now = a.sq
    log(f"current: TP {int(d['matched'].sum())}, SQ {sq_now:.4f}")
    if a.pq is not None:
        log(f"tier break-even k/N > PQ/(2*SQ) = {a.pq:.4f}/(2*{sq_now:.4f}) = "
            f"{a.pq / (2 * sq_now):.4f}; the lowest tier 1/3 is 0.3333")

    g = (d.groupby("tier")
         .agg(n_gt=("filament_id", "size"), tp=("matched", "sum"),
              support=("support", "first"), n_copies=("n_copies", "first"))
         .reset_index())
    g["recall"] = g["tp"] / g["n_gt"]
    g["clusters"] = g["n_gt"] / g["support"].clip(lower=1)

    log("\nper-tier recall and the marginal value of predicting that tier")
    log(f"{'tier':>6s} {'GT rows':>8s} {'clusters':>9s} {'recall':>7s} "
        f"{'marginal':>9s}")
    for _, r in g.sort_values("tier").iterrows():
        k, N = int(r.support), int(r.n_copies)
        marg = 2.0 * sq_now * k / N
        log(f"{r.tier:>6s} {int(r.n_gt):8d} {r.clusters:9.0f} {r.recall:7.3f} "
            f"{marg:9.3f}")
    log("  marginal = 2*SQ*k/N: the change in PQ from predicting one more cluster "
        "of this tier.\n  Predict while it exceeds the PQ you already have.")

    def policy_pq(pred_tiers, sq, recall=1.0):
        num = den = 0.0
        for _, r in g.iterrows():
            k, N, n_gt = int(r.support), int(r.n_copies), int(r.n_gt)
            n_cl = n_gt / max(k, 1)
            if r.tier in pred_tiers:
                hit = n_cl * recall
                num += sq * k * hit
                den += k * hit + 0.5 * (N - k) * hit    # TP + FP on the hits
                den += 0.5 * k * (n_cl - hit)           # FN on the misses
            else:
                den += 0.5 * k * n_cl                   # every row an FN
        return num / den if den else 0.0

    def solve(sq):
        tiers = list(g["tier"])
        chosen, pq = set(tiers), 0.0
        for _ in range(50):
            pq = policy_pq(chosen, sq)
            new = set()
            for _, r in g.iterrows():
                k, N = int(r.support), int(r.n_copies)
                if 2.0 * sq * k / N > pq:
                    new.add(r.tier)
            if new == chosen:
                break
            chosen = new
        return chosen, pq

    chosen, pq = solve(sq_now)
    log(f"\nmetric-optimal policy at our SQ ({sq_now:.4f}):")
    log(f"  predict: {sorted(chosen)}")
    log(f"  skip:    {sorted(set(g['tier']) - chosen)}")
    log(f"  PQ with PERFECT recall inside the predicted tiers: {pq:.4f}")
    log("  (our actual PQ at this operating point: 0.4105)")

    log(f"\nwhat would reach {TARGET}?")
    for sq in np.arange(0.66, 0.93, 0.02):
        ch, p = solve(float(sq))
        flag = "  <- reaches the target" if p >= TARGET else ""
        log(f"  SQ {sq:.2f} -> PQ {p:.4f} (perfect recall, {len(ch)} tiers kept){flag}")

    log("\nreading: with recall already near its metric-optimal policy, SQ is the "
        "only free parameter left. Ours is 0.66-0.68 in every lane we have built, "
        "including native resolution. If 0.55 needs an SQ far above that, the gap "
        "is agreement with the annotator's drawing, not detection.")

    json.dump(dict(sq_now=sq_now, optimal_tiers=sorted(chosen),
                   pq_perfect_recall=pq, sq_overridden=a.sq is not None),
              open(OUT / "what_would_055_take.json", "w"), indent=1)
    log(f"\nsaved -> {OUT/'what_would_055_take.json'}")


if __name__ == "__main__":
    main()
