"""The keep threshold is calibrated to the wrong PQ, and only the leaderboard can show it.

The metric-derived value rule says a candidate is worth keeping when its marginal
contribution exceeds PQ/2, because adding a prediction costs n/2 in the
denominator and pays k*IoU in the numerator. The PQ in that rule is the ACHIEVED
PQ -- and ours differs between the two places it gets used:

    local pooled PQ   0.4286      what the threshold was tuned against
    leaderboard PQ    0.38        what the pipeline actually achieves on test

A lower achieved PQ means a lower bar, so a threshold tuned at 0.4286 prunes
candidates that are net-positive at 0.38. Local validation cannot see this: on
local data the local PQ is correct, so the local optimum is genuinely optimal
there. It is only wrong where it is applied.

This is the rare change that is locally near-free and only leaderboard-measurable,
which is what makes it worth a submission slot rather than another local sweep.

Method. For each pair of adjacent thresholds in the sweep, compute the MARGINAL
value of the candidates that lie between them:

    marginal = d(sum IoU) / d(denominator)

and keep going down while marginal > PQ_target/2 * 2 = PQ_target. (The rule's
fixed point in these pooled terms is marginal == PQ, since PQ itself is
sum IoU / denominator -- adding a batch worth more than the current average
raises it.)

Usage: python src/eda/threshold_for_test_pq.py
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
LOCAL_PQ = 0.4286        # 5-fold pooled, the number the threshold was tuned at
LB_PQ = 0.38             # what sub007 actually scored


def log(*a):
    print(*a, flush=True)


def denom(r):
    return r["tp"] + 0.5 * r["fp"] + 0.5 * r["fn"]


def main():
    s = pd.read_csv(ROOT / "outputs" / "semantic" / "fold0_fused_sweep.csv")
    s = s[s["conf"] == 0.40].sort_values("p_sel", ascending=False).reset_index(drop=True)
    s["denom"] = s.apply(denom, axis=1)
    s["sum_iou"] = s["pq"] * s["denom"]

    log("sweep at conf=0.40, walking DOWN in threshold (each row admits more):")
    log(f"{'p_sel':>6s} {'pq':>7s} {'tp':>5s} {'fp':>5s} {'kept':>5s} "
        f"{'d(sumIoU)':>10s} {'d(denom)':>9s} {'marginal':>9s}  verdict")
    best_local, best_lb = None, None
    for i in range(len(s)):
        r = s.loc[i]
        if i == 0:
            log(f"{r.p_sel:6.2f} {r.pq:7.4f} {int(r.tp):5d} {int(r.fp):5d} "
                f"{int(r.n_kept):5d} {'-':>10s} {'-':>9s} {'-':>9s}  (start)")
            continue
        p = s.loc[i - 1]
        dnum, dden = r.sum_iou - p.sum_iou, r.denom - p.denom
        m = dnum / dden if dden > 0 else np.nan
        v_local = "keep" if m > LOCAL_PQ else "reject"
        v_lb = "keep" if m > LB_PQ else "reject"
        flag = "  <-- local says stop, test says keep" if (m <= LOCAL_PQ and m > LB_PQ) else ""
        log(f"{r.p_sel:6.2f} {r.pq:7.4f} {int(r.tp):5d} {int(r.fp):5d} "
            f"{int(r.n_kept):5d} {dnum:10.1f} {dden:9.1f} {m:9.3f}  "
            f"local:{v_local:6s} test:{v_lb:6s}{flag}")
        if m > LOCAL_PQ:
            best_local = r.p_sel
        if m > LB_PQ:
            best_lb = r.p_sel

    log(f"\nthreshold the LOCAL PQ ({LOCAL_PQ}) justifies : p_sel {best_local}")
    log(f"threshold the TEST  PQ ({LB_PQ}) justifies : p_sel {best_lb}")

    if best_lb is not None and best_local is not None and best_lb < best_local:
        a = s[s.p_sel == best_local].iloc[0]
        b = s[s.p_sel == best_lb].iloc[0]
        dnum, dden = b.sum_iou - a.sum_iou, b.denom - a.denom
        m = dnum / dden
        # project onto the leaderboard's own scale: the extra batch enters a pool
        # whose current average is LB_PQ, not the local one
        frac = dden / a.denom
        proj = (LB_PQ + m * frac) / (1 + frac)
        log(f"\nthe batch between them: {int(b.n_kept - a.n_kept)} extra candidates, "
            f"TP +{int(b.tp - a.tp)}, FP +{int(b.fp - a.fp)}, marginal value {m:.3f}")
        log(f"  local effect  {b.pq - a.pq:+.4f}  (it is below the local average, so it costs)")
        log(f"  test  projection {LB_PQ:.4f} -> {proj:.4f}  ({proj - LB_PQ:+.4f})")
        log(f"\n  honest size check: {proj - LB_PQ:+.4f} on a 2-decimal leaderboard "
            f"{'MAY NOT be visible' if proj - LB_PQ < 0.005 else 'should be visible'}")
        json.dump(dict(local_thr=float(best_local), test_thr=float(best_lb),
                       marginal=float(m), projected=float(proj),
                       local_cost=float(b.pq - a.pq)),
                  open(OUT / "threshold_for_test_pq.json", "w"), indent=1)
        log(f"\nsaved -> {OUT/'threshold_for_test_pq.json'}")


if __name__ == "__main__":
    main()
