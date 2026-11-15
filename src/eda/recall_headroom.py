"""How much PQ headroom is left in recall, and is it worth chasing?

Splits pooled-OOF false negatives by annotator consensus support (how many
copies of that image drew the filament) and computes the PQ that perfect recall
of each support tier would give at our measured SQ. Combined with the marginal
rule (predicting a filament drawn by k of N copies pays off only while
PQ < 2k/(N+k) * IoU), this says which misses are worth fixing.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from selector import fast_pq

OUT = ROOT / "outputs" / "eda"
cand = pd.read_parquet(ROOT / "outputs" / "selector" / "pooled_candidates.parquet")
gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
clusters = pd.read_parquet(OUT / "annotator_clusters.parquet")
total_gt = len(gt)

# map each GT annotation -> its cluster support (multi-copy files only)
ann2support = {}
for r in clusters.itertuples():
    for aid in r.member_ann_ids.split(","):
        ann2support[aid] = int(r.support)
gt["support"] = gt["ann_id"].map(ann2support)
n_copies = gt.groupby("file_name")["entry_id"].nunique()
gt["n_copies"] = gt["file_name"].map(n_copies)
gt.loc[gt["n_copies"] == 1, "support"] = 1          # single-copy files: support 1 of 1

print("GT population by (n_copies, support):", flush=True)
tab = gt.groupby(["n_copies", "support"]).size()
print(tab.to_string(), flush=True)

# which GT rows are currently matched? re-derive from the best pooled policy
best = cand[(cand["conf"] >= 0.30) & (cand["p_sel"] >= 0.35)]
r_now = fast_pq(best, total_gt)
print(f"\ncurrent pooled policy: PQ={r_now['pq']:.4f} TP={r_now['tp']} "
      f"FP={r_now['fp']} FN={r_now['fn']}", flush=True)

sq = r_now["sq"]
tp, fp = r_now["tp"], r_now["fp"]

# hypothetical: additionally detect every GT of a given support tier that is
# currently missed. Each such detection converts one FN into one TP at SQ,
# and (for multi-copy images where other copies lack it) adds FP.
print("\nheadroom by support tier (perfect recall of that tier, SQ kept):", flush=True)
fn_total = r_now["fn"]
for tier in [3, 2, 1]:
    tier_gt = gt[(gt["support"] == tier)]
    # assume the current miss rate applies uniformly within the tier
    share = len(tier_gt) / total_gt
    fn_tier = int(round(fn_total * share))          # approximation
    # detecting them: +fn_tier TP (each at SQ), -fn_tier FN
    # for a filament drawn by `tier` of N copies, the same prediction is an FP in
    # (N - tier) copies -> extra FP per detected instance in multi-copy images
    extra_fp = 0
    for ncop in [2, 3]:
        sub = tier_gt[tier_gt["n_copies"] == ncop]
        if len(sub):
            extra_fp += (ncop - tier) / max(tier, 1) * len(sub) / max(len(tier_gt), 1) * fn_tier
    tp2 = tp + fn_tier
    fp2 = fp + extra_fp
    fn2 = fn_total - fn_tier
    den = tp2 + 0.5 * fp2 + 0.5 * fn2
    pq2 = (sq * tp2) / den
    print(f"  support>={tier}: GT rows {len(tier_gt)} ({share:.1%}), "
          f"est. current misses {fn_tier}, extra FP {extra_fp:.0f} "
          f"-> PQ {pq2:.4f} (delta {pq2 - r_now['pq']:+.4f})", flush=True)

# marginal profitability rule at the current score level
print("\nmarginal rule: predicting a filament drawn by k of N copies pays while "
      f"PQ < 2k/(N+k)*IoU  (IoU~{sq:.2f}, PQ={r_now['pq']:.3f}):", flush=True)
for ncop in [1, 2, 3]:
    for k in range(1, ncop + 1):
        thr = 2 * k / (ncop + k) * sq
        verdict = "PREDICT" if r_now["pq"] < thr else "skip"
        print(f"  N={ncop} k={k}: threshold {thr:.3f} -> {verdict}", flush=True)
