"""Paired cross-fit comparison of two pools, because the reported spread is not the error.

`fused_shrink_eval` prints "cross-fitted +0.0417 +/- 0.0050", and the number
after the sign is the standard DEVIATION over the 20 evaluation splits, not the
standard error of the mean. Reading it as an error bar makes a +0.0040 difference
look like noise when it may be several standard errors.

More importantly the two arms are evaluated on the SAME splits -- crossfit_gain
seeds its permutation at 13 and both pools cover the same fold-0 stems -- so the
comparison is paired. The quantity that decides it is the standard deviation of
the per-split DIFFERENCE, which cancels the split-to-split variation that
dominates each arm's spread.

This refits the selector out-of-fold on each pool, runs the same 10x2 protocol,
and reports the paired difference with a t statistic and a sign test. It exists
because the next decision costs sixteen GPU hours and should not rest on a
misread error bar.

Usage:
  python src/eda/paired_crossfit.py --a fused_pool_f0_base.parquet \
      --b fused_pool_f0_blobens.parquet --shrink 0.10 --true-frac
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament.pq import score
from filament.rle import decode_rle, encode_mask
from selector import fast_pq
from selector_v5 import BASE, SRC, CROSS, oof_probs

OUT = ROOT / "outputs" / "eda"
SEL = ROOT / "outputs" / "selector"
BASE_POL = (0.40, 128)


def log(*a):
    print(*a, flush=True)


def prepare(pool_name, shrink, true_frac, width_gate=None):
    """Shrink, relabel with the exact scorer, and fit the selector out-of-fold."""
    from mask_shrink import shrink_frac, shrink_true_frac, shrink_width_gated
    if width_gate:
        sh = lambda m, f: shrink_width_gated(m, max_width=width_gate, nominal=f)
    else:
        sh = shrink_true_frac if true_frac else shrink_frac
    cand = pd.read_parquet(SEL / pool_name)
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    files = set(cand["stem"] + ".jpeg")
    gt_sub = gt[gt["file_name"].isin(files)]
    new_rle, areas = [], []
    for r in cand.itertuples():
        m2 = sh(decode_rle(r.rle).astype(bool), shrink)
        new_rle.append(encode_mask(m2.astype(np.uint8)))
        areas.append(int(m2.sum()))
    cand["rle"], cand["area"] = new_rle, areas
    cand["log10_area"] = np.log10(np.maximum(cand["area"], 1))
    pred = cand[["cand_id", "rle"]].rename(
        columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
    res = score(gt_sub[["filament_id", "segmentation_rle"]], pred, return_details=True)
    agg = res["matches"].groupby("pred_filament_id").agg(
        weight=("gt_filament_id", "size"),
        n_entries_hit=("annotator_image", "nunique"), iou_sum=("iou", "sum"))
    for c in ["weight", "n_entries_hit", "iou_sum"]:
        cand[c] = cand["cand_id"].map(agg[c]).fillna(0)
    cand["y"] = (cand["weight"] >= 1).astype(int)
    cand["p_sel"] = oof_probs(cand, BASE + SRC + CROSS)
    stems = gt_sub["file_name"].str.replace(".jpeg", "", regex=False)
    return cand, len(gt_sub), stems


def per_split(cand, total_gt, gt_stems, files, splits):
    """PQ on each held-out half, with the policy chosen on the other half."""
    grid = [(c, p) for c in (0.0, 0.2, 0.25, 0.3, 0.35, 0.4)
            for p in np.arange(0.05, 0.70, 0.05)]
    out = []
    for tr_h, ev_h in splits:
        ctr, cev = cand[cand["stem"].isin(tr_h)], cand[cand["stem"].isin(ev_h)]
        gt_tr = int(gt_stems.isin(tr_h).sum())
        gt_ev = total_gt - gt_tr
        best, best_pq = None, -1.0
        for c, p in grid:
            k = ctr[(ctr["conf"] >= c) & (ctr["p_sel"] >= p)]
            if k.empty:
                continue
            q = fast_pq(k, gt_tr)["pq"]
            if q > best_pq:
                best, best_pq = (c, p), q
        k_ev = cev[(cev["conf"] >= best[0]) & (cev["p_sel"] >= best[1])]
        out.append(fast_pq(k_ev, gt_ev)["pq"])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="control pool")
    ap.add_argument("--b", required=True, help="candidate pool")
    ap.add_argument("--shrink", type=float, default=0.10)
    ap.add_argument("--true-frac", action="store_true")
    ap.add_argument("--width-gate", type=float, default=None,
                    help="arm B only: production erosion applied only to masks "
                         "thinner than this mean width (px); thicker masks untouched")
    a = ap.parse_args()

    t0 = time.time()
    ca, ga, sa = prepare(a.a, a.shrink, a.true_frac)
    cb, gb, sb = prepare(a.b, a.shrink, a.true_frac, a.width_gate)
    log(f"prepared both pools ({time.time()-t0:.0f}s): "
        f"A {len(ca)} candidates, B {len(cb)}")

    # the SAME splits for both arms, over the stems they share
    files = np.array(sorted(set(ca["stem"]) & set(cb["stem"])))
    log(f"shared stems: {len(files)} (A has {ca['stem'].nunique()}, "
        f"B has {cb['stem'].nunique()})")
    rng = np.random.default_rng(13)
    splits = []
    for _ in range(10):
        perm = rng.permutation(len(files))
        half = set(files[perm[:len(files) // 2]])
        splits += [(half, set(files) - half), (set(files) - half, half)]

    pa = per_split(ca, ga, sa, files, splits)
    pb = per_split(cb, gb, sb, files, splits)
    d = pb - pa
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    t = d.mean() / se if se > 0 else float("nan")
    wins = int((d > 0).sum())

    log(f"\n{'':22s} {'mean PQ':>9s} {'sd':>8s} {'se':>8s}")
    log(f"  A {Path(a.a).stem[:18]:20s} {pa.mean():9.4f} {pa.std(ddof=1):8.4f} "
        f"{pa.std(ddof=1)/np.sqrt(n):8.4f}")
    log(f"  B {Path(a.b).stem[:18]:20s} {pb.mean():9.4f} {pb.std(ddof=1):8.4f} "
        f"{pb.std(ddof=1)/np.sqrt(n):8.4f}")
    log(f"\nPAIRED difference over the same {n} splits:")
    log(f"  mean {d.mean():+.4f}   sd {d.std(ddof=1):.4f}   se {se:.4f}   "
        f"t = {t:+.2f}")
    log(f"  B beats A on {wins} of {n} splits")
    log(f"  95% interval: [{d.mean()-1.96*se:+.4f}, {d.mean()+1.96*se:+.4f}]")
    log("\n  the per-arm sd is dominated by which images land in the held-out")
    log("  half, and that cancels in the pairing -- which is why the paired sd")
    log("  is the number that decides, not the one the harness prints.")

    json.dump(dict(a=a.a, b=a.b, mean_a=float(pa.mean()), mean_b=float(pb.mean()),
                   paired_mean=float(d.mean()), paired_sd=float(d.std(ddof=1)),
                   paired_se=float(se), t=float(t), wins=wins, n=n),
              open(OUT / "paired_crossfit.json", "w"), indent=1)
    log(f"\nwrote {OUT/'paired_crossfit.json'}")


if __name__ == "__main__":
    main()
