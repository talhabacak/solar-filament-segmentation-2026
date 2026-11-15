"""Experiment A: move de-overlap from pool BUILD time to SELECTION time.

The measured precondition. Build-time greedy de-overlap carves every candidate
against every earlier one, so a good candidate's mask is damaged by neighbours
the selector would never keep. The whole-mask (no-deov) pool's oracle measured
0.5321 against 0.5078 -- the damage is worth ~0.024 at the ceiling -- and the
multithresh failure isolated the mechanism exactly (pixel theft baked into the
stored mask, unrecoverable downstream).

Why the earlier no-deov attempt failed and this one is different: it changed the
POOL the selector both trained and selected on, and the learner could not
exploit the bigger oracle (the sixth capture failure). Here the change is
confined to the SUBMISSION step: candidates stay whole through featurisation and
selection, and only the ~7 masks per image that the selector actually keeps are
de-overlapped against each other, in selector-confidence order. With so few kept
masks, mutual overlap is rare, and the carving that remains happens between
masks that are all wanted -- not against hundreds of rejected proposals.

Scoring is EXACT throughout: with overlapping candidates the fast_pq shortcut's
disjointness assumption fails, so every policy is scored by the organiser-parity
scorer on the de-overlapped kept set. Slower, honest.

Compared against the production baseline 0.4286 (shrunk pool, build-time deov,
cross-fitted). The selector here is trained OOF on the whole-mask pool the same
way, so the comparison isolates when the carving happens.

Usage: python src/models/eval_seldeov.py [--shrink 0.05]
"""

import argparse
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
from mask_shrink import shrink_frac
from selector_v5 import BASE, SRC, CROSS, oof_probs

OUT = ROOT / "outputs" / "selector"
BASELINE = 0.4286


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shrink", type=float, default=0.05)
    ap.add_argument("--pool", default="fused_pool_nodeov.parquet")
    a = ap.parse_args()

    cand = pd.read_parquet(OUT / a.pool)
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt_sub = gt[gt["file_name"].isin(set(cand["stem"] + ".jpeg"))]
    log(f"{len(cand)} whole-mask candidates, {len(gt_sub)} GT rows")

    t0 = time.time()
    new_rle, areas = [], []
    for r in cand.itertuples():
        m = shrink_frac(decode_rle(r.rle).astype(bool), a.shrink)
        new_rle.append(encode_mask(m.astype(np.uint8)))
        areas.append(int(m.sum()))
    cand["rle"], cand["area"] = new_rle, areas
    cand["log10_area"] = np.log10(np.maximum(cand["area"], 1))
    log(f"shrunk by {a.shrink:.0%} ({time.time()-t0:.0f}s)")

    # label the whole-mask pool with the exact scorer (overlaps allowed here --
    # labels are per candidate and the scorer handles the matrices)
    pred_df = cand[["cand_id", "rle"]].rename(
        columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
    res = score(gt_sub[["filament_id", "segmentation_rle"]], pred_df,
                return_details=True)
    m = res["matches"]
    agg = m.groupby("pred_filament_id").agg(weight=("gt_filament_id", "size"),
                                            iou_sum=("iou", "sum"))
    cand["weight"] = cand["cand_id"].map(agg["weight"]).fillna(0)
    cand["y"] = (cand["weight"] >= 1).astype(int)
    log(f"labels: {int(cand['y'].sum())}/{len(cand)} positive")

    feats = BASE + SRC + CROSS
    cand["p_sel"] = oof_probs(cand, feats)
    from sklearn.metrics import roc_auc_score
    log(f"selector OOF AUC on whole-mask pool: "
        f"{roc_auc_score(cand['y'], cand['p_sel']):.4f}")

    def exact_policy(conf_t, p_t):
        preds = []
        for stem, g in cand[(cand["conf"] >= conf_t)
                            & (cand["p_sel"] >= p_t)].groupby("stem"):
            g = g.sort_values("p_sel", ascending=False)
            taken = None
            k = 0
            for r in g.itertuples():
                mm = decode_rle(r.rle).astype(bool)
                if taken is None:
                    taken = np.zeros_like(mm)
                mm = mm & ~taken
                if mm.sum() < 64:
                    continue
                taken |= mm
                k += 1
                preds.append((f"{stem}_{k}", encode_mask(mm.astype(np.uint8))))
        pdf = pd.DataFrame(preds, columns=["filament_id", "segmentation_rle"])
        s = score(gt_sub[["filament_id", "segmentation_rle"]], pdf)
        return s, len(pdf)

    log(f"\n{'conf':>5s} {'p_sel':>6s} {'PQ':>7s} {'SQ':>7s} {'RQ':>7s} "
        f"{'TP':>5s} {'FP':>5s} {'FN':>5s} {'n':>6s}")
    best = None
    for conf_t, p_t in ((0.40, 0.35), (0.35, 0.35), (0.30, 0.40), (0.30, 0.35),
                        (0.40, 0.30)):
        s, n = exact_policy(conf_t, p_t)
        log(f"{conf_t:5.2f} {p_t:6.2f} {s['pq']:7.4f} {s['sq']:7.4f} "
            f"{s['rq']:7.4f} {s['tp']:5d} {s['fp']:5d} {s['fn']:5d} {n:6d}")
        if best is None or s["pq"] > best[0]:
            best = (s["pq"], conf_t, p_t)

    log(f"\nbest selection-time de-overlap PQ: {best[0]:.4f} "
        f"(conf {best[1]}, p {best[2]})")
    log(f"production baseline (build-time deov): {BASELINE:.4f}  "
        f"({best[0]-BASELINE:+.4f})")
    log("NOTE: these policies are in-sample over a small grid; if the gain "
        "clears +0.005, rerun cross-fitted before adopting.")


if __name__ == "__main__":
    main()
