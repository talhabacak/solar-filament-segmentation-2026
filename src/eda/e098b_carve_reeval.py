"""E098b -- why the carve-retention filter (§12) won locally and lost on the board.

§12 measured "drop a candidate that keeps less than tau of its area during
greedy de-overlap" at +0.0052 PQ, 5/5 folds, and sub026 then ranked BELOW
sub025 on Kaggle's true-score ordering. Reading
outputs/eda/_fast_carve_filter_eval.py shows two defects, and this script
prices each one on the same base so the disagreement stops being a mystery.

Defect 1 -- the carving is frozen. That script builds `carved_records` ONCE
(lines 44-66): it walks the candidates in confidence order, carves each against
`taken`, and adds every kept candidate's pixels to `taken`. The tau sweep then
merely filters that fixed list (lines 69-74). So a candidate rejected by tau
still consumed its pixels, and the candidates behind it stay carved against a
mask that was never emitted. Deployment cannot do that: dropping a candidate
frees its pixels, the next candidate keeps more area, and one that was under
the 64 px floor can now clear it. That is exactly what the board saw -- the
test prediction count rose 1179 -> 1182 and the three extra rows were false
positives. Arm `rerun` re-runs the greedy pass per tau with the rule inline,
which is what a submission actually does.

Defect 2 -- the base population is oracle-filtered. Line 34 keeps only clusters
whose vote50 mask already matches ground truth at IoU > 0.5, and no selector
threshold is applied at all. Production is the opposite: no GT, and
conf >= 0.40 with p_sel >= 0.60 has already removed most weak candidates. Two
mechanisms aimed at the same weak-candidate population do not add up. Arm
`nogt` drops the GT filter to show the effect of that choice alone.

Arms (all five folds, exact scorer, identical ordering by confidence):
  frozen  their procedure, reproduced, as a control
  rerun   the same rule applied inside a fresh greedy pass per tau
  nogt    `rerun` on every cluster, without the IoU > 0.5 keep

Usage: python src/eda/e098b_carve_reeval.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament import IMG_H, IMG_W
from filament.pq import score
from filament.rle import decode_rle, encode_mask

OUT = ROOT / "outputs" / "eda"
TAUS = [0.0, 0.50, 0.60, 0.70, 0.75, 0.80]


def log(*a):
    print(*a, flush=True)


def run_fold(fold, gt_keep):
    vb = pd.read_parquet(OUT / f"variant_bank_fold{fold}.parquet")
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt = gt[gt["stem"].isin(set(vb["stem"]))]
    gt_df = gt[["filament_id", "segmentation_rle"]]
    rle_of = dict(zip(vb.index, vb["rle"]))

    def dec(i):
        return decode_rle(rle_of[i]).astype(bool)

    vb["row"] = vb.index
    base = vb[vb["variant"] == "vote50"].set_index("cluster_id")
    sel = base[base["iou"] > 0.5] if gt_keep else base
    by_stem = {}
    for cid, r in sel.iterrows():
        by_stem.setdefault(r["stem"], []).append((float(r["conf"]), int(r["row"])))
    for lst in by_stem.values():
        lst.sort(key=lambda t: -t[0])

    # --- frozen: one carving pass, then filter by ratio (the §12 procedure)
    frozen = []
    for stem, lst in by_stem.items():
        taken = np.zeros((IMG_H, IMG_W), bool)
        for conf, i in lst:
            m = dec(i)
            a0 = int(m.sum())
            mm = m & ~taken
            a1 = int(mm.sum())
            if a1 < 64:
                continue
            frozen.append((stem, a1 / a0, encode_mask(mm.astype(np.uint8))))
            taken |= mm

    gt_cache = {}
    res = {"frozen": {}, "rerun": {}}
    for tau in TAUS:
        preds, k = [], {}
        for stem, ratio, rle in frozen:
            if ratio < tau:
                continue
            k[stem] = k.get(stem, 0) + 1
            preds.append((f"{stem}_{k[stem]}", rle))
        r = score(gt_df, pd.DataFrame(preds, columns=["filament_id", "segmentation_rle"]), gt_cache=gt_cache)
        res["frozen"][tau] = dict(n=len(preds), pq=r["pq"], tp=r["tp"], fp=r["fp"], fn=r["fn"])

        # --- rerun: the rule lives INSIDE the greedy pass, so a rejected
        # candidate's pixels stay available to the candidates behind it
        preds = []
        for stem, lst in by_stem.items():
            taken = np.zeros((IMG_H, IMG_W), bool)
            kk = 0
            for conf, i in lst:
                m = dec(i)
                a0 = int(m.sum())
                mm = m & ~taken
                a1 = int(mm.sum())
                if a1 < 64 or a1 / a0 < tau:
                    continue
                kk += 1
                preds.append((f"{stem}_{kk}", encode_mask(mm.astype(np.uint8))))
                taken |= mm
        r = score(gt_df, pd.DataFrame(preds, columns=["filament_id", "segmentation_rle"]), gt_cache=gt_cache)
        res["rerun"][tau] = dict(n=len(preds), pq=r["pq"], tp=r["tp"], fp=r["fp"], fn=r["fn"])
    return res


def report(name, per_fold):
    log(f"\n  {name}")
    log(f"  {'tau':>6}{'mean PQ':>10}{'Δ':>10}{'+folds':>8}{'preds':>8}{'TP':>7}{'FP':>7}{'FN':>7}")
    b = [per_fold[f][0.0]["pq"] for f in range(5)]
    for tau in TAUS:
        p = [per_fold[f][tau]["pq"] for f in range(5)]
        d = [p[f] - b[f] for f in range(5)]
        log(f"  {tau:>6.2f}{np.mean(p):>10.4f}{np.mean(d):>+10.4f}{sum(x > 0 for x in d):>6}/5"
            f"{sum(per_fold[f][tau]['n'] for f in range(5)):>8}"
            f"{sum(per_fold[f][tau]['tp'] for f in range(5)):>7}"
            f"{sum(per_fold[f][tau]['fp'] for f in range(5)):>7}"
            f"{sum(per_fold[f][tau]['fn'] for f in range(5)):>7}")
    return {str(t): [per_fold[f][t]["pq"] for f in range(5)] for t in TAUS}


def main():
    out = {}
    for gt_keep, tag in ((True, "gtkeep"), (False, "nogt")):
        per_fold = {"frozen": {}, "rerun": {}}
        for f in range(5):
            r = run_fold(f, gt_keep)
            per_fold["frozen"][f] = r["frozen"]
            per_fold["rerun"][f] = r["rerun"]
            log(f"fold {f} done ({tag})")
        out[f"{tag}_frozen"] = report(f"{tag}: frozen carving, filtered afterwards (the §12 procedure)", per_fold["frozen"])
        out[f"{tag}_rerun"] = report(f"{tag}: rule inside a fresh greedy pass (what a submission does)", per_fold["rerun"])
    json.dump(out, open(OUT / "e098b_carve_reeval.json", "w"), indent=1, default=float)
    log(f"\nwrote {OUT/'e098b_carve_reeval.json'}")


if __name__ == "__main__":
    main()
