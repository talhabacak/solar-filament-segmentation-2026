"""E130 -- what does a HUMAN score on this metric, and what is the ceiling?

Two numbers the report needs and the team keeps guessing at.

1. Human ceiling. Take one annotator's entry as the prediction and score it with
   the exact competition scorer against the OTHER annotators of the same image
   only (the self entry is excluded, otherwise every instance matches itself at
   IoU 1 and the number is meaningless). Averaged over every choice of held-out
   annotator on every multi-copy image, this is what a careful human expert
   scores against their colleagues -- the practical ceiling for any method that
   predicts a single mask set.

2. Our ceiling at current mask quality. On the same images, score our deployed
   predictions (honest out-of-fold pool at the production policy) the same way,
   and also compute the perfect-recall bound: what PQ we would reach if every
   ground-truth row we currently miss were recovered at our matched-instance
   quality (SQ), i.e. PQ_max = SQ * TP_possible / TP_possible = SQ, with the
   realistic version bounded by the annotator agreement itself.

Usage: python src/eda/human_ceiling.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from filament.pq import score  # noqa: E402

OUT = ROOT / "outputs" / "eda"
CONF, PCUT = 0.40, 0.60


def log(*a):
    print(*a, flush=True)


def main():
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    train = set(folds["file_name"].str.replace(".jpeg", "", regex=False))
    gt = gt[gt["stem"].isin(train)]
    ncop = gt.groupby("stem")["entry_id"].nunique()
    multi = sorted(ncop[ncop > 1].index)
    log(f"{len(multi)} multi-copy images, {int(ncop[multi].sum())} entries on them")

    # 1. human ceiling: each entry predicts, scored against the other entries only
    preds, gts = [], []
    for stem in multi:
        g = gt[gt["stem"] == stem]
        entries = sorted(g["entry_id"].unique())
        for held in entries:
            # entry_id is already "<annotator>-<stem>", and the scorer reads the
            # image as everything after the FIRST '-', so a synthetic image key
            # is appended to the stem to keep each held-out round separate
            ann = held.split("-", 1)[0]
            img = f"{stem}x{ann}"
            for i, r in enumerate(g.loc[g["entry_id"] == held].itertuples()):
                preds.append((f"{img}_{i}", r.segmentation_rle))
            for j, r in enumerate(g.loc[g["entry_id"] != held].itertuples()):
                other = r.entry_id.split("-", 1)[0]
                gts.append((f"{other}-{img}_{j}", r.segmentation_rle))
    pdf = pd.DataFrame(preds, columns=["filament_id", "segmentation_rle"])
    gdf = pd.DataFrame(gts, columns=["filament_id", "segmentation_rle"])
    r = score(gdf, pdf)
    log(f"\nHUMAN ceiling (one annotator vs the others): PQ {r['pq']:.4f}  SQ {r['sq']:.4f}  "
        f"RQ {r['rq']:.4f}  TP {r['tp']} FP {r['fp']} FN {r['fn']}")

    # 2. our deployed predictions on the same images, scored the same way
    cand = pd.read_parquet(OUT / "e115b_labelled_pool.parquet").merge(
        pd.read_parquet(OUT / "e115b_p_oof.parquet"), on="cand_id")
    keep = cand[(cand["conf"] >= CONF) & (cand["p_oof"] >= PCUT) & (cand["stem"].isin(multi))]
    ours = score(gt[gt["stem"].isin(multi)][["filament_id", "segmentation_rle"]],
                 keep[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"}))
    log(f"OUR pipeline on the same images (all copies): PQ {ours['pq']:.4f}  SQ {ours['sq']:.4f}  "
        f"RQ {ours['rq']:.4f}  TP {ours['tp']} FP {ours['fp']} FN {ours['fn']}")

    # 3. perfect-recall bound at our current mask quality
    sq = ours["sq"]
    tp, fp, fn = ours["tp"], ours["fp"], ours["fn"]
    perfect_recall = (sq * (tp + fn)) / ((tp + fn) + 0.5 * fp)
    no_fp = (sq * tp) / (tp + 0.5 * fn)
    log(f"\nbounds at our current mask quality (SQ {sq:.4f}) on these images:")
    log(f"  every missed row recovered, FP unchanged : PQ {perfect_recall:.4f}")
    log(f"  every false positive removed, FN unchanged: PQ {no_fp:.4f}")
    json.dump(dict(human=dict(pq=r["pq"], sq=r["sq"], rq=r["rq"], tp=r["tp"], fp=r["fp"], fn=r["fn"]),
                   ours=dict(pq=ours["pq"], sq=ours["sq"], rq=ours["rq"], tp=tp, fp=fp, fn=fn),
                   bound_perfect_recall=float(perfect_recall), bound_no_fp=float(no_fp)),
              open(OUT / "human_ceiling.json", "w"), indent=1, default=float)
    log(f"wrote {OUT / 'human_ceiling.json'}")


if __name__ == "__main__":
    main()
