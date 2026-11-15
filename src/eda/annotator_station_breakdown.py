"""Where does our model do badly — which annotators, which stations?

Two grouping variables the pipeline has never been broken down by:

  annotator  the GT entry id carries who drew it, so every image scored against
             several annotators gives a per-annotator PQ for the SAME
             predictions. Differences are then purely about drawing style, not
             about image difficulty.
  station    the stem's site letter (B/C/L/M/T/U) — different telescopes,
             different seeing, different atmospheric conditions.

The annotator axis matters because the target IS an annotation convention: we
measured that snapping to the physical boundary costs -0.07, that a single
annotator scored against the others reaches SQ 0.6337 while the consensus mask
reaches 0.7903. If some annotators are systematically harder to match, that is
either a style we could learn or noise we should stop chasing.

The station axis matters because it is the one grouping the fine-tune can be
rebalanced along, and because a station-specific weakness would be a domain gap
the GONG pretraining could close.

Also reports whether the two interact, and whether per-annotator difficulty
correlates with how much that annotator draws (instance count, area, boundary
complexity) — a heavier hand is a different problem from a rarer one.

Usage: python src/eda/annotator_station_breakdown.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament.pq import score

OUT = ROOT / "outputs" / "eda"


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="outputs/selector/fused_pool_all.parquet")
    ap.add_argument("--conf", type=float, default=0.40)
    a = ap.parse_args()

    cand = pd.read_parquet(ROOT / a.pool)
    cand = cand[(cand["conf"] >= a.conf) & (cand["area"] >= 128)]
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    files = set(cand["stem"] + ".jpeg")
    gt_sub = gt[gt["file_name"].isin(files)].copy()

    # entry_id is "<annotatorEntryId>-<imageStem>"; the prefix identifies the
    # annotator, the stem's 15th character is the station
    # The prefix is GGBBAA: group 01-06, batch 01-04, annotator-slot 01-03
    # (docs/02). Annotator identity is the group+slot pair; the batch is just
    # which set of images that person was given, so grouping by the raw prefix
    # would split one person across batches and hide them.
    pref = gt_sub["entry_id"].str.split("-").str[0]
    gt_sub["group"] = pref.str[:2]
    gt_sub["batch"] = pref.str[2:4]
    gt_sub["slot"] = pref.str[4:6]
    gt_sub["annotator"] = pref.str[:2] + "/" + pref.str[4:6]
    gt_sub["stem"] = gt_sub["file_name"].str.replace(".jpeg", "", regex=False)
    gt_sub["station"] = gt_sub["stem"].str[14]
    log(f"{len(gt_sub)} GT rows | {gt_sub['annotator'].nunique()} annotators "
        f"(group+slot), {gt_sub['slot'].nunique()} slots, "
        f"{gt_sub['group'].nunique()} groups, {gt_sub['station'].nunique()} "
        f"stations | {len(cand)} predictions")
    log("  GT rows per slot: "
        + str(gt_sub['slot'].value_counts().sort_index().to_dict())
        + "  <- slot 01 draws far more than slot 03")

    pred = cand[["cand_id", "rle"]].rename(
        columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
    r = score(gt_sub[["filament_id", "segmentation_rle"]], pred, return_details=True)
    log(f"overall PQ {r['pq']:.4f} SQ {r['sq']:.4f} "
        f"TP {r['tp']} FP {r['fp']} FN {r['fn']}")

    # PQ decomposes as SQ x RQ and both halves matter differently:
    #   SQ = mean IoU of the true positives     -> is the MASK right
    #   RQ = TP / (TP + FP/2 + FN/2)            -> is the DETECTION right
    # Doing this per group needs the per-ENTRY table, not per-GT-row: an FP is
    # defined against an annotator-image entry (a prediction matching no GT in
    # THAT entry), and every entry belongs to exactly one annotator and one
    # station, so the attribution is exact rather than apportioned.
    ent = r["entries"].copy()
    pref = ent["annotator_image"].str.split("-").str[0]
    ent["group"] = pref.str[:2]
    ent["batch"] = pref.str[2:4]
    ent["slot"] = pref.str[4:6]
    ent["annotator"] = pref.str[:2] + "/" + pref.str[4:6]
    ent["stem"] = ent["annotator_image"].str.split("-", n=1).str[1]
    ent["station"] = ent["stem"].str[14]

    m = r["matches"]
    hit = set(m["gt_filament_id"])
    gt_sub["matched"] = gt_sub["filament_id"].isin(hit)
    iou_of = m.groupby("gt_filament_id")["iou"].max().to_dict()
    gt_sub["iou"] = gt_sub["filament_id"].map(iou_of).fillna(0.0)

    def decompose(g):
        tp, fp, fn = g["tp"].sum(), g["fp"].sum(), g["fn"].sum()
        num, den = g["tp_iou_sum"].sum(), tp + 0.5 * fp + 0.5 * fn
        return pd.Series(dict(
            n_entries=len(g), n_gt=int(g["n_gt"].sum()),
            tp=int(tp), fp=int(fp), fn=int(fn),
            SQ=float(num / tp) if tp else 0.0,
            RQ=float(tp / den) if den else 0.0,
            PQ=float(num / den) if den else 0.0))

    res = {}
    for key in ("annotator", "slot", "group", "station"):
        g = ent.groupby(key).apply(decompose, include_groups=False)
        g = g.sort_values("PQ")
        log("")
        log(f"per {key} — PQ = SQ x RQ (our model's predictions, scored "
            f"against each {key}'s own entries):")
        log(g.to_string(float_format=lambda v: f"{v:.4f}"))
        res[key] = g.reset_index().to_dict("records")
        log(f"  PQ spread {g['PQ'].max()-g['PQ'].min():.4f} | "
            f"SQ spread {g['SQ'].max()-g['SQ'].min():.4f} | "
            f"RQ spread {g['RQ'].max()-g['RQ'].min():.4f}")


    # does an annotator's difficulty track how much they draw?
    ann = gt_sub.groupby("annotator").agg(
        recall=("matched", "mean"), n_gt=("filament_id", "size"),
        median_area=("area", "median"),
        per_image=("filament_id", "size"))
    ann["per_image"] = ann["per_image"] / gt_sub.groupby("annotator")["stem"].nunique()
    if len(ann) > 2:
        from scipy.stats import spearmanr
        for col in ("n_gt", "median_area", "per_image"):
            rho = spearmanr(ann["recall"], ann[col]).statistic
            log(f"\ncorr(recall, {col}) across annotators: rho {rho:+.4f}")
            res[f"corr_recall_{col}"] = float(rho)

    # do stations and annotators interact, or is each independent?
    cross = gt_sub.pivot_table(index="station", columns="annotator",
                               values="matched", aggfunc="mean")
    log("\nrecall by station x annotator (blank = that pair never occurs):")
    log(cross.to_string(float_format=lambda v: f"{v:.3f}", na_rep="  -  "))

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "annotator_station_breakdown.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    cross.to_csv(OUT / "recall_station_x_annotator.csv")
    log(f"\nsaved -> {OUT/'annotator_station_breakdown.json'}")


if __name__ == "__main__":
    main()
