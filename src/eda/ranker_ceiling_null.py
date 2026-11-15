"""E152 -- is the +0.0321 ranking ceiling real, or does it only exist for the annotator it was fitted to?

E150 measured what a perfect ranking is worth at the deployed budget: 0.4422 to
0.4743. E151 then fed the selector pixels and captured none of it (-0.0003,
TIED). Before spending more on the ranking axis, the standing rule from E147
applies: a ceiling built by ranking on the ground truth is a ceiling plus
whatever selection on annotator-specific noise contributes, and the way to
separate them is a split the choice was not made from.

The same 296 multi-annotator images allow it. For a candidate, "is this true and
how much is it worth" is computed twice, once against the FIRST annotator's rows
and once against the REST. Then three orderings are scored on the held-out
annotator, all keeping the deployed number of candidates per image:

    the production selector's own ordering       the floor
    ordered by the value realised on annotator A the transferable ceiling
    ordered by the value realised on annotator B the within-annotator ceiling,
                                                 i.e. what E150 reported

If the second sits near the first, most of the +0.0321 is a ranking that fits one
annotator's opinion and no other, and no predictor can reach it. If it sits near
the third, the ceiling is real and E151 simply failed to capture it.

That distinction decides whether the ranking axis is worth a second attempt, so
it is worth measuring before making one.

Usage: python src/eda/ranker_ceiling_null.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/eda", "src/models"):
    sys.path.insert(0, str(ROOT / p))
from filament.pq import score  # noqa: E402

OUT = ROOT / "outputs" / "eda"
GATE = (0.40, 0.60)


def log(*a):
    print(*a, flush=True)


def value_against(cand, gt_rows):
    """Per-candidate realised value (summed IoU over matched rows) against one
    annotator group, using the competition scorer's own matching."""
    pred = cand[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
    res = score(gt_rows[["filament_id", "segmentation_rle"]], pred, return_details=True)
    m = res["matches"]
    return m.groupby("pred_filament_id")["iou"].sum().reindex(cand["cand_id"]).fillna(0.0).to_numpy()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    # A is one annotator and B is the rest, so on a three-copy image B is backed by
    # two opinions and A by one. That asymmetry alone could make A's ordering look
    # worse than it is, so the roles are swappable and both directions are reported.
    ap.add_argument("--reverse", action="store_true", help="choose on the group, score on the single annotator")
    ap.add_argument("--pairs-only", action="store_true", help="restrict to two-annotator images, where the split is symmetric")
    A_ = ap.parse_args()
    a = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    a = a.merge(pd.read_parquet(OUT / "t020pox_oof_scores.parquet"), on="cand_id", how="left")
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    ncop = gt.groupby("stem")["entry_id"].nunique()
    multi = set(ncop[ncop == 2].index) if A_.pairs_only else set(ncop[ncop >= 2].index)
    a = a[a["stem"].isin(multi) & (a["conf"] >= GATE[0])].copy()
    g = gt[gt["stem"].isin(multi)]
    log(f"{len(a)} gated candidates on {a['stem'].nunique()} multi-annotator images")

    first = g.groupby("stem")["entry_id"].transform("min")
    gA = g[g["entry_id"] == first]
    gB = g[g["entry_id"] != first]
    if A_.reverse:
        gA, gB = gB, gA
    log(f"annotator A: {len(gA)} rows, annotator B group: {len(gB)} rows")

    a["vA"] = value_against(a, gA)
    a["vB"] = value_against(a, gB)
    log(f"candidates with value on A {int((a['vA'] > 0).sum())}, on B {int((a['vB'] > 0).sum())}, "
        f"on both {int(((a['vA'] > 0) & (a['vB'] > 0)).sum())}")

    dep = a[a["p"] >= GATE[1]]
    k = dep.groupby("stem").size()
    log(f"deployed keeps {len(dep)} of them\n")

    def top_by(col):
        parts = []
        for s, n in k.items():
            d = a[a["stem"] == s]
            parts.append(d.nlargest(int(n), col))
        return pd.concat(parts) if parts else a.iloc[0:0]

    def sc_on_B(df, name):
        r = score(gB[["filament_id", "segmentation_rle"]],
                  df[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"}))
        log(f"  {name:42s} PQ {r['pq']:.4f}  TP {r['tp']:4d} FP {r['fp']:4d}")
        return r["pq"]

    log("scored on the HELD-OUT annotator group, same budget per image:")
    floor = sc_on_B(dep, "the production selector's ordering")
    transfer = sc_on_B(top_by("vA"), "ordered by value on the OTHER annotator")
    within = sc_on_B(top_by("vB"), "ordered by value on the SAME annotator")

    # the same question for the OTHER oracle in the project: E142's pool ceiling,
    # which keeps every candidate that is true rather than a fixed number
    log("\nthe pool ceiling (keep every true candidate, no budget) on the same split:")
    poolA = sc_on_B(a[a["vA"] > 0], "true on the OTHER annotator")
    poolB = sc_on_B(a[a["vB"] > 0], "true on the SAME annotator")

    frac = (transfer - floor) / (within - floor) if within > floor else float("nan")
    log(f"\n  within-annotator ceiling  {within - floor:+.4f}   (this is what E150 measured)")
    log(f"  transferable ceiling      {transfer - floor:+.4f}")
    log(f"  fraction that transfers   {100 * frac:.0f}%")
    log(f"\n  scaled to E150's full-set +0.0321, the reachable part is "
        f"{0.0321 * frac:+.4f}")
    pfrac = (poolA - floor) / (poolB - floor) if poolB > floor else float("nan")
    log(f"\n  pool ceiling, within annotator   {poolB - floor:+.4f}")
    log(f"  pool ceiling, transferable       {poolA - floor:+.4f}  ({100 * pfrac:.0f}%)")
    out_json = OUT / f"ranker_ceiling_null{'_rev' if A_.reverse else ''}{'_pairs' if A_.pairs_only else ''}.json"
    json.dump(dict(floor=float(floor), transfer=float(transfer), within=float(within),
                   transferred_fraction=float(frac), reachable_of_0321=float(0.0321 * frac),
                   pool_within=float(poolB), pool_transfer=float(poolA),
                   pool_transferred_fraction=float(pfrac)),
              open(out_json, "w"), indent=1, default=float)
    log(f"\nwrote {OUT / 'ranker_ceiling_null.json'}")


if __name__ == "__main__":
    main()
