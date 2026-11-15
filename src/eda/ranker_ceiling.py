"""E150 -- how much is left in the RANKING, as opposed to the operating point?

Two results from E142-E143 look like they contradict each other:

  E142  a perfect selector on today's pool scores 0.5127 against the deployed
        0.4422, so +0.0705 sits inside the candidate set;
  E143  the true candidates the gate drops are individually below the derived
        break-even (iou_sum > 0.5*c*PQ), so admitting them cannot pay.

Both are right, and the resolution names the open axis. E143 is about the
OPERATING POINT: given this ranking, moving the gate down loses. E142's oracle
does something else -- it keeps every true candidate AND drops every false one,
which is a perfect RANKING, not a lower gate.

So the question that matters is what a better ranker is worth while the budget
and the operating point stay exactly where they are. This measures it by keeping
the same NUMBER of candidates the deployed policy keeps, per image, but choosing
them by an increasingly good ranking:

  deployed         the production score
  +noise removed   the production score with candidates re-ranked by the true
                   value they realise, ONLY within the band the gate is sensitive
                   to, i.e. what a perfect tie-break would give
  perfect ranking  same count, chosen by realised value -- the ceiling of the
                   ranker axis at today's budget
  perfect + budget the oracle from E142, which also changes how many are kept

The gap between the first and the third is what a better selector can still buy
without touching candidates, masks or the operating point. The gap between the
third and the fourth is what only a different budget could add, and E143 already
says that part is unavailable.

Usage: python src/eda/ranker_ceiling.py
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


def main():
    a = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    a = a.merge(pd.read_parquet(OUT / "t020pox_oof_scores.parquet"), on="cand_id", how="left")
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    g = gt[gt["stem"].isin(set(a["stem"]))]
    a["value"] = a["y"] * a["iou_sum"]

    def sc(df, name):
        r = score(g[["filament_id", "segmentation_rle"]],
                  df[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"}))
        log(f"{name:34s} n {len(df):5d}  PQ {r['pq']:.4f}  SQ {r['sq']:.4f}  RQ {r['rq']:.4f}  "
            f"TP {r['tp']:5d} FP {r['fp']:5d}")
        return r

    gated = a[a["conf"] >= GATE[0]]
    dep = gated[gated["p"] >= GATE[1]]
    base = sc(dep, "deployed")

    # same number kept per image, chosen by the realised value: the ranker ceiling
    k = dep.groupby("stem").size()
    perfect = (gated.sort_values("value", ascending=False)
               .groupby("stem", group_keys=False)
               .apply(lambda d: d.head(int(k.get(d.name, 0))), include_groups=False)
               .reset_index())
    if "cand_id" not in perfect.columns:
        perfect = gated.loc[gated.index.isin(
            np.concatenate([gated[gated["stem"] == s].sort_values("value", ascending=False)
                            .head(int(k.get(s, 0))).index.to_numpy() for s in k.index]))]
    pr = sc(perfect, "perfect ranking, same budget")

    allt = a[a["y"] == 1]
    orc = sc(allt, "perfect ranking and budget")

    log("")
    log(f"what a better RANKER can buy at today's budget and operating point: "
        f"{pr['pq'] - base['pq']:+.4f}")
    log(f"what only a different BUDGET could add on top:                     "
        f"{orc['pq'] - pr['pq']:+.4f}")
    log(f"  (E143 showed the second part is unavailable: those candidates are")
    log(f"   individually below the break-even iou_sum > 0.5*c*PQ)")

    # where the ranker loses: the swaps a perfect ranking would make
    dep_ids, pf_ids = set(dep["cand_id"]), set(perfect["cand_id"])
    dropped = gated[gated["cand_id"].isin(dep_ids - pf_ids)]
    added = gated[gated["cand_id"].isin(pf_ids - dep_ids)]
    log(f"\nthe perfect ranking swaps {len(dropped)} of {len(dep)} kept candidates "
        f"({len(dropped) / len(dep):.1%})")
    log(f"  dropped: {(dropped['y'] == 0).mean():.1%} of them false, "
        f"mean realised value {dropped['value'].mean():.3f}")
    log(f"  added:   {(added['y'] == 0).mean():.1%} of them false, "
        f"mean realised value {added['value'].mean():.3f}")
    log(f"\ncurrent selector AUC is 0.907 on this pool; the swap rate above is what "
        f"the remaining {1 - 0.907:.3f} of it costs")
    json.dump(dict(deployed=base["pq"], perfect_ranking=pr["pq"], perfect_both=orc["pq"],
                   ranker_headroom=float(pr["pq"] - base["pq"]),
                   budget_headroom=float(orc["pq"] - pr["pq"]),
                   swap_rate=float(len(dropped) / len(dep))),
              open(OUT / "ranker_ceiling.json", "w"), indent=1, default=float)
    log(f"\nwrote {OUT / 'ranker_ceiling.json'}")


if __name__ == "__main__":
    main()
