"""Read the board's true ordering of our own submissions, and decode it.

The public score is displayed and returned to two decimals, so fourteen of our
twenty submissions read 0.38 and three read 0.39 -- the display separates almost
nothing. But the submissions endpoint takes a sort argument,
SUBMISSION_SORT_BY_PUBLIC_SCORE, and Kaggle sorts on the true unrounded value.
The returned ORDER therefore carries information the returned VALUES do not.

Verified before use: three consecutive calls give an identical order, and the
0.38 block comes back as 010, 011, 012, 007, 008, 017, 009, 014, 016, 013 --
neither date order nor its reverse. The ranking is real.

What this buys. Each submission was built to differ from a named predecessor by
exactly one design decision, so a full ordering turns a pile of ties into a set
of A/B results. This encodes the design of every arm and mechanically reports
every pair separated by one axis, rather than letting me pick the pairs that
suit a story. Pairs differing in two or more axes are listed separately and
explicitly marked unreadable -- I have already once read a two-axis pair as if
it were one (sub010 over sub011 as "feature hygiene helps") and had to retract
it when sub017 isolated that axis.

One honest limit: an ordering gives direction, never magnitude. Adjacent arms
may be separated by 0.0001 or by 0.004, and nothing here can tell which.

Usage: python src/eda/board_ordering.py
"""

import json
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "eda"
COMP = "filament-segmentation-2026"

# The design of each arm, from the submission descriptions written at build time.
#
# `p` is NOT comparable across the two_stage boundary: the two-stage score is
# mapped through its training CDF, so its p is a quantile (0.55 = keep the top
# 45%), while the plain classifier's p is a raw probability. Pairs that cross
# that boundary are treated as differing in the scoring rule, not the threshold.
DESIGN = {
    "sub007": dict(semantic="prod5", shrink="prod", feats=32, policy=(0.40, 0.35), two_stage=False),
    "sub008": dict(semantic="recipe3", shrink="prod", feats=32, policy=(0.40, 0.35), two_stage=False),
    "sub009": dict(semantic="recipe3_mixed", shrink="prod", feats=32, policy=(0.40, 0.35), two_stage=False),
    "sub010": dict(semantic="recipe3", shrink="prod", feats=23, policy=(0.40, 0.35), two_stage=False),
    "sub011": dict(semantic="recipe3", shrink="true010", feats=32, policy=(0.40, 0.40), two_stage=False),
    "sub012": dict(semantic="blob_1of4", shrink="true010", feats=32, policy=(0.40, 0.40), two_stage=False),
    "sub013": dict(semantic="blob_1of2", shrink="true010", feats=32, policy=(0.40, 0.40), two_stage=False),
    "sub014": dict(semantic="blob_1of2", shrink="true010", feats=32, policy=(0.40, 0.55), two_stage=True),
    "sub015": dict(semantic="recipe3", shrink="true010", feats=32, policy=(0.40, 0.55), two_stage=True),
    "sub016": dict(semantic="recipe3", shrink="true010", feats=23, policy=(0.35, 0.50), two_stage=True),
    "sub017": dict(semantic="recipe3", shrink="true010", feats=23, policy=(0.40, 0.55), two_stage=True),
    "sub018": dict(semantic="recipe3", shrink="prod", feats=32, policy=(0.40, 0.55), two_stage=True),
    "sub019": dict(semantic="blob5_1of2", shrink="true010", feats=32, policy=(0.40, 0.55), two_stage=True),
    "sub020": dict(semantic="blob5_1of2", shrink="prod", feats=32, policy=(0.40, 0.55), two_stage=True),
    "sub021": dict(semantic="blob5_pure", shrink="true010", feats=32, policy=(0.40, 0.55), two_stage=True),
    "sub023": dict(semantic="blob5_1of2", shrink="true019", feats=32, policy=(0.40, 0.55), two_stage=True),
    "sub024": dict(semantic="blob5_65", shrink="prod", feats=32, policy=(0.40, 0.55), two_stage=True),
}
AXES = ["semantic", "shrink", "feats", "policy", "two_stage"]
MEANING = {
    "semantic": "semantic ensemble",
    "shrink": "mask shrink rule",
    "feats": "selector feature set",
    "policy": "keep threshold",
    "two_stage": "scoring rule (p vs p x E[iou])",
}


def log(*a):
    print(*a, flush=True)


def fetch():
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.competitions.types.competition_enums import SubmissionSortBy
    api = KaggleApi()
    api.authenticate()
    orders = []
    for _ in range(2):
        s = api.competition_submissions(
            COMP, sort=SubmissionSortBy.SUBMISSION_SORT_BY_PUBLIC_SCORE, page_size=100)
        orders.append([(r.file_name, r.public_score) for r in s])
    if orders[0] != orders[1]:
        log("WARNING: two calls disagreed -- the sort is not deterministic, "
            "so the order cannot be read as a ranking. Stopping.")
        sys.exit(1)
    return orders[0]


def main():
    rows = fetch()
    log(f"{len(rows)} submissions, sorted by TRUE public score "
        f"(displayed value is rounded to 2 decimals)\n")
    rank, seen = {}, set()
    log(f"  {'#':>3}  {'submission':<38}{'shown':>7}")
    for i, (fn, sc) in enumerate(rows, 1):
        tag = fn[:6]
        log(f"  {i:>3}. {fn[:38]:<38}{sc:>7}"
            + ("   (duplicate file)" if tag in seen else ""))
        seen.add(tag)
        if tag in DESIGN and tag not in rank:      # first copy is the best copy
            rank[tag] = i

    known = sorted(rank, key=lambda t: rank[t])
    log(f"\n{len(known)} of them carry a recorded design.\n")

    single, multi = [], []
    for a, b in combinations(known, 2):
        diff = [k for k in AXES if DESIGN[a][k] != DESIGN[b][k]]
        (single if len(diff) == 1 else multi).append((a, b, diff))

    log("SINGLE-VARIABLE PAIRS -- each is a clean A/B on the board")
    log(f"  {'winner':<8}{'loser':<8}{'axis':<12}{'from':>16} {'to':>16}")
    readings = []
    for a, b, diff in single:
        w, l = (a, b) if rank[a] < rank[b] else (b, a)
        k = diff[0]
        readings.append(dict(winner=w, loser=l, axis=k,
                             won_with=str(DESIGN[w][k]), lost_with=str(DESIGN[l][k]),
                             rank_w=rank[w], rank_l=rank[l]))
        log(f"  {w:<8}{l:<8}{k:<12}{str(DESIGN[l][k]):>16} > {str(DESIGN[w][k]):>16}"
            .replace(" > ", "  ->  "))

    log("\nby axis -- what the board says about each design decision")
    for k in AXES:
        rs = [r for r in readings if r["axis"] == k]
        if not rs:
            continue
        log(f"\n  {k} ({MEANING[k]})")
        for r in rs:
            log(f"    {r['won_with']:>14} beats {r['lost_with']:<14} "
                f"({r['winner']} #{r['rank_w']} over {r['loser']} #{r['rank_l']})")

    log(f"\n{len(multi)} pairs differ in two or more axes and are NOT readable; "
        f"listing the closest few for the record:")
    for a, b, diff in sorted(multi, key=lambda t: abs(rank[t[0]] - rank[t[1]]))[:6]:
        w, l = (a, b) if rank[a] < rank[b] else (b, a)
        log(f"    {w} #{rank[w]} over {l} #{rank[l]} -- differs in {', '.join(diff)}")

    json.dump(dict(order=[{"file": f, "shown": s} for f, s in rows],
                   rank=rank, single=readings,
                   multi=[dict(a=a, b=b, axes=d) for a, b, d in multi]),
              open(OUT / "board_ordering.json", "w"), indent=1)
    log(f"\nwrote {OUT/'board_ordering.json'}")
    log("\nDirection only. An ordering cannot say whether two adjacent arms are")
    log("separated by 0.0001 or by 0.004, so no delta here should be summed.")


if __name__ == "__main__":
    main()
