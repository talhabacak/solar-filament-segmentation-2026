"""Can a GT-free proposer find the merge pairs the oracle found?

`pair_union_recovery.py` measured the oracle side: for every fold-0 ground-truth
row, the best single-candidate IoU against the best two-candidate UNION IoU, and
found 57 rows rescued with mean union IoU 0.6009 -- a ceiling of about +0.017 to
+0.022. Every number for this lane since has been oracle-side. What was never
measured is the only thing that decides whether the lane can exist: whether a
proposer that CANNOT see the ground truth picks the same pairs.

Why this lane is worth the measurement after thirteen failures. The metric's
denominator is TP + 0.5*FP + 0.5*FN, so a rule that ADDS predictions must clear a
precision bar to pay for the false positives it creates -- which is the bar all
thirteen additive mechanisms failed. A merge takes two predictions to one, so the
denominator FALLS. The three outcomes are asymmetric in our favour:

    FP + FP  ->  FP     numerator 0,      denominator -0.5   free
    FP + FP  ->  TP     numerator +IoU,   denominator -0.5   strongly positive
    TP + FP  ->  fails  numerator -IoU,   denominator -0.5   the only loss

So the lane is free or profitable on two of three arms and loses only when it
swallows a true positive. That is a different shape from every mechanism in the
ledger. It is not a guarantee: the per-instance offset lane was also
non-additive and still died, because its picker correlated 0.2177 with its own
oracle. The risk here is entirely in the picker, which is what this measures.

The proposer uses the geometry a fragmented filament actually has -- two
elongated pieces lying along a common line with a short gap -- and nothing else:

    gap          nearest-pixel distance between the two masks
    collinear    angle between the two principal axes
    on-axis      angle between the centroid-joining line and each axis, which is
                 what separates "two pieces of one filament" from "two parallel
                 filaments side by side"

Three counts decide it, and they are counted against the DEPLOYED keep set
because a merge that rescues a row nothing kept is worth nothing:

    n_rescue     union clears IoU > 0.5 on a GT row neither member reached
    n_free       both members are false positives and the union still is
    n_destroy    the merge swallows a member that was already a true positive

Kill criteria, written before the run: n_destroy >= n_rescue, or n_rescue < 30
on fold 0 (less than half the oracle's 57).

Usage: python src/eda/merge_proposer_precondition.py
"""

import itertools
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament.rle import decode_rle

OUT = ROOT / "outputs" / "eda"
PSEL, CONF = 0.35, 0.40
MAX_GAP = 60          # px at 2048; a filament is 10-12 px wide
MAX_AXIS_ANGLE = 30   # degrees between the two principal axes
MAX_ON_AXIS = 35      # degrees between the joining line and each axis
MIN_ELONG = 2.0       # both members must actually be elongated


def log(*a):
    print(*a, flush=True)


def props(m):
    """Centroid, principal-axis angle and elongation of a binary mask."""
    ys, xs = np.nonzero(m)
    if len(ys) < 8:
        return None
    cy, cx = ys.mean(), xs.mean()
    y, x = ys - cy, xs - cx
    cov = np.array([[(x * x).mean(), (x * y).mean()],
                    [(x * y).mean(), (y * y).mean()]])
    w, v = np.linalg.eigh(cov)
    if w[1] <= 1e-9:
        return None
    ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
    elong = float(np.sqrt(max(w[1], 1e-9) / max(w[0], 1e-9)))
    return dict(cy=cy, cx=cx, ang=ang, elong=elong,
                box=(int(ys.min()), int(ys.max()) + 1,
                     int(xs.min()), int(xs.max()) + 1), area=int(m.sum()))


def ang_diff(a, b):
    """Smallest angle between two undirected axes, in degrees."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def gap_px(ma, mb, pa, pb):
    """Nearest-pixel distance between two masks, computed in a joint crop."""
    y0 = min(pa["box"][0], pb["box"][0])
    y1 = max(pa["box"][1], pb["box"][1])
    x0 = min(pa["box"][2], pb["box"][2])
    x1 = max(pa["box"][3], pb["box"][3])
    a = ma[y0:y1, x0:x1]
    b = mb[y0:y1, x0:x1]
    if np.logical_and(a, b).any():
        return 0.0
    d = cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, 3)
    return float(d[b].min()) if b.any() else np.inf


def iou(a, b):
    i = int(np.logical_and(a, b).sum())
    u = int(np.logical_or(a, b).sum())
    return i / u if u else 0.0


def main():
    cand = pd.read_parquet(ROOT / "outputs" / "selector" /
                           "fused_pool_all_scored.parquet")
    # the scored pool already carries the mask; no join needed
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    f0 = set(folds.loc[folds["fold"] == 0, "file_name"]
             .str.replace(".jpeg", "", regex=False))
    cand = cand[cand["stem"].isin(f0)].copy()
    cand["kept"] = ((cand["p_BASE+SRC+CROSS"] >= PSEL) &
                    (cand["conf"] >= CONF))
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt = gt[gt["stem"].isin(f0)]
    log(f"fold 0: {len(cand)} candidates ({int(cand['kept'].sum())} kept by the "
        f"deployed policy), {len(gt)} ground-truth rows")
    log(f"proposer: gap <= {MAX_GAP}px, axis angle <= {MAX_AXIS_ANGLE}deg, "
        f"on-axis <= {MAX_ON_AXIS}deg, elongation >= {MIN_ELONG}\n")

    gt_by_stem = {s: list(g.itertuples()) for s, g in gt.groupby("stem")}
    rows, t0 = [], time.time()
    n_pairs_seen = n_proposed = 0
    for k, (stem, g) in enumerate(cand.groupby("stem")):
        g = g.reset_index(drop=True)
        masks = [decode_rle(r).astype(bool) for r in g["rle"]]
        pr = [props(m) for m in masks]
        gts = [(r.filament_id, r.entry_id, decode_rle(r.segmentation_rle).astype(bool))
               for r in gt_by_stem.get(stem, [])]
        # per-member best IoU per GT row, needed to classify the merge outcome
        base = [[iou(m, gm) for _, _, gm in gts] for m in masks]
        for i, j in itertools.combinations(range(len(masks)), 2):
            n_pairs_seen += 1
            pi, pj = pr[i], pr[j]
            if pi is None or pj is None:
                continue
            if pi["elong"] < MIN_ELONG or pj["elong"] < MIN_ELONG:
                continue
            if ang_diff(pi["ang"], pj["ang"]) > MAX_AXIS_ANGLE:
                continue
            join = np.degrees(np.arctan2(pj["cy"] - pi["cy"], pj["cx"] - pi["cx"]))
            if (ang_diff(join, pi["ang"]) > MAX_ON_AXIS
                    or ang_diff(join, pj["ang"]) > MAX_ON_AXIS):
                continue
            if gap_px(masks[i], masks[j], pi, pj) > MAX_GAP:
                continue
            n_proposed += 1
            uni = np.logical_or(masks[i], masks[j])
            for t, (fid, _eid, gm) in enumerate(gts):
                a, b = base[i][t], base[j][t]
                u = iou(uni, gm)
                rows.append(dict(stem=stem, gt_id=fid,
                                 ci=g.at[i, "cand_id"], cj=g.at[j, "cand_id"],
                                 kept_i=bool(g.at[i, "kept"]),
                                 kept_j=bool(g.at[j, "kept"]),
                                 iou_i=a, iou_j=b, iou_union=u))
        if (k + 1) % 30 == 0:
            log(f"  {k+1} stems, {n_proposed} pairs proposed "
                f"({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "merge_proposer_precondition.parquet", index=False)
    log(f"\n{n_proposed} pairs proposed out of {n_pairs_seen} possible "
        f"({100*n_proposed/max(n_pairs_seen,1):.2f}%), "
        f"{time.time()-t0:.0f}s")

    # a pair only matters if the deployed policy keeps BOTH members: merging a
    # candidate that is already dropped changes nothing that is submitted
    d = df[df["kept_i"] & df["kept_j"]]
    log(f"pairs where the deployed policy keeps both members: "
        f"{d[['ci','cj']].drop_duplicates().shape[0]}")

    rescue = d[(d["iou_union"] > 0.5) & (d["iou_i"] <= 0.5) & (d["iou_j"] <= 0.5)]
    destroy = d[(d["iou_union"] <= 0.5) &
                ((d["iou_i"] > 0.5) | (d["iou_j"] > 0.5))]
    swap = d[(d["iou_union"] > 0.5) &
             ((d["iou_i"] > 0.5) | (d["iou_j"] > 0.5))]
    log(f"\n  n_rescue  (union clears 0.5, neither member did) : "
        f"{rescue['gt_id'].nunique():4d} GT rows via "
        f"{rescue[['ci','cj']].drop_duplicates().shape[0]} pairs")
    log(f"  n_destroy (union fails, a member was already TP)  : "
        f"{destroy['gt_id'].nunique():4d} GT rows")
    log(f"  n_neutral (union clears, a member already did)    : "
        f"{swap['gt_id'].nunique():4d} GT rows")
    if len(rescue):
        log(f"  rescued rows' union IoU: median "
            f"{rescue['iou_union'].median():.3f}")

    log("\nKILL CRITERIA (written before the run):")
    nr = rescue["gt_id"].nunique()
    nd = destroy["gt_id"].nunique()
    log(f"  n_destroy >= n_rescue ?  {nd} >= {nr}  -> "
        f"{'KILL' if nd >= nr else 'pass'}")
    log(f"  n_rescue < 30 ?          {nr} < 30    -> "
        f"{'KILL' if nr < 30 else 'pass'}")
    log(f"  (oracle over ALL pairs rescued 57; this proposer sees no GT)")

    json.dump(dict(n_proposed=n_proposed, n_pairs_seen=n_pairs_seen,
                   n_rescue=int(nr), n_destroy=int(nd),
                   n_neutral=int(swap["gt_id"].nunique())),
              open(OUT / "merge_proposer_precondition.json", "w"), indent=1)
    log(f"\nwrote {OUT/'merge_proposer_precondition.json'}")


if __name__ == "__main__":
    main()
