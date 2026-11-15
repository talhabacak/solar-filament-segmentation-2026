"""Can multi-threshold extraction reach the faint filaments no candidate ever proposed?

The budget analysis (what_would_055_take.py) reduced the whole gap to one number:
1762 consensus-tier GT rows that NO candidate in our pool ever reached at
IoU > 0.5, and they are systematically fainter, thinner and smaller than what we
catch. The semantic lane extracts components at a single threshold, so a filament
whose peak probability is 0.35 simply never becomes a candidate. That is a hard
zero which no selector downstream can fix.

Why not just lower the threshold? The ledger already measured that: loosening the
candidate thresholds REDUCED reachability (0.8232 -> 0.8179), because bright
masks grow at low threshold and merge with neighbours, and a merged blob matches
nothing at IoU > 0.5. So the extraction here is seeded instead:

  1. components at the HIGH threshold are seeds -- these are today's candidates
  2. at each LOW threshold, support pixels that touch a seed are assigned to that
     seed (the bright object keeps its identity, slightly grown)
  3. a low-threshold component containing NO seed at all is a NEW candidate --
     exactly the faint object that never crossed the high bar. It cannot merge
     into a bright neighbour because any blob touching a seed was consumed by
     step 2.

Measured against the never-proposed GT list from error_profile_fn_gt.parquet,
fold 0 (the fold with cached OOF probability maps for every variant), with the
question asked per GT row: does ANY new candidate reach IoU > 0.5? Reachability
is a ceiling, not a score -- the selector still has to keep the right ones -- but
a hard zero becoming a reachable candidate is the only way this population can
ever contribute.

Usage: python src/eda/multithresh_reach.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament.rle import decode_rle

OUT = ROOT / "outputs" / "eda"
SEM = ROOT / "outputs" / "semantic"
HIGH = 0.5
LOWS = (0.35, 0.25, 0.15)
MIN_AREA = 100


def log(*a):
    print(*a, flush=True)


def seeded_extract(prob, high=HIGH, lows=LOWS, min_area=MIN_AREA):
    """Return (seed_masks, new_masks): today's components plus seedless faint ones."""
    hb = (prob > high).astype(np.uint8)
    n_hi, lab_hi = cv2.connectedComponents(hb, connectivity=8)
    seeds = [(lab_hi == i) for i in range(1, n_hi)
             if (lab_hi == i).sum() >= min_area]

    new = []
    claimed = hb.astype(bool)
    for lo in lows:
        lb = (prob > lo).astype(np.uint8)
        n_lo, lab_lo = cv2.connectedComponents(lb, connectivity=8)
        # a low component is seedless iff it contains no high pixel
        has_seed = np.zeros(n_lo, bool)
        has_seed[np.unique(lab_lo[claimed])] = True
        for i in range(1, n_lo):
            if has_seed[i]:
                continue
            m = lab_lo == i
            if m.sum() < min_area:
                continue
            new.append(m)
            claimed |= m          # do not re-emit the same faint blob at a lower bar
            has_seed[i] = True
    return seeds, new


def main():
    d = pd.read_parquet(OUT / "error_profile_fn_gt.parquet")
    d["matched"] = d["matched"].astype(bool)
    d["in_raw_pool"] = d["in_raw_pool"].astype(bool)
    if "n_copies" not in d:
        d["n_copies"] = d.groupby("stem")["entry_id"].transform("nunique")
    if "support" not in d:
        d["support"] = d.groupby(["stem", "ann_id"])["entry_id"].transform("nunique")

    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    f0 = {f.rsplit(".", 1)[0] for f in folds.loc[folds["fold"] == 0, "file_name"]}
    z = np.load(SEM / "fold0_probs_cldice.npz")
    stems = [s for s in z.files if s in f0]
    log(f"fold 0: {len(stems)} cached OOF probability maps")

    # the profile table does not carry the RLE; join it back from gt_full
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    d = d.merge(gt[["filament_id", "segmentation_rle"]], on="filament_id", how="left")

    dd = d[d["stem"].isin(stems)]
    never = dd[~dd["matched"] & ~dd["in_raw_pool"]]
    log(f"fold-0 GT rows {len(dd)}, matched {int(dd['matched'].sum())}, "
        f"never-proposed {len(never)}\n")

    rows = []
    n_new_total = 0
    for k, stem in enumerate(stems):
        prob = z[stem].astype(np.float32) / 255.0
        seeds, new = seeded_extract(prob)
        n_new_total += len(new)
        sub = never[never["stem"] == stem]
        for _, r in sub.iterrows():
            g = decode_rle(r["segmentation_rle"]).astype(bool)
            best_new = 0.0
            for m in new:
                inter = (m & g).sum()
                if inter == 0:
                    continue
                iou = inter / (m.sum() + g.sum() - inter)
                best_new = max(best_new, iou)
            best_seed = 0.0
            for m in seeds:
                inter = (m & g).sum()
                if inter == 0:
                    continue
                iou = inter / (m.sum() + g.sum() - inter)
                best_seed = max(best_seed, iou)
            rows.append(dict(stem=stem, filament_id=r["filament_id"],
                             tier=f"{int(r['support'])}/{int(r['n_copies'])}",
                             area=r.get("mask_area", np.nan),
                             contrast=r.get("contrast", np.nan),
                             iou_new=best_new, iou_seed=best_seed))
        if (k + 1) % 30 == 0:
            log(f"  {k+1}/{len(stems)} images "
                f"({len(rows)} never-proposed rows checked)")

    rf = pd.DataFrame(rows)
    log(f"\n{n_new_total} seedless low-threshold candidates emitted over "
        f"{len(stems)} images ({n_new_total/len(stems):.1f}/image)")
    got = rf["iou_new"] > 0.5
    close = (rf["iou_new"] > 0.3) & ~got
    log(f"\nof {len(rf)} never-proposed GT rows on fold 0:")
    log(f"  now REACHED by a new faint candidate (IoU > 0.5): {int(got.sum()):4d} "
        f"({100*got.mean():.1f}%)")
    log(f"  close (IoU 0.3-0.5)                             : {int(close.sum()):4d} "
        f"({100*close.mean():.1f}%)")
    log(f"  still untouched (IoU <= 0.3)                    : "
        f"{int((rf['iou_new'] <= 0.3).sum()):4d}")
    if got.any():
        log("\nreached ones, by consensus tier:")
        log(rf[got]["tier"].value_counts().to_string())

    # what would the reached ones be worth? marginal 2*SQ*(k/N) each, using the
    # actual IoU the new candidate achieves rather than the pool SQ
    if got.any():
        gain_num = 0.0
        for _, r in rf[got].iterrows():
            k_, N_ = (int(x) for x in r["tier"].split("/"))
            gain_num += r["iou_new"] * k_ / 1.0
        log(f"\nsum of IoU that becomes claimable on fold 0: {gain_num:.1f}")
        log("  (enters the PQ numerator if the selector keeps them; the FP cost "
        "of the unreached copies is already counted in the marginal rule)")

    OUT.mkdir(parents=True, exist_ok=True)
    rf.to_parquet(OUT / "multithresh_reach.parquet", index=False)
    log(f"\nsaved -> {OUT/'multithresh_reach.parquet'}")


if __name__ == "__main__":
    main()
