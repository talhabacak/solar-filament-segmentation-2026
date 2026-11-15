"""Can hysteresis recover the faint filaments a single threshold cannot?

The failure this targets. 30% of ground-truth rows are unreachable: no pool
candidate reaches IoU > 0.5 at any confidence. Fold-0's 440 such rows are FAINT
(median contrast 0.089 against 0.119 for matched rows) but otherwise ORDINARY --
median area 1352 against 1490, spine 119 against 123 -- and 75.5% sit on entries
the annotators agreed about, 13.6% at full 3/3 consensus.

Why the obvious fix already failed. Simply lowering the extraction threshold was
measured and made reachability WORSE (0.8179 against 0.8232): the faint filament
appears, but so does a low-probability bridge to its neighbour, and the two merge
into one component that matches neither at IoU > 0.5. Multi-threshold extraction
failed for a related reason (-0.0161): nested variants of one filament steal each
other's pixels during the pool's greedy de-overlap.

The untried mechanism. Hysteresis separates the two jobs a single threshold is
being asked to do at once. A HIGH threshold decides what exists (seeds); a LOW
threshold decides how far each thing extends (flood); a watershed grows each seed
through the flood region so two seeds can never merge into one component. That is
structurally the fix for the measured merge failure rather than a re-tuning of
the knob that failed.

Note this is NOT the distance-transform watershed already present in
semantic_candidates.py and correctly left off (that one splits a fat blob by
shape, and merge errors are only 4.9% of our FN). This one grows seeds through a
probability landscape.

The precondition, which is what this measures and what decides whether the
mechanism can work at all:

  1. Does a never-proposed row contain a HIGH-probability core at all? If it has
     no seed, no amount of flooding reaches it and hysteresis is the wrong tool --
     the honest answer would then be that the model does not represent these
     filaments, and only training changes that.
  2. If it has no core of its own, does it at least carry enough LOW-probability
     mass to be found by a lower seed level?
  3. Is its low-probability region CONNECTED to a neighbouring filament? That is
     the merge risk the watershed has to break, and if it is rare the whole
     mechanism is unnecessary; if it is universal the watershed is essential.

Usage: python src/eda/hysteresis_precondition.py
"""

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

SEM = ROOT / "outputs" / "semantic"
OUT = ROOT / "outputs" / "eda"
PROBS = "fold0_probs_cldice.npz"
HI, LO = 0.70, 0.30
MIN_SEED = 200


def log(*a):
    print(*a, flush=True)


def main():
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    f0 = set(folds.loc[folds["fold"] == 0, "file_name"]
             .str.replace(".jpeg", "", regex=False))
    d = pd.read_parquet(OUT / "error_profile_fn_gt.parquet")
    d["matched"] = d["matched"].astype(bool)
    d["in_raw_pool"] = d["in_raw_pool"].astype(bool)
    d = d[d["stem"].isin(f0)]
    never = d[~d["matched"] & ~d["in_raw_pool"]]
    match = d[d["matched"]]
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt = gt.set_index("filament_id")["segmentation_rle"]
    z = np.load(SEM / PROBS)
    log(f"fold-0: {len(never)} never-proposed rows, {len(match)} matched rows")
    log(f"probability map {PROBS}; seed level {HI}, flood level {LO}\n")

    rows, t0 = [], time.time()
    for grp, frame in (("never", never), ("matched", match)):
        for stem, g in frame.groupby("stem"):
            if stem not in z:
                continue
            prob = np.asarray(z[stem]).astype(np.float32) / 255.0
            hi = (prob >= HI).astype(np.uint8)
            lo = (prob >= LO).astype(np.uint8)
            n_lo, lab_lo = cv2.connectedComponents(lo, connectivity=8)
            n_hi, lab_hi, stats_hi, _ = cv2.connectedComponentsWithStats(
                hi, connectivity=8)
            for r in g.itertuples():
                m = decode_rle(gt.loc[r.filament_id]).astype(bool)
                if not m.any():
                    continue
                pv = prob[m]
                # does a high-probability seed of usable size live inside it?
                seeds = np.unique(lab_hi[m])
                seeds = [s for s in seeds if s != 0
                         and stats_hi[s, cv2.CC_STAT_AREA] >= MIN_SEED]
                # which flood component covers it, and what else does that
                # component cover -- the merge risk the watershed must break
                floods = np.unique(lab_lo[m])
                floods = [f for f in floods if f != 0]
                flood_area = int(sum((lab_lo == f).sum() for f in floods)) \
                    if floods else 0
                rows.append(dict(
                    grp=grp, filament_id=r.filament_id, stem=stem,
                    gt_area=int(m.sum()), p_max=float(pv.max()),
                    p_p90=float(np.percentile(pv, 90)), p_mean=float(pv.mean()),
                    frac_over_lo=float((pv >= LO).mean()),
                    frac_over_hi=float((pv >= HI).mean()),
                    has_seed=bool(len(seeds)), n_flood=len(floods),
                    flood_area=flood_area,
                    flood_bloat=flood_area / max(int(m.sum()), 1)))
        log(f"  {grp} done ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "hysteresis_precondition.parquet", index=False)
    a, b = df[df.grp == "never"], df[df.grp == "matched"]

    log("\n1. DOES THE MODEL FIRE UNDER THEM AT ALL?")
    log(f"{'':22s} {'never-proposed':>16s} {'matched':>10s}")
    for lbl, col in (("max prob inside", "p_max"), ("90th pct inside", "p_p90"),
                     ("mean prob inside", "p_mean"),
                     (f"frac over {LO}", "frac_over_lo"),
                     (f"frac over {HI}", "frac_over_hi")):
        log(f"  {lbl:20s} {a[col].median():16.3f} {b[col].median():10.3f}")

    log(f"\n2. IS THERE A SEED (a >={HI} component of >={MIN_SEED}px inside it)?")
    log(f"  never-proposed with a seed: {int(a['has_seed'].sum())} of {len(a)} "
        f"({100*a['has_seed'].mean():.1f}%)")
    log(f"  matched with a seed       : {int(b['has_seed'].sum())} of {len(b)} "
        f"({100*b['has_seed'].mean():.1f}%)")
    log("  -> rows WITHOUT a seed cannot be recovered by hysteresis at this")
    log("     seed level; they are a representation failure, not extraction.")

    log(f"\n3. MERGE RISK: how much does the {LO} flood component bloat past the "
        f"true filament?")
    for q in (0.25, 0.5, 0.75, 0.9):
        log(f"  never-proposed flood_area / gt_area  p{int(q*100):02d}: "
            f"{a['flood_bloat'].quantile(q):8.1f}   "
            f"matched: {b['flood_bloat'].quantile(q):8.1f}")
    log(f"  never-proposed touching a flood blob >5x its own area: "
        f"{100*(a['flood_bloat'] > 5).mean():.1f}% "
        f"(matched {100*(b['flood_bloat'] > 5).mean():.1f}%)")

    seeded = a[a["has_seed"]]
    log(f"\nVERDICT INPUTS")
    log(f"  recoverable-in-principle by hysteresis (has a seed): "
        f"{len(seeded)} of {len(a)} never-proposed rows")
    log(f"  of those, median flood bloat {seeded['flood_bloat'].median():.1f}x "
        f"-- the watershed has to undo this much over-extension")
    json.dump(dict(n_never=len(a), n_matched=len(b),
                   never_with_seed=int(a["has_seed"].sum()),
                   matched_with_seed=int(b["has_seed"].sum()),
                   never_p_max_median=float(a["p_max"].median()),
                   matched_p_max_median=float(b["p_max"].median()),
                   never_flood_bloat_median=float(a["flood_bloat"].median())),
              open(OUT / "hysteresis_precondition.json", "w"), indent=1)
    log(f"\nwrote {OUT/'hysteresis_precondition.json'}")


if __name__ == "__main__":
    main()
