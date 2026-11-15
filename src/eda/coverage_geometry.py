"""Where is the missing coverage -- at the ends, or along the flanks?

The measurement this refines. At p >= 0.7 the model claims 47.3% of a
never-reached filament's area against 79.6% of a matched one's, so half of these
objects is simply not being drawn. That number says HOW MUCH is missing and says
nothing about WHERE, and the two possibilities call for opposite conclusions:

  lateral      the mask is too narrow along the filament's flanks. This is the
               width axis, and it is closed -- `sq_headroom` found no global
               offset beats identity, and `nearmiss_predictability` found the
               per-instance offset unpredictable (correlation 0.2177, deploying
               a predictor costs 0.0284). Nothing more to get here.
  longitudinal the mask stops short of the filament's ENDS. A uniform
               morphological offset cannot fix this -- growing a mask by 3px
               everywhere adds width faster than length -- and neither can a
               lower threshold, which was measured to lose. It would be a
               genuinely open mechanism, and the one the multi-annotator
               core/extent formulation is built for.

Method. Each ground-truth instance is projected onto its own principal axis, so
"along the filament" and "across it" become two coordinates independent of how
the filament happens to be oriented on the disk. Missed ground-truth pixels are
then binned by their position along that axis: the outer fifth at each end
against the middle three fifths. If the model were failing uniformly, the miss
rate would be flat across those bins; a rising miss rate toward the ends is
truncation.

The comparison population matters. Matched instances are scored the same way,
because a miss rate that is high at the ends for EVERY instance is a property of
how annotators draw filament tips, not a defect specific to the ones we lose.

Usage: python src/eda/coverage_geometry.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament.rle import decode_rle

OUT = ROOT / "outputs" / "eda"
SEM = ROOT / "outputs" / "semantic"
PROBS = "fold0_probs_cldice.npz"
HI = 0.7
N_BINS = 5


def log(*a):
    print(*a, flush=True)


def axis_coords(m):
    """Coordinates of a mask's pixels along and across its principal axis."""
    ys, xs = np.nonzero(m)
    if len(ys) < 12:
        return None
    cy, cx = ys.mean(), xs.mean()
    y, x = ys - cy, xs - cx
    cov = np.array([[(x * x).mean(), (x * y).mean()],
                    [(x * y).mean(), (y * y).mean()]])
    w, v = np.linalg.eigh(cov)
    if w[1] <= 1e-9:
        return None
    major = v[:, 1]                       # (dx, dy) of the long axis
    minor = v[:, 0]
    along = x * major[0] + y * major[1]
    across = x * minor[0] + y * minor[1]
    return ys, xs, along, across, float(np.sqrt(w[1] / max(w[0], 1e-9)))


def main():
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    f0 = set(folds.loc[folds["fold"] == 0, "file_name"]
             .str.replace(".jpeg", "", regex=False))
    t = pd.read_parquet(OUT / "true_unreachable.parquet")
    unreach = set(t.loc[t["best"] <= 0.5, "filament_id"])
    d = pd.read_parquet(OUT / "error_profile_fn_gt.parquet")
    d = d[d["stem"].isin(f0)].copy()
    d["grp"] = np.where(d["filament_id"].isin(unreach), "unreachable",
                        np.where(d["matched"].astype(bool), "matched", "other"))
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt = gt.set_index("filament_id")["segmentation_rle"]
    z = np.load(SEM / PROBS)
    log(f"fold 0: " + ", ".join(f"{k} {v}" for k, v in
                                d["grp"].value_counts().items()))
    log(f"binning each instance along its own principal axis into {N_BINS} "
        f"slices; a pixel is MISSED if the model's probability is below {HI}\n")

    rows, t0 = [], time.time()
    for stem, g in d.groupby("stem"):
        if stem not in z:
            continue
        prob = np.asarray(z[stem]).astype(np.float32) / 255.0
        for r in g.itertuples():
            m = decode_rle(gt.loc[r.filament_id]).astype(bool)
            ac = axis_coords(m)
            if ac is None:
                continue
            ys, xs, along, across, elong = ac
            p = prob[ys, xs]
            hit = p >= HI
            # normalise the along-axis coordinate to [0, 1] and slice it
            lo, hi = along.min(), along.max()
            if hi - lo < 1e-6:
                continue
            u = (along - lo) / (hi - lo)
            b = np.clip((u * N_BINS).astype(int), 0, N_BINS - 1)
            for k in range(N_BINS):
                sel = b == k
                if sel.sum() < 3:
                    continue
                rows.append(dict(grp=r.grp, filament_id=r.filament_id, bin=k,
                                 n=int(sel.sum()),
                                 covered=float(hit[sel].mean()),
                                 elong=elong))
            # lateral check: coverage of the outer half of the WIDTH
            aw = np.abs(across)
            outer = aw >= np.percentile(aw, 60)
            rows.append(dict(grp=r.grp, filament_id=r.filament_id, bin=-1,
                             n=int(outer.sum()),
                             covered=float(hit[outer].mean()), elong=elong))
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "coverage_geometry.parquet", index=False)
    log(f"measured {df['filament_id'].nunique()} instances "
        f"({time.time()-t0:.0f}s)\n")

    log("COVERAGE ALONG THE FILAMENT (bin 0 and 4 are the two ends)")
    piv = df[df["bin"] >= 0].pivot_table(index="bin", columns="grp",
                                         values="covered", aggfunc="mean")
    log(piv.to_string(float_format=lambda v: f"{v:.3f}"))
    for g in piv.columns:
        ends = (piv.loc[0, g] + piv.loc[N_BINS - 1, g]) / 2
        mid = piv.loc[1:N_BINS - 2, g].mean()
        log(f"  {g:12s} ends {ends:.3f}  middle {mid:.3f}  "
            f"deficit at the ends {ends - mid:+.3f}")

    log("\nCOVERAGE ACROSS THE FILAMENT (outer 40% of the width)")
    lat = df[df["bin"] == -1].groupby("grp")["covered"].mean()
    log(lat.to_string(float_format=lambda v: f"{v:.3f}"))

    log("\nreading: a large end-deficit that is BIGGER for unreachable instances")
    log("than for matched ones means we truncate the filaments we lose, which no")
    log("uniform offset and no threshold can repair. A flat profile with a low")
    log("flank number instead means it is the width axis, and that is closed.")

    json.dump({g: dict(
        ends=float((piv.loc[0, g] + piv.loc[N_BINS - 1, g]) / 2),
        middle=float(piv.loc[1:N_BINS - 2, g].mean()),
        flanks=float(lat[g])) for g in piv.columns},
        open(OUT / "coverage_geometry.json", "w"), indent=1)
    log(f"\nwrote {OUT/'coverage_geometry.json'}")


if __name__ == "__main__":
    main()
