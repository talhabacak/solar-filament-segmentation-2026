"""Which extraction scheme recovers the unreachable rows, and what does it cost?

`hysteresis_precondition.py` overturned the assumption this work started from.
The model is NOT blind to the 440 never-proposed ground-truth rows: the median
maximum probability inside one is 1.000, identical to a matched row. What differs
is COVERAGE -- the model fires over 47.3% of a never-proposed filament's area at
p >= 0.7 against 79.6% for a matched one. The component we extract is roughly
half the annotated object, which is exactly how an instance lands below IoU 0.5
while being perfectly well detected.

It also killed the merge story. Lowering the threshold was assumed to fail by
bridging neighbours, but the p >= 0.3 flood component is a median 0.7x the true
filament's area -- SMALLER than the object, not larger -- and only 2.7% of these
rows touch a blob more than 5x their own size. Merging is not what is stopping a
lower threshold from working.

So the open question is narrow and answerable: given a probability map that fires
over half of these objects, is there an EXTRACTION that recovers them, and what
does it cost on the rows we already match? Four schemes, scored on both
populations, because a scheme that gains 60 never-proposed rows and breaks 80
matched ones is a loss and the single-population framing would hide that:

  A  connected components at 0.5              the current operating point
  B  connected components at 0.3              the naive relaxation
  C  hysteresis: seeds at 0.7, flooded to 0.3, watershed so two seeds can never
     merge into one component
  D  per-row oracle threshold over the grid   the ceiling of ANY scheme that
                                              picks one global level per image

D matters because if the oracle over thresholds barely beats A, then no amount of
threshold engineering helps and the deficit is in the probability map itself --
which would make this a training problem, not a post-processing one.

Usage: python src/eda/extraction_schemes.py
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
MIN_AREA = 200
GRID = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def log(*a):
    print(*a, flush=True)


def cc(binm, min_area=MIN_AREA):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        binm.astype(np.uint8), connectivity=8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return lab, keep


def hysteresis(prob, hi=0.7, lo=0.3, min_area=MIN_AREA):
    """Seeds at `hi` grown through the `lo` region by watershed.

    Watershed is what makes this different from thresholding at `lo`: two seeds
    inside one flood blob come out as two labels, so recovering faint extent
    cannot fuse two filaments into a component that matches neither.
    """
    lo_m = (prob >= lo).astype(np.uint8)
    seed_lab, seed_keep = cc((prob >= hi), min_area)
    if not seed_keep:
        return np.zeros_like(seed_lab), []
    markers = np.zeros(prob.shape, np.int32)
    for j, s in enumerate(seed_keep, start=1):
        markers[seed_lab == s] = j
    markers[(lo_m == 0) & (markers == 0)] = len(seed_keep) + 1   # background
    img = cv2.cvtColor((prob * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.watershed(img, markers)
    out = np.where((markers > 0) & (markers <= len(seed_keep)), markers, 0)
    keep = [j for j in range(1, len(seed_keep) + 1)
            if int((out == j).sum()) >= min_area]
    return out, keep


def best_iou(gt, lab, keep):
    """Best IoU between one ground-truth mask and any labelled component."""
    if not keep:
        return 0.0
    ids, counts = np.unique(lab[gt], return_counts=True)
    best, g = 0.0, int(gt.sum())
    ks = set(keep)
    for i, c in zip(ids, counts):
        if i == 0 or int(i) not in ks:
            continue
        a = int((lab == i).sum())
        u = a + g - int(c)
        if u:
            best = max(best, int(c) / u)
    return best


def main():
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    f0 = set(folds.loc[folds["fold"] == 0, "file_name"]
             .str.replace(".jpeg", "", regex=False))
    d = pd.read_parquet(OUT / "error_profile_fn_gt.parquet")
    d["matched"] = d["matched"].astype(bool)
    d["in_raw_pool"] = d["in_raw_pool"].astype(bool)
    d = d[d["stem"].isin(f0)]
    frames = {"never-proposed": d[~d["matched"] & ~d["in_raw_pool"]],
              "matched": d[d["matched"]]}
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt = gt.set_index("filament_id")["segmentation_rle"]
    z = np.load(SEM / PROBS)
    log(f"fold-0: " + ", ".join(f"{k} {len(v)}" for k, v in frames.items()))

    rows, t0 = [], time.time()
    stems = sorted(set(d["stem"]) & set(z.files))
    for n, stem in enumerate(stems):
        prob = np.asarray(z[stem]).astype(np.float32) / 255.0
        schemes = {}
        for t in GRID:
            schemes[f"cc{t}"] = cc(prob >= t)
        schemes["hyst"] = hysteresis(prob)
        sub = d[d["stem"] == stem]
        for r in sub.itertuples():
            m = decode_rle(gt.loc[r.filament_id]).astype(bool)
            if not m.any():
                continue
            grp = ("never-proposed" if (not r.matched and not r.in_raw_pool)
                   else ("matched" if r.matched else "other"))
            rec = dict(grp=grp, filament_id=r.filament_id)
            for k, (lab, keep) in schemes.items():
                rec[k] = best_iou(m, lab, keep)
            rows.append(rec)
        if (n + 1) % 40 == 0:
            log(f"  {n+1}/{len(stems)} stems ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "extraction_schemes.parquet", index=False)
    cols = [f"cc{t}" for t in GRID] + ["hyst"]
    df["oracle_thr"] = df[[f"cc{t}" for t in GRID]].max(axis=1)

    log(f"\nrows matched at IoU > 0.5 by each scheme "
        f"({time.time()-t0:.0f}s)\n")
    log(f"{'scheme':10s} {'never-proposed':>15s} {'matched':>9s} {'NET vs cc0.5':>13s}")
    base_n = int((df[df.grp == 'never-proposed']["cc0.5"] > 0.5).sum())
    base_m = int((df[df.grp == 'matched']["cc0.5"] > 0.5).sum())
    for c in cols + ["oracle_thr"]:
        n = int((df[df.grp == "never-proposed"][c] > 0.5).sum())
        m = int((df[df.grp == "matched"][c] > 0.5).sum())
        log(f"{c:10s} {n:15d} {m:9d} {(n-base_n)+(m-base_m):+13d}")

    log(f"\nbaseline cc0.5: {base_n} never-proposed and {base_m} matched rows.")
    log("NET is the change in total rows reached, which is what PQ responds to:")
    log("a scheme that gains faint rows but drops matched ones is a loss.")

    json.dump({c: dict(never=int((df[df.grp == "never-proposed"][c] > 0.5).sum()),
                       matched=int((df[df.grp == "matched"][c] > 0.5).sum()))
               for c in cols + ["oracle_thr"]},
              open(OUT / "extraction_schemes.json", "w"), indent=1)
    log(f"\nwrote {OUT/'extraction_schemes.json'}")


if __name__ == "__main__":
    main()
