"""Does multi-threshold semantic extraction raise the pool's reachability?

The trail: 21% of the never-proposed consensus GT sits under a CLEAN thr-0.5
component of the very probability maps the pool already uses. The pool extracts
at thr 0.3 / min_area 256 -- and at 0.3 neighbouring components merge (the
ledger's measured "loosening reduces reachability" effect), while 100-256 px
candidates are discarded outright. So the loss is in extraction parameters, not
in the model.

The candidate fix is a UNION over thresholds: extract components at 0.3, 0.5 and
0.7 with min_area 100 and let all of them into the pool. A component that is
clean at 0.5 enters even if its 0.3 parent is a merged blob; the selector
already carries cross-lane features to pick among them. Adding candidates can
only raise reachability -- the question is by how much and where.

This measures, on fold 0 against ALL GT rows (not just the never-proposed):

  reachability of: current extraction (0.3/256) alone,
                   each threshold alone,
                   the union, with and without the min_area drop
  and the tier breakdown of what the union adds.

Reachability is the ceiling, not the score: the selector still has to keep the
right ones and the de-overlap has to keep nested variants from cutting each
other. Those are the next measurement if this one moves.

Usage: python src/eda/multithresh_pool.py
"""

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


def log(*a):
    print(*a, flush=True)


def components(prob, thr, min_area):
    n, lab = cv2.connectedComponents((prob >= thr).astype(np.uint8), connectivity=8)
    out = []
    for k in range(1, n):
        m = lab == k
        if m.sum() >= min_area:
            out.append(m)
    return out


def main():
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    f0 = {f.rsplit(".", 1)[0] for f in folds.loc[folds["fold"] == 0, "file_name"]}
    z = np.load(SEM / "fold0_probs_cldice.npz")
    stems = [s for s in z.files if s in f0]

    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.rsplit(".", n=1).str[0]
    gt = gt[gt["stem"].isin(stems)].copy()
    ent = gt[["entry_id", "stem"]].drop_duplicates()
    ncop = ent.groupby("stem")["entry_id"].nunique()
    gt["n_copies"] = gt["stem"].map(ncop)
    log(f"fold 0: {len(stems)} images, {len(gt)} GT rows\n")

    VARIANTS = {
        "current  0.30/256": [(0.30, 256)],
        "thr 0.30/100": [(0.30, 100)],
        "thr 0.50/100": [(0.50, 100)],
        "thr 0.70/100": [(0.70, 100)],
        "union .3/.5/.7 @100": [(0.30, 100), (0.50, 100), (0.70, 100)],
    }
    reach = {k: np.zeros(len(gt), bool) for k in VARIANTS}
    best_iou = {k: np.zeros(len(gt)) for k in VARIANTS}
    gt = gt.reset_index(drop=True)

    for si, stem in enumerate(stems):
        prob = z[stem].astype(np.float32) / 255.0
        comp_cache = {}
        for name, spec in VARIANTS.items():
            cands = []
            for thr, ma in spec:
                if (thr, ma) not in comp_cache:
                    comp_cache[(thr, ma)] = components(prob, thr, ma)
                cands += comp_cache[(thr, ma)]
            idx = gt.index[gt["stem"] == stem]
            for i in idx:
                g = decode_rle(gt.loc[i, "segmentation_rle"]).astype(bool)
                bi = 0.0
                ga = g.sum()
                for m in cands:
                    inter = (m & g).sum()
                    if inter == 0:
                        continue
                    bi = max(bi, inter / (m.sum() + ga - inter))
                best_iou[name][i] = bi
                reach[name][i] = bi > 0.5
        if (si + 1) % 30 == 0:
            log(f"  {si+1}/{len(stems)} images")

    log(f"\n{'variant':>22s} {'reach':>7s} {'d vs current':>13s}")
    base = reach["current  0.30/256"].mean()
    for name in VARIANTS:
        r = reach[name].mean()
        log(f"{name:>22s} {r:7.4f} {r-base:+13.4f}")

    add = reach["union .3/.5/.7 @100"] & ~reach["current  0.30/256"]
    lost = ~reach["union .3/.5/.7 @100"] & reach["current  0.30/256"]
    log(f"\nunion adds {int(add.sum())} GT rows, loses {int(lost.sum())} "
        f"(a union cannot lose -- nonzero here would be a bug)")
    if add.any():
        gadd = gt[add]
        sup = gadd.groupby(["stem", "ann_id"])["entry_id"].transform("nunique")
        tier = sup.astype(str) + "/" + gadd["n_copies"].astype(str)
        log("\nwhat the union adds, by consensus tier:")
        log(tier.value_counts().to_string())
        log(f"\nmedian area of added rows: {gadd['area'].median():.0f} px "
            f"(matched population median 1496)")
        # value estimate with the corrected marginal rule
        val = 0.0
        for t, n in tier.value_counts().items():
            k, N = (int(x) for x in t.split("/"))
            pass
        iou_gain = float(best_iou["union .3/.5/.7 @100"][add].sum())
        log(f"sum of newly claimable IoU on fold 0: {iou_gain:.1f} "
            f"(x5 folds, into a denominator of ~7000)")

    df = pd.DataFrame({name: best_iou[name] for name in VARIANTS})
    df["filament_id"] = gt["filament_id"]
    df.to_parquet(OUT / "multithresh_pool.parquet", index=False)
    log(f"\nsaved -> {OUT/'multithresh_pool.parquet'}")


if __name__ == "__main__":
    main()
