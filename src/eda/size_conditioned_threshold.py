"""A per-component threshold that depends on the component's own size.

The size-conditioned document's one proposal that needs no retraining (its §8):
extract at a base threshold, then re-extract each component at a threshold set by
its own area -- higher for small components so they shrink, lower for large ones
so they grow. It is materially different from the size-conditioned SHRINK already
rejected, because a shrink can only remove pixels while a lower threshold can add
them, and the failure this is aimed at (large ground truth under-covered) needs
addition.

What is already known and constrains the expectation:

  * the size bias the document rests on is mostly a selection artifact. Read
    unconditionally the model tracks size well -- across ground-truth quintiles
    the predicted/true area ratio runs 1.22, 1.13, 1.07, 1.02, 0.90, and the
    log-area spread of predictions is 0.938 of the truth's. The 2.50 and 0.52
    in the outcome table come from conditioning on failure.
  * but that residual 1.22-to-0.90 tilt is real, and it is exactly the shape a
    size-conditioned threshold could remove.
  * the ceiling is known: a per-image ORACLE threshold over the same grid was
    measured at +92 net rows reached (extraction_schemes.py). A rule using only
    component area cannot beat an oracle that picks freely, so +92 bounds it.

Scored the way every extraction change here is scored -- net rows reached at
IoU > 0.5 over BOTH the matched and the unreachable populations, because a rule
that gains faint rows by breaking matched ones is a loss.

Usage: python src/eda/size_conditioned_threshold.py
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

OUT = ROOT / "outputs" / "eda"
SEM = ROOT / "outputs" / "semantic"
PROBS = "fold0_probs_cldice.npz"
BASE_THR, MIN_AREA = 0.3, 256
PAD = 40

# (small area cut, small threshold, large area cut, large threshold)
RULES = {
    "production fixed 0.30": None,
    "doc rule, mild":  (1000, 0.40, 2500, 0.24),
    "doc rule, as written": (1000, 0.55, 2500, 0.22),
    "inverted (sanity)": (1000, 0.22, 2500, 0.55),
}


def log(*a):
    print(*a, flush=True)


def boxed(m):
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return None
    return ((int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1),
            m[ys.min():ys.max() + 1, xs.min():xs.max() + 1], int(m.sum()))


def iou(a, b):
    if a is None or b is None:
        return 0.0
    (ay0, ay1, ax0, ax1), asub, asum = a
    (by0, by1, bx0, bx1), bsub, bsum = b
    y0, y1 = max(ay0, by0), min(ay1, by1)
    x0, x1 = max(ax0, bx0), min(ax1, bx1)
    if y0 >= y1 or x0 >= x1:
        return 0.0
    i = int(np.logical_and(asub[y0 - ay0:y1 - ay0, x0 - ax0:x1 - ax0],
                           bsub[y0 - by0:y1 - by0, x0 - bx0:x1 - bx0]).sum())
    u = asum + bsum - i
    return i / u if u else 0.0


def components(prob, rule):
    """Extract at the base threshold, then re-extract each piece at its own."""
    base = (prob >= BASE_THR).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(base, connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_AREA:
            continue
        m = lab == i
        if rule is None:
            out.append(m)
            continue
        a_small, t_small, a_large, t_large = rule
        area = int(m.sum())
        t = t_small if area < a_small else (t_large if area > a_large else BASE_THR)
        if t == BASE_THR:
            out.append(m)
            continue
        # re-threshold inside a padded box, then keep the piece that still covers
        # this component's centroid -- a lower threshold can otherwise merge two
        # filaments into one blob and the merge would be scored as this component
        ys, xs = np.nonzero(m)
        y0, y1 = max(0, ys.min() - PAD), min(prob.shape[0], ys.max() + PAD + 1)
        x0, x1 = max(0, xs.min() - PAD), min(prob.shape[1], xs.max() + PAD + 1)
        sub = (prob[y0:y1, x0:x1] >= t).astype(np.uint8)
        n2, lab2 = cv2.connectedComponents(sub, connectivity=8)
        cy, cx = int(round(ys.mean())) - y0, int(round(xs.mean())) - x0
        cy = min(max(cy, 0), sub.shape[0] - 1)
        cx = min(max(cx, 0), sub.shape[1] - 1)
        k = lab2[cy, cx]
        if k == 0:
            out.append(m)
            continue
        new = np.zeros_like(m)
        new[y0:y1, x0:x1] = lab2 == k
        # refuse a re-threshold that swallows more than three times the original,
        # which is the merge this guard exists for
        out.append(new if new.sum() <= 3 * area and new.sum() >= MIN_AREA else m)
    return out


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
    gtr = gt.set_index("filament_id")["segmentation_rle"]
    z = np.load(SEM / PROBS)
    log(f"fold 0: " + ", ".join(f"{k} {v}" for k, v in
                                d["grp"].value_counts().items()))
    log(f"base extraction {BASE_THR}, min area {MIN_AREA}\n")

    res = {k: [] for k in RULES}
    t0 = time.time()
    for stem, g in d.groupby("stem"):
        if stem not in z:
            continue
        prob = np.asarray(z[stem]).astype(np.float32) / 255.0
        gts = [(r.filament_id, r.grp,
                boxed(decode_rle(gtr.loc[r.filament_id]).astype(bool)))
               for r in g.itertuples()]
        for name, rule in RULES.items():
            comps = [boxed(c) for c in components(prob, rule)]
            comps = [c for c in comps if c is not None]
            for fid, grp, gm in gts:
                res[name].append(dict(filament_id=fid, grp=grp,
                                      iou=max((iou(gm, c) for c in comps),
                                              default=0.0)))
    log(f"scored {len(RULES)} rules ({time.time()-t0:.0f}s)\n")

    log(f"{'rule':26s} {'matched':>9s} {'unreach':>9s} {'other':>7s} {'NET':>7s}")
    base_counts = None
    for name in RULES:
        df = pd.DataFrame(res[name])
        c = {g: int((s["iou"] > 0.5).sum()) for g, s in df.groupby("grp")}
        tot = sum(c.values())
        if base_counts is None:
            base_counts = tot
        log(f"  {name:24s} {c.get('matched',0):9d} {c.get('unreachable',0):9d} "
            f"{c.get('other',0):7d} {tot - base_counts:+7d}")
    log("\n  NET is the change in total rows reached against the production rule.")
    log("  Ceiling for reference: a per-image ORACLE threshold over the same grid")
    log("  was worth +92 rows, and a rule keyed only on area cannot beat it.")
    log("  The inverted rule is a sanity arm: if it scores like the intended one,")
    log("  the mechanism is threshold slack rather than size conditioning.")

    json.dump({k: {g: int((pd.DataFrame(v)["iou"][pd.DataFrame(v)["grp"] == g] > 0.5).sum())
                   for g in ("matched", "unreachable", "other")}
               for k, v in res.items()},
              open(OUT / "size_conditioned_threshold.json", "w"), indent=1)
    log(f"\nwrote {OUT/'size_conditioned_threshold.json'}")


if __name__ == "__main__":
    main()
