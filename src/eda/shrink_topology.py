"""Does the production shrink break filaments at the waist, and does protecting it pay?

Talha's observation, and it is a sharp one. Growing masks hurts and shrinking
helps -- measured twice now, at the uniform 5% shrink and again at the fusion
vote. But `shrink_frac` erodes by a distance-transform quantile: it keeps pixels
whose distance to the boundary exceeds the frac-quantile of that distance. That
rule is not uniform in EFFECT. A thick body loses a thin rind, while a narrow
waist -- where every pixel is already close to the boundary -- can be removed
wholesale. A filament with a thin middle does not get slimmer; it gets cut in
two, and the pixels lost are a chunk rather than a rind.

Note what this does and does not cost. Nothing downstream re-extracts connected
components, so a split candidate stays ONE prediction carrying two blobs and its
IoU changes only by the pixels actually removed. The question is therefore not
"does splitting confuse the pipeline" but "does the quantile rule delete more
than it means to, and specifically at the waist".

Three geometries, scored against each candidate's best-matching ground truth:

  A  production            distance-quantile shrink exactly as deployed
  B  medial-axis protected shrink UNION the original's skeleton -- the whole
                           1px medial axis is immune, so connectivity can never
                           break and every waist survives
  C  bridge protected      Talha's proposal in its narrow form: the skeleton is
                           added back ONLY inside components the shrink actually
                           splits, so nothing changes for candidates that survive
                           intact

C is the honest test of the idea. B is its upper bound: if B does not beat A,
protecting connectivity cannot help and C cannot either, because C is B
restricted to a subset of candidates.

Usage: python src/eda/shrink_topology.py
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
sys.path.insert(0, str(ROOT / "src" / "models"))

from skimage.morphology import skeletonize

from filament.rle import decode_rle

OUT = ROOT / "outputs" / "eda"
SEL = ROOT / "outputs" / "selector"
FRAC = 0.05


def log(*a):
    print(*a, flush=True)


def skeleton(m):
    """1px medial axis, thinned so that it stays CONNECTED.

    The whole test rests on this: if the skeleton of a connected mask is itself
    connected, adding it back guarantees the shrunk mask cannot be split. A
    morphological skeleton (iterated open-and-subtract) does NOT have that
    property -- on a dumbbell it returns 7 disconnected pieces where thinning
    returns 1 -- so using it would have measured a bridge that does not bridge.
    OpenCV's ximgproc is absent in this environment; scikit-image's Zhang-Suen
    thinning is present and is what is used.
    """
    return skeletonize(m.astype(bool))


def ncomp(m):
    return cv2.connectedComponents(m.astype(np.uint8), connectivity=8)[0] - 1


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


def shrink_quantile(m, frac, min_keep=64):
    u = m.astype(np.uint8)
    if u.sum() < min_keep:
        return m
    dist = cv2.distanceTransform(u, cv2.DIST_L2, 3)
    t = np.quantile(dist[u > 0], frac)
    out = dist > t
    return m if out.sum() < min_keep else out


def bridge_protect(orig, shrunk, sk):
    """Restore the medial axis only inside components the shrink actually split."""
    n_o, lab_o = cv2.connectedComponents(orig.astype(np.uint8), connectivity=8)
    out = shrunk.copy()
    touched = 0
    for c in range(1, n_o):
        comp = lab_o == c
        before, after = 1, ncomp(shrunk & comp)
        if after > before or after == 0:
            out |= (sk & comp)
            touched += 1
    return out, touched


def main():
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    f0 = set(folds.loc[folds["fold"] == 0, "file_name"]
             .str.replace(".jpeg", "", regex=False))
    cand = pd.read_parquet(SEL / "fused_pool_all.parquet")
    cand = cand[cand["stem"].isin(f0)]
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt = gt[gt["stem"].isin(f0)]
    gt_by = {s: [decode_rle(r).astype(bool) for r in g["segmentation_rle"]]
             for s, g in gt.groupby("stem")}
    log(f"fold 0: {len(cand)} candidates, shrink frac {FRAC}")

    rows, t0 = [], time.time()
    for k, (stem, g) in enumerate(cand.groupby("stem")):
        gts = [boxed(m) for m in gt_by.get(stem, [])]
        gts = [x for x in gts if x is not None]
        for r in g.itertuples():
            m = decode_rle(r.rle).astype(bool)
            if not m.any():
                continue
            n0 = ncomp(m)
            s = shrink_quantile(m, FRAC)
            n1 = ncomp(s)
            sk = skeleton(m)
            b = s | sk
            c, touched = bridge_protect(m, s, sk)
            gm = max(gts, key=lambda x: iou(boxed(m), x)) if gts else None
            rows.append(dict(
                stem=stem, cand_id=r.cand_id, area=int(m.sum()),
                n_before=n0, n_after=n1, split=int(n1 > n0),
                bridged=touched,
                area_A=int(s.sum()), area_B=int(b.sum()), area_C=int(c.sum()),
                iou_orig=iou(boxed(m), gm), iou_A=iou(boxed(s), gm),
                iou_B=iou(boxed(b), gm), iou_C=iou(boxed(c), gm)))
        if (k + 1) % 40 == 0:
            log(f"  {k+1} stems ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "shrink_topology.parquet", index=False)
    log(f"\nmeasured {len(df)} candidates ({time.time()-t0:.0f}s)\n")

    log("1. DOES THE SHRINK BREAK THINGS?")
    log(f"  candidates already multi-component before shrinking: "
        f"{int((df['n_before'] > 1).sum())}")
    log(f"  candidates the shrink SPLITS                       : "
        f"{int(df['split'].sum())} ({100*df['split'].mean():.1f}%)")
    if df["split"].any():
        sp = df[df["split"] == 1]
        log(f"  their median area {sp['area'].median():.0f}px against "
            f"{df['area'].median():.0f}px overall")
        log(f"  area they lose to the shrink: "
            f"{100*(1 - sp['area_A'].sum()/sp['area'].sum()):.1f}% "
            f"against {100*(1 - df['area_A'].sum()/df['area'].sum()):.1f}% overall")

    log("\n2. WHAT IT COSTS, and what protecting it recovers")
    log(f"{'geometry':28s} {'mean IoU':>9s} {'rows > 0.5':>11s} {'total px':>12s}")
    for lbl, col, acol in (("unshrunk", "iou_orig", "area"),
                           ("A production shrink", "iou_A", "area_A"),
                           ("B medial axis protected", "iou_B", "area_B"),
                           ("C bridge protected", "iou_C", "area_C")):
        log(f"  {lbl:26s} {df[col].mean():9.4f} {int((df[col] > 0.5).sum()):11d} "
            f"{int(df[acol].sum()):12d}")

    if df["split"].any():
        sp = df[df["split"] == 1]
        log(f"\n3. ON THE SPLIT CANDIDATES ONLY ({len(sp)} of them)")
        for lbl, col in (("unshrunk", "iou_orig"), ("A production", "iou_A"),
                         ("B medial axis", "iou_B"), ("C bridge", "iou_C")):
            log(f"  {lbl:26s} mean IoU {sp[col].mean():.4f}  "
                f"rows > 0.5 {int((sp[col] > 0.5).sum()):4d}")

    log("\nreading: C only differs from A on the split candidates, so if the")
    log("row-count line does not move there, the idea has nothing to act on.")

    json.dump({k: dict(mean_iou=float(df[c].mean()),
                       rows_over_50=int((df[c] > 0.5).sum()))
               for k, c in (("unshrunk", "iou_orig"), ("A", "iou_A"),
                            ("B", "iou_B"), ("C", "iou_C"))} |
              dict(n=len(df), n_split=int(df["split"].sum())),
              open(OUT / "shrink_topology.json", "w"), indent=1)
    log(f"\nwrote {OUT/'shrink_topology.json'}")


if __name__ == "__main__":
    main()
