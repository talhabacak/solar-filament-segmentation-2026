"""Does candidate density explain why the board wants more shrink than the cross-fit?

The shrink axis now has two independent single-variable board confirmations, on
two different semantic fields: production ring erosion (removes 18.7% of area)
beats true-fractional 0.10 (removes 10.1%), sub020 over sub019 and sub018 over
sub015. The local paired cross-fit says the opposite by +0.0031 on two separate
pools, and the first mechanism I proposed for the disagreement -- that test masks
over-cover more -- was refuted: test masks are SMALLER than out-of-fold ones
(median ratios 0.955 and 0.973).

Second candidate mechanism, and it sits exactly in the local measurement's blind
spot. The out-of-fold pool is built from ONE detector cache per fold and carries
13.7 candidates per image; the submitted pool concatenates FIVE and carries 55.5.
At four times the density, more predictions crowd the same ground-truth row, and
under PQ only one of them can match -- every other overlapping prediction becomes
a false positive charged at 0.5. Trimming masks separates crowded predictions,
so a larger shrink should pay MORE at high density, which is a benefit the
out-of-fold pool is too sparse to show.

This measures the crowding directly: among the instances a policy actually keeps,
how much do they overlap each other, out-of-fold against test, and how much does
the larger shrink relieve it.

If test-side crowding is materially higher and the production shrink relieves it,
the board's preference has a mechanism and the amount is worth pushing further.
If crowding is the same, this explanation joins the over-coverage one and the
axis stays empirical -- two board confirmations with no mechanism behind them.

Usage: python src/eda/overlap_density.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament.rle import decode_rle

OUT = ROOT / "outputs" / "eda"
SUB = ROOT / "outputs" / "submissions"


def log(*a):
    print(*a, flush=True)


def crowding(masks):
    """Per-image crowding among kept instances.

    Returns the fraction of instances that overlap at least one other, the mean
    best pairwise IoU, and the share of claimed pixels claimed more than once.
    """
    n = len(masks)
    if n < 2:
        return 0.0, 0.0, 0.0
    boxes = []
    for m in masks:
        ys, xs = np.nonzero(m)
        boxes.append((ys.min(), ys.max(), xs.min(), xs.max()))
    touched = np.zeros(n, bool)
    best = np.zeros(n)
    for i in range(n):
        for j in range(i + 1, n):
            # bounding boxes first: most pairs are far apart and the full
            # intersection over a 2048^2 pair is not worth paying for
            ai, bi, ci, di = boxes[i]
            aj, bj, cj, dj = boxes[j]
            if bi < aj or bj < ai or di < cj or dj < ci:
                continue
            inter = np.logical_and(masks[i], masks[j]).sum()
            if inter == 0:
                continue
            union = masks[i].sum() + masks[j].sum() - inter
            iou = inter / union
            touched[i] = touched[j] = True
            best[i], best[j] = max(best[i], iou), max(best[j], iou)
    stack = np.zeros(masks[0].shape, np.uint8)
    for m in masks:
        stack += m.astype(np.uint8)
    claimed = (stack > 0).sum()
    multi = (stack > 1).sum()
    return touched.mean(), best.mean(), (multi / claimed if claimed else 0.0)


def from_csv(path):
    d = pd.read_csv(path)
    d["stem"] = d["filament_id"].str.rsplit("_", n=1).str[0]
    return {s: [decode_rle(r).astype(bool) for r in g["segmentation_rle"]]
            for s, g in d.groupby("stem")}


def report(name, by_stem):
    rows = [crowding(v) for v in by_stem.values() if len(v) >= 2]
    if not rows:
        return None
    a = np.array(rows)
    n_per = np.mean([len(v) for v in by_stem.values()])
    log(f"  {name:<28}{len(by_stem):>6}{n_per:>9.2f}"
        f"{100*a[:, 0].mean():>13.1f}%{a[:, 1].mean():>11.4f}"
        f"{100*a[:, 2].mean():>13.2f}%")
    return dict(images=len(by_stem), per_image=float(n_per),
                frac_touching=float(a[:, 0].mean()),
                mean_best_iou=float(a[:, 1].mean()),
                double_claimed=float(a[:, 2].mean()))


def main():
    log("crowding among KEPT instances\n")
    log(f"  {'arm':<28}{'images':>6}{'inst/img':>9}"
        f"{'overlapping':>14}{'mean IoU':>11}{'double-claimed':>14}")

    res = {}
    for tag, fn in (("sub019 test (true .10)", "sub019_blob5_ts.csv"),
                    ("sub020 test (prod .187)", "sub020_blob5_prodshrink.csv"),
                    ("sub015 test (true .10)", "sub015_recipe3_tf010_ts.csv"),
                    ("sub018 test (prod .187)", "sub018_prodshrink_ts.csv")):
        p = SUB / fn
        if p.exists():
            r = report(tag, from_csv(p))
            if r:
                res[tag] = r

    # out-of-fold side, same policy, so the two sets survived an equivalent
    # decision and the only structural difference left is how many candidates
    # were competing in the first place
    import joblib
    from mask_shrink import shrink_true_frac
    SEL = ROOT / "outputs" / "selector"
    b = joblib.load(SEL / "selector_v5_shrunkblob5ts.joblib")
    cand = pd.read_parquet(SEL / "fused_pool_all_blob5.parquet")
    areas = np.fromiter(
        (int(shrink_true_frac(decode_rle(r).astype(bool), 0.10).sum())
         for r in cand["rle"]), dtype=np.int64, count=len(cand))
    cand["area"] = areas
    cand["log10_area"] = np.log10(np.maximum(areas, 1))
    X = cand[b["features"]].to_numpy(np.float64)
    p = b["model"].predict_proba(X)[:, 1]
    if b.get("iou_model") is not None:
        p = p * np.clip(b["iou_model"].predict(X), 0.0, None)
        ref = b.get("two_stage_cdf")
        if ref is not None:
            p = np.searchsorted(ref, p, side="right") / len(ref)
    pol = b["policy"]
    keep = cand[(cand["conf"] >= pol["conf"]).to_numpy() & (p >= pol["p"])]
    by = {}
    for s, g in keep.groupby("stem"):
        by[s] = [shrink_true_frac(decode_rle(r).astype(bool), 0.10)
                 for r in g["rle"]]
    r = report("sub019 OOF (true .10)", by)
    if r:
        res["sub019 OOF (true .10)"] = r

    log("\nthe two comparisons:")
    t, o = res.get("sub019 test (true .10)"), res.get("sub019 OOF (true .10)")
    if t and o:
        log(f"  density   out-of-fold {o['per_image']:.2f} inst/img"
            f"  ->  test {t['per_image']:.2f}")
        log(f"  crowding  out-of-fold {100*o['frac_touching']:.1f}% overlapping"
            f"  ->  test {100*t['frac_touching']:.1f}%")
        log(f"  double-claimed pixels {100*o['double_claimed']:.2f}%"
            f"  ->  {100*t['double_claimed']:.2f}%")
    a, bb = res.get("sub019 test (true .10)"), res.get("sub020 test (prod .187)")
    if a and bb:
        log(f"\n  does the larger shrink relieve it, on the same test frames?")
        log(f"    overlapping instances {100*a['frac_touching']:.1f}%"
            f"  ->  {100*bb['frac_touching']:.1f}%")
        log(f"    double-claimed pixels {100*a['double_claimed']:.2f}%"
            f"  ->  {100*bb['double_claimed']:.2f}%")

    json.dump(res, open(OUT / "overlap_density.json", "w"), indent=1)
    log(f"\nwrote {OUT/'overlap_density.json'}")


if __name__ == "__main__":
    main()
