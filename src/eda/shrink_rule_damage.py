"""What the two shrink rules actually do to a thin filament.

The board has now ranked three arms that differ only in the shrink:

    sub020  production ring erosion, removes 18.7%   #1
    sub019  fractional rule at 0.10, removes 10.1%   #2
    sub023  fractional rule at 0.19, removes 19.1%   #10

That kills the framing I had been carrying. I read "production beats fractional
0.10" as the board wanting MORE removal, since 18.7% > 10.1%. sub023 removes the
same 19% by the fractional rule and falls eight places, so more removal is not
what wins -- within the fractional rule more is clearly worse, on the board and
locally alike (0.4416 at 0.10 against 0.4324 at 0.19). What wins is the RULE.

The two rules differ in where they take the pixels from.

  production  = one-pixel morphological erosion. Every mask loses its outer ring
                and nothing else, so a two-pixel-wide filament keeps a spine.
  fractional  = remove the lowest-ranked f of pixels by blurred distance-to-edge.
                In a NARROW part of a mask every pixel has a low distance, so the
                rule takes the whole width there while leaving a wide blob almost
                untouched.

If that is right, the fractional rule should damage thin candidates
disproportionately: fragmenting them into pieces, or deleting them outright, at a
rate that rises as the mask gets thinner -- while erosion should be roughly
neutral in thinness. Thinness here is area / skeleton length, i.e. the mean width
of the structure, which is the quantity the mechanism is about.

This is worth measuring rather than asserting, because it is the first mechanism
proposed for this axis that concerns the operators themselves rather than a
train/test difference -- and the two train/test stories I proposed before were
both measured and refuted.

Usage: python src/eda/shrink_rule_damage.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament.rle import decode_rle
from mask_shrink import shrink_frac, shrink_true_frac

OUT = ROOT / "outputs" / "eda"
SEL = ROOT / "outputs" / "selector"


def log(*a):
    print(*a, flush=True)


def skeleton_len(m):
    """Cheap proxy for spine length: pixels surviving a thinning."""
    try:
        sk = cv2.ximgproc.thinning((m * 255).astype(np.uint8))
        return int((sk > 0).sum())
    except Exception:
        # ximgproc is optional; fall back to the perimeter/2 estimate, which for
        # an elongated structure is close to its spine length
        c, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE)
        return max(int(sum(len(x) for x in c)) // 2, 1)


def damage(m, out):
    """What the operator did: area kept, pieces, and whether it vanished."""
    a0 = int(m.sum())
    a1 = int(out.sum())
    if a1 == 0:
        return 0.0, 0, True
    n, _ = cv2.connectedComponents(out.astype(np.uint8), connectivity=8)
    return a1 / a0, n - 1, False


def main():
    cand = pd.read_parquet(SEL / "fused_pool_all_blob5.parquet")
    rng = np.random.default_rng(13)
    idx = rng.choice(len(cand), size=min(1500, len(cand)), replace=False)
    sub = cand.iloc[idx]
    log(f"{len(sub)} candidates sampled from the production pool\n")

    rows = []
    for r in sub.itertuples():
        m = decode_rle(r.rle).astype(bool)
        a = int(m.sum())
        if a < 64:
            continue
        sk = skeleton_len(m)
        width = a / max(sk, 1)          # mean width in pixels
        rec = dict(area=a, skel=sk, width=width)
        for tag, fn, f in (("prod", shrink_frac, 0.05),
                           ("frac10", shrink_true_frac, 0.10),
                           ("frac19", shrink_true_frac, 0.19)):
            kept, pieces, gone = damage(m, fn(m, f))
            rec[f"{tag}_kept"] = kept
            rec[f"{tag}_pieces"] = pieces
            rec[f"{tag}_gone"] = gone
            rec[f"{tag}_frag"] = pieces > 1
        rows.append(rec)

    d = pd.DataFrame(rows)
    d.to_parquet(OUT / "shrink_rule_damage.parquet", index=False)
    log(f"{len(d)} candidates measured, mean width "
        f"{d['width'].median():.1f} px (p10 {d['width'].quantile(0.1):.1f}, "
        f"p90 {d['width'].quantile(0.9):.1f})\n")

    log("overall, per rule")
    log(f"  {'rule':<10}{'area kept':>11}{'fragmented':>12}{'vanished':>10}")
    for tag, name in (("prod", "production"), ("frac10", "fractional .10"),
                      ("frac19", "fractional .19")):
        log(f"  {name:<10}{d[f'{tag}_kept'].mean():>11.3f}"
            f"{100*d[f'{tag}_frag'].mean():>11.1f}%{100*d[f'{tag}_gone'].mean():>9.1f}%")

    log("\nby mean width -- the mechanism says the fractional rule should hurt")
    log("thin candidates disproportionately and erosion should not")
    d["wq"] = pd.qcut(d["width"], 5, labels=False, duplicates="drop")
    log(f"  {'width band':>12}{'n':>6}"
        f"{'prod frag':>11}{'frac10 frag':>13}{'frac19 frag':>13}"
        f"{'prod gone':>11}{'frac19 gone':>13}")
    for q, g in d.groupby("wq"):
        log(f"  {g['width'].median():>12.1f}{len(g):>6}"
            f"{100*g['prod_frag'].mean():>10.1f}%{100*g['frac10_frag'].mean():>12.1f}%"
            f"{100*g['frac19_frag'].mean():>12.1f}%"
            f"{100*g['prod_gone'].mean():>10.1f}%{100*g['frac19_gone'].mean():>12.1f}%")

    thin = d[d["wq"] == 0]
    fat = d[d["wq"] == d["wq"].max()]
    log(f"\n  thinnest fifth (width ~{thin['width'].median():.1f} px):")
    log(f"    production      fragments {100*thin['prod_frag'].mean():.1f}%, "
        f"deletes {100*thin['prod_gone'].mean():.1f}%")
    log(f"    fractional .19  fragments {100*thin['frac19_frag'].mean():.1f}%, "
        f"deletes {100*thin['frac19_gone'].mean():.1f}%")
    log(f"  widest fifth (width ~{fat['width'].median():.1f} px):")
    log(f"    production      fragments {100*fat['prod_frag'].mean():.1f}%, "
        f"deletes {100*fat['prod_gone'].mean():.1f}%")
    log(f"    fractional .19  fragments {100*fat['frac19_frag'].mean():.1f}%, "
        f"deletes {100*fat['frac19_gone'].mean():.1f}%")

    log("\n  how much of a mask each rule takes, by width band")
    log(f"  {'width':>12}{'prod':>9}{'frac10':>9}{'frac19':>9}")
    for q, g in d.groupby("wq"):
        log(f"  {g['width'].median():>12.1f}{1-g['prod_kept'].mean():>9.3f}"
            f"{1-g['frac10_kept'].mean():>9.3f}{1-g['frac19_kept'].mean():>9.3f}")

    json.dump(dict(n=len(d),
                   overall={t: dict(kept=float(d[f"{t}_kept"].mean()),
                                    frag=float(d[f"{t}_frag"].mean()),
                                    gone=float(d[f"{t}_gone"].mean()))
                            for t in ("prod", "frac10", "frac19")},
                   thin_frag={t: float(thin[f"{t}_frag"].mean())
                              for t in ("prod", "frac10", "frac19")}),
              open(OUT / "shrink_rule_damage.json", "w"), indent=1)
    log(f"\nwrote {OUT/'shrink_rule_damage.json'}")


if __name__ == "__main__":
    main()
