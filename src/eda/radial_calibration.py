"""Does a confidence of 0.6 mean the same thing at disk centre and at the limb?

Talha's question: the disk is bright at centre and darkens toward the limb, and
filaments are foreshortened there, so the model's job is not the same across the
disk. Should confidence be adjusted by local brightness or radius?

The question has to be sharpened before it can be answered, because the selector
ALREADY takes r_norm as a feature and the ledger already records that relaxing a
radial threshold buys +0.0000. So "does the pipeline know about radius" is
settled -- it does. The open question is different and stronger:

    is the selector's OUTPUT calibrated across the disk?

If P(true positive | p_sel) depends on r_norm, then a single global keep
threshold is systematically too strict in one zone and too loose in another, and
a zone-conditional threshold would be free money. If it does not depend on
r_norm, the selector has already absorbed the effect and there is nothing left --
which is what a gradient-boosted model with both conf and r_norm available is
supposed to do, but with ~700 images it is not guaranteed.

Three things are measured, in increasing order of what they would license:

  1. reliability of the RAW detector/semantic conf by radial zone. A spread here
     is expected and is exactly what the selector exists to fix.
  2. reliability of p_sel, the selector's output, by radial zone. A spread HERE
     is a real miscalibration and is actionable.
  3. the same for local BRIGHTNESS rather than radius, since limb darkening and
     radius are correlated but not identical -- a bright active region near the
     centre is dark-adjacent in a way the radius does not capture.

The actionable form, if a spread survives: fit P(y | p_sel, zone) and threshold
on the calibrated probability instead of on p_sel. That is a post-processing
change with no retraining.

Usage: python src/eda/radial_calibration.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "outputs" / "eda"
SEL = ROOT / "outputs" / "selector"
PSEL = "p_BASE+SRC+CROSS"


def log(*a):
    print(*a, flush=True)


def reliability(df, score, group, bins=(0.0, 0.2, 0.4, 0.6, 0.8, 1.01)):
    """Empirical P(y=1) per score bin, split by group."""
    d = df.copy()
    d["sbin"] = pd.cut(d[score], bins=bins, right=False,
                       labels=[f"{bins[i]:.1f}-{bins[i+1]:.1f}"
                               for i in range(len(bins) - 1)])
    t = d.pivot_table(index="sbin", columns=group, values="y",
                      aggfunc="mean", observed=True)
    n = d.pivot_table(index="sbin", columns=group, values="y",
                      aggfunc="size", observed=True)
    return t, n


def main():
    cand = pd.read_parquet(SEL / "fused_pool_all_scored.parquet")
    log(f"pool {len(cand)} candidates, positives {int(cand['y'].sum())}")

    # radial zones: the disk is masked at r <= 0.97, and foreshortening only
    # bites in the outer third, so the cut points are physical rather than equal-N
    cand["zone"] = pd.cut(cand["r_norm"], [0, 0.35, 0.7, 0.9, 1.01],
                          labels=["centre", "mid", "outer", "limb"])
    log("\ncandidates per zone: " +
        ", ".join(f"{k} {v}" for k, v in cand["zone"].value_counts().sort_index().items()))
    log("base positive rate per zone:")
    log("  " + cand.groupby("zone", observed=True)["y"].mean()
        .to_string(float_format=lambda v: f"{v:.3f}").replace("\n", "\n  "))

    log("\n1. RAW conf -- P(true positive) by confidence bin and zone")
    t, n = reliability(cand, "conf", "zone")
    log(t.to_string(float_format=lambda v: f"{v:.3f}"))
    log("   counts:")
    log("   " + n.to_string().replace("\n", "\n   "))

    log("\n2. SELECTOR p_sel -- the one that matters")
    t2, n2 = reliability(cand, PSEL, "zone")
    log(t2.to_string(float_format=lambda v: f"{v:.3f}"))
    log("   counts:")
    log("   " + n2.to_string().replace("\n", "\n   "))
    spread = (t2.max(axis=1) - t2.min(axis=1))
    log(f"\n   spread across zones per p_sel bin: "
        + ", ".join(f"{i} {v:.3f}" for i, v in spread.items()))
    log(f"   largest spread on a bin with >=100 candidates in every zone: "
        f"{spread[(n2 >= 100).all(axis=1)].max() if (n2 >= 100).all(axis=1).any() else float('nan'):.3f}")

    # 3. real image brightness. NOT sem_ring_mean -- that column is the SEMANTIC
    # MODEL's own probability in a ring, so using it would have measured the
    # model against its own context rather than against the sun's illumination.
    # The disk image is read from the cache and the background level is taken in
    # a ring around each candidate, which is what limb darkening actually moves.
    import cv2
    import json as _json
    from filament.rle import decode_rle
    cache = ROOT / "data" / "semantic_cache"
    idx = _json.loads((cache / "index_768.json").read_text())
    imgs = np.load(cache / "images_768.npy", mmap_mode="r")
    frow = {f.rsplit(".", 1)[0]: i for i, f in enumerate(idx["files"])}
    ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    bright = {}
    for stem, g in cand.groupby("stem"):
        if stem not in frow:
            continue
        im = np.asarray(imgs[frow[stem]]).astype(np.float32)
        for r in g.itertuples():
            m = decode_rle(r.rle).astype(np.uint8)
            sm = cv2.resize(m, (768, 768), interpolation=cv2.INTER_AREA) > 0.5
            if not sm.any():
                continue
            halo = (cv2.dilate(sm.astype(np.uint8), ring) > 0) & ~sm
            if halo.any():
                bright[r.cand_id] = float(im[halo].mean())
    cand["bright_val"] = cand["cand_id"].map(bright)
    b = cand.dropna(subset=["bright_val"]).copy()
    b["bright"] = pd.qcut(b["bright_val"], 4,
                          labels=["B1 dark", "B2", "B3", "B4 bright"])
    log("")
    log(f"3. SELECTOR p_sel by LOCAL IMAGE BRIGHTNESS ({len(b)} candidates)")
    log("   quartile mean DN: " + ", ".join(
        f"{k} {v:.1f}" for k, v in
        b.groupby("bright", observed=True)["bright_val"].mean().items()))
    t3, n3 = reliability(b, PSEL, "bright")
    log(t3.to_string(float_format=lambda v: f"{v:.3f}"))
    log("   counts:")
    log("   " + n3.to_string().replace(chr(10), chr(10) + "   "))
    sp3 = (t3.max(axis=1) - t3.min(axis=1))
    ok3 = (n3 >= 100).all(axis=1)
    log("   spread per bin: " + ", ".join(f"{i} {v:.3f}" for i, v in sp3.items()))
    log(f"   largest spread on a bin with >=100 everywhere: "
        f"{sp3[ok3].max() if ok3.any() else float('nan'):.3f}")

    # what a zone-conditional threshold would be worth, measured the honest way
    log("\n4. WOULD A ZONE-CONDITIONAL THRESHOLD PAY?")
    sys.path.insert(0, str(ROOT / "src" / "models"))
    from selector import fast_pq
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    total_gt = int(gt["stem"].isin(set(cand["stem"])).sum())
    grid = np.round(np.arange(0.15, 0.71, 0.05), 2)
    base = max(((t, fast_pq(cand[cand[PSEL] >= t], total_gt)["pq"]) for t in grid),
               key=lambda x: x[1])
    log(f"  single global threshold : best {base[1]:.4f} at p_sel >= {base[0]}")
    best_per = {}
    for z, g in cand.groupby("zone", observed=True):
        best_per[z] = max(grid, key=lambda t: fast_pq(g[g[PSEL] >= t],
                                                      max(int(total_gt * len(g) / len(cand)), 1))["pq"])
    keep = pd.concat([g[g[PSEL] >= best_per[z]]
                      for z, g in cand.groupby("zone", observed=True)])
    per = fast_pq(keep, total_gt)["pq"]
    log(f"  per-zone thresholds     : {per:.4f}  {best_per}")
    log(f"  difference              : {per - base[1]:+.4f}")
    log("\n  NOTE this per-zone number is IN-SAMPLE: four thresholds were chosen")
    log("  on the same data they are scored on, so it is an upper bound and the")
    log("  honest value is lower. If even the upper bound is under +0.005 the")
    log("  axis is closed without a cross-fitted run.")

    json.dump(dict(global_best=float(base[1]), global_thr=float(base[0]),
                   per_zone_insample=float(per),
                   per_zone_thresholds={str(k): float(v) for k, v in best_per.items()}),
              open(OUT / "radial_calibration.json", "w"), indent=1)
    log(f"\nwrote {OUT/'radial_calibration.json'}")


if __name__ == "__main__":
    main()
