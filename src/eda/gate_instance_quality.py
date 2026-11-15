"""The gate for an instance-level change: per-instance quality, not pixel-Dice.

Why this file exists rather than reading val pixel-Dice off a training log. Twice
in this project a mechanism was judged on a proxy that could not see the
population it targeted -- `eval_tiled.py` demanded a single lane beat the whole
pipeline's oracle, and the physical-channels arm was rejected on val pixel-Dice
in a 0.42% foreground regime where a faint minority moves it by less than the
run-to-run noise. An instance-balanced loss is exactly the kind of change that
trap is built for: it is designed to redistribute quality ACROSS instances at
roughly constant pixel count, which is the one thing an image-level average is
constructed not to notice.

So the gate is the measurement the mechanism claims to move:

  1. mean PER-INSTANCE Dice, split by whether the pipeline currently reaches the
     instance. The baseline reads 0.6472 overall -- 0.7725 on matched rows
     against 0.4478 on never-proposed ones, with 44.5% of the latter under 0.5.
  2. COVERAGE, the fraction of each ground-truth instance the model claims at
     p >= 0.7. The baseline reads 0.743 on matched rows against 0.456 on
     never-proposed ones, and that 0.456 is the number this whole line of work
     is trying to raise.
  3. REACH at the production threshold, counted on BOTH populations, because a
     change that recovers faint rows while breaking matched ones is a loss and
     the single-population view hides it.
  4. image-level pixel-Dice, reported last and only for continuity with the
     ledger -- it is not the gate.

Usage:
  python src/eda/gate_instance_quality.py \
      --base outputs/semantic/fold0_probs_cldice.npz \
      --cand outputs/semantic/fold0_probs_blob.npz
"""

import argparse
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
THR_PROD, MIN_AREA, HI = 0.3, 256, 0.7


def log(*a):
    print(*a, flush=True)


def cc(binm, min_area=MIN_AREA):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        binm.astype(np.uint8), connectivity=8)
    return lab, [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]


def best_iou(gt, lab, keep):
    if not keep:
        return 0.0
    ids, counts = np.unique(lab[gt], return_counts=True)
    best, g, ks = 0.0, int(gt.sum()), set(keep)
    for i, c in zip(ids, counts):
        if i == 0 or int(i) not in ks:
            continue
        u = int((lab == i).sum()) + g - int(c)
        if u:
            best = max(best, int(c) / u)
    return best


def measure(npz_path, rows, gt):
    z = np.load(npz_path)
    out, inter, union = [], 0, 0
    for stem, g in rows.groupby("stem"):
        if stem not in z:
            continue
        prob = np.asarray(z[stem]).astype(np.float32)
        if prob.max() > 1.5:
            prob /= 255.0
        lab, keep = cc(prob >= THR_PROD)
        pred = np.isin(lab, keep) if keep else np.zeros(prob.shape, bool)
        fg = np.zeros(prob.shape, bool)
        for r in g.itertuples():
            m = decode_rle(gt.loc[r.filament_id]).astype(bool)
            if not m.any():
                continue
            fg |= m
            iou = best_iou(m, lab, keep)
            out.append(dict(grp=r.grp, filament_id=r.filament_id, iou=iou,
                            dice=2 * iou / (1 + iou),
                            coverage=float((prob[m] >= HI).mean())))
        inter += int((pred & fg).sum())
        union += int(pred.sum() + fg.sum())
    return pd.DataFrame(out), (2 * inter / max(union, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="outputs/semantic/fold0_probs_cldice.npz")
    ap.add_argument("--cand", required=True)
    ap.add_argument("--fold", type=int, default=0)
    a = ap.parse_args()

    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    f0 = set(folds.loc[folds["fold"] == a.fold, "file_name"]
             .str.replace(".jpeg", "", regex=False))
    d = pd.read_parquet(OUT / "error_profile_fn_gt.parquet")
    d["matched"] = d["matched"].astype(bool)
    d["in_raw_pool"] = d["in_raw_pool"].astype(bool)
    d = d[d["stem"].isin(f0)].copy()
    d["grp"] = np.where(d["matched"], "matched",
                        np.where(d["in_raw_pool"], "in-pool", "never-proposed"))
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt = gt.set_index("filament_id")["segmentation_rle"]

    res = {}
    for tag, p in (("baseline", a.base), ("candidate", a.cand)):
        t0 = time.time()
        df, pd_dice = measure(ROOT / p if not Path(p).is_absolute() else p, d, gt)
        res[tag] = (df, pd_dice)
        log(f"{tag:10s} {Path(p).name}: {len(df)} instances "
            f"({time.time()-t0:.0f}s)")

    b, c = res["baseline"][0], res["candidate"][0]
    log(f"\n{'':28s} {'baseline':>10s} {'candidate':>10s} {'delta':>9s}")

    def line(lbl, fb, fc, fmt="{:10.4f}", d_fmt="{:+9.4f}"):
        log(f"  {lbl:26s} " + fmt.format(fb) + " " + fmt.format(fc)
            + " " + d_fmt.format(fc - fb))

    log("\n1. PER-INSTANCE DICE  (the gate)")
    for g in ("matched", "never-proposed", "in-pool"):
        bb, cc_ = b[b.grp == g], c[c.grp == g]
        if not len(bb):
            continue
        line(f"{g} (n={len(bb)})", bb["dice"].mean(), cc_["dice"].mean())
    line("ALL instances", b["dice"].mean(), c["dice"].mean())
    line("frac under Dice 0.5", (b["dice"] < 0.5).mean(), (c["dice"] < 0.5).mean())

    log(f"\n2. COVERAGE  (fraction of each instance claimed at p >= {HI})")
    for g in ("matched", "never-proposed"):
        bb, cc_ = b[b.grp == g], c[c.grp == g]
        if not len(bb):
            continue
        line(f"{g}", bb["coverage"].mean(), cc_["coverage"].mean())

    log(f"\n3. REACH at the production threshold {THR_PROD} "
        f"(rows at IoU > 0.5)")
    tot_b = tot_c = 0
    for g in ("matched", "never-proposed", "in-pool"):
        bb, cc_ = b[b.grp == g], c[c.grp == g]
        if not len(bb):
            continue
        nb, nc = int((bb["iou"] > 0.5).sum()), int((cc_["iou"] > 0.5).sum())
        tot_b += nb
        tot_c += nc
        log(f"  {g:26s} {nb:10d} {nc:10d} {nc-nb:+9d}")
    log(f"  {'NET rows reached':26s} {tot_b:10d} {tot_c:10d} {tot_c-tot_b:+9d}")
    log("  (net is what PQ responds to: gaining faint rows while losing matched")
    log("   ones is a loss, and only this line can tell the difference)")

    log(f"\n4. image-level pixel-Dice  (for continuity with the ledger, NOT the gate)")
    line("pixel-Dice", res["baseline"][1], res["candidate"][1])

    json.dump(dict(
        base=a.base, cand=a.cand,
        per_instance_dice={g: dict(
            base=float(b[b.grp == g]["dice"].mean()),
            cand=float(c[c.grp == g]["dice"].mean()))
            for g in b["grp"].unique()},
        reach_net=dict(base=tot_b, cand=tot_c),
        pixel_dice=dict(base=res["baseline"][1], cand=res["candidate"][1])),
        open(OUT / "gate_instance_quality.json", "w"), indent=1)
    log(f"\nwrote {OUT/'gate_instance_quality.json'}")


if __name__ == "__main__":
    main()
