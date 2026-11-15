"""E134 -- disc and limb as two separate problems: where exactly does each lose?

Per-band PQ (ledger 22a) says the disc scores 0.445 and the limb 0.367, with SQ
flat everywhere and only RQ collapsing. That says the masks are fine and the
objects are missing, but not WHERE they are lost. This walks every ground-truth
row through the pipeline and attributes it to the stage that failed:

  reached          the deployed keep set matches it at IoU > 0.5           (TP)
  selector         a pool candidate matches it at IoU > 0.5 but was not kept
  boundary         the best pool candidate overlaps it at 0.05 < IoU <= 0.5
  not proposed     no pool candidate overlaps it beyond IoU 0.05

and, for the rows that were proposed at all, which lane produced the best
candidate (detector-only, semantic-only, or both) -- so "the detector never saw
it" and "the fusion lost it" stop being the same sentence.

False positives are attributed the same way: a kept prediction with no overlap
is a hallucination, one overlapping below the threshold is a boundary failure.

Everything is computed on the champion configuration (semantic threshold 0.2
pool, 42-feature selector, out-of-fold scores) with the exact scorer's matching
rule, and reported for r <= 0.8, 0.8-0.9 and > 0.9.

Usage: python src/eda/band_failure_anatomy.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from filament.rle import decode_rle  # noqa: E402

OUT = ROOT / "outputs" / "eda"
GATE = (0.40, 0.60)
BANDS = [("disc r<=0.8", 0.0, 0.8), ("limb 0.8-0.9", 0.8, 0.9), ("far limb >0.9", 0.9, 1.01)]


def log(*a):
    print(*a, flush=True)


def main():
    a = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    a = a.merge(pd.read_parquet(OUT / "t020pox_oof_scores.parquet"), on="cand_id", how="left")
    keepmask = (a["conf"] >= GATE[0]) & (a["p"] >= GATE[1])
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt = gt[gt["stem"].isin(set(a["stem"]))]
    rg = pd.read_parquet(OUT / "limb_geometry.parquet").set_index("filament_id")["r_norm"]
    gt["r"] = gt["filament_id"].map(rg)
    gt = gt[gt["r"].notna()]
    log(f"pool {len(a)} candidates ({int(keepmask.sum())} kept), {len(gt)} GT rows")

    rows, fps = [], []
    for n, (stem, g) in enumerate(gt.groupby("stem")):
        c = a[a["stem"] == stem]
        if c.empty:
            for r in g.itertuples():
                rows.append(dict(r_norm=r.r, stage="not proposed", lane="none"))
            continue
        masks = [decode_rle(x).astype(bool) for x in c["rle"]]
        kept = keepmask.loc[c.index].to_numpy()
        lanes = list(zip(c["from_det"].to_numpy(), c["from_sem"].to_numpy()))
        best_for_pred = np.zeros(len(masks))
        for r in g.itertuples():
            gm = decode_rle(r.segmentation_rle).astype(bool)
            ga = gm.sum()
            if ga == 0:
                continue
            ious = np.zeros(len(masks))
            for i, m in enumerate(masks):
                inter = np.count_nonzero(m & gm)
                if inter:
                    ious[i] = inter / (m.sum() + ga - inter)
            best_for_pred = np.maximum(best_for_pred, ious)
            j = int(np.argmax(ious)) if len(ious) else -1
            best = ious[j] if j >= 0 else 0.0
            if best > 0.5 and kept[j]:
                stage = "reached"
            elif best > 0.5:
                stage = "selector"
            elif best > 0.05:
                stage = "boundary"
            else:
                stage = "not proposed"
            d, sm = (lanes[j] if j >= 0 else (0, 0))
            lane = "both" if (d and sm) else ("detector" if d else ("semantic" if sm else "none"))
            rows.append(dict(r_norm=r.r, stage=stage, lane=lane if best > 0.05 else "none"))
        for i, m in enumerate(masks):
            if kept[i]:
                fps.append(dict(r_norm=float(c["r_norm"].iloc[i]),
                                kind="true" if best_for_pred[i] > 0.5 else
                                     ("boundary" if best_for_pred[i] > 0.05 else "hallucination")))
        if n % 100 == 0:
            log(f"  {n}/{gt['stem'].nunique()} images")

    d = pd.DataFrame(rows)
    f = pd.DataFrame(fps)
    res = {}
    log(f"\n{'band':16s} {'GT':>6s} " + " ".join(f"{s:>13s}" for s in ("reached", "selector", "boundary", "not proposed")))
    for name, lo, hi in BANDS:
        sub = d[(d["r_norm"] > lo) & (d["r_norm"] <= hi)]
        if sub.empty:
            continue
        vc = sub["stage"].value_counts()
        res[name] = {k: int(vc.get(k, 0)) for k in ("reached", "selector", "boundary", "not proposed")}
        res[name]["n"] = len(sub)
        log(f"{name:16s} {len(sub):6d} " + " ".join(
            f"{vc.get(s, 0):6d} ({vc.get(s, 0)/len(sub):5.1%})" for s in ("reached", "selector", "boundary", "not proposed")))
    log("\nwhich lane produced the best candidate, for rows that were proposed at all:")
    for name, lo, hi in BANDS:
        sub = d[(d["r_norm"] > lo) & (d["r_norm"] <= hi) & (d["lane"] != "none")]
        if sub.empty:
            continue
        vc = sub["lane"].value_counts(normalize=True)
        log(f"  {name:16s} both {vc.get('both', 0):5.1%}  detector-only {vc.get('detector', 0):5.1%}  semantic-only {vc.get('semantic', 0):5.1%}")
    log("\nkept predictions by band and kind:")
    for name, lo, hi in BANDS:
        sub = f[(f["r_norm"] > lo) & (f["r_norm"] <= hi)]
        if sub.empty:
            continue
        vc = sub["kind"].value_counts()
        log(f"  {name:16s} n {len(sub):5d}  true {vc.get('true', 0)/len(sub):5.1%}  "
            f"boundary-FP {vc.get('boundary', 0)/len(sub):5.1%}  hallucination {vc.get('hallucination', 0)/len(sub):5.1%}")
        res[name]["preds"] = {k: int(v) for k, v in vc.items()}
    json.dump(res, open(OUT / "band_failure_anatomy.json", "w"), indent=1, default=float)
    log(f"\nwrote {OUT / 'band_failure_anatomy.json'}")


if __name__ == "__main__":
    main()
