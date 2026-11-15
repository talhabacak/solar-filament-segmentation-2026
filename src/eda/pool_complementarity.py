"""E184 -- selector-free pool reading: what does each configuration find that the others do not?

Talha, 2026-09-13: "when the IoU thresholds change, how right is it to look at PQ directly? The
selector is not fitted for that distribution and may need retraining -- so it may be better to
look at what the two segmentation models find EXTRA relative to each other."

He is right, and it is a flaw in how the threshold sweep was read. `_paired_pools.py` fits the
selector on the PRODUCTION pool of the other folds and applies it to the new pool; that is the
deployment-faithful protocol for a MEMBER swap, where the candidate distribution barely moves,
but a threshold change rebuilds the distribution itself (clustering at 0.4 gives 1098 TP and
1758 FP against production's 910 / 352). A selector fitted on one distribution and applied to
another measures the mismatch as much as the change.

This reads the pool before any selection, so no selector is involved:

  reach          ground-truth copies with a candidate at IoU > 0.5 (what the pool COULD score)
  gained / lost  copies reachable in A but not B, and the reverse -- the complementarity Talha
                 asked for, per pair
  cost           candidates per image (every extra candidate is a chance for the selector to
                 err, so reach is only half the story)
  near band      copies whose best candidate sits in 0.4-0.5, one small step short

Use it for anything that changes the candidate distribution -- thresholds, lanes, NMS, fusion
rules -- and only promote to the full protocol (rebuild all five folds, relabel, refit the
selector) what improves reach without an unreasonable candidate cost.

Usage: python src/eda/pool_complementarity.py df_ref8_t020 sw_cl06_t020 sw_cl07_t020 ...
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from filament.rle import decode_rle  # noqa: E402
from filament.pq import _entry_iou_matrix  # noqa: E402

OUT = ROOT / "outputs" / "eda"


def log(*a):
    print(*a, flush=True)


def reach_table(tag, gt, stems):
    f = OUT / f"fused_pool_f0_{tag}_labelled.parquet"
    if not f.exists():
        log(f"{tag}: pool missing")
        return None
    pool = pd.read_parquet(f)
    best = {}
    for stem in stems:
        gg = gt[gt["stem"] == stem]
        pc = pool[pool["stem"] == stem]
        if not len(gg):
            continue
        if not len(pc):
            for fid in gg["filament_id"]:
                best[fid] = 0.0
            continue
        G = np.stack([decode_rle(r).astype(np.uint8) for r in gg["segmentation_rle"]])
        P = np.stack([decode_rle(r).astype(np.uint8) for r in pc["rle"]])
        iou = _entry_iou_matrix(G, P).max(axis=1)
        for fid, v in zip(gg["filament_id"], iou):
            best[fid] = float(v)
    return pd.Series(best, name=tag), len(pool), pool["stem"].nunique()


def main():
    tags = sys.argv[1:]
    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
    folds["stem"] = folds["file_name"].str.replace(".jpeg", "", regex=False)
    stems = sorted(set(folds.loc[folds["fold"] == 0, "stem"]) & set(gt["stem"]))
    gt = gt[gt["stem"].isin(stems)]
    log(f"fold 0: {len(stems)} images, {len(gt)} ground-truth copies\n")
    series, meta = {}, {}
    for t in tags:
        r = reach_table(t, gt, stems)
        if r is None:
            continue
        series[t], meta[t] = r[0], (r[1], r[2])
    d = pd.DataFrame(series)
    log("per configuration (selector-free):")
    rows = {}
    for t in d.columns:
        v = d[t]
        rows[t] = dict(candidates=meta[t][0], per_image=round(meta[t][0] / max(meta[t][1], 1), 1),
                       reach=int((v > 0.5).sum()), reach_pct=round(float((v > 0.5).mean()), 4),
                       near_040_050=int(((v > 0.4) & (v <= 0.5)).sum()),
                       mean_best=round(float(v.mean()), 4))
    log(pd.DataFrame(rows).T.to_string())
    base = d.columns[0]
    log(f"\ncomplementarity against {base} (ground-truth copies reachable at IoU > 0.5):")
    for t in d.columns[1:]:
        a, b = d[base] > 0.5, d[t] > 0.5
        gained, lost = int((~a & b).sum()), int((a & ~b).sum())
        dc = meta[t][0] - meta[base][0]
        log(f"  {t:22s} gained {gained:4d}   lost {lost:4d}   net {gained - lost:+4d}   "
            f"at {dc:+6d} candidates ({dc / max(gained - lost, 1):+.0f} per net row)" if gained != lost else
            f"  {t:22s} gained {gained:4d}   lost {lost:4d}   net {gained - lost:+4d}   at {dc:+6d} candidates")
    if len(d.columns) > 2:
        log("\npairwise: rows only the row-configuration reaches (not the column one)")
        m = pd.DataFrame(index=d.columns, columns=d.columns, dtype=int)
        for i in d.columns:
            for j in d.columns:
                m.loc[i, j] = int(((d[i] > 0.5) & ~(d[j] > 0.5)).sum())
        log(m.to_string())
    d.to_parquet(OUT / "pool_complementarity.parquet")
    log("\nCOMPLEMENTARITY_DONE")


if __name__ == "__main__":
    main()
