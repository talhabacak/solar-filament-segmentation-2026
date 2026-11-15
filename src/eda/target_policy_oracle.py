"""E127 -- which annotator target could a PERFECT model be trained on?

Talha's proposal: keep only the narrowest of the annotators, treat that as the
truth, and drop the others; maybe the model is confused by the disagreement and
this is what the 5% shrink is really compensating for.

The proposal is answerable without training anything. If a model trained on
target policy X were perfect, its prediction on a multi-copy image would BE the
mask that policy produces. So we can take each policy's masks as the prediction
and score them with the exact competition scorer against ALL copies of that
image. That is the ceiling of the policy: no model trained on it can do better.

Policies scored (multi-copy training images only, so every policy is defined):

    narrowest   the copy with the smallest total annotated area
    widest      the copy with the largest total annotated area
    median      the copy whose area is the median
    union       per-pixel union of the copies, re-instanced by connected component
    intersect   per-pixel intersection, re-instanced
    agree>t     pixels drawn by more than a fraction t of the annotators,
                re-instanced, for t on a grid -- the metric's own optimum is
                t* = IoU/(1+IoU), 0.404 at our matched IoU of 0.679

Reported per policy: pooled PQ, SQ, RQ, TP, FP, FN over the same images, plus
the mean instance area relative to the mean copy area, so "narrow" and "wide"
can be read directly. The comparison is honest in the only way that matters
here: every policy is scored by the same rule the competition uses, against
every annotator.

Usage: python src/eda/target_policy_oracle.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from filament.pq import score  # noqa: E402
from filament.rle import decode_rle, encode_mask  # noqa: E402

OUT = ROOT / "outputs" / "eda"
MIN_AREA = 256
GRID = [0.2, 0.3, 0.34, 0.4, 0.5, 0.6, 0.7]


def log(*a):
    print(*a, flush=True)


def instances(mask):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [(lab == i) for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= MIN_AREA]


def main():
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    train = set(folds["file_name"].str.replace(".jpeg", "", regex=False))
    gt = gt[gt["stem"].isin(train)]
    ncop = gt.groupby("stem")["entry_id"].nunique()
    multi = sorted(ncop[ncop > 1].index)
    log(f"{len(multi)} multi-copy training images ({int((ncop > 1).sum())} of {len(ncop)})")

    policies = ["narrowest", "widest", "median", "union", "intersect"] + [f"agree>{t}" for t in GRID]
    preds = {p: [] for p in policies}
    areas = {p: [] for p in policies}
    copy_area = []
    for n, stem in enumerate(multi):
        g = gt[gt["stem"] == stem]
        entries = sorted(g["entry_id"].unique())
        N = len(entries)
        masks, tot = {}, {}
        acc = np.zeros((2048, 2048), np.float32)
        for e in entries:
            m = np.zeros((2048, 2048), bool)
            for r in g.loc[g["entry_id"] == e, "segmentation_rle"]:
                m |= decode_rle(r).astype(bool)
            masks[e] = m
            tot[e] = int(m.sum())
            acc += m
        copy_area.append(float(np.mean(list(tot.values()))))
        order = sorted(entries, key=lambda e: tot[e])
        chosen = {"narrowest": masks[order[0]], "widest": masks[order[-1]],
                  "median": masks[order[len(order) // 2]],
                  "union": acc > 0, "intersect": acc >= N}
        for t in GRID:
            chosen[f"agree>{t}"] = acc / N > t
        for p, m in chosen.items():
            insts = instances(m)
            areas[p].append(float(np.mean([i.sum() for i in insts])) if insts else 0.0)
            for k, inst in enumerate(insts):
                preds[p].append((f"{stem}_{p.replace('>', '')}{k}", encode_mask(inst.astype(np.uint8))))
        if n % 50 == 0:
            log(f"  {n}/{len(multi)} images")

    gsub = gt[["filament_id", "segmentation_rle"]]
    res = {}
    log(f"\nmean copy area {np.mean(copy_area):.0f} px")
    log(f"{'policy':12s} {'PQ':>8s} {'SQ':>8s} {'RQ':>8s} {'TP':>6s} {'FP':>6s} {'FN':>6s} {'inst area':>10s} {'vs copy':>8s}")
    for p in policies:
        df = pd.DataFrame(preds[p], columns=["filament_id", "segmentation_rle"])
        r = score(gsub, df)
        rel = np.mean(areas[p]) / max(np.mean(copy_area), 1)
        res[p] = dict(pq=r["pq"], sq=r["sq"], rq=r["rq"], tp=r["tp"], fp=r["fp"], fn=r["fn"],
                      mean_inst_area=float(np.mean(areas[p])), rel_area=float(rel))
        log(f"{p:12s} {r['pq']:8.4f} {r['sq']:8.4f} {r['rq']:8.4f} {r['tp']:6d} {r['fp']:6d} {r['fn']:6d} "
            f"{np.mean(areas[p]):10.0f} {rel:8.2f}")
    best = max(res, key=lambda p: res[p]["pq"])
    log(f"\nbest target policy: {best} (PQ {res[best]['pq']:.4f}); "
        f"narrowest is {res['narrowest']['pq']:.4f}, widest {res['widest']['pq']:.4f}")
    json.dump(res, open(OUT / "target_policy_oracle.json", "w"), indent=1, default=float)
    log(f"wrote {OUT / 'target_policy_oracle.json'}")


if __name__ == "__main__":
    main()
