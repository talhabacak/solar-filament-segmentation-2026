"""Measure the PQ ceiling imposed by annotator disagreement.

Protocols (all with the verified exact scorer, global pooling like the official metric):

1. LOAO (leave-one-annotator-out): on each multi-copy file, each annotator copy's
   masks are scored as a "submission" against the remaining copies. This is
   human-level PQ under the official metric.
2. Consensus variants scored against ALL copies of multi-copy files:
   - majority: instance clusters present in >=2 copies, pixelwise-mean>=0.5 mask
   - union: all clusters (support>=1), deduplicated
   - strict: clusters present in ALL copies
3. Pairwise matched-instance IoU stats (bounds SQ) and match rate (bounds RQ).
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament import IMG_H, IMG_W
from filament.pq import score, _entry_iou_matrix
from filament.rle import rles_to_layers, encode_mask

OUT = ROOT / "outputs" / "eda"
gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")

gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
copies = gt.groupby("file_name")["entry_id"].nunique()
multi_files = sorted(copies[copies >= 2].index)
print(f"multi-copy files: {len(multi_files)}")

# ---------- Protocol 1: LOAO ----------
acc = defaultdict(float)
per_pred = []          # one row per (file, predicting annotator copy)
pair_ious = []         # matched-pair IoUs across copies

t0 = time.time()
for fi, fn in enumerate(multi_files):
    sub = gt[gt["file_name"] == fn]
    stem = sub["stem"].iloc[0]
    entries = sorted(sub["entry_id"].unique())
    for entry in entries:
        pred_rows = sub[sub["entry_id"] == entry].copy()
        pred_rows = pred_rows.reset_index(drop=True)
        pred_rows["filament_id"] = [f"{stem}_{k+1}" for k in range(len(pred_rows))]
        gt_rows = sub[sub["entry_id"] != entry]
        r = score(gt_rows[["filament_id", "segmentation_rle"]],
                  pred_rows[["filament_id", "segmentation_rle"]],
                  return_details=True)
        iou_sum = r["sq"] * r["tp"]
        acc["iou_sum"] += iou_sum
        acc["tp"] += r["tp"]; acc["fp"] += r["fp"]; acc["fn"] += r["fn"]
        denom = r["tp"] + 0.5 * r["fp"] + 0.5 * r["fn"]
        per_pred.append(dict(file_name=fn, pred_entry=entry, n_copies=len(entries),
                             tp=r["tp"], fp=r["fp"], fn=r["fn"],
                             iou_sum=iou_sum,
                             pq=(iou_sum / denom) if denom else 0.0))
        if len(r["matches"]):
            pair_ious.extend(r["matches"]["iou"].tolist())
    if (fi + 1) % 50 == 0:
        print(f"  LOAO {fi+1}/{len(multi_files)}  ({time.time()-t0:.0f}s)")

loao_pq = acc["iou_sum"] / (acc["tp"] + 0.5 * acc["fp"] + 0.5 * acc["fn"])
loao = dict(pq=loao_pq,
            sq=acc["iou_sum"] / acc["tp"] if acc["tp"] else 0.0,
            rq=acc["tp"] / (acc["tp"] + 0.5 * acc["fp"] + 0.5 * acc["fn"]),
            tp=int(acc["tp"]), fp=int(acc["fp"]), fn=int(acc["fn"]))
print("LOAO global:", {k: round(v, 4) if isinstance(v, float) else v for k, v in loao.items()})

per_pred_df = pd.DataFrame(per_pred)
per_pred_df.to_parquet(OUT / "annotator_ceiling_per_pred.parquet", index=False)

# ---------- Protocol 2: consensus variants ----------
def consensus_masks(sub: pd.DataFrame, mode: str):
    """Cluster instances across annotator copies; return list of binary masks."""
    entries = sorted(sub["entry_id"].unique())
    n_copies = len(entries)
    layers = rles_to_layers(sub["segmentation_rle"].tolist())
    owner = sub["entry_id"].values                      # copy of each instance
    n = len(sub)
    # pairwise IoU between all instances of different copies
    iou = _entry_iou_matrix(layers, layers)             # (n, n), self-pairs included
    adj = (iou > 0.3)
    for i in range(n):
        adj[i, i] = False
    # union-find over cross-copy links
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] and owner[i] != owner[j]:
                union(i, j)
    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)

    out = []
    for members in clusters.values():
        support = len({owner[i] for i in members})
        if mode == "majority" and support < min(2, n_copies):
            continue
        if mode == "strict" and support < n_copies:
            continue
        # per copy: union of that copy's members, then pixelwise mean over copies present
        per_copy = defaultdict(lambda: np.zeros((IMG_H, IMG_W), dtype=np.uint8))
        for i in members:
            per_copy[owner[i]] |= layers[i].astype(np.uint8)
        stack = np.stack(list(per_copy.values())).astype(np.float32)
        mean = stack.mean(axis=0)
        mask = (mean >= 0.5).astype(np.uint8)
        if mode == "union" and mask.sum() == 0:      # single-copy cluster -> mean=1 anyway
            mask = (mean > 0).astype(np.uint8)
        if mask.sum() > 0:
            out.append((support, mask))
    # disjointness: assign contested pixels to higher-support (then earlier) mask
    out.sort(key=lambda t: -t[0])
    taken = np.zeros((IMG_H, IMG_W), dtype=bool)
    final = []
    for support, m in out:
        m = m.astype(bool) & ~taken
        if m.sum() > 0:
            final.append(m.astype(np.uint8))
            taken |= m
    return final

results = {"loao": loao}
for mode in ["majority", "union", "strict"]:
    acc2 = defaultdict(float)
    t0 = time.time()
    for fi, fn in enumerate(multi_files):
        sub = gt[gt["file_name"] == fn]
        stem = sub["stem"].iloc[0]
        masks = consensus_masks(sub, mode)
        pred_df = pd.DataFrame(
            [(f"{stem}_{k+1}", encode_mask(m)) for k, m in enumerate(masks)],
            columns=["filament_id", "segmentation_rle"])
        if len(pred_df) == 0:
            pred_df = pd.DataFrame([("NONE_1", encode_mask(np.zeros((IMG_H, IMG_W), np.uint8)))],
                                   columns=["filament_id", "segmentation_rle"])
        r = score(sub[["filament_id", "segmentation_rle"]], pred_df)
        acc2["iou_sum"] += r["sq"] * r["tp"]
        acc2["tp"] += r["tp"]; acc2["fp"] += r["fp"]; acc2["fn"] += r["fn"]
        if (fi + 1) % 100 == 0:
            print(f"  {mode} {fi+1}/{len(multi_files)} ({time.time()-t0:.0f}s)")
    denom = acc2["tp"] + 0.5 * acc2["fp"] + 0.5 * acc2["fn"]
    results[mode] = dict(pq=acc2["iou_sum"] / denom if denom else 0.0,
                         sq=acc2["iou_sum"] / acc2["tp"] if acc2["tp"] else 0.0,
                         rq=acc2["tp"] / denom if denom else 0.0,
                         tp=int(acc2["tp"]), fp=int(acc2["fp"]), fn=int(acc2["fn"]))
    print(mode, {k: round(v, 4) if isinstance(v, float) else v for k, v in results[mode].items()})

# ---------- Protocol 3: pairwise agreement stats ----------
pair_ious = np.array(pair_ious)
results["pairwise"] = dict(
    n_matched_pairs=int(len(pair_ious)),
    matched_iou_mean=float(pair_ious.mean()) if len(pair_ious) else 0.0,
    matched_iou_median=float(np.median(pair_ious)) if len(pair_ious) else 0.0,
    matched_iou_p10=float(np.percentile(pair_ious, 10)) if len(pair_ious) else 0.0,
    matched_iou_p90=float(np.percentile(pair_ious, 90)) if len(pair_ious) else 0.0,
)

q = per_pred_df["pq"]
results["loao_per_pred_pq"] = dict(p10=float(q.quantile(0.1)), p50=float(q.median()),
                                   p90=float(q.quantile(0.9)))
for nc in [2, 3]:
    sel = per_pred_df[per_pred_df["n_copies"] == nc]
    d = sel[["tp", "fp", "fn", "iou_sum"]].sum()
    denom = d["tp"] + 0.5 * d["fp"] + 0.5 * d["fn"]
    results[f"loao_{nc}copy"] = dict(pq=float(d["iou_sum"] / denom) if denom else 0.0,
                                     n_preds=int(len(sel)))

with open(OUT / "annotator_ceiling.json", "w") as f:
    json.dump(results, f, indent=1)
print(json.dumps(results, indent=1))
