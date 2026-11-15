"""Ceiling part 2: union & strict consensus variants (lean memory), merge into JSON.

Part 1 (LOAO + majority) crashed mid-run on RAM while training ran concurrently;
its numbers are already in outputs/eda/annotator_ceiling.json. This computes the
remaining variants with aggressive memory hygiene and updates the JSON in place,
writing after EACH mode.
"""

import gc
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


def consensus_masks(sub, mode):
    entries = sorted(sub["entry_id"].unique())
    n_copies = len(entries)
    layers = rles_to_layers(sub["segmentation_rle"].tolist())
    owner = sub["entry_id"].values
    n = len(sub)
    iou = _entry_iou_matrix(layers, layers)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if iou[i, j] > 0.3 and owner[i] != owner[j]:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)

    out = []
    for members in clusters.values():
        support = len({owner[i] for i in members})
        if mode == "strict" and support < n_copies:
            continue
        # accumulate per-copy union then mean without stacking
        acc = np.zeros((IMG_H, IMG_W), dtype=np.float32)
        seen = {}
        for i in members:
            seen.setdefault(owner[i], np.zeros((IMG_H, IMG_W), dtype=bool))
            seen[owner[i]] |= layers[i].astype(bool)
        for m in seen.values():
            acc += m
        mask = (acc / len(seen)) >= 0.5
        if mask.sum() == 0 and mode == "union":
            mask = acc > 0
        if mask.sum() > 0:
            out.append((support, mask))
        del acc, seen
    del layers, iou
    out.sort(key=lambda t: -t[0])
    taken = np.zeros((IMG_H, IMG_W), dtype=bool)
    final = []
    for support, m in out:
        m = m & ~taken
        if m.sum() > 0:
            final.append(m.astype(np.uint8))
            taken |= m
    return final


with open(OUT / "annotator_ceiling.json") as f:
    results = json.load(f)

for mode in ["union", "strict"]:
    if mode in results:
        continue
    acc = defaultdict(float)
    t0 = time.time()
    for fi, fn in enumerate(multi_files):
        sub = gt[gt["file_name"] == fn]
        stem = sub["stem"].iloc[0]
        masks = consensus_masks(sub, mode)
        pred_df = pd.DataFrame(
            [(f"{stem}_{k+1}", encode_mask(m)) for k, m in enumerate(masks)],
            columns=["filament_id", "segmentation_rle"])
        if pred_df.empty:
            pred_df = pd.DataFrame([("NONE_1", encode_mask(np.zeros((IMG_H, IMG_W), np.uint8)))],
                                   columns=["filament_id", "segmentation_rle"])
        r = score(sub[["filament_id", "segmentation_rle"]], pred_df)
        acc["iou_sum"] += r["sq"] * r["tp"]
        acc["tp"] += r["tp"]; acc["fp"] += r["fp"]; acc["fn"] += r["fn"]
        if (fi + 1) % 50 == 0:
            gc.collect()
            print(f"  {mode} {fi+1}/{len(multi_files)} ({time.time()-t0:.0f}s)", flush=True)
    denom = acc["tp"] + 0.5 * acc["fp"] + 0.5 * acc["fn"]
    results[mode] = dict(pq=acc["iou_sum"] / denom if denom else 0.0,
                         sq=acc["iou_sum"] / acc["tp"] if acc["tp"] else 0.0,
                         rq=acc["tp"] / denom if denom else 0.0,
                         tp=int(acc["tp"]), fp=int(acc["fp"]), fn=int(acc["fn"]))
    with open(OUT / "annotator_ceiling.json", "w") as f:
        json.dump(results, f, indent=1)
    print(mode, {k: round(v, 4) if isinstance(v, float) else v
                 for k, v in results[mode].items()}, flush=True)
print("done")
