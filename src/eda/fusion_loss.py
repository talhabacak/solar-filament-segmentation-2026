"""Our own fusion throws away reachable ground truth. Which step does it?

Measured on fold 0: the raw union of detector + semantic candidates reaches
1243/1510 = 0.8232 of the GT rows, but the fused pool that comes out the other
end reaches only 0.705 pooled. Roughly 0.12 of reachability is destroyed by our
own post-processing — a bigger loss than any extra model would recover (adding
the 1-band net was worth +0.0139).

Three steps could be responsible, and they need very different fixes:

  cluster    candidates with IoU>0.5 are merged into one confidence-weighted
             consensus mask. If two of our candidates each matched a DIFFERENT
             annotator's version of the same filament, the merged mask can match
             neither at the strict 0.5 threshold.
  vote       the merged mask is thresholded at 0.5 of the weighted votes, which
             can shrink it below what any member was.
  deoverlap  masks are made pixel-disjoint greedily in confidence order, so a
             lower-confidence candidate is carved up by whatever was placed
             first. Disjointness is a submission requirement; the greedy ORDER
             is not.

This reports reachability after each step so the fix targets the right one.

Usage: python src/eda/fusion_loss.py --fold 0
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament import IMG_H, IMG_W
from filament.pq import score, _entry_iou_matrix
from filament.rle import decode_rle, encode_mask
from semantic_candidates import prob_to_instances, R_MAX

OUT = ROOT / "outputs" / "eda"
SEM_THR, SEM_MIN_AREA = 0.3, 256


def log(*a):
    print(*a, flush=True)


def cluster(items):
    """Union-find at IoU>0.5, exactly as build_fused_pool does."""
    masks = np.stack([m for _, _, m in items]).astype(np.uint8)
    iou = _entry_iou_matrix(masks, masks)
    parent = list(range(len(items)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ii, jj = np.nonzero(iou > 0.5)
    for i, j in zip(ii, jj):
        if i < j:
            ri, rj = find(int(i)), find(int(j))
            if ri != rj:
                parent[rj] = ri
    groups = defaultdict(list)
    for i in range(len(items)):
        groups[find(i)].append(i)
    return masks, list(groups.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    a = ap.parse_args()

    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    stems = [f.rsplit(".", 1)[0]
             for f in sorted(folds.loc[folds["fold"] == a.fold, "file_name"])]
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt_f = gt[gt["file_name"].isin({s + ".jpeg" for s in stems})]
    N = len(gt_f)

    det = pd.read_parquet(ROOT / "outputs" / "oof_cache" / f"fold{a.fold}_preds_e013.parquet")
    det = det[(det["conf"] >= 0.05) & (det["r_norm"] <= R_MAX)]
    if "view" in det.columns:
        det = det[det["view"] == "o"]
    z = np.load(ROOT / "outputs" / "semantic" / f"fold{a.fold}_probs_cldice.npz")

    stage = {k: [] for k in ("raw", "clustered", "deoverlapped")}
    t0 = time.time()
    for n, stem in enumerate(stems):
        items = []
        for r in det[det["stem"] == stem].sort_values("conf", ascending=False).itertuples():
            items.append(("det", float(r.conf), decode_rle(r.rle).astype(bool)))
        prob = z[stem].astype(np.float32) / 255.0
        for c in prob_to_instances(prob, SEM_THR, SEM_MIN_AREA):
            items.append(("sem", float(c["conf"]), c["mask"].astype(bool)))
        if not items:
            continue

        for k, (_, _, m) in enumerate(items):
            stage["raw"].append((f"{stem}_r{k}", encode_mask(m.astype(np.uint8))))

        masks, groups = cluster(items)
        fused = []
        for members in groups:
            confs = np.array([items[i][1] for i in members], np.float32)
            w = confs / confs.sum()
            acc = np.tensordot(w, masks[members].astype(np.float32), axes=1)
            m = acc >= 0.5
            if not m.any():
                m = masks[members[int(np.argmax(confs))]].astype(bool)
            fused.append((float(confs.max()), m))
        fused.sort(key=lambda t: -t[0])
        for k, (_, m) in enumerate(fused):
            stage["clustered"].append((f"{stem}_c{k}", encode_mask(m.astype(np.uint8))))

        taken = np.zeros((IMG_H, IMG_W), bool)
        for k, (_, m) in enumerate(fused):
            mm = m & ~taken
            if not mm.any():
                continue
            taken |= mm
            stage["deoverlapped"].append((f"{stem}_d{k}", encode_mask(mm.astype(np.uint8))))
        if (n + 1) % 40 == 0:
            log(f"  {n+1}/{len(stems)} images ({time.time()-t0:.0f}s)")

    gt_cache, res = {}, {}
    prev = None
    for name in ("raw", "clustered", "deoverlapped"):
        df = pd.DataFrame(stage[name], columns=["filament_id", "segmentation_rle"])
        r = score(gt_f[["filament_id", "segmentation_rle"]], df,
                  gt_cache=gt_cache, return_details=True)
        hit = r["matches"]["gt_filament_id"].nunique()
        delta = "" if prev is None else f"  ({(hit-prev)/N:+.4f})"
        log(f"{name:14s} preds {len(df):5d}  reach {hit:5d}/{N} = {hit/N:.4f}{delta}  "
            f"PQ {r['pq']:.4f} SQ {r['sq']:.4f}")
        res[name] = dict(n_preds=len(df), reached=hit, reach=hit / N,
                         pq=r["pq"], sq=r["sq"])
        prev = hit

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"fusion_loss_fold{a.fold}.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    log(f"saved -> {OUT / f'fusion_loss_fold{a.fold}.json'}")


if __name__ == "__main__":
    main()
