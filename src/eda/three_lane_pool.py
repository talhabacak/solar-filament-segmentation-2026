"""Does the seed-refined lane survive fusion, or get absorbed like the others?

Raw reachability on fold 0 rises from 0.8232 (detector + semantic) to 0.8417
with the seed-refined lane — 28 GT rows the current two never reach, 9 of them
Unidentifiable. But raw reachability has repeatedly failed to survive: the
1-band model added 21 rows and was not worth training four more folds, and
deferring de-overlap added 0.0496 of pool reachability for exactly zero realised
PQ.

So this measures the thing that actually decides it — oracle PQ through the real
pipeline (cluster at IoU>0.5, confidence-weighted consensus, de-overlap, then
selection by the metric-derived value rule) with and without the third lane.
Oracle selection on both sides, so the selector is not what is being compared.

The seed lane's confidence is inherited from the detector box it came from,
which is the only score it has.

Usage: python src/eda/three_lane_pool.py --fold 0
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament import IMG_H, IMG_W
from filament.pq import score, _entry_iou_matrix
from filament.rle import decode_rle, encode_mask
from semantic_candidates import prob_to_instances, R_MAX
from apply_seed_refiner import load, refine_image, bbox_of

DATA = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
OUT = ROOT / "outputs" / "eda"
SEM_THR, SEM_MIN_AREA = 0.3, 256


def log(*a):
    print(*a, flush=True)


def fuse(items):
    """Cluster at IoU>0.5, confidence-weighted vote, greedy de-overlap."""
    if not items:
        return []
    masks = np.stack([m for _, m in items]).astype(np.uint8)
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

    fused = []
    for members in groups.values():
        confs = np.array([items[i][0] for i in members], np.float32)
        acc = np.tensordot(confs / confs.sum(), masks[members].astype(np.float32), axes=1)
        m = acc >= 0.5
        if not m.any():
            m = masks[members[int(np.argmax(confs))]].astype(bool)
        fused.append((float(confs.max()), m))
    fused.sort(key=lambda t: -t[0])

    taken = np.zeros((IMG_H, IMG_W), bool)
    kept = []
    for c, m in fused:
        mm = m & ~taken
        if mm.any():
            taken |= mm
            kept.append((c, mm))
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--weights", default="outputs/runs_seed/seed_b3_512/best.pt")
    a = ap.parse_args()

    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    stems = [f.rsplit(".", 1)[0]
             for f in sorted(folds.loc[folds["fold"] == a.fold, "file_name"])]
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt_f = gt[gt["file_name"].isin({s + ".jpeg" for s in stems})]
    N = len(gt_f)
    ncop = {f.rsplit(".", 1)[0]: int(v)
            for f, v in gt_f.groupby("file_name")["entry_id"].nunique().items()}

    det = pd.read_parquet(ROOT / "outputs" / "oof_cache" / f"fold{a.fold}_preds_e013.parquet")
    det = det[(det["conf"] >= 0.05) & (det["r_norm"] <= R_MAX)]
    if "view" in det.columns:
        det = det[det["view"] == "o"]
    z = np.load(ROOT / "outputs" / "semantic" / f"fold{a.fold}_probs_cldice.npz")
    model, S, pad, ctx = load(ROOT / a.weights)

    two, three = {}, {}
    t0 = time.time()
    for n, stem in enumerate(stems):
        g = det[det["stem"] == stem]
        items = [(float(c), decode_rle(r).astype(bool))
                 for c, r in zip(g["conf"], g["rle"])]
        for c in prob_to_instances(z[stem].astype(np.float32) / 255.0,
                                   SEM_THR, SEM_MIN_AREA):
            items.append((float(c["conf"]), c["mask"].astype(bool)))
        two[stem] = fuse(items)

        img = cv2.imread(str(DATA / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        extra = []
        if img is not None and len(g):
            dm = [decode_rle(r).astype(bool) for r in g["rle"]]
            boxes = [bbox_of(m) for m in dm]
            ok = [i for i, b in enumerate(boxes) if b is not None]
            confs = g["conf"].to_numpy()
            for i, r in zip(ok, refine_image(model, S, pad, img,
                                             [boxes[i] for i in ok],
                                             context=ctx)):
                if r is not None:
                    extra.append((float(confs[i]), r))
        three[stem] = fuse(items + extra)
        if (n + 1) % 40 == 0:
            log(f"  {n+1}/{len(stems)} images ({time.time()-t0:.0f}s)")

    gt_cache, res = {}, {}
    for name, pool in (("two lanes", two), ("three lanes", three)):
        rows, ids = [], []
        for stem, kept in pool.items():
            for k, (c, m) in enumerate(kept):
                rows.append((f"{stem}_{k}", encode_mask(m.astype(np.uint8))))
                ids.append((stem, k, c))
        df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
        r = score(gt_f[["filament_id", "segmentation_rle"]], df, gt_cache=gt_cache,
                  return_details=True)
        hit = r["matches"]["gt_filament_id"].nunique()
        iou_sum = r["matches"].groupby("pred_filament_id")["iou"].sum().to_dict()
        meta = pd.DataFrame(ids, columns=["stem", "k", "conf"])
        meta["cand_id"] = df["filament_id"]
        meta["rle"] = df["segmentation_rle"]
        meta["value"] = (meta["cand_id"].map(iou_sum).fillna(0.0) /
                         meta["stem"].map(ncop).fillna(1).clip(lower=1))
        # oracle: keep by value > PQ/2, solved as a fixed point
        q = 0.53
        for _ in range(6):
            keep = meta[meta["value"] > q / 2.0]
            rr = score(gt_f[["filament_id", "segmentation_rle"]],
                       keep[["cand_id", "rle"]].rename(
                           columns={"cand_id": "filament_id", "rle": "segmentation_rle"}),
                       gt_cache=gt_cache, return_details=True)
            if abs(rr["pq"] - q) < 1e-3:
                break
            q = rr["pq"]
        log(f"{name:12s} pool {len(meta):5d}  reach {hit:4d}/{N}={hit/N:.4f}  "
            f"oracle PQ {rr['pq']:.4f} SQ {rr['sq']:.4f} TP {rr['tp']:4d} "
            f"FP {rr['fp']:4d} FN {rr['fn']:4d} (kept {len(keep)})")
        res[name] = dict(pool=len(meta), reach=hit / N, pq=rr["pq"], sq=rr["sq"],
                         tp=rr["tp"], fp=rr["fp"], fn=rr["fn"], kept=len(keep))

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"three_lane_pool_fold{a.fold}.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    log(f"saved -> {OUT / f'three_lane_pool_fold{a.fold}.json'}")


if __name__ == "__main__":
    main()
