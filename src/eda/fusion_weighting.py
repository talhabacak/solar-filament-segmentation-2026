"""Whose boundary should a two-lane cluster keep?

78% of matched candidates are clusters where both lanes fired, and those carry
almost all the value (mean 0.6023 against 0.4859 for detector-only and 0.4584
for semantic-only). Their mask is a confidence-weighted vote, and detector
confidence averages 0.8833 against the semantic lane's much lower scores — so
the vote is pulled toward the detector's mask.

That matters because the two lanes produce boundaries at different resolutions.
YOLO's prototype masks live at input/4, so at imgsz 1792 a mask is decided on a
448x448 grid and then upsampled to 2048 — a 4.5x stretch. The semantic lane
predicts at 1536 and is upsampled 1.33x. If the detector is the better DETECTOR
and the semantic map the better SEGMENTER, the fusion should localise with one
and cut with the other.

SQ is where the value is (0.6666 now, 0.7903 for the consensus mask, and perfect
masks at the current matching would be PQ 0.7618), so a better cut is worth more
than anything selection can still give.

Policies compared on the same clusters, scored exactly:
  conf-vote    current: weighted vote >= 0.5
  equal-vote   unweighted majority of members
  sem-member   the semantic member's own mask when the cluster has one
  det-member   the detector member's own mask (control)
  sem-recut    the detector's localisation, re-cut by thresholding the semantic
               probability inside a dilated neighbourhood of the fused mask
  union/inter  the extremes, as bounds

Usage: python src/eda/fusion_weighting.py --fold 0
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

OUT = ROOT / "outputs" / "eda"
SEM_THR, SEM_MIN_AREA = 0.3, 256
GROW = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
POLICIES = ["conf-vote", "equal-vote", "sem-member", "det-member", "sem-recut",
            "union", "inter"]


def log(*a):
    print(*a, flush=True)


def sem_recut(fused, prob, thr):
    """Keep the detector's localisation, take the semantic map's boundary.

    The candidate is grown a little, the semantic probability is thresholded
    inside that window, and the component overlapping the original mask is kept.
    Falls back to the original when the semantic map says nothing there.
    """
    box = cv2.dilate(fused.astype(np.uint8), GROW) > 0
    cut = (prob >= thr) & box
    if not cut.any():
        return fused
    n, lab = cv2.connectedComponents(cut.astype(np.uint8), connectivity=8)
    best, best_ov = None, 0
    for k in range(1, n):
        c = lab == k
        ov = int((c & fused).sum())
        if ov > best_ov:
            best, best_ov = c, ov
    return best if best is not None and best.sum() >= 64 else fused


def build(stems, det, z):
    per_image = {}
    t0 = time.time()
    for n, stem in enumerate(stems):
        items = []
        for r in det[det["stem"] == stem].sort_values("conf", ascending=False).itertuples():
            items.append(("det", float(r.conf), decode_rle(r.rle).astype(bool)))
        prob = z[stem].astype(np.float32) / 255.0
        for c in prob_to_instances(prob, SEM_THR, SEM_MIN_AREA):
            items.append(("sem", float(c["conf"]), c["mask"].astype(bool)))
        if not items:
            per_image[stem] = ([], prob)
            continue
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
        cl = []
        for members in groups.values():
            cl.append(dict(conf=max(items[i][1] for i in members),
                           members=[(items[i][0], items[i][1], items[i][2])
                                    for i in members],
                           stack=masks[members]))
        per_image[stem] = (cl, prob)
        if (n + 1) % 40 == 0:
            log(f"  built {n+1}/{len(stems)} ({time.time()-t0:.0f}s)")
    return per_image


def mask_for(policy, c, prob):
    stack = c["stack"].astype(np.float32)
    confs = np.array([cf for _, cf, _ in c["members"]], np.float32)
    sems = [m for s, _, m in c["members"] if s == "sem"]
    dets = [m for s, _, m in c["members"] if s == "det"]
    if policy == "conf-vote":
        m = np.tensordot(confs / confs.sum(), stack, axes=1) >= 0.5
    elif policy == "equal-vote":
        m = stack.mean(axis=0) >= 0.5
    elif policy == "union":
        m = stack.max(axis=0) > 0
    elif policy == "inter":
        m = stack.min(axis=0) > 0
    elif policy == "sem-member":
        m = sems[0] if sems else (np.tensordot(confs / confs.sum(), stack, axes=1) >= 0.5)
    elif policy == "det-member":
        m = dets[0] if dets else (np.tensordot(confs / confs.sum(), stack, axes=1) >= 0.5)
    else:                                       # sem-recut
        base = np.tensordot(confs / confs.sum(), stack, axes=1) >= 0.5
        if not base.any():
            base = stack[int(np.argmax(confs))] > 0
        return sem_recut(base, prob, SEM_THR)
    if not m.any():
        m = stack[int(np.argmax(confs))] > 0
    return m


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

    per_image = build(stems, det, z)
    gt_cache, res = {}, {}
    for policy in POLICIES:
        rows = []
        for stem, (cl, prob) in per_image.items():
            for ci, c in enumerate(cl):
                m = mask_for(policy, c, prob)
                rows.append((f"{stem}_{ci}", encode_mask(m.astype(np.uint8))))
        df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
        r = score(gt_f[["filament_id", "segmentation_rle"]], df, gt_cache=gt_cache,
                  return_details=True)
        hit = r["matches"]["gt_filament_id"].nunique()
        log(f"{policy:11s} preds {len(df):5d}  reach {hit:4d}/{N}={hit/N:.4f}  "
            f"SQ {r['sq']:.4f}  PQ {r['pq']:.4f}  TP {r['tp']:4d} FP {r['fp']:5d} "
            f"FN {r['fn']:4d}")
        res[policy] = dict(n=len(df), reach=hit / N, sq=r["sq"], pq=r["pq"],
                           tp=r["tp"], fp=r["fp"], fn=r["fn"])

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"fusion_weighting_fold{a.fold}.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    log(f"saved -> {OUT / f'fusion_weighting_fold{a.fold}.json'}")


if __name__ == "__main__":
    main()
