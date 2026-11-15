"""Is the consensus mask the right representative of a cluster?

Two measurements set this up. Clustering at IoU>0.5 costs fold 0 -0.0483 of
reachability, and de-overlapping before selection another -0.0490. But simply
keeping the raw pool and de-overlapping later is worse, not better: oracle PQ
0.4382 against 0.5321, because the raw pool holds several candidates per
filament and greedy de-overlap carves the runners-up into leftovers that match
nothing (FP 1037 vs 94).

That is forced by the metric, not an artefact. Two pixel-disjoint masks cannot
both exceed IoU 0.5 against the same GT — their intersections would have to sum
past |GT| — which is exactly why disjointness is required. So the pipeline must
emit ONE mask per filament, and clustering is the right shape of operation.

What is not established is the choice of representative. Today each cluster is
replaced by a confidence-weighted vote thresholded at 0.5, which can land
between two members and match neither at the strict threshold. The alternatives:

  consensus   what we do now
  best        the member with the highest value (oracle upper bound on picking
              a representative — a learnable ranking, not a free lunch)
  largest     the member with the most pixels, a cheap non-oracle rule
  highest     the member with the highest producer confidence, also non-oracle

Selection is the derived value rule on both sides, then de-overlap, so the
comparison isolates the representative choice.

Usage: python src/eda/cluster_representative.py --fold 0
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


def opt_id(stem, ci, key):
    """Scorable id for one cluster option.

    pq.py takes the image stem as everything before the FIRST underscore
    (split("_", n=1)), so everything after it must stay underscore-free or the
    prediction is silently attributed to no image and matches nothing.
    """
    return f"{stem}_{ci}x{key}"


def clusters_for(stem, det, prob):
    """Cluster one image's candidates and return (members, consensus mask)."""
    items = []
    for r in det.sort_values("conf", ascending=False).itertuples():
        items.append((float(r.conf), decode_rle(r.rle).astype(bool)))
    for c in prob_to_instances(prob, SEM_THR, SEM_MIN_AREA):
        items.append((float(c["conf"]), c["mask"].astype(bool)))
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

    out = []
    for members in groups.values():
        confs = np.array([items[i][0] for i in members], np.float32)
        w = confs / confs.sum()
        acc = np.tensordot(w, masks[members].astype(np.float32), axes=1)
        cons = acc >= 0.5
        if not cons.any():
            cons = masks[members[int(np.argmax(confs))]].astype(bool)
        out.append(dict(conf=float(confs.max()),
                        members=[(float(items[i][0]), items[i][1]) for i in members],
                        consensus=cons))
    return out


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
    ncop = {f.rsplit(".", 1)[0]: int(v)
            for f, v in gt_f.groupby("file_name")["entry_id"].nunique().items()}

    det = pd.read_parquet(ROOT / "outputs" / "oof_cache" / f"fold{a.fold}_preds_e013.parquet")
    det = det[(det["conf"] >= 0.05) & (det["r_norm"] <= R_MAX)]
    if "view" in det.columns:
        det = det[det["view"] == "o"]
    z = np.load(ROOT / "outputs" / "semantic" / f"fold{a.fold}_probs_cldice.npz")

    log("clustering...")
    per_image, t0 = {}, time.time()
    for n, s in enumerate(stems):
        per_image[s] = clusters_for(s, det[det["stem"] == s],
                                    z[s].astype(np.float32) / 255.0)
        if (n + 1) % 40 == 0:
            log(f"  {n+1}/{len(stems)} ({time.time()-t0:.0f}s)")

    # value of EVERY option (each member and each consensus), scored once
    # The scorer takes the image stem as everything before the FIRST underscore
    # (pq.py: split("_", n=1)), so the option key must not contain one.
    rows = []
    for s, cl in per_image.items():
        for ci, c in enumerate(cl):
            rows.append((opt_id(s, ci, "cons"), encode_mask(c["consensus"].astype(np.uint8))))
            for mi, (_, m) in enumerate(c["members"]):
                rows.append((opt_id(s, ci, f"m{mi}"), encode_mask(m.astype(np.uint8))))
    allopt = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    log(f"scoring {len(allopt)} options...")
    gt_cache = {}
    r = score(gt_f[["filament_id", "segmentation_rle"]], allopt, gt_cache=gt_cache,
              return_details=True)
    val = r["matches"].groupby("pred_filament_id")["iou"].sum().to_dict()
    log(f"  options reach {r['matches']['gt_filament_id'].nunique()}/{N}")

    def value(s, ci, key):
        return val.get(opt_id(s, ci, key), 0.0) / max(ncop.get(s, 1), 1)

    def pick(policy, s, c, ci):
        if policy == "consensus":
            return c["consensus"], value(s, ci, "cons")
        if policy == "largest":
            mi = int(np.argmax([m.sum() for _, m in c["members"]]))
        elif policy == "highest":
            mi = int(np.argmax([cf for cf, _ in c["members"]]))
        else:                                   # best = oracle representative
            opts = [("cons", c["consensus"])] + [(f"m{i}", m)
                                                 for i, (_, m) in enumerate(c["members"])]
            k = max(range(len(opts)), key=lambda i: value(s, ci, opts[i][0]))
            return opts[k][1], value(s, ci, opts[k][0])
        return c["members"][mi][1], value(s, ci, f"m{mi}")

    res = {}
    for policy in ("consensus", "largest", "highest", "best"):
        q = 0.45
        for _ in range(10):
            out = []
            for s, cl in per_image.items():
                chosen = []
                for ci, c in enumerate(cl):
                    m, v = pick(policy, s, c, ci)
                    if v > q / 2.0:
                        chosen.append((v, m))
                chosen.sort(key=lambda t: -t[0])
                taken = np.zeros((IMG_H, IMG_W), bool)
                for v, m in chosen:
                    mm = m & ~taken
                    if mm.sum() < 64:
                        continue
                    taken |= mm
                    out.append((f"{s}_{len(out)}", encode_mask(mm.astype(np.uint8))))
            df = pd.DataFrame(out, columns=["filament_id", "segmentation_rle"])
            rr = score(gt_f[["filament_id", "segmentation_rle"]], df,
                       gt_cache=gt_cache, return_details=True)
            if abs(rr["pq"] - q) < 1e-4:
                break
            q = rr["pq"]
        hit = rr["matches"]["gt_filament_id"].nunique()
        log(f"{policy:10s} kept {len(df):4d}  reach {hit:4d}/{N}={hit/N:.4f}  "
            f"PQ {rr['pq']:.4f} SQ {rr['sq']:.4f} TP {rr['tp']:4d} FP {rr['fp']:4d} "
            f"FN {rr['fn']:4d}")
        res[policy] = dict(kept=len(df), reach=hit / N, pq=rr["pq"], sq=rr["sq"],
                           tp=rr["tp"], fp=rr["fp"], fn=rr["fn"])

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"cluster_representative_fold{a.fold}.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    log(f"saved -> {OUT / f'cluster_representative_fold{a.fold}.json'}")


if __name__ == "__main__":
    main()
