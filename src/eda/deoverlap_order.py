"""De-overlapping before selection sets the ceiling. What if we did it after?

fusion_loss measured where fold 0's reachability goes:

    raw candidates            0.8232
    after IoU>0.5 clustering  0.7748   (-0.0483)
    after greedy de-overlap   0.7258   (-0.0490)

Both steps run BEFORE the selector sees anything, so the pool the selector picks
from has already lost 10% of the ground truth it could have reached. But
pixel-disjointness is a requirement on the SUBMISSION, not on a candidate pool —
there is no reason to enforce it early, and every reason not to: a carved-up
low-confidence mask usually matches nothing at the strict IoU>0.5 threshold.

Compare, with oracle selection on both sides so the selector's quality is not
what is being measured:

  current   cluster -> de-overlap -> oracle-select -> score
  proposed  oracle-select from RAW -> de-overlap only the winners -> score
  keep-all  oracle-select from raw + consensus masks -> de-overlap winners

Selection uses the derived value rule (iou_sum/n > PQ/2), and de-overlap orders
by that same value, since with oracle labels that is the natural ranking.

Usage: python src/eda/deoverlap_order.py --fold 0
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


def build_variants(stems, det, z):
    """Per image: raw candidate masks, cluster-consensus masks, and the current
    pipeline's clustered+de-overlapped masks."""
    raw, cons, current = {}, {}, {}
    t0 = time.time()
    for n, stem in enumerate(stems):
        items = []
        for r in det[det["stem"] == stem].sort_values("conf", ascending=False).itertuples():
            items.append((float(r.conf), decode_rle(r.rle).astype(bool)))
        prob = z[stem].astype(np.float32) / 255.0
        for c in prob_to_instances(prob, SEM_THR, SEM_MIN_AREA):
            items.append((float(c["conf"]), c["mask"].astype(bool)))
        if not items:
            raw[stem] = cons[stem] = current[stem] = []
            continue
        raw[stem] = list(items)

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
            w = confs / confs.sum()
            acc = np.tensordot(w, masks[members].astype(np.float32), axes=1)
            m = acc >= 0.5
            if not m.any():
                m = masks[members[int(np.argmax(confs))]].astype(bool)
            fused.append((float(confs.max()), m))
        fused.sort(key=lambda t: -t[0])
        cons[stem] = fused

        taken = np.zeros((IMG_H, IMG_W), bool)
        cur = []
        for c, m in fused:
            mm = m & ~taken
            if mm.any():
                taken |= mm
                cur.append((c, mm))
        current[stem] = cur
        if (n + 1) % 40 == 0:
            log(f"  built {n+1}/{len(stems)} images ({time.time()-t0:.0f}s)")
    return raw, cons, current


def label_pool(pool, gt_f, gt_cache):
    """Score a {stem: [(conf, mask)]} pool and return per-candidate iou_sum/n."""
    rows, ids = [], []
    for stem, items in pool.items():
        for k, (c, m) in enumerate(items):
            cid = f"{stem}_{k}"
            rows.append((cid, encode_mask(m.astype(np.uint8))))
            ids.append((stem, k, c))
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    r = score(gt_f[["filament_id", "segmentation_rle"]], df, gt_cache=gt_cache,
              return_details=True)
    m = r["matches"]
    iou_sum = m.groupby("pred_filament_id")["iou"].sum().to_dict()
    ncop = gt_f.groupby("file_name")["entry_id"].nunique()
    ncop = {f.rsplit(".", 1)[0]: int(v) for f, v in ncop.items()}
    meta = pd.DataFrame(ids, columns=["stem", "k", "conf"])
    meta["cand_id"] = df["filament_id"]
    meta["rle"] = df["segmentation_rle"]
    meta["iou_sum"] = meta["cand_id"].map(iou_sum).fillna(0.0)
    meta["n_copies"] = meta["stem"].map(ncop).fillna(1)
    meta["value"] = meta["iou_sum"] / meta["n_copies"].clip(lower=1)
    hit = m["gt_filament_id"].nunique()
    return meta, r, hit


def oracle_select_deoverlap(meta, pool, q):
    """Keep by value rule, then make the winners disjoint in value order."""
    keep = meta[meta["value"] > q / 2.0].sort_values("value", ascending=False)
    out = []
    for stem, g in keep.groupby("stem", sort=False):
        taken = np.zeros((IMG_H, IMG_W), bool)
        for row in g.itertuples():
            m = pool[stem][row.k][1]
            mm = m & ~taken
            if mm.sum() < 64:
                continue
            taken |= mm
            out.append((f"{stem}_{len(out)}", encode_mask(mm.astype(np.uint8))))
    return pd.DataFrame(out, columns=["filament_id", "segmentation_rle"])


def fixed_point(meta, pool, gt_f, gt_cache, q0=0.40, iters=12):
    q = q0
    best = None
    for _ in range(iters):
        df = oracle_select_deoverlap(meta, pool, q)
        r = score(gt_f[["filament_id", "segmentation_rle"]], df, gt_cache=gt_cache,
                  return_details=True)
        if abs(r["pq"] - q) < 1e-4:
            return r, df
        q, best = r["pq"], (r, df)
    return best


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

    raw, cons, current = build_variants(stems, det, z)
    keepall = {s: raw[s] + cons[s] for s in raw}
    gt_cache = {}
    res = {}

    for name, pool in (("current (cluster+deoverlap first)", current),
                       ("proposed (raw, deoverlap after)", raw),
                       ("keep-all (raw + consensus)", keepall)):
        meta, r0, hit = label_pool(pool, gt_f, gt_cache)
        r, df = fixed_point(meta, pool, gt_f, gt_cache)
        log(f"{name:36s} pool {len(meta):5d} reach {hit:5d}/{N}={hit/N:.4f} | "
            f"oracle PQ {r['pq']:.4f} SQ {r['sq']:.4f} TP {r['tp']:4d} "
            f"FP {r['fp']:4d} FN {r['fn']:4d} (kept {len(df)})")
        res[name] = dict(pool=len(meta), reach=hit / N, pq=r["pq"], sq=r["sq"],
                         tp=r["tp"], fp=r["fp"], fn=r["fn"], kept=len(df))

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"deoverlap_order_fold{a.fold}.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    log(f"saved -> {OUT / f'deoverlap_order_fold{a.fold}.json'}")


if __name__ == "__main__":
    main()
