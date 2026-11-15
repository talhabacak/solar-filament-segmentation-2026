"""E093/E094 -- immutable candidate bank and the legal best-variant oracle.

Why. The raw detector reaches 1243 of fold-0's 1510 ground-truth rows; the built
pool reaches 1096. The 147 in between are lost AFTER the model: 72 in
clustering/fusion (the confidence-weighted 0.5 vote pulls a fused mask toward
the intersection of its members, and the rows it loses sit at best-IoU median
0.437, p90 0.490 -- just under the threshold), 75 more in greedy de-overlap.
Every later selector then chooses among masks that have already been mutated,
so it cannot recover them. The deep-search package's central proposal, and the
one thing on its list that costs no GPU: keep every cluster's members
immutable, build several mask VARIANTS per cluster, and ask -- with ground truth
-- how much of the 147 comes back if one variant per cluster may be chosen.

This is a precondition measurement, not a model. It answers, on fold 0, with
the exact competition scorer:

  production-oracle  : one vote-0.5 mask per cluster, greedy de-overlap as
                       shipped, keep it iff some GT matches at IoU > 0.5.
                       (the 0.5321 number in the ledger, recomputed here)
  cluster-only oracle: per cluster, the variant with the best GT IoU, or
                       reject; overlaps between chosen masks IGNORED. Upper
                       bound on what any choose-one-or-reject rule can do.
  legal oracle       : same choice, then a non-overlap pass in descending
                       matched-IoU order -- a mask that collides is carved
                       against the taken pixels and kept only if it still
                       clears 0.5. This is a valid submission.

Gate (written before running, from docs/chatgpt_deep_search/05, E094):
  GO         legal PQ >= production-oracle + 0.003 and >= +25 reached rows
  STRONG GO  >= +0.006 and >= +50 rows
  STOP       < +0.003, or the legal pass destroys most of the cluster-only gain

Variants per cluster (all from immutable member masks; nothing ever steals
pixels from another cluster before the final legal pass):
  vote50   confidence-weighted vote at 0.5           -- production
  vote65   the same vote at 0.65 (tighter)
  top      the highest-confidence raw member
  semtop   the member with the highest mean semantic probability
  union    union of all members (loosest)
  semcut   union restricted to semantic prob >= 0.3
  er_vote  vote50 after the production one-pixel erosion
  er_top   top after the production one-pixel erosion

Usage: python src/eda/candidate_bank_oracle.py --fold 0
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
from filament.rle import decode_rle, encode_mask, rles_to_layers
from build_fused_pool import (semantic_probs_for_fold, prob_to_instances,
                              SEM_THR, SEM_MIN_AREA, R_MAX, CACHE)
from mask_shrink import shrink_frac

OUT = ROOT / "outputs" / "eda"
VARIANTS = ("vote50", "vote65", "top", "semtop", "union", "semcut", "er_vote", "er_top")


def log(*a):
    print(*a, flush=True)


def clusters_for(items):
    """Union-find at IoU > 0.5 -- identical to build_fused_pool._fuse_body."""
    masks = np.stack([m for _, _, m, _ in items]).astype(np.uint8)
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
    cl = defaultdict(list)
    for i in range(len(items)):
        cl[find(i)].append(i)
    return masks, list(cl.values())


def variants_for(members, items, masks, prob):
    confs = np.array([items[i][1] for i in members], dtype=np.float32)
    w = confs / confs.sum()
    acc = np.tensordot(w, masks[members].astype(np.float32), axes=1)
    top = masks[members[int(np.argmax(confs))]].astype(bool)
    v = {}
    m50 = acc >= 0.5
    v["vote50"] = m50 if m50.any() else top
    m65 = acc >= 0.65
    v["vote65"] = m65 if m65.any() else v["vote50"]
    v["top"] = top
    sem_means = [float(prob[masks[i].astype(bool)].mean()) for i in members]
    v["semtop"] = masks[members[int(np.argmax(sem_means))]].astype(bool)
    uni = masks[members].any(axis=0)
    v["union"] = uni
    sc = uni & (prob >= 0.3)
    v["semcut"] = sc if sc.sum() >= 64 else v["vote50"]
    v["er_vote"] = shrink_frac(v["vote50"], 0.05)
    v["er_top"] = shrink_frac(top, 0.05)
    conf = float(confs.max())
    return v, conf


def best_gt_iou(mask, gt_layers, gt_areas=None):
    """Best IoU of `mask` against any GT layer, computed on the mask's bbox crop.

    Exact: intersection lies inside the mask's bbox, and the GT areas are the
    full-frame ones (precomputed once per image), so the ratio is unchanged.
    ~50x cheaper than the full-frame matrix, which made fold 0 take 26 s/image.
    """
    if not mask.any() or len(gt_layers) == 0:
        return 0.0, -1
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    mc = mask[y0:y1, x0:x1]
    gc = gt_layers[:, y0:y1, x0:x1].astype(bool)
    inter = (gc & mc[None]).sum(axis=(1, 2)).astype(np.float64)
    if gt_areas is None:
        gt_areas = gt_layers.reshape(len(gt_layers), -1).sum(axis=1)
    union = gt_areas + int(mc.sum()) - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    k = int(np.argmax(iou))
    return float(iou[k]), k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--det-tag", default="_e013")
    ap.add_argument("--sem-tag", default="_cldice")
    a = ap.parse_args()

    import build_fused_pool as bfp
    bfp._SEM_TAG = a.sem_tag
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    files = sorted(folds.loc[folds["fold"] == a.fold, "file_name"])
    stems = [f.rsplit(".", 1)[0] for f in files]
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt_f = gt[gt["file_name"].isin(files)]
    probs = semantic_probs_for_fold(a.fold, stems)
    det = pd.read_parquet(CACHE / f"fold{a.fold}_preds{a.det_tag}.parquet")
    det = det[(det["conf"] >= 0.05) & (det["r_norm"] <= R_MAX)]
    if "view" in det.columns:
        det = det[det["view"] == "o"]
    log(f"fold {a.fold}: {len(stems)} images, {len(gt_f)} GT rows, "
        f"{len(det)} detector candidates")

    # per-image GT layers (all annotator entries together; the scorer separates them)
    preds = {k: [] for k in ("prod", "cluster", "legal")}
    bank_rows, var_rows, t0 = [], [], time.time()
    n_clusters = 0
    for n, stem in enumerate(stems):
        g = gt_f[gt_f["file_name"] == f"{stem}.jpeg"]
        gt_layers = rles_to_layers(g["segmentation_rle"].tolist()) if len(g) else np.zeros((0, IMG_H, IMG_W), np.uint8)
        gt_areas = gt_layers.reshape(len(gt_layers), -1).sum(axis=1).astype(np.float64) if len(gt_layers) else np.zeros(0)
        prob = probs[stem]
        dg = det[det["stem"] == stem]
        items = []
        for j, r in enumerate(dg.sort_values("conf", ascending=False).itertuples()):
            items.append(("det", float(r.conf), decode_rle(r.rle).astype(bool), j))
        for c in prob_to_instances(prob, SEM_THR, SEM_MIN_AREA):
            items.append(("sem", float(c["conf"]), c["mask"].astype(bool), -1))
        if not items:
            continue
        masks, cls = clusters_for(items)
        n_clusters += len(cls)

        # --- production path: vote50, greedy de-overlap by conf, oracle keep
        fused = []
        chosen = []
        for members in cls:
            v, conf = variants_for(members, items, masks, prob)
            fused.append((conf, v["vote50"]))
            # --- bank: best variant per cluster
            ious = {k: best_gt_iou(v[k], gt_layers, gt_areas)[0] for k in VARIANTS}
            best = max(((ious[k], k) for k in VARIANTS), key=lambda t: t[0])
            cid = f"{stem}_c{len(bank_rows)}"
            srcs = {items[i][0] for i in members}
            det_confs = [items[i][1] for i in members if items[i][0] == "det"]
            sub = None
            for k in VARIANTS:
                mk = v[k]
                if not mk.any():
                    continue
                ys_, xs_ = np.nonzero(mk)
                pv = prob[mk]
                var_rows.append(dict(
                    stem=stem, cluster_id=cid, variant=k, rle=encode_mask(mk.astype(np.uint8)),
                    iou=ious[k], is_best=int(k == best[1]), area=int(mk.sum()),
                    r_norm=float(np.hypot(xs_.mean() - 1023.5, ys_.mean() - 1023.5) / 900.0),
                    conf=conf, n_members=len(members), n_det=len(det_confs),
                    from_det=int("det" in srcs), from_sem=int("sem" in srcs),
                    det_conf_mean=float(np.mean(det_confs)) if det_confs else 0.0,
                    sem_prob_mean=float(pv.mean()), sem_prob_p90=float(np.percentile(pv, 90)),
                    sem_frac_over_50=float((pv >= 0.5).mean()),
                    area_ratio_to_vote50=float(mk.sum() / max(int(v["vote50"].sum()), 1))))
            base_iou = best_gt_iou(v["vote50"], gt_layers, gt_areas)[0]
            bank_rows.append(dict(stem=stem, n_members=len(members), conf=conf,
                                  best_variant=best[1], best_iou=best[0],
                                  vote50_iou=base_iou))
            if best[0] > 0.5:
                chosen.append((best[0], v[best[1]]))
        fused.sort(key=lambda t: -t[0])
        taken = np.zeros((IMG_H, IMG_W), bool)
        k = 0
        for conf, m in fused:
            mm = m & ~taken
            if not mm.any():
                continue
            taken |= mm
            if best_gt_iou(mm, gt_layers, gt_areas)[0] > 0.5:
                k += 1
                preds["prod"].append((f"{stem}_{k}", encode_mask(mm.astype(np.uint8))))

        # --- cluster-only oracle (overlaps ignored)
        for k, (iou_, m) in enumerate(chosen, 1):
            preds["cluster"].append((f"{stem}_{k}", encode_mask(m.astype(np.uint8))))
        # --- legal oracle: descending IoU, carve against taken, keep iff still > 0.5
        chosen.sort(key=lambda t: -t[0])
        taken = np.zeros((IMG_H, IMG_W), bool)
        k = 0
        for iou_, m in chosen:
            mm = m & ~taken
            if not mm.any():
                continue
            if (mm != m).any():
                if best_gt_iou(mm, gt_layers, gt_areas)[0] <= 0.5:
                    continue
            taken |= mm
            k += 1
            preds["legal"].append((f"{stem}_{k}", encode_mask(mm.astype(np.uint8))))
        if (n + 1) % 20 == 0:
            log(f"  {n+1}/{len(stems)} images ({time.time()-t0:.0f}s)")

    bank = pd.DataFrame(bank_rows)
    bank.to_parquet(OUT / f"candidate_bank_fold{a.fold}.parquet", index=False)
    vb = pd.DataFrame(var_rows)
    vb.to_parquet(OUT / f"variant_bank_fold{a.fold}.parquet", index=False)
    log(f"variant bank: {len(vb)} rows ({vb['cluster_id'].nunique()} clusters) "
        f"-> variant_bank_fold{a.fold}.parquet")
    log(f"\n{n_clusters} clusters; variant chosen by the oracle:")
    log(bank["best_variant"].value_counts().to_string())
    log(f"  clusters whose best variant beats vote50 by > 0.05 IoU: "
        f"{int((bank.best_iou - bank.vote50_iou > 0.05).sum())}")
    log(f"  clusters crossing 0.5 only with a non-production variant: "
        f"{int(((bank.best_iou > 0.5) & (bank.vote50_iou <= 0.5)).sum())}")

    gt_df = gt_f[["filament_id", "segmentation_rle"]]
    res = {}
    log(f"\nexact scorer, fold {a.fold} ({len(gt_f)} GT rows)")
    log(f"  {'arm':<20}{'preds':>7}{'PQ':>9}{'SQ':>9}{'RQ':>9}{'TP':>7}{'FP':>6}{'FN':>6}")
    for arm in ("prod", "cluster", "legal"):
        pdf = pd.DataFrame(preds[arm], columns=["filament_id", "segmentation_rle"])
        r = score(gt_df, pdf)
        res[arm] = dict(n=len(pdf), pq=r["pq"], sq=r["sq"], rq=r["rq"],
                        tp=r["tp"], fp=r["fp"], fn=r["fn"])
        log(f"  {arm:<20}{len(pdf):>7}{r['pq']:>9.4f}{r['sq']:>9.4f}{r['rq']:>9.4f}"
            f"{r['tp']:>7}{r['fp']:>6}{r['fn']:>6}")
    d_cl = res["cluster"]["pq"] - res["prod"]["pq"]
    d_lg = res["legal"]["pq"] - res["prod"]["pq"]
    log(f"\n  cluster-only oracle vs production-oracle: {d_cl:+.4f} PQ, "
        f"FN {res['prod']['fn']} -> {res['cluster']['fn']}")
    log(f"  LEGAL oracle vs production-oracle:        {d_lg:+.4f} PQ, "
        f"FN {res['prod']['fn']} -> {res['legal']['fn']}  "
        f"(legal pass keeps {100*d_lg/max(d_cl,1e-9):.0f}% of the cluster-only gain)")
    rows_back = res["prod"]["fn"] - res["legal"]["fn"]
    verdict = ("STRONG GO" if d_lg >= 0.006 and rows_back >= 50 else
               "GO" if d_lg >= 0.003 and rows_back >= 25 else "STOP")
    log(f"\n  gate (E094): legal +{d_lg:.4f} PQ, +{rows_back} rows -> {verdict}")
    json.dump(dict(fold=a.fold, res=res, delta_cluster=d_cl, delta_legal=d_lg,
                   rows_back=rows_back, verdict=verdict,
                   variant_counts=bank["best_variant"].value_counts().to_dict()),
              open(OUT / f"candidate_bank_oracle_fold{a.fold}.json", "w"), indent=1)
    log(f"\nwrote {OUT/f'candidate_bank_oracle_fold{a.fold}.json'}")


if __name__ == "__main__":
    main()
