"""Build the fused (detector + semantic) candidate pool for every fold, with
cross-lane agreement features — the input to selector v5.

Why: on fold 0 the fused pool's oracle is PQ 0.5262 against a realised 0.4383, so
+0.0879 sits in SELECTION alone — larger than every modelling gain we have made.
The previous selector never saw the strongest available evidence: what the OTHER
lane thinks about a candidate's pixels. A detector box that the semantic model
also lights up is very different from one it ignores, and vice versa.

Features added over the earlier fused pool:
  sem_prob_mean/max/p90/frac_over_50 inside the mask   (cross-lane agreement)
  sem_prob_ring_mean, sem_prob_contrast                (is the signal localised?)
  n_det_models, det_conf_mean/std/max                  (detector-fold consensus)
  member_iou_mean/min                                  (cluster tightness)
  nn_dist, n_within_200px                              (neighbourhood context)
  img_cand_density, img_sem_area_frac                  (per-image activity)

Usage: python src/models/build_fused_pool.py --folds 0,1,2,3,4
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament import IMG_H, IMG_W
from filament.pq import score, _entry_iou_matrix
from filament.rle import decode_rle, encode_mask
from selector import mask_features
from semantic_candidates import load_model, predict_prob, prob_to_instances
from maskops import bbox, iou_matrix_boxed, nonzero_boxed, union_box

CACHE = ROOT / "outputs" / "oof_cache"
OUT = ROOT / "outputs" / "selector"
IMG_DIR = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
R_MAX = 0.97
SEM_THR, SEM_MIN_AREA = 0.3, 256
_VOTE = 0.5          # fusion vote threshold; see _fuse_body
# E183: candidates are grouped into one object when they overlap at more than this. 0.5 was
# inherited from the first pool build and never swept. It is the knob that decides whether two
# nearby filaments become one candidate (the 'merged_into' mechanism, ledger 29a) or stay two.
_CLUSTER_IOU = 0.5
_LANES = {"det", "sem"}   # which lanes contribute candidates (see --lanes)
# Multi-threshold semantic extraction (docs: multithresh_pool.py). At 0.3 alone,
# neighbouring components merge and 100-256px candidates are dropped; the union
# of thresholds raises fold-0 reachability 0.6656 -> 0.6974 (+48 GT rows, all
# low-consensus, median 766 px). Nested variants of the same object are handled
# by the existing IoU>0.5 union-find in _fuse_body: near-identical components
# cluster and fuse, while a clean 0.5-component inside a merged 0.3-blob stays a
# separate candidate -- which is the point.
SEM_SPECS_MULTI = ((0.3, 100), (0.5, 100), (0.7, 100))

BASE = ["conf", "log10_area", "r_norm", "abs_lat", "bbox_aspect", "eccentricity",
        "boundary_complexity", "skel_len", "width_proxy", "n_cands_img", "rank_conf"]
SRC = ["det_conf", "sem_conf", "from_det", "from_sem", "n_sources"]
CROSS = ["sem_prob_mean", "sem_prob_max", "sem_prob_p90", "sem_frac_over_50",
         "sem_ring_mean", "sem_contrast",
         "n_det_models", "det_conf_mean", "det_conf_std", "det_conf_max",
         "member_iou_mean", "member_iou_min",
         "nn_dist", "n_within_200px", "img_cand_density", "img_sem_area_frac"]
FEATS_V5 = BASE + SRC + CROSS


def log(*a):
    print(*a, flush=True)


_SEM_TAG = "_cldice"      # which cached semantic field the pool is built from
_CARVE_DROPS = 0          # cluster masks rejected by the fusion-stage carve rule


def semantic_probs_for_fold(fold, stems):
    """Probability map per stem from that fold's held-out U-Net."""
    cache = ROOT / "outputs" / "semantic" / f"fold{fold}_probs{_SEM_TAG}.npz"
    if cache.exists():
        z = np.load(cache)
        if all(s in z.files for s in stems):
            log(f"  reusing cached semantic probs for fold {fold}")
            return {s: z[s].astype(np.float32) / 255.0 for s in stems}
    if _SEM_TAG != "_cldice":
        # A blended or committee field cannot be regenerated from the single
        # production model. Regenerating silently produced a clDice-only field
        # under the _blob5ens tag on the GPU box (2026-09-06) and a whole
        # five-fold twin had to be rerun. Fail loudly instead.
        raise FileNotFoundError(f"semantic cache {cache.name} missing or incomplete for tag {_SEM_TAG}; "
                                "push the composed field instead of regenerating it")
    w = ROOT / "outputs" / "runs_semantic" / f"sem_b3_f{fold}_1536_cldice" / "best.pt"
    model, imgsz, channels = load_model(str(w))
    out = {}
    t0 = time.time()
    for i, s in enumerate(stems):
        out[s] = predict_prob(model, imgsz, s, tta=True, channels=channels)
    del model
    torch.cuda.empty_cache()
    np.savez_compressed(cache, **{s: (v * 255).astype(np.uint8) for s, v in out.items()})
    log(f"  semantic probs for fold {fold}: {len(stems)} images ({time.time()-t0:.0f}s)")
    return out


RING_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))


def fuse_and_featurise(stem, det_group, prob, fold=-1, deoverlap=True,
                       sem_specs=((SEM_THR, SEM_MIN_AREA),),
                       carve_tau=0.0):
    """Cluster one image's detector + semantic candidates and emit v5 features.

    Shared by the OOF pool builder and the test-time submission so the features a
    selector is trained on are exactly the features it is applied to.

    deoverlap=False leaves the cluster masks whole; the caller is then
    responsible for making the SELECTED masks disjoint before writing them.
    Measured worth on fold 0 under oracle selection: 0.5321 -> 0.5476.
    """
    rows = []
    items = []                                       # (source, conf, mask, model_id)
    for j, r in enumerate(det_group.sort_values("conf", ascending=False).itertuples()):
        # lane identity (docs/32 section 3.3): an extra instance lane (Mask2Former) enters
        # the pool as a detector row; without this column the selector cannot learn
        # that its confidence means something different. Clustering and every existing
        # feature are untouched -- the lane only feeds the four *_m2f/e013 features.
        items.append(("det", float(r.conf), decode_rle(r.rle).astype(bool), j, getattr(r, "lane", "det")))
    for thr, min_area in sem_specs:
        for c in prob_to_instances(prob, thr, min_area):
            items.append(("sem", float(c["conf"]), c["mask"].astype(bool), -1, "sem"))
    if items:
        _fuse_body(stem, items, prob, RING_K, rows, fold, deoverlap, carve_tau=carve_tau)
    return rows


def build_fold(fold, deoverlap=True, sem_specs=None,
               det_tag="_e013", carve_tau=0.0):
    if sem_specs is None:
        sem_specs = ((SEM_THR, SEM_MIN_AREA),)
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    stems = [f.rsplit(".", 1)[0] for f in sorted(folds.loc[folds["fold"] == fold, "file_name"])]
    probs = semantic_probs_for_fold(fold, stems)

    det = pd.read_parquet(CACHE / f"fold{fold}_preds{det_tag}.parquet")
    det = det[(det["conf"] >= 0.05) & (det["r_norm"] <= R_MAX)]
    if "det" not in _LANES:
        det = det.iloc[0:0]
    if "view" in det.columns:
        det = det[det["view"] == "o"]

    rows = []
    t0 = time.time()
    for n, stem in enumerate(stems):
        rows.extend(fuse_and_featurise(stem, det[det["stem"] == stem], probs[stem],
                                       fold, deoverlap, sem_specs, carve_tau=carve_tau))
        if (n + 1) % 40 == 0:
            log(f"  fold {fold}: {n+1}/{len(stems)} images ({time.time()-t0:.0f}s)")
    df = pd.DataFrame(rows)
    df["log10_area"] = np.log10(df["area"])
    if carve_tau > 0.0:
        global _CARVE_DROPS
        log(f"  fold {fold}: carve rule (tau {carve_tau}) rejected {_CARVE_DROPS} cluster masks at fusion")
        _CARVE_DROPS = 0
    return df


def _fuse_body(stem, items, prob, ring_k, rows, fold, deoverlap=True, carve_tau=0.0):
    """Union-find clustering at IoU>0.5, confidence-weighted fusion, greedy
    de-overlap, then the v5 feature row per surviving instance."""
    # Every full-frame operation below runs on bounding boxes instead (src/models/maskops.py);
    # the values are the same to the bit, which outputs/eda/_verify_lossless_fusion.py checks
    # against the pool the selector was fitted on and the submitted CSV.
    masks = [it[2] for it in items]
    boxes = [bbox(m) for m in masks]
    iou = iou_matrix_boxed(masks, boxes)
    parent = list(range(len(items)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ii, jj = np.nonzero(iou > _CLUSTER_IOU)
    for i, j in zip(ii, jj):
        if i < j:
            ri, rj = find(int(i)), find(int(j))
            if ri != rj:
                parent[rj] = ri
    clusters = defaultdict(list)
    for i in range(len(items)):
        clusters[find(i)].append(i)

    fused = []
    for members in clusters.values():
        srcs = {items[i][0] for i in members}
        confs = np.array([items[i][1] for i in members], dtype=np.float32)
        w = confs / confs.sum()
        # the vote on the union box of the members: outside it every member is zero, so the
        # weighted sum is zero there and below the threshold. Same tensordot, same member
        # order, same float32 weights -- only the pixel count differs.
        ub = union_box([boxes[i] for i in members])
        if ub is None:
            ub = (0, 1, 0, 1)
        uy0, uy1, ux0, ux1 = ub
        acc = np.tensordot(w, np.stack([masks[i][uy0:uy1, ux0:ux1] for i in members]).astype(np.float32), axes=1)
        # Confidence-weighted majority vote. The 0.5 default makes a pixel
        # survive only if the members carrying more than half the confidence
        # agree on it, which pulls a fused mask toward the INTERSECTION of its
        # members -- and where two members disagree about a filament's extent
        # that systematically loses area. Measured on fold 0: of the 147
        # ground-truth rows the raw detector reaches but the built pool does
        # not, 72 are already gone before de-overlap, and their best IoU in the
        # de-overlap-free pool sits at median 0.437 and p90 0.490, with none of
        # them lacking an overlapping candidate. Clustering finds them; this
        # vote pushes the mask under the threshold. Exposed so the trade against
        # matched-instance IoU can be measured rather than assumed.
        mw = acc >= _VOTE
        if mw.any():
            m = np.zeros((IMG_H, IMG_W), dtype=bool)
            m[uy0:uy1, ux0:ux1] = mw
            mbox = bbox(m)
        else:
            m = masks[members[int(np.argmax(confs))]]
            mbox = boxes[members[int(np.argmax(confs))]]
        det_confs = [items[i][1] for i in members if items[i][0] == "det"]
        lanes = [items[i][4] if len(items[i]) > 4 else items[i][0] for i in members]
        m2f_confs = [items[i][1] for i, l in zip(members, lanes) if l == "m2f"]
        e013_confs = [items[i][1] for i, l in zip(members, lanes) if l == "det"]
        sub = iou[np.ix_(members, members)]
        off = sub[~np.eye(len(members), dtype=bool)] if len(members) > 1 else np.array([1.0])
        fused.append(dict(
            mask=m, box=mbox,
            det_conf=max(det_confs, default=0.0),
            sem_conf=max([items[i][1] for i in members if items[i][0] == "sem"], default=0.0),
            from_det=int("det" in srcs), from_sem=int("sem" in srcs),
            n_sources=len(srcs),
            n_det_models=len(det_confs),
            det_conf_mean=float(np.mean(det_confs)) if det_confs else 0.0,
            det_conf_std=float(np.std(det_confs)) if len(det_confs) > 1 else 0.0,
            det_conf_max=float(np.max(det_confs)) if det_confs else 0.0,
            from_m2f=int(len(m2f_confs) > 0), m2f_conf_max=max(m2f_confs, default=0.0),
            n_e013_models=len(e013_confs), e013_conf_max=max(e013_confs, default=0.0),
            member_iou_mean=float(off.mean()), member_iou_min=float(off.min()),
            conf=max(max(det_confs, default=0.0),
                     max([items[i][1] for i in members if items[i][0] == "sem"], default=0.0))))
    fused.sort(key=lambda d: -d["conf"])

    if deoverlap:
        # Measured cost of doing this HERE rather than after selection (fold 0,
        # oracle selection on both sides): PQ 0.5321 with de-overlap first
        # against 0.5476 with it last, because a carved-up runner-up usually
        # matches nothing at the strict IoU>0.5 threshold. Clustering already
        # guarantees one mask per filament, so nothing needs disjointness until
        # the submission is written.
        taken = np.zeros((IMG_H, IMG_W), dtype=bool)
        kept = []
        for d in fused:
            if d["box"] is None:
                continue
            y0, y1, x0, x1 = d["box"]
            win = d["mask"][y0:y1, x0:x1]
            orig_area = int(win.sum())
            mw = win & ~taken[y0:y1, x0:x1]
            if not mw.any():
                continue
            if carve_tau > 0.0 and (int(mw.sum()) / orig_area) < carve_tau:
                global _CARVE_DROPS
                _CARVE_DROPS += 1
                continue
            taken[y0:y1, x0:x1] |= mw
            mm = np.zeros((IMG_H, IMG_W), dtype=bool)
            mm[y0:y1, x0:x1] = mw
            d["mask"] = mm
            d["box"] = bbox(mm)
            kept.append(d)
    else:
        kept = fused

    sem_area_frac = float((prob >= 0.5).mean())
    cents = []
    for d in kept:
        d["yx"] = nonzero_boxed(d["mask"], d["box"])
        ys, xs = d["yx"]
        cents.append((xs.mean(), ys.mean()))
    cents = np.array(cents) if cents else np.zeros((0, 2))

    for k, d in enumerate(kept, start=1):
        m = d["mask"]
        f = mask_features(m)
        if f["r_norm"] > R_MAX:
            continue
        ys, xs = d["yx"]
        y0, y1 = max(0, ys.min() - 12), min(IMG_H, ys.max() + 13)
        x0, x1 = max(0, xs.min() - 12), min(IMG_W, xs.max() + 13)
        sub_m = m[y0:y1, x0:x1]
        sub_p = prob[y0:y1, x0:x1]
        ring = (cv2.dilate(sub_m.astype(np.uint8), ring_k) > 0) & ~sub_m
        pv = sub_p[sub_m]
        rv = sub_p[ring] if ring.sum() > 10 else pv
        d2 = np.hypot(cents[:, 0] - xs.mean(), cents[:, 1] - ys.mean()) if len(cents) else np.array([])
        d2 = d2[d2 > 1e-6]
        rows.append(dict(
            stem=stem, fold=fold, cand_id=f"{stem}_{k}",
            rle=encode_mask(m.astype(np.uint8)),
            conf=d["conf"], det_conf=d["det_conf"], sem_conf=d["sem_conf"],
            from_det=d["from_det"], from_sem=d["from_sem"], n_sources=d["n_sources"],
            n_det_models=d["n_det_models"], det_conf_mean=d["det_conf_mean"],
            det_conf_std=d["det_conf_std"], det_conf_max=d["det_conf_max"],
            from_m2f=d["from_m2f"], m2f_conf_max=d["m2f_conf_max"],
            n_e013_models=d["n_e013_models"], e013_conf_max=d["e013_conf_max"],
            member_iou_mean=d["member_iou_mean"], member_iou_min=d["member_iou_min"],
            sem_prob_mean=float(pv.mean()), sem_prob_max=float(pv.max()),
            sem_prob_p90=float(np.percentile(pv, 90)),
            sem_frac_over_50=float((pv >= 0.5).mean()),
            sem_ring_mean=float(rv.mean()),
            sem_contrast=float(pv.mean() - rv.mean()),
            nn_dist=float(d2.min()) if len(d2) else 2048.0,
            n_within_200px=int((d2 < 200).sum()),
            n_cands_img=len(kept), rank_conf=float(k),
            img_cand_density=len(kept) / 100.0, img_sem_area_frac=sem_area_frac,
            **f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--no-deoverlap", action="store_true",
                    help="leave cluster masks whole; the caller must make the "
                         "SELECTED masks disjoint (measured +0.0155 at oracle)")
    ap.add_argument("--out", default="fused_pool_all.parquet")
    ap.add_argument("--multithresh", action="store_true",
                    help="semantic extraction as the union of 0.3/0.5/0.7 at "
                         "min_area 100 (fold-0 reachability +0.0318; measured "
                         "NEGATIVE at realized PQ -- kept for experiments only)")
    ap.add_argument("--sem-thr", type=float, default=0.3,
                    help="semantic extraction threshold. 0.3 is production and was "
                         "swept optimal for the clDice field; a better-calibrated "
                         "field may want a higher one, since the cost of a lane is "
                         "the candidates it adds as well as the ones it recovers.")
    ap.add_argument("--sem-tag", default="_cldice",
                    help="suffix of the cached semantic probability field to fuse "
                         "from, e.g. _bloblocal. The default is the production "
                         "1536 clDice model.")
    ap.add_argument("--cluster-iou", type=float, default=0.5,
                    help="E183: overlap above which two candidates become one object. Never "
                         "swept before; higher keeps neighbouring filaments separate, lower "
                         "merges more")
    ap.add_argument("--vote", type=float, default=0.5,
                    help="confidence-weighted fusion vote threshold; 0.5 is "
                         "production. Lower grows the fused mask toward the "
                         "union of its members, higher toward the intersection")
    ap.add_argument("--lanes", default="det,sem",
                    help="which lanes may contribute candidates: det, sem or both. "
                         "Used to measure what each lane is worth end to end "
                         "(pool -> selector -> PQ), not just in reachability.")
    ap.add_argument("--det-tag", default="_e013",
                    help="detector OOF cache tag, e.g. _e016 for the 2048 models")
    ap.add_argument("--carve-tau", type=float, default=0.0,
                    help="fusion-stage retention rule: drop a cluster mask that keeps "
                         "less than this fraction of its area after carving against "
                         "higher-confidence masks (sub026 used 0.70). 0 = off "
                         "(production sub025). A rejected mask frees its pixels for "
                         "the masks behind it, exactly as submit_v2 does.")
    a = ap.parse_args()
    global _VOTE, _SEM_TAG, SEM_THR, _LANES, _CLUSTER_IOU
    _LANES = set(x.strip() for x in a.lanes.split(",") if x.strip())
    _VOTE = a.vote
    _CLUSTER_IOU = a.cluster_iou
    _SEM_TAG = a.sem_tag
    SEM_THR = a.sem_thr
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")

    parts = []
    for fold in [int(x) for x in a.folds.split(",")]:
        specs = SEM_SPECS_MULTI if a.multithresh else ((SEM_THR, SEM_MIN_AREA),)
        if "sem" not in _LANES:
            specs = ()
        df = build_fold(fold, not a.no_deoverlap, specs, a.det_tag, carve_tau=a.carve_tau)
        log(f"fold {fold}: {len(df)} fused candidates")
        parts.append(df)
    cand = pd.concat(parts, ignore_index=True)

    files = set(cand["stem"] + ".jpeg")
    gt_sub = gt[gt["file_name"].isin(files)]
    pred_df = cand[["cand_id", "rle"]].rename(
        columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
    res = score(gt_sub[["filament_id", "segmentation_rle"]], pred_df, return_details=True)
    log(f"ALL fused (pooled): PQ={res['pq']:.4f} TP={res['tp']} FP={res['fp']} FN={res['fn']}")
    m = res["matches"]
    agg = m.groupby("pred_filament_id").agg(weight=("gt_filament_id", "size"),
                                            n_entries_hit=("annotator_image", "nunique"),
                                            iou_sum=("iou", "sum"))
    cand = cand.merge(agg, left_on="cand_id", right_index=True, how="left")
    for c in ["weight", "n_entries_hit", "iou_sum"]:
        cand[c] = cand[c].fillna(0)
    cand["y"] = (cand["weight"] >= 1).astype(int)
    ncop = gt_sub.groupby("file_name")["entry_id"].nunique()
    cand["n_copies"] = cand["stem"].map({f.rsplit(".", 1)[0]: int(v) for f, v in ncop.items()})
    log(f"labels: {int(cand['y'].sum())}/{len(cand)} positive")
    cand.to_parquet(OUT / a.out, index=False)
    log(f"saved -> {OUT / a.out}")


if __name__ == "__main__":
    main()
