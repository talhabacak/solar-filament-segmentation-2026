"""Detector NMS through the selector, five folds (measured on 15 November 2026, after submission).

Why: every out-of-fold number of the submitted system is read on detector candidates made with
NMS IoU 0.5, while the test-time candidates are made with 0.05 (report, Section 2.4). The earlier
NMS measurements each covered part of this: the week-one sweep on one detector before fusion
(src/models/nms_sweep.py; 0.05 best, +0.0105), E218 (src/eda/e218_nms_tighter.py, ledger 54a;
0.3-0.7 in the fused pool by reach, no selector, fold 0) and src/eda/nms_mismatch_f0.py (0.05 and
0.10 through the selector, fold 0 only). This reads 0.05, 0.10, 0.3, 0.5 and 0.7 end to end on
all 707 training images: fused pool -> out-of-fold selector -> keep gate -> pooled PQ.

Steps (inputs live in the development repository; pass it as --root):
  infer --fold k   the fold's detector once with NMS effectively off (iou 0.95, max_det 3000) on
                   its held-out images, exactly as e218_nms_tighter.infer but on the GPU; fold 0
                   reuses E218's CPU run. Writes oof_cache/raw95_e013_f{k}_0of1.parquet.
  build            torchvision NMS per image at each threshold, max_det 300, empty masks dropped
                   (e218_nms_tighter.build) -> oof_cache/fold{k}_preds_e013o{tag}.parquet.
                   GUARD: the 0.5 arm against the production cache fold{k}_preds_e013 (count per
                   image and mask IoU in confidence order, E218's thresholds 0.95 / 0.98); on fold
                   0 the 0.05 / 0.10 / 0.5 arms must equal nms_mismatch_f0's and E218's files.
  pools (shell)    for each tag: build_fused_pool.py --folds 0,1,2,3,4 --sem-tag _blob5ens
                   --sem-thr 0.2 --det-tag _e013o{tag} --out fused_pool_all_sw_o{tag}_t020.parquet
                   then outputs/eda/_label_one_pool.py fused_pool_all_sw_o{tag}_t020
  eval             per fold f, the selector fitted on the other four folds of the shipped
                   labelled pool with the inner leave-one-fold-out quantile reference
                   (e115c_threshold_ci.oof_scores). GUARD: on the shipped rows it must reproduce
                   outputs/eda/t020pox_oof_scores.parquet. Each arm's candidates get the 42
                   features (extra features recomputed), keep conf >= 0.40 and p >= 0.60, pooled PQ
                   over 707 images, paired image bootstrap (4000) of every arm against the 0.5 arm.

Result (outputs/eda/nms_sweep_5fold.json; build guards in nms_sweep_5fold_build.json):
  guards   offline 0.5 reproduces every fold's production cache (count per image 0.971-0.993,
           mean mask IoU 0.988-0.998); on fold 0 the 0.05/0.10/0.5 caches equal the earlier files;
           the five selectors reproduce the shipped out-of-fold scores exactly (max |dp| = 0); the
           rebuilt 0.5 pool reads the shipped pool's PQ to -0.0001.
  arms     against 0.5 (PQ 0.4421, 10043 candidates, 4152 matching):
             0.05  -0.0016 [-0.0037, +0.0006]  2/5 folds up   9130 candidates, 4149 matching
             0.10  -0.0001 [-0.0021, +0.0020]  3/5            9330, 4150
             0.3   -0.0002 [-0.0017, +0.0013]  3/5           10011, 4150
             0.7   -0.0003 [-0.0017, +0.0012]  3/5            9886, 4148
  reading  the axis is flat through the selector; the test-time setting costs at most a sub-floor
           -0.0016, so the submission is unchanged. Fold 0's +0.0045 for 0.10 does not survive.
           Out of fold each image has one detector's candidates and at test time five: this
           prices NMS on the out-of-fold pool, not on the five-detector test pool.
The fold 1-4 raw runs were made on an RTX 5070 Ti, one fold at a time (four concurrent processes
exhausted its 16 GB); fold 0's is E218's CPU run.

Usage: python src/eda/nms_sweep_5fold.py infer --fold 1 [--root DIR]
       python src/eda/nms_sweep_5fold.py build [--root DIR]
       python src/eda/nms_sweep_5fold.py eval  [--root DIR] [--jobs 10]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[2]
ARMS = {"005": 0.05, "010": 0.10, "03": 0.3, "05": 0.5, "07": 0.7}
DISK_C, DISK_R = 1023.5, 900.0


def infer(root, fold, device):
    import cv2
    import torch  # noqa: F401
    for p in ("src", "src/eda"):
        sys.path.insert(0, str(root / p))
    from filament import IMG_H, IMG_W
    from filament.rle import encode_mask
    from ultralytics import YOLO
    from ultralytics.utils import ops
    from e217_yolo_soft_masks import soft_process_mask
    ops.process_mask = soft_process_mask                     # as E218: sigmoid kept, > 0.5 below
    out = root / "outputs" / "oof_cache" / f"raw95_e013_f{fold}_0of1.parquet"
    if out.exists():
        print(f"{out.name} exists, skip")
        return
    model = YOLO(str(root / "outputs" / "runs" / f"e013_yolo11m_f{fold}_1792" / "weights" / "best.pt"))
    folds = pd.read_parquet(root / "data" / "splits" / "folds_v1.parquet")
    img_dir = root / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
    files = sorted(folds.loc[folds["fold"] == fold, "file_name"])
    rows, t0 = [], time.time()
    for n, fn in enumerate(files):
        stem = fn.rsplit(".", 1)[0]
        src = cv2.cvtColor(cv2.imread(str(img_dir / fn), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
        res = model.predict(src, imgsz=1792, conf=0.05, iou=0.95, max_det=3000, verbose=False, device=device)[0]
        if res.masks is not None:
            soft = res.masks.data.cpu().numpy()
            for m, c, b in zip(soft, res.boxes.conf.cpu().numpy(), res.boxes.xyxy.cpu().numpy()):
                mm = cv2.resize((m > 0.5).astype(np.uint8), (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR) >= 0.5
                ys, xs = np.nonzero(mm)
                rows.append(dict(stem=stem, conf=float(c), x0=float(b[0]), y0=float(b[1]), x1=float(b[2]),
                                 y1=float(b[3]), area=int(mm.sum()),
                                 r_norm=float(np.hypot(xs.mean() - DISK_C, ys.mean() - DISK_C) / DISK_R) if mm.any() else np.nan,
                                 rle=encode_mask(mm.astype(np.uint8)) if mm.any() else ""))
        if (n + 1) % 20 == 0 or n + 1 == len(files):
            print(f"  fold {fold}: {n + 1}/{len(files)} {time.time() - t0:.0f}s raw {len(rows)}", flush=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"wrote {out.name}: {len(rows)} raw detections")


def build(root):
    import torch
    import torchvision
    sys.path.insert(0, str(root / "src"))
    from filament.rle import decode_rle
    cache = root / "outputs" / "oof_cache"
    report = {}
    for fold in range(5):
        raw = pd.concat([pd.read_parquet(p) for p in sorted(cache.glob(f"raw95_e013_f{fold}_*of*.parquet"))],
                        ignore_index=True)
        rep = dict(raw=len(raw), images=int(raw["stem"].nunique()), arms={})
        for tag, t in ARMS.items():
            keep = []
            for _, g in raw.groupby("stem"):
                boxes = torch.tensor(g[["x0", "y0", "x1", "y1"]].to_numpy(), dtype=torch.float32)
                idx = torchvision.ops.nms(boxes, torch.tensor(g["conf"].to_numpy(), dtype=torch.float32), t)[:300]
                keep.append(g.iloc[idx.numpy()])
            d = pd.concat(keep)
            d = d[d["rle"] != ""][["stem", "conf", "area", "r_norm", "rle"]].reset_index(drop=True)
            d.to_parquet(cache / f"fold{fold}_preds_e013o{tag}.parquet", index=False)
            rep["arms"][tag] = len(d)
            if fold == 0 and tag in ("005", "010", "05"):
                ref = pd.read_parquet(cache / f"fold0_preds_e013nms{tag}.parquet").reset_index(drop=True)
                same = len(ref) == len(d) and bool((ref == d).all().all())
                rep[f"equal_to_earlier_nms{tag}"] = same
                if not same:
                    sys.exit(f"fold 0 arm {tag} differs from the earlier file")
        # guard: offline 0.5 against production
        prod = pd.read_parquet(cache / f"fold{fold}_preds_e013.parquet")
        d5 = pd.read_parquet(cache / f"fold{fold}_preds_e013o05.parquet")
        g_rows = []
        for stem, gp in prod.groupby("stem"):
            gp = gp.sort_values("conf", ascending=False)
            gn = d5[d5["stem"] == stem].sort_values("conf", ascending=False)
            ious = []
            for (_, rp), (_, rn) in zip(gp.iterrows(), gn.iterrows()):
                p, q = decode_rle(rp["rle"]).astype(bool), decode_rle(rn["rle"]).astype(bool)
                ious.append((p & q).sum() / max((p | q).sum(), 1))
            g_rows.append(dict(n_prod=len(gp), n_new=len(gn), iou=float(np.mean(ious)) if ious else 1.0))
        g = pd.DataFrame(g_rows)
        rep["guard"] = dict(count_match=float((g.n_prod == g.n_new).mean()), mean_iou=float(g.iou.mean()),
                            total_prod=int(g.n_prod.sum()), total_new=int(g.n_new.sum()))
        rep["guard"]["passed"] = bool(rep["guard"]["count_match"] >= 0.95 and rep["guard"]["mean_iou"] >= 0.98)
        report[fold] = rep
        print(f"fold {fold}: raw {rep['raw']}, arms {rep['arms']}, guard {rep['guard']}", flush=True)
    json.dump(report, open(HERE / "outputs" / "eda" / "nms_sweep_5fold_build.json", "w"), indent=1)


def _extras_for_stem(root, img_path, items):
    """Extra features for one image's candidates of every arm; the structure tensor once."""
    for p in ("src", "src/models", "src/eda"):
        if str(Path(root) / p) not in sys.path:
            sys.path.insert(0, str(Path(root) / p))
    import cv2
    from filament.rle import decode_rle
    from extra_features import extra_features, structure_tensor
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    T = structure_tensor(img)
    rows = []
    for key, rle in items:
        d = extra_features(decode_rle(rle).astype(bool), img, T)
        d["key"] = key
        rows.append(d)
    return rows


def evaluate(root, jobs):
    import joblib
    from joblib import Parallel, delayed
    for p in ("src", "src/models", "src/eda"):
        sys.path.insert(0, str(root / p))
    from extra_features import EXTRA
    from e115c_threshold_ci import make_cls, make_reg, per_image, pq

    out_dir = root / "outputs" / "eda"
    img_dir = root / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
    feats = list(joblib.load(root / "outputs/selector/selector_t020pox_bag5.joblib")["features"])
    A = pd.read_parquet(out_dir / "fused_pool_all_blob5_t020_labelled.parquet")
    A = A.merge(pd.read_parquet(out_dir / "extra_features_t020.parquet")[["cand_id"] + EXTRA], on="cand_id", how="left")
    folds = pd.read_parquet(root / "data/splits/folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    A["fold"] = A["stem"].map(fold_of).astype(int)
    fold = A["fold"].to_numpy()
    X, y, v = A[feats].to_numpy(np.float64), A["y"].to_numpy(), A["iou_sum"].to_numpy(np.float64)

    # the five out-of-fold selectors of e115c_threshold_ci.oof_scores
    sel = {}
    for f in range(5):
        tr = np.where(fold != f)[0]
        inner = np.zeros(len(tr))
        for g in [k for k in range(5) if k != f]:
            itr = np.where((fold != f) & (fold != g))[0]
            msk = fold[tr] == g
            ci = make_cls().fit(X[itr], y[itr])
            ri = make_reg().fit(X[itr][y[itr] == 1], v[itr][y[itr] == 1])
            inner[msk] = ci.predict_proba(X[tr][msk])[:, 1] * np.clip(ri.predict(X[tr][msk]), 0, None)
        sel[f] = (make_cls().fit(X[tr], y[tr]), make_reg().fit(X[tr][y[tr] == 1], v[tr][y[tr] == 1]), np.sort(inner))
        print(f"selector for fold {f} fitted", flush=True)

    def score_p(pool):
        p = np.zeros(len(pool))
        fo = pool["fold"].to_numpy()
        for f, (c, r, ref) in sel.items():
            m = fo == f
            if m.any():
                Xp = pool.loc[m, feats].to_numpy(np.float64)
                raw = c.predict_proba(Xp)[:, 1] * np.clip(r.predict(Xp), 0, None)
                p[m] = np.searchsorted(ref, raw, side="right") / len(ref)
        return p

    A["p"] = score_p(A)
    shipped = A[["cand_id"]].merge(pd.read_parquet(out_dir / "t020pox_oof_scores.parquet"), on="cand_id", how="left")["p"]
    guard = float(np.abs(A["p"].to_numpy() - shipped.to_numpy()).max())
    print(f"guard: max |p - shipped out-of-fold p| = {guard:.2e}", flush=True)
    if not guard <= 1e-9:
        sys.exit("the out-of-fold selectors do not reproduce the shipped scores")

    pools = {}
    for tag in ARMS:
        P = pd.read_parquet(out_dir / f"fused_pool_all_sw_o{tag}_t020_labelled.parquet")
        P["fold"] = P["stem"].map(fold_of).astype(int)
        P = P.drop(columns=[e for e in EXTRA if e in P.columns])
        P["key"] = tag + ":" + P["cand_id"].astype(str)
        pools[tag] = P
    allc = pd.concat([P[["stem", "key", "rle"]] for P in pools.values()], ignore_index=True)
    t0 = time.time()
    res_rows = Parallel(n_jobs=jobs, verbose=0)(
        delayed(_extras_for_stem)(str(root), img_dir / f"{stem}.jpeg", list(zip(g["key"], g["rle"])))
        for stem, g in allc.groupby("stem", sort=False))
    ex = pd.DataFrame([r for rows in res_rows for r in rows])
    print(f"extra features for {len(ex)} candidates ({time.time() - t0:.0f}s)", flush=True)
    for tag, P in pools.items():
        P = P.merge(ex[["key"] + list(EXTRA)], on="key", how="left")
        P["p"] = score_p(P)
        pools[tag] = P

    gt = pd.read_parquet(root / "outputs/gt_cache/gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    stems = np.array(sorted(fold_of))
    gt = gt[gt["stem"].isin(set(stems))]
    ncopy = gt.groupby("stem")["entry_id"].nunique().reindex(stems).fillna(1).to_numpy()
    stem_fold = np.array([fold_of[s] for s in stems])
    arms = {"shipped": A, **{f"nms_{ARMS[t]:g}": pools[t] for t in ARMS}}
    stats, res = {}, {}
    for nm, pool in arms.items():
        keep = pool[(pool["conf"] >= 0.40) & (pool["p"] >= 0.60)]
        st, exs = per_image(keep[["cand_id", "rle", "stem"]], gt, stems, ncopy)
        stats[nm] = st
        per_fold = {int(f): float(pq(st["tp"][stem_fold == f].sum(), st["fp"][stem_fold == f].sum(),
                                     st["fn"][stem_fold == f].sum(), st["iou"][stem_fold == f].sum())) for f in range(5)}
        res[nm] = dict(candidates=len(pool), positives=int(pool["y"].sum()), kept=len(keep), pq=float(exs["pq"]),
                       tp=int(exs["tp"]), fp=int(exs["fp"]), fn=int(exs["fn"]),
                       pq_decomposed=float(pq(st["tp"].sum(), st["fp"].sum(), st["fn"].sum(), st["iou"].sum())),
                       pq_per_fold=per_fold)
        print(f"{nm:9s} candidates {len(pool):6d}  positives {res[nm]['positives']:5d}  kept {len(keep):5d}  "
              f"PQ {exs['pq']:.4f}  TP {exs['tp']} FP {exs['fp']} FN {exs['fn']}  per fold "
              + " ".join(f"{x:.4f}" for x in per_fold.values()), flush=True)
    rng = np.random.default_rng(0)
    base = stats["nms_0.5"]
    for nm in arms:
        if nm == "nms_0.5":
            continue
        s, d = stats[nm], np.empty(4000)
        for b in range(4000):
            i = rng.integers(0, len(stems), len(stems))
            d[b] = (pq(s["tp"][i].sum(), s["fp"][i].sum(), s["fn"][i].sum(), s["iou"][i].sum())
                    - pq(base["tp"][i].sum(), base["fp"][i].sum(), base["fn"][i].sum(), base["iou"][i].sum()))
        lo, hi = np.percentile(d, [2.5, 97.5])
        obs = res[nm]["pq_decomposed"] - res["nms_0.5"]["pq_decomposed"]
        folds_up = sum(res[nm]["pq_per_fold"][f] > res["nms_0.5"]["pq_per_fold"][f] for f in range(5))
        res[nm]["delta_vs_nms_0.5"] = dict(observed=float(obs), ci95=[float(lo), float(hi)],
                                           p_gt0=float((d > 0).mean()), folds_up=int(folds_up))
        print(f"  {nm} - nms_0.5: {obs:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  P(>0) {(d > 0).mean():.3f}  "
              f"folds up {folds_up}/5", flush=True)
    out = HERE / "outputs" / "eda" / "nms_sweep_5fold.json"
    json.dump(dict(images=len(stems), gt_copies=len(gt), keep_gate=[0.40, 0.60], guard_max_abs_p=guard,
                   arms=res), open(out, "w"), indent=1)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["infer", "build", "eval"])
    ap.add_argument("--root", default=str(HERE), help="repository holding the inputs (see the docstring)")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--device", default="0")
    ap.add_argument("--jobs", type=int, default=10)
    a = ap.parse_args()
    root = Path(a.root).resolve()
    if a.step == "infer":
        infer(root, a.fold, a.device)
    elif a.step == "build":
        build(root)
    else:
        evaluate(root, a.jobs)


if __name__ == "__main__":
    main()
