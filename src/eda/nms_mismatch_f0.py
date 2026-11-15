"""Does the detector NMS of the test caches cost PQ? (fold 0, measured on 15 November 2026)

The submitted system makes its out-of-fold training candidates with detector NMS IoU 0.5 and its
test-time candidates with 0.05 (report, Section 2.4; the 0.05 dates from week one, where it gave
a single detector +0.0105 by removing duplicate boxes). The selector therefore judges test pools
built under a setting it was never fitted on. This measures what that costs, on fold 0.

Earlier NMS measurements, none of which noticed the mismatch: the week-one sweep on one detector
before fusion (src/models/nms_sweep.py; 0.05 best, plateau 0.01-0.15); sub005 at 0.05, which read
the same leaderboard 0.36 as sub004; and E218 (src/eda/e218_nms_tighter.py, ledger 54a), which
measured 0.3-0.7 in the fused pool by reach, without the selector, and found the axis flat. What
this adds is 0.05 and 0.10 in the fused pool, through the selector.

Candidates: E218's fold-0 detector run with NMS effectively off (src/eda/e218_nms_tighter.py
infer, `outputs/oof_cache/raw95_e013_f0_0of1.parquet`), to which NMS is applied offline at 0.5,
0.10 and 0.05 (`build`, the construction of e218_nms_tighter.build; the 0.5 arm is checked equal
to E218's own file). Each arm is then fused and labelled the production way:

  python src/models/build_fused_pool.py --folds 0 --sem-tag _blob5ens --sem-thr 0.2 \
      --det-tag _e013nms{05,010,005} --out fused_pool_f0_sw_nms{05,010,005}_t020.parquet
  python outputs/eda/_label_one_pool.py fused_pool_f0_sw_nms{05,010,005}_t020
  (a development-repository helper around src/eda/two_stage_selector.label_pool)

Selector: the out-of-fold one for fold 0, i.e. e115c_threshold_ci.oof_scores for f = 0 (fitted on
folds 1-4 of the shipped labelled pool, quantile reference from inner leave-one-fold-out raws).
GUARD: on the shipped fold-0 rows it must reproduce outputs/eda/t020pox_oof_scores.parquet.
Keep conf >= 0.40 and p >= 0.60; pooled PQ on the 142 fold-0 images; paired image bootstrap of
each arm against the 0.5 arm (4000 resamples).

Result (outputs/eda/nms_mismatch_f0.json): guard exact (max |dp| = 0). NMS 0.05 removes 188 of
1927 candidates and none of the matching ones, and changes PQ by +0.0007 [-0.0043, +0.0055]: the
mismatch does not cost. NMS 0.10 reads +0.0045 [+0.0000, +0.0088]; one fold, chosen after the
fact among several arms, below the five-fold promotion protocol of the ledger, and not adopted.

Superseded by the five-fold reading of the same instrument, src/eda/nms_sweep_5fold.py
(outputs/eda/nms_sweep_5fold.json): on all 707 images 0.05 reads -0.0016 [-0.0037, +0.0006] and
0.10 -0.0001 [-0.0021, +0.0020] against 0.5. Fold 0's +0.0045 for 0.10 does not survive.

The inputs (the raw detector run, the fold-0 semantic caches the pools are fused from, the
ground-truth cache and the training images) are in the development repository and are not
shipped here; the run that wrote the JSON passed `--root` pointing at it.

Usage: python src/eda/nms_mismatch_f0.py build [--root DIR]
       python src/eda/nms_mismatch_f0.py eval  [--root DIR]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[2]
ARMS = {"005": 0.05, "010": 0.10, "05": 0.5}


def build(root):
    import torch
    import torchvision
    cache = root / "outputs" / "oof_cache"
    raw = pd.concat([pd.read_parquet(p) for p in sorted(cache.glob("raw95_e013_f0_*of*.parquet"))], ignore_index=True)
    print(f"raw detections {len(raw)} over {raw['stem'].nunique()} images")
    for tag, t in ARMS.items():
        keep = []
        for _, g in raw.groupby("stem"):
            boxes = torch.tensor(g[["x0", "y0", "x1", "y1"]].to_numpy(), dtype=torch.float32)
            idx = torchvision.ops.nms(boxes, torch.tensor(g["conf"].to_numpy(), dtype=torch.float32), t)[:300]
            keep.append(g.iloc[idx.numpy()])
        d = pd.concat(keep)
        d = d[d["rle"] != ""][["stem", "conf", "area", "r_norm", "rle"]]
        out = cache / f"fold0_preds_e013nms{tag}.parquet"
        if tag == "05":                       # E218 wrote this one; it must come out the same
            ref = pd.read_parquet(out)
            same = len(ref) == len(d) and (ref.reset_index(drop=True) == d.reset_index(drop=True)).all().all()
            print(f"nms {t}: {len(d)} candidates, equal to E218's file: {same}")
            continue
        d.to_parquet(out, index=False)
        print(f"nms {t}: {len(d)} candidates -> {out.name}")


def evaluate(root):
    import cv2
    import joblib
    for p in ("src", "src/models", "src/eda"):
        sys.path.insert(0, str(root / p))
    from filament.rle import decode_rle
    from extra_features import EXTRA, extra_features, structure_tensor
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

    # e115c.oof_scores, the f = 0 iteration
    tr = np.where(fold != 0)[0]
    inner = np.zeros(len(tr))
    for g in range(1, 5):
        itr = np.where((fold != 0) & (fold != g))[0]
        msk = fold[tr] == g
        ci = make_cls().fit(X[itr], y[itr])
        ri = make_reg().fit(X[itr][y[itr] == 1], v[itr][y[itr] == 1])
        inner[msk] = ci.predict_proba(X[tr][msk])[:, 1] * np.clip(ri.predict(X[tr][msk]), 0, None)
    ref = np.sort(inner)
    c = make_cls().fit(X[tr], y[tr])
    r = make_reg().fit(X[tr][y[tr] == 1], v[tr][y[tr] == 1])

    def score_p(pool):
        Xp = pool[feats].to_numpy(np.float64)
        raw = c.predict_proba(Xp)[:, 1] * np.clip(r.predict(Xp), 0, None)
        return np.searchsorted(ref, raw, side="right") / len(ref)

    A0 = A[A["fold"] == 0].copy()
    A0["p"] = score_p(A0)
    shipped = A0[["cand_id"]].merge(pd.read_parquet(out_dir / "t020pox_oof_scores.parquet"), on="cand_id")["p"]
    guard = float(np.abs(A0["p"].to_numpy() - shipped.to_numpy()).max())
    print(f"guard: max |p - shipped out-of-fold p| on fold 0 = {guard:.2e}")
    if guard > 1e-9:
        sys.exit("the fold-0 selector does not reproduce the shipped out-of-fold scores")

    def arm(name):
        pool = pd.read_parquet(out_dir / f"fused_pool_f0_sw_nms{name}_t020_labelled.parquet")
        rows = []
        for stem, gg in pool.groupby("stem", sort=False):
            img = cv2.imread(str(img_dir / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            T = structure_tensor(img)
            for cid, rle in zip(gg["cand_id"], gg["rle"]):
                d = extra_features(decode_rle(rle).astype(bool), img, T)
                d["cand_id"] = cid
                rows.append(d)
        ex = pd.DataFrame(rows)[["cand_id"] + list(EXTRA)]
        pool = pool.drop(columns=[e for e in EXTRA if e in pool.columns]).merge(ex, on="cand_id", how="left")
        pool["p"] = score_p(pool)
        return pool

    gt = pd.read_parquet(root / "outputs/gt_cache/gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    stems = np.array(sorted(s for s, k in fold_of.items() if k == 0))
    gt = gt[gt["stem"].isin(set(stems))]
    ncopy = gt.groupby("stem")["entry_id"].nunique().reindex(stems).fillna(1).to_numpy()
    pools = {"shipped_f0": A0, "nms_0.5": arm("05"), "nms_0.10": arm("010"), "nms_0.05": arm("005")}
    stats, res = {}, {}
    for nm, pool in pools.items():
        keep = pool[(pool["conf"] >= 0.40) & (pool["p"] >= 0.60)]
        st, ex = per_image(keep, gt, stems, ncopy)
        stats[nm] = st
        res[nm] = dict(candidates=len(pool), positives=int(pool["y"].sum()), kept=len(keep), pq=float(ex["pq"]),
                       tp=int(ex["tp"]), fp=int(ex["fp"]), fn=int(ex["fn"]),
                       pq_decomposed=float(pq(st["tp"].sum(), st["fp"].sum(), st["fn"].sum(), st["iou"].sum())))
        print(f"{nm:10s} candidates {len(pool):5d}  positives {res[nm]['positives']:4d}  kept {len(keep):4d}  "
              f"PQ {ex['pq']:.4f}  TP {ex['tp']} FP {ex['fp']} FN {ex['fn']}")
    rng = np.random.default_rng(0)
    base = stats["nms_0.5"]
    for nm in ("nms_0.10", "nms_0.05", "shipped_f0"):
        s, d = stats[nm], np.empty(4000)
        for b in range(4000):
            i = rng.integers(0, len(stems), len(stems))
            d[b] = (pq(s["tp"][i].sum(), s["fp"][i].sum(), s["fn"][i].sum(), s["iou"][i].sum())
                    - pq(base["tp"][i].sum(), base["fp"][i].sum(), base["fn"][i].sum(), base["iou"][i].sum()))
        lo, hi = np.percentile(d, [2.5, 97.5])
        obs = res[nm]["pq_decomposed"] - res["nms_0.5"]["pq_decomposed"]
        res[nm]["delta_vs_nms_0.5"] = dict(observed=float(obs), ci95=[float(lo), float(hi)], p_gt0=float((d > 0).mean()))
        print(f"  {nm} - nms_0.5: {obs:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  P(>0) {(d > 0).mean():.3f}")
    out = HERE / "outputs" / "eda" / "nms_mismatch_f0.json"
    json.dump(dict(fold=0, images=len(stems), gt_copies=len(gt), keep_gate=[0.40, 0.60],
                   guard_max_abs_p=guard, arms=res), open(out, "w"), indent=1)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "eval"])
    ap.add_argument("--root", default=str(HERE), help="repository holding the inputs (see the docstring)")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    build(root) if a.step == "build" else evaluate(root)


if __name__ == "__main__":
    main()
