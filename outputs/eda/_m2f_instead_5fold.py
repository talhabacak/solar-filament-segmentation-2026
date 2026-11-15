"""E222 -- Mask2Former INSTEAD of YOLO as the detector lane (Talha, 2026-10-04). Derived from
_lane_5fold.py (ledger 27t), which measured M2F as an ADDED lane (+0.0020). Same protocol: the
selector is refitted on the other folds of the same pool and read out of fold on all 707 images.
  A  production pool (YOLO11m-seg e013 + semantic), production features
  D  Mask2Former (Swin-T, 1024) + semantic, production features -- YOLO removed
Confound stated: the M2F models run at 1024 px, the detector at 1792.

Lane identity, the deployment-faithful read (ledger 27m follow-up).

With Mask2Former candidates on every fold, the selector can be fitted the production way (on
the other folds of the SAME pool) and read out of fold on all 707 images:
  A  production pool (e013 + semantic), production features          -- the deployed machine
  B  e013+M2F pool, production features                               -- the lane, blind
  C  e013+M2F pool, production features + lane identity               -- the lane, seen
Paired image bootstrap A->B, B->C, A->C. Extra features (EXTRA) are computed for the new pool
on the fly; the lane features come from the pool builder. Usage:
  python outputs/eda/_lane_5fold.py fused_pool_all_e013m2fL_t020_labelled.parquet
"""
import sys, json, pathlib, numpy as np, pandas as pd, joblib, cv2
ROOT = pathlib.Path(".").resolve()
for p in ("src", "src/eda", "src/models"): sys.path.insert(0, str(ROOT / p))
from filament.pq import score
from filament.rle import decode_rle
from e115c_threshold_ci import make_cls, make_reg, per_image, pq
from extra_features import extra_features, structure_tensor, EXTRA
OUT = ROOT / "outputs" / "eda"; IMG = ROOT / "data/MAGFiLO_1.0_Kaggle_2026/train/train_images"
LANE = ["from_m2f", "m2f_conf_max", "n_e013_models", "e013_conf_max"]
base = list(joblib.load(ROOT / "outputs/selector/selector_v5_shrunkblob5prodts.joblib")["features"]) + list(EXTRA)
folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet"); gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
stems = np.array(sorted(fold_of)); ncopy = gt.groupby("stem")["entry_id"].nunique().reindex(stems).fillna(1).to_numpy()


def with_extra(a, cache_name):
    f = OUT / cache_name
    if f.exists():
        return a.merge(pd.read_parquet(f)[["cand_id"] + list(EXTRA)], on="cand_id", how="left")
    rows = []
    for stem, gg in a.groupby("stem"):
        img = cv2.imread(str(IMG / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE).astype(np.float32); T = structure_tensor(img)
        for cid, rle in zip(gg["cand_id"], gg["rle"]):
            d = extra_features(decode_rle(rle).astype(bool), img, T); d["cand_id"] = cid; rows.append(d)
    ex = pd.DataFrame(rows); ex.to_parquet(f, index=False)
    return a.merge(ex, on="cand_id", how="left")


def oof(a, feats):
    a = a.copy(); a["fold"] = a["stem"].map(fold_of).astype(int); sel = []
    for F in range(5):
        trn = a[a["fold"] != F]; tst = a[a["fold"] == F].copy()
        X, y, v = trn[feats].to_numpy(np.float64), trn["y"].to_numpy(), trn["iou_sum"].to_numpy(np.float64)
        inner = np.zeros(len(trn))
        for g in [k for k in range(5) if k != F]:
            itr = (trn["fold"] != g).to_numpy(); msk = (trn["fold"] == g).to_numpy()
            c = make_cls().fit(X[itr], y[itr]); r = make_reg().fit(X[itr][y[itr] == 1], v[itr][y[itr] == 1])
            inner[msk] = c.predict_proba(X[msk])[:, 1] * np.clip(r.predict(X[msk]), 0, None)
        ref = np.sort(inner); cls = make_cls().fit(X, y); reg = make_reg().fit(X[y == 1], v[y == 1])
        Zt = tst[feats].to_numpy(np.float64)
        tst["p"] = np.searchsorted(ref, cls.predict_proba(Zt)[:, 1] * np.clip(reg.predict(Zt), 0, None), side="right") / len(ref)
        sel.append(tst[(tst["conf"] >= 0.40) & (tst["p"] >= 0.60)])
    sub = pd.concat(sel)
    r = score(gt[["filament_id", "segmentation_rle"]], sub[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"}))
    return r, per_image(sub, gt, stems, ncopy)[0]


prod = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
prod = prod.merge(pd.read_parquet(OUT / "extra_features_t020.parquet")[["cand_id"] + list(EXTRA)], on="cand_id", how="left")
lane = with_extra(pd.read_parquet(OUT / sys.argv[1]), "extra_features_m2fonly_t020.parquet")
print(f"pools: production {len(prod)} rows, M2F-only {len(lane)} rows; positives {int(prod['y'].sum())} / {int(lane['y'].sum())}", flush=True)
res, per = {}, {}
for name, (a, feats) in {"A prod": (prod, base), "D m2f instead": (lane, base)}.items():
    r, pp = oof(a, feats); res[name] = r; per[name] = pp
    print(f"{name:13s} PQ {r['pq']:.4f} SQ {r['sq']:.4f} RQ {r['rq']:.4f} TP {r['tp']} FP {r['fp']} FN {r['fn']}", flush=True)
rng = np.random.default_rng(0); n = len(stems)
out = {}
for A, B in (("A prod", "D m2f instead"),):
    pa, pb = per[A], per[B]; boot = np.empty(4000)
    for i in range(4000):
        idx = rng.integers(0, n, n)
        boot[i] = (pq(pb["tp"][idx].sum(), pb["fp"][idx].sum(), pb["fn"][idx].sum(), pb["iou"][idx].sum())
                   - pq(pa["tp"][idx].sum(), pa["fp"][idx].sum(), pa["fn"][idx].sum(), pa["iou"][idx].sum()))
    d = res[B]["pq"] - res[A]["pq"]
    print(f"  {A} -> {B}: delta {d:+.4f}  95% CI [{np.quantile(boot, .025):+.4f}, {np.quantile(boot, .975):+.4f}]  P(delta>0) {(boot > 0).mean():.3f}", flush=True)
    out[f"{A}->{B}"] = dict(delta=d, lo=float(np.quantile(boot, .025)), hi=float(np.quantile(boot, .975)), p=float((boot > 0).mean()))
json.dump(dict(arms={k: {kk: float(vv) for kk, vv in v.items() if kk in ("pq", "sq", "rq", "tp", "fp", "fn")} for k, v in res.items()}, paired=out),
          open(OUT / "e222_m2f_instead_5fold.json", "w"), indent=1)
print("E222_DONE")
