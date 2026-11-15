"""docs/v2 §15 asks whether random image folds leak near-duplicate frames across train and
validation. 27% of training images have a cross-fold neighbour within 24 h (E166). If that
inflates the OOF number, images WITH such a neighbour should score higher than images without,
at the same deployed selection. Per-image PQ terms of the deployed fold-0..4 selection are not
cached here, so this uses the production OOF pool at the deployed gate (selector fitted on the
other folds, as everywhere) and compares the pooled PQ of the two image groups, with a
bootstrap over images for the difference. Also reports the test set's distance to the nearest
training frame, which is what matters for transfer."""
import sys, os, pathlib, numpy as np, pandas as pd, joblib
ROOT = pathlib.Path(".").resolve()
for p in ("src", "src/eda", "src/models"): sys.path.insert(0, str(ROOT / p))
from e115c_threshold_ci import make_cls, make_reg, per_image, pq
from extra_features import EXTRA
OUT = ROOT / "outputs" / "eda"
feats = list(joblib.load(ROOT / "outputs/selector/selector_v5_shrunkblob5prodts.joblib")["features"]) + list(EXTRA)
folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
folds["stem"] = folds["file_name"].str.replace(".jpeg", "", regex=False)
folds["t"] = pd.to_datetime(folds["stem"].str[:14], format="%Y%m%d%H%M%S")
fold_of = dict(zip(folds["stem"], folds["fold"]))
gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet"); gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
prod = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
prod = prod.merge(pd.read_parquet(OUT / "extra_features_t020.parquet")[["cand_id"] + EXTRA], on="cand_id", how="left")
prod["fold"] = prod["stem"].map(fold_of).astype(int)
stems = np.array(sorted(fold_of)); ncopy = gt.groupby("stem")["entry_id"].nunique().reindex(stems).fillna(1).to_numpy()
sel = []
for F in range(5):
    trn = prod[prod["fold"] != F]; tst = prod[prod["fold"] == F].copy()
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
pi = per_image(sub, gt, stems, ncopy)[0]
# nearest cross-fold training neighbour in hours
t = folds.set_index("stem").loc[stems, "t"].to_numpy(); f = folds.set_index("stem").loc[stems, "fold"].to_numpy()
near = np.full(len(stems), 1e9)
for i in range(len(stems)):
    other = f != f[i]
    if other.any():
        near[i] = np.min(np.abs((t[other] - t[i]).astype("timedelta64[m]").astype(float))) / 60.0
has = near <= 24.0
def PQ(idx):
    return pq(pi["tp"][idx].sum(), pi["fp"][idx].sum(), pi["fn"][idx].sum(), pi["iou"][idx].sum())
print(f"images with a cross-fold neighbour within 24 h: {has.sum()} of {len(stems)}")
print(f"PQ  with neighbour {PQ(np.flatnonzero(has)):.4f}   without {PQ(np.flatnonzero(~has)):.4f}   all {PQ(np.arange(len(stems))):.4f}")
rng = np.random.default_rng(0); a, b = np.flatnonzero(has), np.flatnonzero(~has); boot = np.empty(4000)
for i in range(4000):
    boot[i] = PQ(rng.choice(a, len(a))) - PQ(rng.choice(b, len(b)))
print(f"difference (with - without) {PQ(a) - PQ(b):+.4f}  95% CI [{np.quantile(boot, .025):+.4f}, {np.quantile(boot, .975):+.4f}]")
for lo, hi in ((0, 1), (1, 6), (6, 24), (24, 24 * 7), (24 * 7, 1e9)):
    m = (near >= lo) & (near < hi)
    if m.sum() >= 10:
        print(f"  neighbour {lo:>5.0f}-{hi:<7.0f} h: {m.sum():3d} images  PQ {PQ(np.flatnonzero(m)):.4f}")
test = sorted(fn[:-5] for fn in os.listdir(ROOT / "data/MAGFiLO_1.0_Kaggle_2026/test/test_images"))
tt = pd.to_datetime(pd.Series(test).str[:14], format="%Y%m%d%H%M%S").to_numpy()
tn = [np.min(np.abs((t - x).astype("timedelta64[m]").astype(float))) / 60.0 for x in tt]
print(f"test images: nearest training frame median {np.median(tn):.1f} h, {np.mean(np.array(tn) <= 24):.0%} within 24 h")
print("NEAR_DUP_DONE")
