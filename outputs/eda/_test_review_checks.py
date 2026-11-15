"""Three measurements behind ledger 51a: do the patterns seen in the 180-image review point to a lever?

1. Band lever. The review's recurring loss is long, clear filaments left in the marginal band
   (conf >= 0.40, 0.40 <= p < 0.60), mostly near the limb or at high latitude. Out of fold, each band
   candidate is valued by the break-even rule of report eq. 5, v = iou_sum - PQ * n_copies / 2
   (adding it raises PQ when v > 0), summed by length, limb distance and latitude; the subgroup that
   matches the visual pattern is then split into what sub042's cascade already admits and the rest.
2. The unproposed dark structure in 20220714185352Th (ledger 51, finding 2): its position on the
   2048 frame, to settle whether it lies inside the +-70 deg annotation scope.
3. Station mix and dark discs. Four late T-station frames look dark on the contact sheets; per-image
   OOF PQ by station and by disc brightness, and the PQ shift from reweighting training to the test mix.

Diagnosis only: nothing here feeds back into the submission. Part 3 rescores every image (minutes).
"""
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/eda", "src/models"):
    sys.path.insert(0, str(ROOT / p))
from e115c_threshold_ci import per_image, pq  # noqa: E402
from extra_features import EXTRA  # noqa: E402
from final_pair_selection import cascade_admit  # noqa: E402

OUT = ROOT / "outputs/eda"
PQ = 0.4431                                       # sub036 out of fold
C, R = 1023.5, 904.1

a = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
a = a.merge(pd.read_parquet(OUT / "extra_features_t020.parquet")[["cand_id"] + EXTRA], on="cand_id")
folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
a["fold"] = a["stem"].map(fold_of).astype(int)
a = a.merge(pd.read_parquet(OUT / "bag5_oof_scores.parquet"), on="cand_id")
kept = (a.conf >= .40) & (a.p >= .60)
D = a.loc[kept, "iou_sum"].sum() / PQ             # PQ denominator implied by the kept masks
a["v"] = a.iou_sum - PQ * a.n_copies / 2

print("1. BAND LEVER (dPQ = sum v / denominator)")
band = (a.conf >= .40) & (a.p >= .40) & (a.p < .60)
b = a[band].copy()
print(f"  band {len(b)}  positive rate {b.y.mean():.3f}  whole band {b.v.sum() / D:+.4f}")
for col in ("skel_len", "r_norm", "abs_lat"):
    t = b.groupby(pd.qcut(b[col], 5), observed=True).agg(n=("v", "size"), pos=("y", "mean"), v=("v", "sum"))
    print(f"  {col}: " + "  ".join(f"{str(i)} {r.pos:.2f} {r.v / D:+.4f}" for i, r in t.iterrows()))
cfeats = list(joblib.load(ROOT / "outputs/selector/selector_v5_shrunkblob5prodts.joblib")["features"]) + list(EXTRA)
adm = cascade_admit(a, a.p.to_numpy(), cfeats)
tgt = band & (a.skel_len >= b.skel_len.quantile(.8)) & (a.abs_lat > 30)
rest = tgt & ~adm
print(f"  cascade (sub042) admits {int(adm.sum())}, {a.v[adm].sum() / D:+.4f}")
print(f"  long (top fifth) and |lat| > 30: {int(tgt.sum())}, positive {a.y[tgt].mean():.3f}, {a.v[tgt].sum() / D:+.4f}; "
      f"already admitted by the cascade {int((tgt & adm).sum())}; the rest {a.v[rest].sum() / D:+.4f} "
      "(folds " + " ".join(f"{a.v[rest & (a.fold == k)].sum() / D:+.4f}" for k in range(5)) + ")")

print("\n2. DARK STRUCTURE IN 20220714185352Th")
img = cv2.imread(str(ROOT / "data/MAGFiLO_1.0_Kaggle_2026/test/test_images/20220714185352Th.jpeg"),
                 cv2.IMREAD_GRAYSCALE).astype(np.float32)
yy, xx = np.mgrid[0:2048, 0:2048]
r = np.hypot(xx - C, yy - C) / R
dark = (img < cv2.GaussianBlur(img, (0, 0), 25) - 18) & (r > .80) & (r < .99) & (xx > C) & (yy > C)
n, _, stats, cent = cv2.connectedComponentsWithStats(dark.astype(np.uint8))
i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
x, y = (cent[i] - C) / R
lon = np.degrees(np.arcsin(np.clip(x / np.sqrt(1 - min(y * y, .998)), -1, 1)))
print(f"  largest dark component: {stats[i, cv2.CC_STAT_AREA]} px, centre r {np.hypot(x, y):.3f}, longitude {lon:.1f} deg")

print("\n3. STATION MIX AND DARK DISCS (per-image OOF PQ of sub036)")
gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet")
gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
stems = np.array(sorted(fold_of))
gt = gt[gt["stem"].isin(set(stems))]
ncopy = gt.groupby("stem")["entry_id"].nunique().reindex(stems).fillna(1).to_numpy()
st = per_image(a[kept][["cand_id", "rle", "stem"]], gt, stems, ncopy)[0]
C4 = ["tp", "fp", "fn", "iou"]
P = pd.DataFrame({"stem": stems, **{c: st[c] for c in C4}})
q = pd.read_parquet(OUT / "image_quality.parquet")
tr = q[q.split == "train"].merge(P, on="stem")
te = q[q.split == "test"]
tot = pq(*tr[C4].sum().to_numpy())
print(f"  local PQ {tot:.4f}")
for s, g in tr.groupby("site"):
    print(f"  {s}: train {len(g):3d} ({len(g) / len(tr):.3f})  test {int((te.site == s).sum()):3d} "
          f"({(te.site == s).mean():.3f})  PQ {pq(*g[C4].sum().to_numpy()):.4f}  disk_mean median "
          f"{g.disk_mean.median():.1f} / {te[te.site == s].disk_mean.median():.1f}")
w = tr.site.map({s: (te.site == s).mean() / (tr.site == s).mean() for s in tr.site.unique()}).to_numpy()
print(f"  station-reweighted shift {pq(*(tr[C4].to_numpy() * w[:, None]).sum(0)) - tot:+.4f}")
thr = tr.disk_mean.quantile(.10)
d = (tr.disk_mean < thr).to_numpy()
fte = (te.disk_mean < thr).mean()
X = tr[C4].to_numpy()
rng = np.random.default_rng(0)
gaps = [pq(*X[i][d[i]].sum(0)) - pq(*X[i][~d[i]].sum(0))
        for i in (rng.integers(0, len(tr), len(tr)) for _ in range(2000))]
w = np.where(d, fte / d.mean(), (1 - fte) / (1 - d.mean()))
print(f"  dark disc (disk_mean < train p10 {thr:.1f}): train {d.mean():.3f}, test {fte:.3f}; "
      f"PQ {pq(*X[d].sum(0)):.4f} vs {pq(*X[~d].sum(0)):.4f}, 95% CI of the gap "
      f"[{np.quantile(gaps, .025):+.4f}, {np.quantile(gaps, .975):+.4f}]; reweighted shift "
      f"{pq(*(X * w[:, None]).sum(0)) - tot:+.4f}")
