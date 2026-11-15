"""WHERE does the high-frequency loss happen -- model, selector, or mask? And is hf new?

hf (Laplacian variance on the disk) is the only image-level covariate that predicts our PQ
(Spearman -0.140, p 2e-4, pooled PQ 0.464 -> 0.416 across quartiles). Two things must be settled
before it is worth wiring anywhere:

  1. ANATOMY. Decompose the loss by hf quartile into the four outcomes of the error atlas plus
     the false-positive rate, and split PQ into SQ and RQ. "never proposed" rising with hf is a
     generation problem; "selector" rising is a selection problem; "shape" rising is mask
     quality; FP rising is precision.
  2. NOVELTY. hf may just be crowding wearing a different name, and crowding is already in the
     selector as n_cands_img / img_cand_density / img_sem_area_frac. Partial out the density
     features and see whether anything survives.
"""
import sys, pathlib, numpy as np, pandas as pd
ROOT = pathlib.Path(".").resolve()
for p in ("src", "src/eda", "src/models"): sys.path.insert(0, str(ROOT/p))
from e115c_threshold_ci import per_image, pq
from scipy.stats import spearmanr

q = pd.read_parquet("outputs/eda/image_quality.parquet")
q = q[q.split == "train"][["stem", "hf", "limb_width", "disk_sd", "disk_mean"]]
pool = pd.read_parquet("outputs/eda/fused_pool_all_blob5_t020_labelled.parquet")
pool = pool.merge(pd.read_parquet("outputs/eda/bag5_oof_scores.parquet"), on="cand_id")
gt = pd.read_parquet("outputs/gt_cache/gt_full.parquet")
gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
gt = gt[gt["stem"].isin(set(pool["stem"]))]
stems = np.array(sorted(set(pool["stem"])))
ncopy = gt.groupby("stem")["entry_id"].nunique().reindex(stems).fillna(1).to_numpy()
keep = (pool["conf"] >= 0.40) & (pool["p"] >= 0.60)
st = per_image(pool[keep][["cand_id", "rle", "stem"]], gt, stems, ncopy)[0]
d = pd.DataFrame({"stem": stems, "tp": st["tp"], "fp": st["fp"], "fn": st["fn"], "iou": st["iou"]}).merge(q, on="stem")
img = pool.groupby("stem").agg(aday=("cand_id", "size"), sem_alan=("img_sem_area_frac", "mean"),
                               yogunluk=("img_cand_density", "mean")).reset_index()
d = d.merge(img, on="stem")
d["ngt"] = d.tp + d.fn
print(f"{len(d)} goruntu\n")
print("hf ile diger eksenler arasindaki iliski (Spearman):")
for c in ["limb_width", "disk_sd", "aday", "ngt", "yogunluk", "sem_alan"]:
    print(f"  hf ~ {c:12s} rho {spearmanr(d.hf, d[c]).statistic:+.3f}")

print("\n\n(1) ANATOMI -- hf ceyregine gore")
a = pd.read_parquet("outputs/eda/error_atlas.parquet").merge(q, on="stem")
a["_q"] = pd.qcut(a.hf, 4, labels=["Q1 dusuk", "Q2", "Q3", "Q4 yuksek"])
t = pd.crosstab(a["_q"], a["outcome"], normalize="index").round(3)
t["kopya"] = a.groupby("_q", observed=True).size()
print(t.to_string())
d["_q"] = pd.qcut(d.hf, 4, labels=["Q1 dusuk", "Q2", "Q3", "Q4 yuksek"])
g = d.groupby("_q", observed=True).apply(lambda s: pd.Series({
    "n": len(s), "hf": s.hf.median(), "GT/gor": s.ngt.median(), "aday/gor": s.aday.median(),
    "PQ": pq(s.tp.sum(), s.fp.sum(), s.fn.sum(), s.iou.sum()),
    "SQ": s.iou.sum()/max(s.tp.sum(), 1), "RQ": s.tp.sum()/(s.tp.sum()+s.fp.sum()/2+s.fn.sum()/2),
    "FP/GT": s.fp.sum()/max(s.ngt.sum(), 1), "recall": s.tp.sum()/max(s.ngt.sum(), 1)}), include_groups=False)
print("\n" + g.round(3).to_string())

print("\n\n(2) YENI MI? -- yogunluk kontrol edilince hf hala ongoruyor mu")
d["pq_img"] = d.iou / (d.tp + d.fp/2 + d.fn/2).clip(lower=1e-9)
d = d[(d.tp + d.fp + d.fn) > 0]
import numpy.linalg as la
def partial(y, x, ctrl):
    X = np.column_stack([np.ones(len(d)), d[ctrl].to_numpy(float)])
    ry = d[y].to_numpy(float) - X @ la.lstsq(X, d[y].to_numpy(float), rcond=None)[0]
    rx = d[x].to_numpy(float) - X @ la.lstsq(X, d[x].to_numpy(float), rcond=None)[0]
    return spearmanr(rx, ry)
r0 = spearmanr(d.hf, d.pq_img)
print(f"  ham            rho {r0.statistic:+.3f}  p {r0.pvalue:.2e}")
for ctrl, lab in ((["aday"], "aday sayisi"), (["ngt"], "GT sayisi"),
                  (["aday", "yogunluk", "sem_alan"], "seciciDEKI yogunluk ozellikleri"),
                  (["aday", "ngt", "yogunluk", "sem_alan", "disk_sd", "limb_width"], "hepsi")):
    r = partial("pq_img", "hf", ctrl)
    print(f"  {lab:32s} kismi rho {r.statistic:+.3f}  p {r.pvalue:.2e}")
