"""Is the late-year deficit real, and is it a DOMAIN shift or a harder population?

2021 reads -0.066 and 2022 -0.047 against the pool, but n=28 and n=12, so individually neither
clears its own CI. Pooled they are 40 images and the test set weights them 2.0x and 2.3x. Two
questions decide whether augmentation has a target:
  1. is the combined late-year deficit significant (paired image bootstrap against the rest)?
  2. is it DOMAIN (same filaments, model transfers worse) or POPULATION (different filaments --
     solar minimum has fewer, smaller, fainter ones, which no augmentation fixes)?
"""
import sys, pathlib, numpy as np, pandas as pd
ROOT = pathlib.Path(".").resolve()
for p in ("src", "src/eda", "src/models"): sys.path.insert(0, str(ROOT/p))
from e115c_threshold_ci import per_image, pq
pool = pd.read_parquet("outputs/eda/fused_pool_all_blob5_t020_labelled.parquet")
pool = pool.merge(pd.read_parquet("outputs/eda/bag5_oof_scores.parquet"), on="cand_id")
gt = pd.read_parquet("outputs/gt_cache/gt_full.parquet")
gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
gt = gt[gt["stem"].isin(set(pool["stem"]))]
stems = np.array(sorted(set(pool["stem"])))
ncopy = gt.groupby("stem")["entry_id"].nunique().reindex(stems).fillna(1).to_numpy()
keep = (pool["conf"] >= 0.40) & (pool["p"] >= 0.60)
st = per_image(pool[keep][["cand_id", "rle", "stem"]], gt, stems, ncopy)[0]
d = pd.DataFrame({"stem": stems, "tp": st["tp"], "fp": st["fp"], "fn": st["fn"], "iou": st["iou"]})
d["year"] = d["stem"].str[:4].astype(int)
rng = np.random.default_rng(0)
def boot(sub, n=4000):
    a = sub[["tp","fp","fn","iou"]].to_numpy()
    return np.array([pq(*a[rng.integers(0,len(a),len(a))].sum(0)) for _ in range(n)])
for cut, lab in ((2021, "2021-2022"), (2019, "2019-2022"), (2016, "2016-2022")):
    late, early = d[d.year>=cut], d[d.year<cut]
    bl, be = boot(late), boot(early)
    dd = pq(*late[["tp","fp","fn","iou"]].sum()) - pq(*early[["tp","fp","fn","iou"]].sum())
    diff = bl - be
    print(f"{lab:10s} n={len(late):3d}  PQ {pq(*late[['tp','fp','fn','iou']].sum()):.4f} vs "
          f"{pq(*early[['tp','fp','fn','iou']].sum()):.4f}   fark {dd:+.4f}  "
          f"CI [{np.percentile(diff,2.5):+.4f}, {np.percentile(diff,97.5):+.4f}]  "
          f"P(fark<0) {(diff<0).mean():.3f}")
print("\nPOPULASYON MU, ALAN MI? gec yillarin GT'si farkli mi:")
a = pd.read_parquet("outputs/eda/error_atlas.parquet")
a["year"] = a["stem"].str[:4].astype(int); a["gec"] = a["year"] >= 2021
g = a.groupby("gec").agg(kopya=("fid","size"), goruntu=("stem","nunique"), alan=("log_area","median"),
                         kontrast=("contrast","median"), genislik=("width","median"),
                         iskelet=("skel_len","median"), kalabalik=("crowding","median"),
                         r=("r_norm","median"), n_kopya=("n_copies","median"),
                         uyum=("agree_iou", lambda s:(s==0).mean()))
g["kopya/goruntu"] = (g["kopya"]/g["goruntu"]).round(2)
print(g.round(3).to_string())
print("\nsonuc okumasi: 'uyum' = hicbir ikinci anotatorun cizmedigi kopya orani")
