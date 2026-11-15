"""E157 -- how much can the private half differ from the public one, by chance alone?

The test set is 180 images and the public leaderboard is about half of it, so the
private score is computed on roughly 90 images we never see. Talha's question is
whether we can blow up there.

The answer is a variance question and it can be measured directly, because our
out-of-fold predictions cover 703 images scored by the same scorer. Repeatedly
splitting those into random halves of 90 and scoring each half gives the spread a
90-image PQ carries, and repeating it for two of our submissions gives the spread
of the DIFFERENCE, which is what actually decides a rank.

Two quantities are reported:

  spread of one arm's PQ over 90-image draws        how far public and private can
                                                    sit apart for the same model
  spread of the difference between two arms         whether a real +0.004 change
                                                    survives being measured on 90

The second is the one that matters: a change smaller than that spread is a coin
toss on the private board no matter how well it is measured locally.
"""
import sys, json, pathlib, numpy as np, pandas as pd
ROOT = pathlib.Path(".").resolve()
for p in ("src", "src/eda", "src/models"): sys.path.insert(0, str(ROOT / p))
from e115c_threshold_ci import per_image, pq
OUT = ROOT / "outputs" / "eda"
GATE = (0.40, 0.60)
a = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
a = a.merge(pd.read_parquet(OUT / "t020pox_oof_scores.parquet"), on="cand_id", how="left")
gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet")
gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
gt = gt[gt["stem"].isin(set(a["stem"]))]
stems = np.array(sorted(set(a["stem"])))
ncopy = gt.groupby("stem")["entry_id"].nunique().reindex(stems).fillna(1).to_numpy()
keep = a[(a["conf"] >= GATE[0]) & (a["p"] >= GATE[1])]
st = per_image(keep[["cand_id", "rle", "stem"]], gt, stems, ncopy)[0]
# a slightly different arm: the same selector one notch down, which is the size of
# change this project actually argues about
keep2 = a[(a["conf"] >= GATE[0]) & (a["p"] >= 0.58)]
st2 = per_image(keep2[["cand_id", "rle", "stem"]], gt, stems, ncopy)[0]
full = pq(st["tp"].sum(), st["fp"].sum(), st["fn"].sum(), st["iou"].sum())
full2 = pq(st2["tp"].sum(), st2["fp"].sum(), st2["fn"].sum(), st2["iou"].sum())
print(f"on all {len(stems)} images: arm A {full:.4f}, arm B {full2:.4f}, difference {full2-full:+.4f}")
rng = np.random.default_rng(0)
for n in (90, 180):
    va, vd = [], []
    for _ in range(4000):
        i = rng.choice(len(stems), n, replace=False)
        x = pq(st["tp"][i].sum(), st["fp"][i].sum(), st["fn"][i].sum(), st["iou"][i].sum())
        y = pq(st2["tp"][i].sum(), st2["fp"][i].sum(), st2["fn"][i].sum(), st2["iou"][i].sum())
        va.append(x); vd.append(y - x)
    va, vd = np.array(va), np.array(vd)
    print(f"\n{n}-image draws:")
    print(f"  one arm's PQ:     sd {va.std():.4f}, 95% range [{np.percentile(va,2.5):.4f}, "
          f"{np.percentile(va,97.5):.4f}] around {full:.4f}")
    print(f"  the difference:   sd {vd.std():.4f}, 95% range [{np.percentile(vd,2.5):+.4f}, "
          f"{np.percentile(vd,97.5):+.4f}] around {full2-full:+.4f}")
    for d in (0.003, 0.005, 0.010, 0.020):
        print(f"    a true +{d:.3f} change is measured with the right SIGN on {n} images "
              f"{100*(1 - 0.5*(1 + np.math.erf(-d/(vd.std()*np.sqrt(2))))):.0f}% of the time"
              if False else f"    a true +{d:.3f} change: P(seen positive on {n} images) "
              f"= {100*np.mean(np.random.default_rng(1).normal(d, vd.std(), 20000) > 0):.0f}%")
json.dump(dict(full=float(full), n_images=len(stems)), open(OUT / "private_lb_risk.json", "w"), indent=1)
