"""Annotator decision-structure mining for MAGFiLO Kaggle 2026.

1. Prefix semantics (GGBBAA = group / batch / annotator-slot), person mapping.
2. Cross-copy instance matching (IoU>0.3 union-find) on the 296 multi-copy files.
3. Chirality agreement, pair confusion matrix, Cohen kappa (L vs R), existence-vs-category.
4. Per-annotator fingerprints + generosity ranking + ANOVA on count residuals.
5. Support-prediction dataset + LogisticRegression / GradientBoosting AUC (GroupKFold by file).
6. Hemispheric chirality preference check.

RAM-lean: IoU is computed directly on compressed RLEs via pycocotools.mask.iou
(exact, no full-frame decode); validated once against filament.rle.rles_to_layers.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from filament.rle import rles_to_layers  # noqa: E402

DATA = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"
OUT = ROOT / "outputs" / "eda"
OUT.mkdir(parents=True, exist_ok=True)

CX, CY, R = 1023.5, 1023.5, 900.0  # confirmed fixed disk geometry
CAT_NAME = {1: "Left", 2: "Right", 3: "Unidentifiable"}
IOU_THR = 0.3

P = lambda *a: print(*a, flush=True)  # noqa: E731

summary: dict = {}

# ----------------------------------------------------------------------------
# Load artifacts
# ----------------------------------------------------------------------------
man = pd.read_parquet(OUT / "manifest_entries.parquet")
gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
morph = pd.read_parquet(OUT / "morphology_instances.parquet")
P(f"loaded manifest {man.shape}, gt {gt.shape}, morph {morph.shape}")

# fixed-geometry radial / latitude coordinates (recomputed, NOT the per-image fits)
morph["r_norm_fix"] = np.hypot(morph.centroid_x - CX, morph.centroid_y - CY) / R
dyn = np.clip((CY - morph.centroid_y) / R, -1, 1)  # + = north
morph["dy_norm_fix"] = dyn
morph["lat_deg_fix"] = np.degrees(np.arcsin(dyn))

# ----------------------------------------------------------------------------
# 1. Prefix semantics
# ----------------------------------------------------------------------------
P("\n=== 1. prefix semantics ===")
man["group"] = man.annotator_prefix.str[:2]
man["batch"] = man.annotator_prefix.str[2:4]
man["aslot"] = man.annotator_prefix.str[4:6]
man["person"] = man.group + "-" + man.aslot

copies = man.groupby("file_name").entry_id.count().rename("n_copies")
man = man.merge(copies, on="file_name")
multi_files = copies[copies > 1].index
mm = man[man.file_name.isin(multi_files)]

same4 = mm.groupby("file_name").annotator_prefix.apply(lambda s: s.str[:4].nunique() == 1)
distinct_slot = mm.groupby("file_name").annotator_prefix.apply(
    lambda s: s.str[4:6].nunique() == len(s))
summary["prefix"] = {
    "n_prefixes": int(man.annotator_prefix.nunique()),
    "n_groups": int(man.group.nunique()),
    "n_batches": int(man.batch.nunique()),
    "n_annotator_slots": int(man.aslot.nunique()),
    "n_persons_group_x_slot": int(man.person.nunique()),
    "n_group_batch_pools": int(man.annotator_prefix.str[:4].nunique()),
    "n_multi_copy_files": int(len(multi_files)),
    "pct_multi_files_sharing_first4": float(100 * same4.mean()),
    "pct_multi_files_distinct_slots": float(100 * distinct_slot.mean()),
    "copies_dist": {int(k): int(v) for k, v in copies.value_counts().sort_index().items()},
}
P(json.dumps(summary["prefix"], indent=1))
entries_per_person = man.groupby("person").entry_id.count().sort_values(ascending=False)
P(entries_per_person.to_string())

# ----------------------------------------------------------------------------
# 2. Cross-copy instance matching (union-find, IoU on RLE)
# ----------------------------------------------------------------------------
P("\n=== 2. cross-copy matching ===")
gt_multi = gt[gt.file_name.isin(multi_files)]
P(f"instances in multi-copy files: {len(gt_multi)}")


def find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


cluster_rows = []
validated = False
for fi, (fname, g) in enumerate(gt_multi.groupby("file_name", sort=True)):
    g = g.reset_index(drop=True)
    n = len(g)
    rle_dicts = [{"size": [2048, 2048], "counts": c.encode("ascii")}
                 for c in g.segmentation_rle]
    iou = mask_util.iou(rle_dicts, rle_dicts, [0] * n)  # (n, n) exact IoU
    if not validated:  # one-time cross-check vs full decode per task spec
        layers = rles_to_layers(list(g.segmentation_rle)).astype(bool)
        for a in range(min(n, 4)):
            for b in range(min(n, 4)):
                inter = np.logical_and(layers[a], layers[b]).sum()
                uni = np.logical_or(layers[a], layers[b]).sum()
                ref = inter / uni if uni else 0.0
                assert abs(ref - iou[a, b]) < 1e-6, (fname, a, b, ref, iou[a, b])
        del layers
        validated = True
        P(f"IoU validation vs rles_to_layers passed on {fname}")
    parent = list(range(n))
    ents = g.entry_id.values
    for a in range(n):
        for b in range(a + 1, n):
            if ents[a] != ents[b] and iou[a, b] > IOU_THR:
                ra, rb = find(parent, a), find(parent, b)
                if ra != rb:
                    parent[rb] = ra
    roots = [find(parent, i) for i in range(n)]
    n_cop = int(copies[fname])
    for k, r0 in enumerate(sorted(set(roots))):
        mem = [i for i in range(n) if roots[i] == r0]
        sub = g.iloc[mem]
        cluster_rows.append({
            "cluster_id": f"{fname.rsplit('.', 1)[0]}_c{k}",
            "file_name": fname,
            "n_copies_file": n_cop,
            "n_members": len(mem),
            "support": int(sub.entry_id.nunique()),
            "member_ann_ids": ",".join(sub.ann_id),
            "member_entry_ids": ",".join(sub.entry_id),
            "member_categories": ",".join(str(c) for c in sub.category_id),
            "member_areas": ",".join(f"{a:.0f}" for a in sub.area),
            "mean_pair_iou": float(np.mean([iou[a, b] for a in mem for b in mem
                                            if a < b and ents[a] != ents[b]]))
            if len(mem) > 1 else np.nan,
        })
    if (fi + 1) % 50 == 0:
        P(f"  {fi + 1}/{len(multi_files)} files")
clus = pd.DataFrame(cluster_rows)
clus.to_parquet(OUT / "annotator_clusters.parquet", index=False)
summary["matching"] = {
    "n_clusters": int(len(clus)),
    "n_singletons": int((clus.support == 1).sum()),
    "n_full_support": int((clus.support == clus.n_copies_file).sum()),
    "support_dist": {int(k): int(v) for k, v in clus.support.value_counts().sort_index().items()},
    "mean_pair_iou_matched": float(clus.mean_pair_iou.mean()),
}
P(json.dumps(summary["matching"], indent=1))

# ----------------------------------------------------------------------------
# 3a. Chirality agreement on matched clusters
# ----------------------------------------------------------------------------
P("\n=== 3. agreement ===")
conf = np.zeros((3, 3), dtype=int)  # unordered pair counts, cats 1..3
n_all_same = 0
n_ge2 = 0
for _, row in clus[clus.n_members >= 2].iterrows():
    cats = [int(c) for c in row.member_categories.split(",")]
    ents = row.member_entry_ids.split(",")
    cross = [(cats[a], cats[b]) for a in range(len(cats)) for b in range(a + 1, len(cats))
             if ents[a] != ents[b]]
    if not cross:
        continue
    n_ge2 += 1
    if len(set(cats)) == 1:
        n_all_same += 1
    for ca, cb in cross:
        i, j = sorted((ca - 1, cb - 1))
        conf[i, j] += 1

sym = conf + conf.T - np.diag(np.diag(conf))  # symmetrized (both orderings)
tot_pairs = int(conf.sum())
same_pairs = int(np.trace(conf))

# Cohen kappa on L-vs-R restricted pairs (symmetrized 2x2)
lr = sym[:2, :2].astype(float)
po = np.trace(lr) / lr.sum()
marg = lr.sum(0) / lr.sum()
pe = float((marg ** 2).sum())
kappa_lr = (po - pe) / (1 - pe)

# kappa over all 3 categories (pair-level, symmetrized)
s3 = sym.astype(float)
po3 = np.trace(s3) / s3.sum()
m3 = s3.sum(0) / s3.sum()
pe3 = float((m3 ** 2).sum())
kappa_all = (po3 - pe3) / (1 - pe3)

summary["agreement"] = {
    "n_clusters_ge2_members": int(n_ge2),
    "pct_all_same_category": float(100 * n_all_same / n_ge2),
    "pair_confusion_unordered_LRU": conf.tolist(),
    "n_pairs": tot_pairs,
    "pct_pairs_same_category": float(100 * same_pairs / tot_pairs),
    "cohen_kappa_LR_pairs": float(kappa_lr),
    "cohen_kappa_all3_pairs": float(kappa_all),
    "n_LR_pairs": int(lr.sum() / 2 + np.trace(lr) / 2),
    "pct_LR_pairs_agree": float(100 * po),
}
P(json.dumps(summary["agreement"], indent=1))

# 3b. existence disagreement vs category
maj_cat = []
for _, row in clus.iterrows():
    cats = [int(c) for c in row.member_categories.split(",")]
    maj_cat.append(Counter(cats).most_common(1)[0][0])
clus["majority_cat"] = maj_cat
clus["full_support"] = clus.support == clus.n_copies_file
tab = pd.crosstab(clus.majority_cat, clus.full_support)
chi2, p_exist, _, _ = stats.chi2_contingency(tab.values)
skip_rate = (1 - clus.groupby("majority_cat").full_support.mean()) * 100
skip_by_ncop = clus.groupby(["n_copies_file", "majority_cat"]).full_support \
    .agg(["mean", "count"])
summary["existence_vs_category"] = {
    "skip_rate_pct_by_cat": {CAT_NAME[k]: float(v) for k, v in skip_rate.items()},
    "chi2": float(chi2), "p_value": float(p_exist),
    "contingency_cat_x_full": tab.values.tolist(),
}
P(json.dumps(summary["existence_vs_category"], indent=1))
P(skip_by_ncop.to_string())
clus.to_parquet(OUT / "annotator_clusters.parquet", index=False)  # resave w/ cols

# ----------------------------------------------------------------------------
# 4. Per-annotator fingerprints
# ----------------------------------------------------------------------------
P("\n=== 4. fingerprints ===")
morph["prefix"] = morph.image_id.str.split("-").str[0]
morph["person"] = morph.prefix.str[:2] + "-" + morph.prefix.str[4:6]
morph["pts_per_karea"] = morph.n_poly_points / morph.poly_area * 1000

fp = man.groupby("person").agg(
    n_entries=("entry_id", "count"),
    mean_inst_per_entry=("num_annotations", "mean"),
    n_inst=("num_annotations", "sum"),
    n_unid=("n_cat_3_Unidentifiable", "sum"),
).reset_index()
fp["pct_unidentifiable"] = 100 * fp.n_unid / fp.n_inst
mstats = morph.groupby("person").agg(
    median_area=("poly_area", "median"),
    median_pts_per_karea=("pts_per_karea", "median"),
    median_width=("width_proxy", "median"),
    median_spine_len=("spine_len", "median"),
).reset_index()
fp = fp.merge(mstats, on="person")

# generosity: count residual vs file mean, multi-copy entries only
mm2 = man[man.n_copies > 1].copy()
fmean = mm2.groupby("file_name").num_annotations.transform("mean")
mm2["count_resid"] = mm2.num_annotations - fmean
gen = mm2.groupby("person").count_resid.agg(["mean", "count"]).rename(
    columns={"mean": "generosity_resid", "count": "n_multi_entries"})
fp = fp.merge(gen, on="person", how="left").sort_values(
    "generosity_resid", ascending=False)
fp.to_parquet(OUT / "annotator_fingerprints.parquet", index=False)
P(fp.to_string(index=False))

groups_r = [g.count_resid.values for _, g in mm2.groupby("person") if len(g) >= 5]
F, p_anova = stats.f_oneway(*groups_r)
summary["fingerprints"] = {
    "n_persons": int(len(fp)),
    "anova_count_resid_F": float(F), "anova_p": float(p_anova),
    "n_persons_in_anova": len(groups_r),
    "most_generous": fp.iloc[0].person,
    "most_generous_resid": float(fp.iloc[0].generosity_resid),
    "least_generous": fp.dropna(subset=["generosity_resid"]).iloc[-1].person,
    "least_generous_resid": float(fp.dropna(subset=["generosity_resid"]).iloc[-1].generosity_resid),
    "pct_unid_range": [float(fp.pct_unidentifiable.min()), float(fp.pct_unidentifiable.max())],
}
P(json.dumps(summary["fingerprints"], indent=1))

# ----------------------------------------------------------------------------
# 5. Support-prediction dataset + models
# ----------------------------------------------------------------------------
P("\n=== 5. support prediction ===")
# polygon perimeter -> boundary complexity, from COCO json (one pass, then freed)
P("computing polygon perimeters from COCO json ...")
with open(DATA / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json") as f:
    coco = json.load(f)
perim = {}
for a in coco["annotations"]:
    tot = 0.0
    for part in a["segmentation"]:
        xy = np.asarray(part, dtype=np.float64).reshape(-1, 2)
        d = np.diff(np.vstack([xy, xy[:1]]), axis=0)
        tot += float(np.hypot(d[:, 0], d[:, 1]).sum())
    perim[a["id"]] = tot
del coco
P(f"perimeters computed for {len(perim)} annotations")

morph["perimeter"] = morph.ann_id.map(perim)
morph["complexity"] = morph.perimeter ** 2 / (4 * math.pi * morph.poly_area)

mfeat = morph.set_index("ann_id")[
    ["poly_area", "width_proxy", "spine_len", "r_norm_fix", "lat_deg_fix",
     "complexity", "eccentricity", "category_id"]]

rows = []
for _, row in clus.iterrows():
    ids = row.member_ann_ids.split(",")
    sub = mfeat.loc[[i for i in ids if i in mfeat.index]]
    if sub.empty:
        continue
    rows.append({
        "cluster_id": row.cluster_id, "file_name": row.file_name,
        "n_copies_file": row.n_copies_file, "support": row.support,
        "support_frac": row.support / row.n_copies_file,
        "target_consensus": int(row.support >= 2),
        "area": float(sub.poly_area.mean()),
        "log10_area": float(np.log10(sub.poly_area.mean())),
        "width_proxy": float(sub.width_proxy.mean()),
        "spine_len": float(sub.spine_len.mean()),
        "r_norm": float(sub.r_norm_fix.mean()),
        "abs_lat_deg": float(sub.lat_deg_fix.abs().mean()),
        "complexity": float(sub.complexity.mean()),
        "eccentricity": float(sub.eccentricity.mean()),
        "mean_category": float(sub.category_id.mean()),
        "frac_unidentifiable": float((sub.category_id == 3).mean()),
    })
ds = pd.DataFrame(rows)
ds.to_parquet(OUT / "support_prediction_dataset.parquet", index=False)
P(f"dataset: {ds.shape}, consensus rate {ds.target_consensus.mean():.3f}")

from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

FEATS = ["log10_area", "width_proxy", "spine_len", "r_norm", "abs_lat_deg",
         "complexity", "eccentricity", "mean_category", "frac_unidentifiable",
         "n_copies_file"]
X = ds[FEATS].fillna(ds[FEATS].median()).values
y = ds.target_consensus.values
files = ds.file_name.values
gkf = GroupKFold(n_splits=5)
oof = {"lr": np.zeros(len(y)), "gb": np.zeros(len(y))}
fold_auc = {"lr": [], "gb": []}
for tr, te in gkf.split(X, y, files):
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                    learning_rate=0.05, random_state=0)
    lr.fit(X[tr], y[tr]); gb.fit(X[tr], y[tr])
    oof["lr"][te] = lr.predict_proba(X[te])[:, 1]
    oof["gb"][te] = gb.predict_proba(X[te])[:, 1]
    fold_auc["lr"].append(roc_auc_score(y[te], oof["lr"][te]))
    fold_auc["gb"].append(roc_auc_score(y[te], oof["gb"][te]))
    P(f"fold done lr={fold_auc['lr'][-1]:.3f} gb={fold_auc['gb'][-1]:.3f}")

m2, m3 = ds.n_copies_file.values == 2, ds.n_copies_file.values == 3
res_models = {}
for name in ("lr", "gb"):
    res_models[name] = {
        "auc_oof": float(roc_auc_score(y, oof[name])),
        "auc_fold_mean": float(np.mean(fold_auc[name])),
        "auc_fold_std": float(np.std(fold_auc[name])),
        "auc_2copy": float(roc_auc_score(y[m2], oof[name][m2])),
        "auc_3copy": float(roc_auc_score(y[m3], oof[name][m3])),
    }
ds["oof_consensus_prob_gb"] = oof["gb"]
ds["oof_consensus_prob_lr"] = oof["lr"]
ds.to_parquet(OUT / "support_prediction_dataset.parquet", index=False)

gb_full = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                     learning_rate=0.05, random_state=0).fit(X, y)
lr_full = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000)).fit(X, y)
imp = sorted(zip(FEATS, gb_full.feature_importances_), key=lambda t: -t[1])
coefs = sorted(zip(FEATS, lr_full[-1].coef_[0]), key=lambda t: -abs(t[1]))
summary["support_model"] = {
    "n_clusters": int(len(ds)), "consensus_rate": float(y.mean()),
    **{f"{k}_{m}": v for k, d in res_models.items() for m, v in d.items()},
    "gb_importances": {k: float(v) for k, v in imp},
    "lr_std_coefs": {k: float(v) for k, v in coefs},
}
P(json.dumps(summary["support_model"], indent=1))

# ----------------------------------------------------------------------------
# 6. Hemispheric chirality preference
# ----------------------------------------------------------------------------
P("\n=== 6. hemispheric preference ===")
ch = morph[morph.category_id.isin([1, 2])].copy()
ch["hemi"] = np.where(ch.dy_norm_fix > 0, "north", "south")
tabh = pd.crosstab(ch.hemi, ch.category_id)
tabh.columns = [CAT_NAME[c] for c in tabh.columns]
chi2h, ph, _, _ = stats.chi2_contingency(tabh.values)
prop = (tabh.T / tabh.sum(1)).T * 100
summary["hemisphere"] = {
    "counts": {h: {c: int(v) for c, v in r.items()} for h, r in tabh.iterrows()},
    "pct_left_north": float(prop.loc["north", "Left"]),
    "pct_left_south": float(prop.loc["south", "Left"]),
    "chi2": float(chi2h), "p_value": float(ph),
}
P(json.dumps(summary["hemisphere"], indent=1))

with open(OUT / "annotator_structure.json", "w") as f:
    json.dump(summary, f, indent=1)

# ----------------------------------------------------------------------------
# markdown report
# ----------------------------------------------------------------------------
s = summary
md = f"""# Annotator decision structure (MAGFiLO Kaggle 2026 subset)

## 1. Entry-id prefix semantics
Prefix `GGBBAA`: **GG = group** (01-06 in this subset, paper has 24), **BB = batch**
(01-04), **AA = annotator slot** within group (01-03). Evidence: all
{s['prefix']['n_multi_copy_files']} multi-copy files share GG+BB across copies
({s['prefix']['pct_multi_files_sharing_first4']:.0f}%) and copies always carry distinct
AA slots ({s['prefix']['pct_multi_files_distinct_slots']:.0f}%).
{s['prefix']['n_prefixes']} prefixes -> {s['prefix']['n_persons_group_x_slot']} distinct
annotators (group x slot), {s['prefix']['n_group_batch_pools']} group-batch pools.
Copies per file: {s['prefix']['copies_dist']}.

Entries per annotator:
{entries_per_person.to_string()}

## 2. Cross-copy instance matching (IoU>{IOU_THR}, union-find, per file)
{s['matching']['n_clusters']} clusters on {s['prefix']['n_multi_copy_files']} files;
support distribution {s['matching']['support_dist']};
{s['matching']['n_full_support']} full-support, {s['matching']['n_singletons']} singletons.
Mean pairwise IoU among matched members: {s['matching']['mean_pair_iou_matched']:.3f}.
-> `annotator_clusters.parquet`

## 3. Agreement
- Clusters with >=2 members: {s['agreement']['n_clusters_ge2_members']};
  **{s['agreement']['pct_all_same_category']:.1f}% all-same-category**.
- Pair-level (n={s['agreement']['n_pairs']}): {s['agreement']['pct_pairs_same_category']:.1f}% same category.
- Unordered pair confusion (rows/cols = L,R,U): {s['agreement']['pair_confusion_unordered_LRU']}
- **Cohen kappa L-vs-R pairs: {s['agreement']['cohen_kappa_LR_pairs']:.3f}**
  (paper reports 0.66 overall); all-3-category pair kappa {s['agreement']['cohen_kappa_all3_pairs']:.3f}.
- Existence vs category: skip rate (support < n_copies) by majority category:
  {s['existence_vs_category']['skip_rate_pct_by_cat']} (chi2={s['existence_vs_category']['chi2']:.1f},
  p={s['existence_vs_category']['p_value']:.2e}) -> Unidentifiable are skipped far more often.

## 4. Annotator fingerprints (-> `annotator_fingerprints.parquet`)
{fp[['person','n_entries','mean_inst_per_entry','pct_unidentifiable','median_area',
     'median_pts_per_karea','median_width','generosity_resid','n_multi_entries']].to_string(index=False)}

Most generous (count residual within shared files): {s['fingerprints']['most_generous']}
({s['fingerprints']['most_generous_resid']:+.2f} inst/entry), least:
{s['fingerprints']['least_generous']} ({s['fingerprints']['least_generous_resid']:+.2f}).
ANOVA on per-entry count residuals across annotators: F={s['fingerprints']['anova_count_resid_F']:.2f},
p={s['fingerprints']['anova_p']:.2e} -> annotator identity is a significant factor.

## 5. Support prediction (-> `support_prediction_dataset.parquet`)
Target: consensus (support>=2) vs singleton; {s['support_model']['n_clusters']} clusters,
consensus rate {s['support_model']['consensus_rate']:.3f}. 5-fold GroupKFold by file:
- LogisticRegression AUC {s['support_model']['lr_auc_oof']:.3f}
  (2-copy {s['support_model']['lr_auc_2copy']:.3f}, 3-copy {s['support_model']['lr_auc_3copy']:.3f})
- GradientBoosting AUC {s['support_model']['gb_auc_oof']:.3f}
  (2-copy {s['support_model']['gb_auc_2copy']:.3f}, 3-copy {s['support_model']['gb_auc_3copy']:.3f})
- GB importances: {json.dumps(s['support_model']['gb_importances'])}
- LR standardized coefs: {json.dumps(s['support_model']['lr_std_coefs'])}

## 6. Hemispheric chirality preference (L/R instances, fixed geometry)
Counts: {s['hemisphere']['counts']}. Left fraction north {s['hemisphere']['pct_left_north']:.1f}%
vs south {s['hemisphere']['pct_left_south']:.1f}% (chi2={s['hemisphere']['chi2']:.1f},
p={s['hemisphere']['p_value']:.2e}).
"""
with open(OUT / "annotator_structure.md", "w", encoding="utf-8") as f:
    f.write(md)
P("\nwrote annotator_structure.md / .json + parquets. DONE")
