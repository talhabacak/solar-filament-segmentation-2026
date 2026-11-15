"""E165 -- what each selector feature is worth, measured out of fold.

The selector is a two-stage HistGradientBoosting pair (p(match) x E[iou_sum | match])
over 42 candidate features. This measures, per feature and per family, two things:

  permutation drop   fit on four folds, then on the held-out fold permute one
                     feature across candidates and re-score; the fall in AUC of the
                     classifier and in the rank correlation of the regressor is
                     that feature's out-of-fold contribution GIVEN the others
  family drop        the same with a whole family permuted together, which is the
                     number that survives when features inside a family are
                     redundant with each other (a single feature's drop can be
                     near zero while its family's is large)

Usage: python src/eda/selector_feature_importance.py
"""
import sys, json, pathlib, numpy as np, pandas as pd, joblib
ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in ("src", "src/eda", "src/models"): sys.path.insert(0, str(ROOT / p))
from e115c_threshold_ci import make_cls, make_reg
from extra_features import EXTRA, PHOTO, ORI
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
OUT = ROOT / "outputs" / "eda"
FAMILIES = {
    "lane & confidence": ["conf", "det_conf", "sem_conf", "from_det", "from_sem", "n_sources", "rank_conf",
                          "n_det_models", "det_conf_mean", "det_conf_std", "det_conf_max",
                          "member_iou_mean", "member_iou_min"],
    "semantic field": ["sem_prob_mean", "sem_prob_max", "sem_prob_p90", "sem_frac_over_50",
                       "sem_ring_mean", "sem_contrast"],
    "size & shape": ["log10_area", "bbox_aspect", "eccentricity", "boundary_complexity", "skel_len", "width_proxy"],
    "position": ["r_norm", "abs_lat"],
    "image context": ["n_cands_img", "nn_dist", "n_within_200px", "img_cand_density", "img_sem_area_frac"],
    "photometry (E123/E125)": list(PHOTO),
    "orientation (E123/E125)": list(ORI),
}
def main():
    feats = list(joblib.load(ROOT / "outputs/selector/selector_v5_shrunkblob5prodts.joblib")["features"]) + list(EXTRA)
    fam = {}
    for k, v in FAMILIES.items():
        for f in v:
            if f in feats: fam[f] = k
    for f in feats:
        fam.setdefault(f, "other")
    a = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    a = a.merge(pd.read_parquet(OUT / "extra_features_t020.parquet")[["cand_id"] + EXTRA], on="cand_id", how="left")
    folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    a["fold"] = a["stem"].map(fold_of).astype(int)
    rng = np.random.default_rng(0)
    drop_cls = {f: [] for f in feats}; drop_reg = {f: [] for f in feats}
    fdrop_cls = {k: [] for k in FAMILIES}; fdrop_reg = {k: [] for k in FAMILIES}
    base_auc, base_rho = [], []
    for F in range(5):
        tr, te = a[a["fold"] != F], a[a["fold"] == F]
        X, y, v = tr[feats].to_numpy(np.float64), tr["y"].to_numpy(), tr["iou_sum"].to_numpy(np.float64)
        cls = make_cls().fit(X, y); reg = make_reg().fit(X[y == 1], v[y == 1])
        Xt, yt, vt = te[feats].to_numpy(np.float64), te["y"].to_numpy(), te["iou_sum"].to_numpy(np.float64)
        pos = yt == 1
        auc0 = roc_auc_score(yt, cls.predict_proba(Xt)[:, 1]); rho0 = spearmanr(reg.predict(Xt[pos]), vt[pos])[0]
        base_auc.append(auc0); base_rho.append(rho0)
        def score_with(cols):
            Xp = Xt.copy()
            for c in cols:
                j = feats.index(c); Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
            return (auc0 - roc_auc_score(yt, cls.predict_proba(Xp)[:, 1]),
                    rho0 - spearmanr(reg.predict(Xp[pos]), vt[pos])[0])
        for f in feats:
            d1, d2 = score_with([f]); drop_cls[f].append(d1); drop_reg[f].append(d2)
        for k, v_ in FAMILIES.items():
            cols = [c for c in v_ if c in feats]
            if cols:
                d1, d2 = score_with(cols); fdrop_cls[k].append(d1); fdrop_reg[k].append(d2)
        print(f"fold {F}: AUC {auc0:.4f}  rho {rho0:.4f}", flush=True)
    rows = [dict(feature=f, family=fam[f], auc_drop=float(np.mean(drop_cls[f])), rho_drop=float(np.mean(drop_reg[f]))) for f in feats]
    df = pd.DataFrame(rows).sort_values("auc_drop", ascending=False)
    df.to_csv(OUT / "selector_feature_importance.csv", index=False)
    print(f"\nbaseline OOF: AUC {np.mean(base_auc):.4f}, regressor Spearman {np.mean(base_rho):.4f}\n")
    print("per feature (mean OOF drop when permuted):")
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    fd = pd.DataFrame([dict(family=k, n=len([c for c in v_ if c in feats]), auc_drop=float(np.mean(fdrop_cls[k])), rho_drop=float(np.mean(fdrop_reg[k]))) for k, v_ in FAMILIES.items() if fdrop_cls[k]]).sort_values("auc_drop", ascending=False)
    fd.to_csv(OUT / "selector_family_importance.csv", index=False)
    print("\nper family (whole family permuted):")
    print(fd.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    json.dump(dict(base_auc=float(np.mean(base_auc)), base_rho=float(np.mean(base_rho))), open(OUT / "selector_feature_importance.json", "w"), indent=1)
if __name__ == "__main__":
    main()
