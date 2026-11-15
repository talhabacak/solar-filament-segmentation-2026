"""Run the existing cascade experiment with exact, reusable match records.

Only prediction selection changes, so IoUs and >0.5 hits remain invariant.
FP/FN count unique matched IDs, including possible many-to-many hits.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
import time
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for part in ("src", "src/eda", "src/models"):
    sys.path.insert(0, str(ROOT / part))
from filament.pq import score as exact


def run_image(args):
    gt, pred = args
    return exact(gt, pred, return_details=True)["matches"]


def main():
    start = time.monotonic()
    pool = pd.read_parquet(ROOT / "outputs/eda/fused_pool_all_blob5_t020_labelled.parquet")
    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet")
    gt["stem"] = gt.file_name.str.replace(".jpeg", "", regex=False)
    gt = gt[gt.stem.isin(pool.stem)]
    gtcols = ["filament_id", "segmentation_rle"]
    pred = pool[["cand_id", "rle"]].rename(columns=dict(cand_id="filament_id", rle="segmentation_rle"))
    jobs = [(g[gtcols], pred[pool.stem == stem]) for stem, g in gt.groupby("stem")]
    matches = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        for i, m in enumerate(executor.map(run_image, jobs), 1):
            if len(m):
                matches.append(m)
            if i % 100 == 0:
                print(f"cached {i}/{len(jobs)} images", flush=True)
    hits = pd.concat(matches, ignore_index=True)
    hits.to_parquet(ROOT / "outputs/eda/sub042_exact_matches.parquet", index=False)

    def cached(g, p, return_details=False, **kwargs):
        entries = g.filament_id.str.split("_", n=1).str[0]
        pids = set(p.filament_id)
        h = hits[hits.gt_filament_id.isin(g.filament_id) & hits.pred_filament_id.isin(pids)]
        unique_entries = entries.unique()
        stem_counts = p.filament_id.str.split("_", n=1).str[0].value_counts()
        total_predictions = sum(int(stem_counts.get(e.split("-", 1)[1], 0)) for e in unique_entries)
        tp = len(h)
        fp = total_predictions - len(h[["annotator_image", "pred_filament_id"]].drop_duplicates())
        fn = len(g) - h.gt_filament_id.nunique()
        iou = float(h.iou.sum())
        den = tp + .5 * (fp + fn)
        out = dict(pq=iou / den if den else 0., sq=iou / tp if tp else 0., rq=tp / den if den else 0., tp=tp, fp=fp, fn=fn)
        if return_details:
            out["matches"] = h.copy()
        return out

    # Direct parity on multiple selection policies, using the original scorer.
    small = jobs[:8]
    g = pd.concat([j[0] for j in small])
    p = pd.concat([j[1] for j in small])
    for frac in (0., .2, .5, 1.):
        selected = p.sample(frac=frac, random_state=42)
        a, b = exact(g, selected), cached(g, selected)
        for key in a:
            assert np.isclose(a[key], b[key], rtol=0, atol=1e-12), (frac, key, a, b)
    print(f"exact subset parity passed; cache took {time.monotonic()-start:.1f}s", flush=True)
    import band_cascade
    import e115c_threshold_ci
    band_cascade.score = cached
    e115c_threshold_ci.score = cached
    band_cascade.main("bag5_oof_scores.parquet", "_bag5_cached")
    # The submitted combination retains sub037's pre-existing q=.85 rule.
    # Measure that exact fixed policy, independently of the nested-q diagnostic.
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from extra_features import EXTRA
    outdir = ROOT / "outputs/eda"
    a = pool.merge(pd.read_parquet(outdir / "extra_features_t020.parquet")[["cand_id"] + EXTRA], on="cand_id", validate="one_to_one")
    a = a.merge(pd.read_parquet(outdir / "bag5_oof_scores.parquet"), on="cand_id", validate="one_to_one")
    splits = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
    fold_of = dict(zip(splits.file_name.str.replace(".jpeg", "", regex=False), splits.fold))
    a["fold"] = a.stem.map(fold_of)
    feats = joblib.load(ROOT / "outputs/selector/selector_t020pox_bag5.joblib")["features"]
    keep = a[(a.conf >= .4) & (a.p >= .6)]
    band = a[(a.conf >= .4) & (a.p >= .4) & (a.p < .6)]
    additions = []
    for k in range(5):
        tr, te = band[band.fold != k], band[band.fold == k]
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=.05, max_leaf_nodes=31,
                                          min_samples_leaf=25, l2_regularization=1., random_state=42)
        m.fit(tr[feats].to_numpy(np.float64), tr.y.to_numpy())
        s = m.predict_proba(te[feats].to_numpy(np.float64))[:, 1]
        additions.append(te[s >= np.quantile(s, .85)])
    trial = pd.concat([keep] + additions)
    stems = np.array(sorted(a.stem.unique()))
    ncopy = gt.groupby("stem").entry_id.nunique().reindex(stems).to_numpy()
    stats0, base = e115c_threshold_ci.per_image(keep, gt, stems, ncopy)
    stats1, result = e115c_threshold_ci.per_image(trial, gt, stems, ncopy)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(4000):
        idx = rng.integers(0, len(stems), len(stems))
        vals = [e115c_threshold_ci.pq(*(s[key][idx].sum() for key in ("tp", "fp", "fn", "iou"))) for s in (stats0, stats1)]
        boot.append(vals[1] - vals[0])
    by_fold = []
    for k in range(5):
        gk = gt[gt.stem.map(fold_of) == k][gtcols]
        vals = [cached(gk, df[df.fold == k][["cand_id", "rle"]].rename(columns=dict(cand_id="filament_id", rle="segmentation_rle"))) for df in (keep, trial)]
        by_fold.append(dict(fold=k, base=vals[0], cascade=vals[1], delta=vals[1]["pq"]-vals[0]["pq"]))
    summary = dict(experiment_id="sub042_bag5_cascade085", q=.85,
                   base={k:v for k,v in base.items() if k != "matches"},
                   cascade={k:v for k,v in result.items() if k != "matches"},
                   delta=result["pq"]-base["pq"], ci=np.percentile(boot, [2.5,97.5]).tolist(),
                   folds=by_fold, admitted=len(trial)-len(keep), seconds=time.monotonic()-start)
    (outdir / "sub042_fixed085_eval.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"total seconds {time.monotonic()-start:.1f}", flush=True)


if __name__ == "__main__":
    main()
