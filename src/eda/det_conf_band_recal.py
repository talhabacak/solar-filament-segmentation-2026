"""E121 -- band-wise recalibration of detector confidence for the selector.

Ledger §16d: at r 0.8-0.9 the held-out detector proposes the filaments almost
as often as on the disc (recall@0.05 74% vs 78%, same mask IoU) but with
lower confidence (median 0.52 vs 0.61); §16c: the selector's rejected-but-
true limb candidates carry a detector member at half the emitted confidence.
So the same raw det_conf does not mean the same P(match) at the limb, and the
selector -- trained mostly on disc candidates -- reads limb detections as
weaker than they are.

Single variable: the selector's three detector-confidence features (det_conf,
det_conf_mean, det_conf_max) are replaced by their OUT-OF-FOLD, BAND-WISE
isotonic recalibration: within each outer fold, an isotonic regression
det_conf -> P(y=1) is fit per radius band (<=0.5, 0.5-0.8, 0.8-0.9, >0.9) on
the TRAINING folds' candidates only, and applied to both training and
held-out candidates. Everything else is E115b's recipe (nested reference,
pinned seeds, same labelling, same GT scope). Control = the untouched 32
features on the same folds.

PRE-REGISTERED: gate = mean Δ over folds >= +0.002 AND >= 4/5 folds
positive; paired image bootstrap for the interval (±0.005 equivalence band);
mechanism check = reach at r 0.8-0.9 must rise (that is where the detector
sees-but-underscores); a gain anywhere else without the limb moving means
the story is wrong and the result is recorded as such.

Usage: python src/eda/det_conf_band_recal.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "eda"))

from filament.pq import score
from filament.rle import decode_rle
from e115c_threshold_ci import per_image, pq, make_cls, make_reg

OUT = ROOT / "outputs" / "eda"
BUNDLE = ROOT / "outputs" / "selector" / "selector_v5_shrunkblob5prodts.joblib"
DISK_C, DISK_R = 1023.5, 900.0
DET = ["det_conf", "det_conf_mean", "det_conf_max"]
EDGES = [0.0, 0.5, 0.8, 0.9, 1.01]
FLOOR, EQUIV_BAND, REPS = 0.002, 0.005, 4000


def log(*a):
    print(*a, flush=True)


def band_of(r):
    return np.digitize(r, EDGES[1:-1])


def recal_fit(cand_tr):
    """Per band, isotonic det_conf -> P(y); returns dict band -> model."""
    from sklearn.isotonic import IsotonicRegression
    models = {}
    b = band_of(cand_tr["r_norm"].to_numpy())
    for k in range(len(EDGES) - 1):
        sel = b == k
        if sel.sum() < 50:
            models[k] = None
            continue
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(cand_tr.loc[sel, "det_conf"].to_numpy(), cand_tr.loc[sel, "y"].to_numpy())
        models[k] = ir
    return models


def recal_apply(cand, models, col):
    b = band_of(cand["r_norm"].to_numpy())
    x = cand[col].to_numpy()
    out = x.copy()
    for k, ir in models.items():
        sel = b == k
        if ir is not None and sel.any():
            out[sel] = ir.predict(x[sel])
    return out


def oof_scores_recal(cand, feats, fold, recal):
    """E115b's nested OOF scoring; if `recal`, the det features are replaced
    per outer fold by their band-wise isotonic recalibration fit on the
    training folds only (inner folds refit their own)."""
    y = cand["y"].to_numpy()
    v = cand["iou_sum"].to_numpy(np.float64)
    p = np.zeros(len(cand))

    def X_of(idx_fit, idx_apply):
        df = cand.copy()
        if recal:
            models = recal_fit(cand.iloc[idx_fit])
            for c in DET:
                df[c] = recal_apply(df, models, c)
        return df[feats].to_numpy(np.float64)[idx_apply]

    for f in range(5):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        inner = np.zeros(len(tr))
        for g in [k for k in range(5) if k != f]:
            itr = np.where((fold != f) & (fold != g))[0]
            msk = fold[tr] == g
            ite = tr[msk]
            Xi_tr, Xi_te = X_of(itr, itr), X_of(itr, ite)
            ci = make_cls().fit(Xi_tr, y[itr])
            ri = make_reg().fit(Xi_tr[y[itr] == 1], v[itr][y[itr] == 1])
            inner[msk] = ci.predict_proba(Xi_te)[:, 1] * np.clip(ri.predict(Xi_te), 0, None)
        ref = np.sort(inner)
        X_tr, X_te = X_of(tr, tr), X_of(tr, te)
        c = make_cls().fit(X_tr, y[tr])
        r = make_reg().fit(X_tr[y[tr] == 1], v[tr][y[tr] == 1])
        raw = c.predict_proba(X_te)[:, 1] * np.clip(r.predict(X_te), 0, None)
        p[te] = np.searchsorted(ref, raw, side="right") / len(ref)
        log(f"    fold {f} scored ({'recal' if recal else 'control'})")
    return p


def exact_at(cand, gt, pcol, fold=None):
    sub = cand[(cand["conf"] >= 0.40) & (cand[pcol] >= 0.60)]
    g = gt
    if fold is not None:
        sub, g = sub[sub["fold"] == fold], gt[gt["fold"] == fold]
    pred = sub[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
    return score(g[["filament_id", "segmentation_rle"]], pred)


def main():
    import joblib
    feats = list(joblib.load(BUNDLE)["features"])
    cand = pd.read_parquet(OUT / "e115b_labelled_pool.parquet")
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    cand["fold"] = cand["stem"].map(fold_of).astype(int)
    gt["fold"] = gt["stem"].map(fold_of)
    gt = gt[gt["fold"].notna()].copy()
    gt["fold"] = gt["fold"].astype(int)
    fold = cand["fold"].to_numpy()

    # how different is the calibration by band? (fit on everything, for reading only)
    m_all = recal_fit(cand)
    for k in range(4):
        ir = m_all[k]
        if ir is not None:
            log(f"  band {k} [{EDGES[k]},{EDGES[k+1]}): P(match | det_conf=0.3) {ir.predict([0.3])[0]:.3f}, "
                f"P(match | 0.5) {ir.predict([0.5])[0]:.3f}, P(match | 0.7) {ir.predict([0.7])[0]:.3f}")

    pc = OUT / "e115b_p_oof.parquet"
    if pc.exists():
        cand["p_ctrl"] = cand["cand_id"].map(pd.read_parquet(pc).set_index("cand_id")["p_oof"]).to_numpy()
        log("control OOF scores from cache")
    else:
        log("control OOF selector:")
        cand["p_ctrl"] = oof_scores_recal(cand, feats, fold, recal=False)
    log("treatment OOF selector (band-recalibrated detector confidence):")
    cand["p_rec"] = oof_scores_recal(cand, feats, fold, recal=True)

    res = {}
    log(f"\n  {'fold':<5}{'ctrl PQ':>9}{'recal PQ':>9}{'Δ':>9}{'TP':>11}{'FP':>11}")
    d = []
    for f in range(5):
        c = exact_at(cand, gt, "p_ctrl", f)
        t = exact_at(cand, gt, "p_rec", f)
        d.append(t["pq"] - c["pq"])
        res[f"fold{f}"] = dict(ctrl=c["pq"], recal=t["pq"], delta=t["pq"] - c["pq"], tp=(c["tp"], t["tp"]), fp=(c["fp"], t["fp"]))
        log(f"  {f:<5}{c['pq']:>9.4f}{t['pq']:>9.4f}{t['pq']-c['pq']:>+9.4f}{c['tp']:>5}>{t['tp']:<5}{c['fp']:>5}>{t['fp']:<5}")
    dm, pos = float(np.mean(d)), sum(x > 0 for x in d)
    verdict = "GO (>= +0.002, >= 4/5): submission candidate, ask Talha" if (dm >= FLOOR and pos >= 4) else "NO GO"
    log(f"  mean Δ {dm:+.4f}, folds positive {pos}/5 -> {verdict}")

    Rg = []
    for rle in gt["segmentation_rle"]:
        m = decode_rle(rle).astype(np.uint8); ys, xs = np.nonzero(m)
        Rg.append(float(np.hypot(xs.mean() - DISK_C, ys.mean() - DISK_C) / DISK_R) if len(xs) else np.nan)
    gt["r"] = Rg
    log("\n  reach by radius band (mechanism check: 0.8-0.9 must rise):")
    for pcol, name in (("p_ctrl", "ctrl"), ("p_rec", "recal")):
        sub = cand[(cand["conf"] >= 0.40) & (cand[pcol] >= 0.60)]
        pred = sub[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
        r = score(gt[["filament_id", "segmentation_rle"]], pred, return_details=True)
        hit = set(r["matches"]["gt_filament_id"])
        parts = []
        for lo, hi in [(0, 0.5), (0.5, 0.8), (0.8, 0.9), (0.9, 1.01)]:
            g = gt[(gt["r"] > lo) & (gt["r"] <= hi)]
            parts.append(f"({lo},{hi}] {100*g['filament_id'].isin(hit).mean():.1f}%")
        log(f"    {name}: " + "  ".join(parts) + f"   pooled PQ {r['pq']:.4f}  TP {r['tp']} FP {r['fp']}")
        res[f"reach_{name}"] = parts

    stems = np.array(sorted(cand["stem"].unique()))
    ncopy = gt.groupby("stem")["entry_id"].nunique().reindex(stems).fillna(1).to_numpy()
    parts = {}
    for pcol in ("p_ctrl", "p_rec"):
        sub = cand[(cand["conf"] >= 0.40) & (cand[pcol] >= 0.60)]
        st, r = per_image(sub, gt, stems, ncopy)
        if abs(pq(st["tp"].sum(), st["fp"].sum(), st["fn"].sum(), st["iou"].sum()) - r["pq"]) > 5e-4:
            parts = None; log("  guard failed; no interval"); break
        parts[pcol] = st
    if parts:
        rng = np.random.default_rng(0); n = len(stems); boot = np.empty(REPS)
        for i in range(REPS):
            idx = rng.integers(0, n, n)
            vals = [pq(s["tp"][idx].sum(), s["fp"][idx].sum(), s["fn"][idx].sum(), s["iou"][idx].sum()) for s in (parts["p_ctrl"], parts["p_rec"])]
            boot[i] = vals[1] - vals[0]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        log(f"  paired image bootstrap: 95% CI [{lo:+.4f}, {hi:+.4f}], sd {boot.std():.4f}, P(Δ>0) {(boot > 0).mean():.3f}")
        res["interval"] = dict(ci=[float(lo), float(hi)], sd=float(boot.std()), p_gt0=float((boot > 0).mean()))
    res.update(mean_delta=dm, folds_positive=pos, verdict=verdict)
    cand[["cand_id", "p_rec"]].to_parquet(OUT / "e121_p_rec.parquet", index=False)
    json.dump(res, open(OUT / "det_conf_band_recal.json", "w"), indent=1, default=float)
    log(f"\nwrote {OUT/'det_conf_band_recal.json'}")


if __name__ == "__main__":
    main()
