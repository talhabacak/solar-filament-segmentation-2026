"""Semantic extraction threshold 0.3 -> 0.2 for the PRODUCTION field, five folds.

Side finding of ledger 17b: on fold 0, with the selector held fixed, the pool
built from the production map at threshold 0.2 scored 0.4429 against 0.4358 at
0.3 (+0.0071), while the committee at 0.2 added nothing on top. The threshold
was swept optimal for the clDice field alone, before the blob member joined it;
a better-calibrated blend may want a lower cut (build_fused_pool --sem-thr
help text says exactly this). One fold cannot carry a threshold decision, so
this is the five-fold twin, both ways the change could be deployed:

  fixed    selector fit per fold on pool A's (production, thr 0.3) training
           folds with the inner LOFO quantile reference, applied to the
           held-out fold of BOTH pools -- deploy the new pool under the old
           bundle (E098e protocol);
  refit    E115b honest OOF scores on each pool separately -- deploy a selector
           refit on the new pool (E098d protocol).

Both pools are label_pool(shrink 0.05) outputs (the deployment order). Exact
scorer at (0.40, 0.60), per fold and pooled over the 703 stems; paired
image-level bootstrap over the union of stems with the decomposition guard.
Pre-registered bands as E098d/e: pays >= +0.003, hurts <= -0.003, otherwise
the 95% interval decides TIED (inside +-0.005) or UNDERPOWERED. Control: pool
A at (0.40, 0.60) fixed must reproduce E115b's 0.4360 within 0.0005.

Usage (on the box): python src/eda/threshold_5fold_twin.py --b outputs/eda/fused_pool_all_blob5_t020_labelled.parquet
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "eda"))

from filament.pq import score  # noqa: E402
from e115c_threshold_ci import per_image, pq, make_cls, make_reg, oof_scores  # noqa: E402

OUT = ROOT / "outputs" / "eda"
SEL = ROOT / "outputs" / "selector"
BUNDLE = SEL / "selector_v5_shrunkblob5prodts.joblib"
A_POOL = OUT / "e098d_fused_pool_all_blob5_rebuild_labelled.parquet"
FLOOR, EQUIV, REPS = 0.003, 0.005, 4000
CONTROL = dict(pq=0.4360, tol=0.0005)
POLICY = (0.40, 0.60)


def log(*a):
    print(*a, flush=True)


def fixed_scores(a, b, feats):
    Xa, ya, va, fa = a[feats].to_numpy(np.float64), a["y"].to_numpy(), a["iou_sum"].to_numpy(np.float64), a["fold"].to_numpy()
    Xb, fb = b[feats].to_numpy(np.float64), b["fold"].to_numpy()
    pa, pb = np.zeros(len(a)), np.zeros(len(b))
    for f in range(5):
        tr = np.where(fa != f)[0]
        inner = np.zeros(len(tr))
        for g in [k for k in range(5) if k != f]:
            itr = np.where((fa != f) & (fa != g))[0]
            msk = fa[tr] == g
            ci = make_cls().fit(Xa[itr], ya[itr])
            ri = make_reg().fit(Xa[itr][ya[itr] == 1], va[itr][ya[itr] == 1])
            Xg = Xa[tr][msk]
            inner[msk] = ci.predict_proba(Xg)[:, 1] * np.clip(ri.predict(Xg), 0, None)
        ref = np.sort(inner)
        c = make_cls().fit(Xa[tr], ya[tr])
        r = make_reg().fit(Xa[tr][ya[tr] == 1], va[tr][ya[tr] == 1])
        for X, fo, p in ((Xa, fa, pa), (Xb, fb, pb)):
            te = np.where(fo == f)[0]
            raw = c.predict_proba(X[te])[:, 1] * np.clip(r.predict(X[te]), 0, None)
            p[te] = np.searchsorted(ref, raw, side="right") / len(ref)
        log(f"    fold {f} fixed selector done")
    return pa, pb


def exact(cand, gt, col, fold=None):
    sub = cand[(cand["conf"] >= POLICY[0]) & (cand[col] >= POLICY[1])]
    g = gt
    if fold is not None:
        sub, g = sub[sub["fold"] == fold], gt[gt["fold"] == fold]
    pred = sub[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
    return score(g[["filament_id", "segmentation_rle"]], pred), len(sub)


def interval(pools, col, gt, stems, ncopy):
    parts = {}
    for name, cand in pools.items():
        sub = cand[(cand["conf"] >= POLICY[0]) & (cand[col] >= POLICY[1])]
        st, r = per_image(sub, gt, stems, ncopy)
        tot = pq(st["tp"].sum(), st["fp"].sum(), st["fn"].sum(), st["iou"].sum())
        log(f"    guard {name}: decomposed {tot:.4f} vs exact {r['pq']:.4f}")
        if abs(tot - r["pq"]) > 5e-4:
            return None
        parts[name] = st
    rng = np.random.default_rng(0)
    n = len(stems)
    boot = np.empty(REPS)
    for i in range(REPS):
        idx = rng.integers(0, n, n)
        v = [pq(s["tp"][idx].sum(), s["fp"][idx].sum(), s["fn"][idx].sum(), s["iou"][idx].sum()) for s in (parts["A"], parts["B"])]
        boot[i] = v[1] - v[0]
    obs = pq(*(parts["B"][k].sum() for k in ("tp", "fp", "fn", "iou"))) - pq(*(parts["A"][k].sum() for k in ("tp", "fp", "fn", "iou")))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(observed=float(obs), ci=[float(lo), float(hi)], sd=float(boot.std()), p_gt0=float((boot > 0).mean()))


def verdict(dm, iv):
    if dm >= FLOOR:
        return "PAYS (>= +0.003)"
    if dm <= -FLOOR:
        return "HURTS (<= -0.003)"
    if iv is None:
        return "no interval (guard failed)"
    lo, hi = iv["ci"]
    if lo > -EQUIV and hi < EQUIV:
        return f"TIED: 95% CI inside +-{EQUIV}"
    return f"UNDERPOWERED: CI [{lo:+.4f}, {hi:+.4f}] reaches past +-{EQUIV}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", required=True, help="labelled five-fold pool to compare against pool A")
    ap.add_argument("--tag", default="t020")
    a_ = ap.parse_args()
    feats = joblib.load(BUNDLE)["features"]
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    gt["fold"] = gt["stem"].map(fold_of)
    gt = gt[gt["fold"].notna()].copy()
    gt["fold"] = gt["fold"].astype(int)
    a = pd.read_parquet(A_POOL)
    b = pd.read_parquet(a_.b if Path(a_.b).exists() else OUT / a_.b)
    for d in (a, b):
        d["fold"] = d["stem"].map(fold_of).astype(int)
    log(f"A (production, thr 0.3): {len(a)} rows / {a['stem'].nunique()} stems; B ({a_.tag}): {len(b)} rows / {b['stem'].nunique()} stems")
    stems = np.array(sorted(set(a["stem"]) | set(b["stem"])))
    gts = gt[gt["stem"].isin(stems)]
    ncopy = gts.groupby("stem")["entry_id"].nunique().reindex(stems).fillna(1).to_numpy()

    log("\nFIXED selector (fit on A's training folds, applied to both):")
    a["p_fixed"], b["p_fixed"] = fixed_scores(a, b, feats)
    log("\nREFIT selector (E115b honest OOF on each pool):")
    a["p_refit"] = oof_scores(a, feats, a["fold"].to_numpy())
    b["p_refit"] = oof_scores(b, feats, b["fold"].to_numpy())

    res = {"b": a_.b}
    for col in ("p_fixed", "p_refit"):
        log(f"\n== {col}  policy (0.40, 0.60)")
        rows = {}
        for name, cand in (("A", a), ("B", b)):
            per = [exact(cand, gts, col, fold=f)[0]["pq"] for f in range(5)]
            r, n = exact(cand, gts, col)
            rows[name] = dict(kept=n, pq=r["pq"], tp=r["tp"], fp=r["fp"], fn=r["fn"], per_fold=per)
            log(f"  {name}: kept {n} PQ {r['pq']:.4f} TP {r['tp']} FP {r['fp']} FN {r['fn']}  per fold " + " ".join(f"{x:.4f}" for x in per))
        d = [rows["B"]["per_fold"][i] - rows["A"]["per_fold"][i] for i in range(5)]
        dm = float(np.mean(d))
        if col == "p_fixed":
            ok = abs(rows["A"]["pq"] - CONTROL["pq"]) <= CONTROL["tol"]
            res["control_ok"] = bool(ok)
            log(f"  control: A pooled {rows['A']['pq']:.4f} vs E115b {CONTROL['pq']} -> {'OK' if ok else 'FAIL: VOID'}")
        iv = interval({"A": a, "B": b}, col, gts, stems, ncopy)
        v = verdict(dm, iv)
        log(f"  delta B - A: mean of folds {dm:+.4f}, folds+ {sum(x > 0 for x in d)}/5, per fold " + " ".join(f"{x:+.4f}" for x in d))
        if iv:
            log(f"  pooled delta {iv['observed']:+.4f}; bootstrap {REPS} reps over {len(stems)} images: 95% CI [{iv['ci'][0]:+.4f}, {iv['ci'][1]:+.4f}], sd {iv['sd']:.4f}, P(delta>0) {iv['p_gt0']:.3f}")
        log(f"  VERDICT ({col}): {v}")
        res[col] = dict(A=rows["A"], B=rows["B"], delta_folds=d, delta_mean=dm, interval=iv, verdict=v)
    out = OUT / f"threshold_5fold_twin_{a_.tag}.json"
    json.dump(res, open(out, "w"), indent=1, default=float)
    log(f"\nwrote {out}")


if __name__ == "__main__":
    main()
