"""Selector REFIT on committee pools -- the last committee hypothesis (ledger 17e-f).

Three board reads put every committee field below sub025 under the production
selector, which was fit on single-model OOF fields and is deployed on the
five-fold-mean test field. The remaining hypothesis is train/deploy
consistency: fit the selector on committee OOF pools (smooth like the test
field) and deploy it with the committee test field. This is the E098d
"adoption" protocol (honest OOF refit on each pool) restricted to the folds
whose committee arms exist.

  A   production labelled pool (E115b / e098d tau-0), all five folds
  B   A with the rows of the committee folds replaced by the labelled
      committee4 pools of those folds (0.5 blob + 0.5 mean(clDice, 2048/B2,
      B5, wide decoder) at 0.3, label_pool order); other folds unchanged

Both pools get E115b honest OOF scores (fit on the other four folds, inner
LOFO reference), production policy (0.40, 0.60), exact scorer; the fixed
selector (fit on A's training folds, applied to both) is reported too. The
number that decides is the delta on the COMMITTEE folds (the others are
identical rows scored by a model whose training set changed slightly).

PRE-REGISTERED: adoption pays if the committee-fold delta (refit) is
>= +0.003 with the paired image bootstrap over those folds' stems P(delta>0)
>= 0.80; otherwise the committee axis closes and the remaining queued arm
runs are cancelled.

Usage: python src/eda/committee_refit_twin.py --committee-folds 0,1
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
from e115c_threshold_ci import per_image, pq, oof_scores  # noqa: E402
from threshold_5fold_twin import fixed_scores, exact, A_POOL, BUNDLE, OUT, POLICY  # noqa: E402

REPS, FLOOR, P_GO = 4000, 0.003, 0.80


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--committee-folds", default="0,1")
    ap.add_argument("--tag", default="committee4")
    a_ = ap.parse_args()
    cf = [int(x) for x in a_.committee_folds.split(",")]
    feats = joblib.load(BUNDLE)["features"]
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    gt["fold"] = gt["stem"].map(fold_of)
    gt = gt[gt["fold"].notna()].copy()
    gt["fold"] = gt["fold"].astype(int)

    a = pd.read_parquet(A_POOL)
    a["fold"] = a["stem"].map(fold_of).astype(int)
    parts = [a[~a["fold"].isin(cf)]]
    for f in cf:
        c = pd.read_parquet(OUT / f"fused_pool_f{f}_{a_.tag}_blob_t030_labelled.parquet")
        c["fold"] = c["stem"].map(fold_of).astype(int)
        c = c[c["fold"] == f]
        parts.append(c[a.columns])
        log(f"fold {f}: production rows {(a['fold'] == f).sum()} -> committee rows {len(c)}")
    b = pd.concat(parts, ignore_index=True)
    log(f"A {len(a)} rows; B {len(b)} rows; committee folds {cf}")
    stems = np.array(sorted(set(a["stem"]) | set(b["stem"])))
    gts = gt[gt["stem"].isin(stems)]

    a["p_fixed"], b["p_fixed"] = fixed_scores(a, b, feats)
    a["p_refit"] = oof_scores(a, feats, a["fold"].to_numpy())
    b["p_refit"] = oof_scores(b, feats, b["fold"].to_numpy())

    res = {"committee_folds": cf}
    for col in ("p_fixed", "p_refit"):
        log(f"\n== {col}  policy {POLICY}")
        rows = {}
        for name, cand in (("A", a), ("B", b)):
            per = [exact(cand, gts, col, fold=f)[0]["pq"] for f in range(5)]
            r, n = exact(cand, gts, col)
            rows[name] = dict(kept=n, pq=r["pq"], tp=r["tp"], fp=r["fp"], fn=r["fn"], per_fold=per)
            log(f"  {name}: kept {n} PQ {r['pq']:.4f} TP {r['tp']} FP {r['fp']} FN {r['fn']}  per fold " + " ".join(f"{x:.4f}" for x in per))
        d = [rows["B"]["per_fold"][i] - rows["A"]["per_fold"][i] for i in range(5)]
        dc = [d[f] for f in cf]
        log(f"  delta B - A per fold " + " ".join(f"{x:+.4f}" for x in d) + f"  | committee folds mean {np.mean(dc):+.4f}")
        # paired bootstrap over the committee folds' stems only
        cst = np.array([s for s in stems if fold_of[s] in cf])
        gsub = gts[gts["stem"].isin(cst)]
        ncopy = gsub.groupby("stem")["entry_id"].nunique().reindex(cst).fillna(1).to_numpy()
        pieces = {}
        for name, cand in (("A", a), ("B", b)):
            sub = cand[(cand["conf"] >= POLICY[0]) & (cand[col] >= POLICY[1]) & cand["stem"].isin(cst)]
            st, r = per_image(sub, gsub, cst, ncopy)
            tot = pq(st["tp"].sum(), st["fp"].sum(), st["fn"].sum(), st["iou"].sum())
            log(f"    guard {name} (committee folds): decomposed {tot:.4f} vs exact {r['pq']:.4f}")
            pieces[name] = (st, r["pq"])
        rng = np.random.default_rng(0)
        n = len(cst)
        boot = np.empty(REPS)
        for i in range(REPS):
            idx = rng.integers(0, n, n)
            v = [pq(s["tp"][idx].sum(), s["fp"][idx].sum(), s["fn"][idx].sum(), s["iou"][idx].sum()) for s, _ in (pieces["A"], pieces["B"])]
            boot[i] = v[1] - v[0]
        obs = pieces["B"][1] - pieces["A"][1]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        p_gt0 = float((boot > 0).mean())
        log(f"  committee-fold pooled delta {obs:+.4f}; bootstrap {REPS} over {n} images: 95% CI [{lo:+.4f}, {hi:+.4f}], sd {boot.std():.4f}, P(delta>0) {p_gt0:.3f}")
        if col == "p_refit":
            verdict = ("PAYS -> keep the queue" if obs >= FLOOR and p_gt0 >= P_GO else
                       "NO -> committee axis closed, cancel the remaining arm runs")
            log(f"  VERDICT (refit, committee folds): {verdict}")
            res["verdict"] = verdict
        res[col] = dict(A=rows["A"], B=rows["B"], delta_folds=d, committee_delta=float(obs), ci=[float(lo), float(hi)], p_gt0=p_gt0)
    out = OUT / f"committee_refit_twin_{a_.tag}_f{''.join(map(str, cf))}.json"
    json.dump(res, open(out, "w"), indent=1, default=float)
    log(f"\nwrote {out}")


if __name__ == "__main__":
    main()
