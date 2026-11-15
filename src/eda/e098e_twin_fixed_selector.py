"""E098e -- the honest twin of sub026 vs sub025, with the selector held FIXED.

The test-side rebuild settled the provenance question: submit_v2 with
--carve-tau 0.70 reproduces sub026 exactly (1182 rows, identical RLE set) and
without the flag reproduces sub025 exactly (1179 rows), and the fusion-stage
counter shows the rule rejecting 354 cluster masks on the 180 test images --
about 2.0 per image against 2.3 per image on the 703 train images. So the rule
fires at the same rate on test; what differs between the twin (E098d, -0.017,
16% of the selected set removed) and the shipped file (1.7% of masks changed)
is not the rule but what E098d did AFTER the rule: it REFIT the selector on
the carved pool. Deployment does not. sub026 was scored by the same bundle as
sub025, fit on the tau-0 pool with the tau-0 quantile reference, applied to a
pool built with the rule inline. Refitting on the carved pool changed the
score distribution and the CDF reference, and the "-0.017" was mostly that
refit, not the rule.

This script is the deployment-faithful version. Per fold of folds_v1:
  - the selector recipe (E115b: HGB 300/0.05/31/25/1.0 seed 42; positives-only
    IoU regressor 300/0.06; score p x E[iou]; quantile reference from INNER
    leave-one-fold-out raws of the training folds) is fit ONCE, on the tau-0
    pool's training folds;
  - that one model and that one reference score the held-out fold of BOTH
    pools: tau 0 (the control, which must reproduce E115b's numbers because
    it is the same computation) and tau 0.70 (the carved pool);
  - the production policy conf >= 0.40, p_sel >= 0.60, exact scorer.
Nothing is matched across pools; the carved pool's candidates simply get
scored by the model deployment would have used.

PRE-REGISTERED, before the run. Same two bands as E098d: action thresholds at
±0.003 (pays / hurts, the transfer floor); equivalence band ±0.005 (the
measured flatness band) for TIED, decided only by the paired image-level
bootstrap (E115c machinery, guard at both pools, images resampled whole over
the union of stems). Control criterion: the tau-0 arm at (0.40, 0.60) must
reproduce E115b within 0.0005 pooled PQ (0.4360; TP 4719, FP 1779) -- it is
the same computation on the same cached pool, so anything else means the
script is wrong, and the twin is VOID. Diagnostics reported, not verdicts:
kept-set sizes, TP/FP/FN, the same Δ at (0.40, 0.55), and the interval's sd
with the sd that would have been needed if UNDERPOWERED.

Population caveat as before: honest OOF train pool, not the test set.

Usage: python src/eda/e098e_twin_fixed_selector.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "eda"))

from filament.pq import score
from e115c_threshold_ci import per_image, pq, make_cls, make_reg

OUT = ROOT / "outputs" / "eda"
BUNDLE = ROOT / "outputs" / "selector" / "selector_v5_shrunkblob5prodts.joblib"
A = OUT / "e098d_fused_pool_all_blob5_rebuild_labelled.parquet"
B = OUT / "e098d_fused_pool_all_blob5_carve70_labelled.parquet"
FLOOR, EQUIV_BAND, REPS = 0.003, 0.005, 4000
CONTROL = dict(pq=0.4360, tol=0.0005)
POLICIES = [(0.40, 0.55), (0.40, 0.60)]


def log(*a):
    print(*a, flush=True)


def fixed_selector_scores(a, b, feats):
    """Fit on pool A's training folds only; score the held-out fold of both."""
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
        log(f"    fold {f}: selector fit on tau-0 training folds, applied to both pools' held-out fold")
    return pa, pb


def exact_at(cand, gt, conf, pcut, fold=None):
    sub = cand[(cand["conf"] >= conf) & (cand["p_oof"] >= pcut)]
    g = gt
    if fold is not None:
        sub, g = sub[sub["fold"] == fold], gt[gt["fold"] == fold]
    pred = sub[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
    return score(g[["filament_id", "segmentation_rle"]], pred), len(sub)


def main():
    import joblib
    feats = joblib.load(BUNDLE)["features"]
    for f in (A, B):
        if not f.exists():
            sys.exit(f"{f} missing -- run e098d_pool_twin.py first (it writes the labelled pools)")
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    gt["fold"] = gt["stem"].map(fold_of)
    gt = gt[gt["fold"].notna()].copy()
    gt["fold"] = gt["fold"].astype(int)
    a, b = pd.read_parquet(A), pd.read_parquet(B)
    for d in (a, b):
        d["fold"] = d["stem"].map(fold_of).astype(int)
    log(f"tau 0 pool {len(a)}, tau 0.70 pool {len(b)}; fitting the FIXED selector on the tau-0 pool:")
    a["p_oof"], b["p_oof"] = fixed_selector_scores(a, b, feats)
    pools = {"tau0": a, "tau070": b}

    res = {}
    log(f"\n  {'policy':<14}{'pool':<8}{'kept':>7}{'PQ':>9}{'TP':>7}{'FP':>7}{'FN':>7}   per-fold PQ")
    for conf, pcut in POLICIES:
        for name, cand in pools.items():
            per = [exact_at(cand, gt, conf, pcut, fold=f)[0]["pq"] for f in range(5)]
            r, n = exact_at(cand, gt, conf, pcut)
            res[f"{name}_{conf}_{pcut}"] = dict(kept=n, pq_pooled=r["pq"], tp=r["tp"], fp=r["fp"], fn=r["fn"],
                                                per_fold=per, pq_mean=float(np.mean(per)))
            log(f"  ({conf:.2f},{pcut:.2f})   {name:<8}{n:>7}{r['pq']:>9.4f}{r['tp']:>7}{r['fp']:>7}{r['fn']:>7}   "
                + " ".join(f"{x:.4f}" for x in per))
        d = [res[f"tau070_{conf}_{pcut}"]["per_fold"][i] - res[f"tau0_{conf}_{pcut}"]["per_fold"][i] for i in range(5)]
        dm = float(np.mean(d))
        is_v = abs(pcut - 0.60) < 1e-9
        if is_v:
            c_ok = abs(res["tau0_0.4_0.6"]["pq_pooled"] - CONTROL["pq"]) <= CONTROL["tol"]
            res["control_ok"] = bool(c_ok)
            log(f"    control: tau-0 pooled {res['tau0_0.4_0.6']['pq_pooled']:.4f} vs E115b {CONTROL['pq']} "
                f"-> {'OK' if c_ok else 'FAIL: twin VOID'}")
        verdict = ("PAYS (>= +0.003)" if dm >= FLOOR else "HURTS (<= -0.003)" if dm <= -FLOOR
                   else "point estimate inside ±0.003 -> the interval decides")
        log(f"    Δ tau0.70 - tau0 at ({conf:.2f},{pcut:.2f}): mean {dm:+.4f}, folds+ {sum(x > 0 for x in d)}/5, per fold "
            + " ".join(f"{x:+.4f}" for x in d) + f"  -> {verdict}" + ("  <- VERDICT" if is_v else "  (diagnostic)"))
        res[f"delta_{conf}_{pcut}"] = dict(mean=dm, per_fold=d, verdict=verdict)
        if is_v:
            stems = np.array(sorted(set(a["stem"]) | set(b["stem"])))
            ncopy = gt.groupby("stem")["entry_id"].nunique().reindex(stems).fillna(1).to_numpy()
            parts = {}
            for name, cand in pools.items():
                sub = cand[(cand["conf"] >= conf) & (cand["p_oof"] >= pcut)]
                st, r = per_image(sub, gt, stems, ncopy)
                tot = pq(st["tp"].sum(), st["fp"].sum(), st["fn"].sum(), st["iou"].sum())
                log(f"    guard {name}: decomposed {tot:.4f} vs exact {r['pq']:.4f} (FP {int(st['fp'].sum())}/{r['fp']})")
                if abs(tot - r["pq"]) > 5e-4:
                    parts = None
                    log("    GUARD FAILED -> no interval")
                    break
                parts[name] = st
            if parts:
                rng = np.random.default_rng(0)
                n = len(stems)
                boot = np.empty(REPS)
                for i in range(REPS):
                    idx = rng.integers(0, n, n)
                    v = [pq(s["tp"][idx].sum(), s["fp"][idx].sum(), s["fn"][idx].sum(), s["iou"][idx].sum())
                         for s in (parts["tau0"], parts["tau070"])]
                    boot[i] = v[1] - v[0]
                obs = (pq(*(parts["tau070"][k].sum() for k in ("tp", "fp", "fn", "iou")))
                       - pq(*(parts["tau0"][k].sum() for k in ("tp", "fp", "fn", "iou"))))
                lo, hi = np.percentile(boot, [2.5, 97.5])
                sd = float(boot.std())
                inside = lo > -EQUIV_BAND and hi < EQUIV_BAND
                needed = max(EQUIV_BAND - abs(obs), 0.0) / 1.96
                if dm >= FLOOR or dm <= -FLOOR:
                    final = verdict
                elif inside:
                    final = f"TIED: 95% interval inside ±{EQUIV_BAND} -> the sub025/sub026 ordering carries no information"
                else:
                    final = (f"UNDERPOWERED: interval reaches {max(-EQUIV_BAND - lo, hi - EQUIV_BAND, 0):.4f} past "
                             f"±{EQUIV_BAND}; sd obtained {sd:.4f}, sd needed {needed:.4f}")
                log(f"    pooled Δ {obs:+.4f}; paired image bootstrap {REPS} reps over {n} images: "
                    f"95% CI [{lo:+.4f}, {hi:+.4f}], sd {sd:.4f}, P(Δ>0) {(boot > 0).mean():.3f}")
                log(f"    FINAL: {final}" + ("" if res.get("control_ok") else "   [CONTROL FAILED -> VOID]"))
                res["interval"] = dict(observed=float(obs), ci=[float(lo), float(hi)], sd=sd,
                                       p_gt0=float((boot > 0).mean()), inside=bool(inside), final=final)
    json.dump(res, open(OUT / "e098e_twin_fixed_selector.json", "w"), indent=1, default=float)
    log(f"\nwrote {OUT/'e098e_twin_fixed_selector.json'}")


if __name__ == "__main__":
    main()
