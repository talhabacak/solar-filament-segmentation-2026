"""Committee pool twin -- does the fold-0 committee map pay at selector PQ?

The instance gate (semantic_ensemble_gate.py) says the mean of the production
fold-0 model and the seven A100 arms reaches +11 rows at threshold 0.20 with
per-instance Dice +0.0076 and pixel-Dice +0.0059. The gate is semantic-only.
What the board scores is the selected pool, so the question is the E098e one:
with the selector held FIXED (fit on the production pool's training folds,
folds 1-4, with the inner leave-one-fold-out quantile reference), does the
held-out fold 0 score higher when its pool is built from the committee map
than when it is built from the production map?

  A  production OOF pool (e098d tau-0 rebuild, 0.5*blob + 0.5*clDice at 0.3),
     all five folds; the selector is fit on folds 1-4 of THIS pool only.
  B  fold-0 pool built by build_fused_pool.py from the committee map
     (0.5*blob + 0.5*mean(clDice, arms)) at the threshold under test.

Both pools' fold-0 candidates are scored by that one model and reference; the
production policy conf >= 0.40, p_sel >= 0.60; exact scorer over fold-0 ground
truth (GT scope: every folds_v1 fold-0 stem, both arms).

PRE-REGISTERED. Control: pool A's fold-0 PQ at (0.40, 0.60) must reproduce
E098e's fold-0 value 0.4358 within 0.0005, or the twin is VOID. This is a
FUNDING decision (training the arms on folds 1-4 is ~20 A100 hours), so the
bands are the transfer floor and the sign: GO when the fold-0 delta is
>= +0.003 with P(delta > 0) >= 0.80 under the paired image bootstrap (4000
reps over the fold-0 stems, per-image decomposition guard at both pools);
NO-GO when the delta is <= -0.003; otherwise OPEN (one fold cannot resolve it)
and the seed-rerun control decides whether the mechanism is variance
reduction the fold ensemble already provides.

Usage:
  python src/eda/committee_pool_twin.py --pool fused_pool_f0_committee_blob_t020.parquet --tag t020
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
from e115c_threshold_ci import per_image, pq, make_cls, make_reg  # noqa: E402

OUT = ROOT / "outputs" / "eda"
SEL = ROOT / "outputs" / "selector"
BUNDLE = SEL / "selector_v5_shrunkblob5prodts.joblib"
A_POOL = OUT / "e098d_fused_pool_all_blob5_rebuild_labelled.parquet"
FOLD, FLOOR, P_GO, REPS = 0, 0.003, 0.80, 4000
# E098e per-fold values of pool A at (0.40, 0.60) -- the control for each fold
E098E_FOLD = [0.4358, 0.4399, 0.4454, 0.4510, 0.4061]
CONTROL = dict(pq=E098E_FOLD[0], tol=0.0005)
POLICIES = [(0.40, 0.55), (0.40, 0.60)]


def log(*a):
    print(*a, flush=True)


def fixed_selector(a, b, feats, fold):
    """Fit on pool A's training folds; score the held-out fold of A and all of B."""
    Xa, ya = a[feats].to_numpy(np.float64), a["y"].to_numpy()
    va, fa = a["iou_sum"].to_numpy(np.float64), a["fold"].to_numpy()
    tr = np.where(fa != fold)[0]
    inner = np.zeros(len(tr))
    for g in [k for k in range(5) if k != fold]:
        itr = np.where((fa != fold) & (fa != g))[0]
        msk = fa[tr] == g
        ci = make_cls().fit(Xa[itr], ya[itr])
        ri = make_reg().fit(Xa[itr][ya[itr] == 1], va[itr][ya[itr] == 1])
        Xg = Xa[tr][msk]
        inner[msk] = ci.predict_proba(Xg)[:, 1] * np.clip(ri.predict(Xg), 0, None)
    ref = np.sort(inner)
    c = make_cls().fit(Xa[tr], ya[tr])
    r = make_reg().fit(Xa[tr][ya[tr] == 1], va[tr][ya[tr] == 1])

    def sc(X):
        raw = c.predict_proba(X)[:, 1] * np.clip(r.predict(X), 0, None)
        return np.searchsorted(ref, raw, side="right") / len(ref)
    te = np.where(fa == fold)[0]
    pa = np.full(len(a), np.nan)
    pa[te] = sc(Xa[te])
    pb = sc(b[feats].to_numpy(np.float64))
    return pa, pb


def load_pool(name):
    """A pool by path, or by file name inside outputs/selector. The deployment-
    faithful pools are the LABELLED ones (two_stage_selector.label_pool: masks
    shrunk 0.05 first, area/log10_area recomputed, labels on the shrunk masks --
    the order submit_v2 uses), which live in outputs/eda."""
    p = Path(name)
    return pd.read_parquet(p if p.exists() else SEL / name)


def exact_at(cand, gt, conf, pcut):
    sub = cand[(cand["conf"] >= conf) & (cand["p_oof"] >= pcut)]
    pred = sub[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
    return score(gt[["filament_id", "segmentation_rle"]], pred), sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="fold-0 committee pool parquet in outputs/selector")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ref", default=None,
                    help="builder-consistent reference: a fold-0 pool built TODAY from the "
                         "production map (outputs/selector). When given, the 'prod' arm is this "
                         "pool scored by the same fixed selector, so builder drift between pool A "
                         "and today's build_fused_pool cannot masquerade as a committee gain; pool "
                         "A's own fold-0 rows are still scored and reported as the control.")
    ap.add_argument("--fold", type=int, default=0, help="held-out fold of the twin (pool B must be that fold)")
    a_ = ap.parse_args()
    global FOLD
    FOLD = a_.fold
    CONTROL["pq"] = E098E_FOLD[FOLD]

    feats = joblib.load(BUNDLE)["features"]
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    stems = np.array(sorted(s for s, f in fold_of.items() if f == FOLD))
    gt0 = gt[gt["stem"].isin(stems)].copy()

    a = pd.read_parquet(A_POOL)
    a["fold"] = a["stem"].map(fold_of).astype(int)
    b = load_pool(a_.pool)
    b = b[b["stem"].isin(stems)].copy()
    missing = [f for f in feats if f not in b.columns]
    if missing:
        sys.exit(f"committee pool lacks selector features: {missing}")
    log(f"A: {len(a)} rows / {a['stem'].nunique()} stems (fold-0 rows {(a['fold'] == FOLD).sum()}); "
        f"B: {len(b)} rows / {b['stem'].nunique()} stems; GT fold-0 rows {len(gt0)} over {len(stems)} stems")

    if a_.ref:
        ref = load_pool(a_.ref)
        ref = ref[ref["stem"].isin(stems)].copy()
        both = pd.concat([b.assign(_arm="committee"), ref.assign(_arm="ref")], ignore_index=True)
        pa, pboth = fixed_selector(a, both, feats, FOLD)
        both["p_oof"] = pboth
        b = both[both["_arm"] == "committee"].copy()
        ref = both[both["_arm"] == "ref"].copy()
    else:
        pa, pb = fixed_selector(a, b, feats, FOLD)
        b["p_oof"] = pb
    a0 = a[a["fold"] == FOLD].copy()
    a0["p_oof"] = pa[a["fold"].to_numpy() == FOLD]
    if a_.ref:
        r_ctl, _ = exact_at(a0, gt0, 0.40, 0.60)
        log(f"pool A fold-0 control at (0.40,0.60): {r_ctl['pq']:.4f} (E098e {CONTROL['pq']}); "
            f"'prod' arm below is the builder-consistent reference {a_.ref} ({len(ref)} rows)")
        pools = {"prod": ref, "committee": b}
        CONTROL["pq"] = r_ctl["pq"]     # the control line then reports drift of today's builder
    else:
        pools = {"prod": a0, "committee": b}

    res = {"pool": a_.pool}
    log(f"\n  {'policy':<14}{'pool':<11}{'kept':>6}{'PQ':>9}{'TP':>6}{'FP':>6}{'FN':>6}{'sumIoU':>9}")
    for conf, pcut in POLICIES:
        rr = {}
        for name, cand in pools.items():
            r, sub = exact_at(cand, gt0, conf, pcut)
            rr[name] = r
            res[f"{name}_{conf}_{pcut}"] = dict(kept=len(sub), pq=r["pq"], tp=r["tp"], fp=r["fp"], fn=r["fn"])
            log(f"  ({conf:.2f},{pcut:.2f})   {name:<11}{len(sub):>6}{r['pq']:>9.4f}{r['tp']:>6}{r['fp']:>6}{r['fn']:>6}"
                f"{r.get('iou_sum', float('nan')):>9.3f}")
        d = rr["committee"]["pq"] - rr["prod"]["pq"]
        res[f"delta_{conf}_{pcut}"] = d
        is_v = abs(pcut - 0.60) < 1e-9
        if is_v:
            c_ok = abs(rr["prod"]["pq"] - CONTROL["pq"]) <= CONTROL["tol"]
            res["control_ok"] = bool(c_ok)
            log(f"    control: prod fold-0 {rr['prod']['pq']:.4f} vs E098e fold-0 {CONTROL['pq']} "
                f"-> {'OK' if c_ok else 'FAIL: twin VOID'}")
        log(f"    delta committee - prod at ({conf:.2f},{pcut:.2f}): {d:+.4f}" + ("  <- VERDICT policy" if is_v else "  (diagnostic)"))
        if is_v:
            ncopy = gt0.groupby("stem")["entry_id"].nunique().reindex(stems).fillna(1).to_numpy()
            parts = {}
            for name, cand in pools.items():
                sub = cand[(cand["conf"] >= conf) & (cand["p_oof"] >= pcut)]
                st, r = per_image(sub, gt0, stems, ncopy)
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
                         for s in (parts["prod"], parts["committee"])]
                    boot[i] = v[1] - v[0]
                lo, hi = np.percentile(boot, [2.5, 97.5])
                p_gt0 = float((boot > 0).mean())
                if d >= FLOOR and p_gt0 >= P_GO:
                    final = f"GO: fold-0 delta {d:+.4f} >= +{FLOOR} with P(delta>0) {p_gt0:.2f} >= {P_GO} -> fund the arms on folds 1-4"
                elif d <= -FLOOR:
                    final = f"NO-GO: fold-0 delta {d:+.4f} <= -{FLOOR}"
                else:
                    final = (f"OPEN: delta {d:+.4f}, P(delta>0) {p_gt0:.2f}; one fold cannot resolve it -- "
                             f"the seed-rerun control decides whether this is variance reduction")
                log(f"    paired image bootstrap {REPS} reps over {n} images: 95% CI [{lo:+.4f}, {hi:+.4f}], "
                    f"sd {boot.std():.4f}, P(delta>0) {p_gt0:.3f}")
                log(f"    FINAL: {final}" + ("" if res.get("control_ok") else "   [CONTROL FAILED -> VOID]"))
                res["interval"] = dict(observed=float(d), ci=[float(lo), float(hi)], sd=float(boot.std()),
                                       p_gt0=p_gt0, final=final)
    out = OUT / f"committee_pool_twin_{a_.tag}.json"
    json.dump(res, open(out, "w"), indent=1, default=float)
    log(f"\nwrote {out}")


if __name__ == "__main__":
    main()
