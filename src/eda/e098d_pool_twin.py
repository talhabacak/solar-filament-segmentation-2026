"""E098d -- the honest local twin of sub026 vs sub025.

sub026 = sub025 + the fusion-stage carve-retention rule (tau 0.70), and the
board ranked it one place below sub025. Every local number for that rule so
far was measured on the wrong object: §12 on a GT-conditioned variant-bank
base with a frozen carving (E098b: the freeze is harmless, the GT filter
makes the gain smaller not larger, and on the unfiltered raw masks the rule is
worth +0.027 because that population is 61% false positive); e098c on the
production pool AFTER fusion, where masks are already disjoint and the rule is
a no-op by construction. Neither is the deployed setting.

This one is. The pool is rebuilt with the rule INSIDE fusion, exactly as
submit_v2 --carve-tau does it, so a rejected mask frees its pixels for the
masks behind it and every downstream feature (area, ring, density) is
recomputed on the resulting masks. Then the two pools -- tau 0 (the sub025
pool, rebuilt as a control that must reproduce fused_pool_all_blob5) and
tau 0.70 -- get the same labelling (shrink 0.05), the same deployment-faithful
out-of-fold selector recipe (E115b), and the exact scorer at the production
policy, per fold and pooled.

PRE-REGISTERED VERDICT, fixed before either pool was built. An earlier
criterion for this question counted FP units removed from a fixed p>=0.60
pool; it is RETIRED because the design changed, not because its number was
inconvenient: the rule now builds a different pool, the selector is refit on
each, and "candidates that would have cleared the threshold" is a cross-pool
identity that cannot be established (the bank-to-pool join failed on 66/140
stems and 617/1854 rows). The verdict is instead the direct comparison, which
needs no matching:

    Δ = PQ(tau 0.70 pool, conf>=0.40, p_oof>=0.60) - PQ(tau 0 pool, same)
    mean over the five folds of folds_v1, exact scorer
    Δ >= +0.003  -> the rule pays at production purity (transfer floor)
    Δ <= -0.003  -> the rule hurts at production purity
    otherwise    -> NOT decided by the point estimate. "Tied" is an
                    equivalence claim and needs an interval (the lesson of
                    E115c): the paired image-level bootstrap of E115c is run
                    on this Δ (703 images resampled whole, pooled PQ
                    recomputed from per-image parts, decomposition guard at
                    BOTH pools). Only a 95% interval contained in the
                    EQUIVALENCE BAND establishes TIED -- which then says the
                    sub025/sub026 ordering is noise-scale and carries no
                    information about the rule. An interval reaching past
                    the band is recorded as UNDERPOWERED, with the sd
                    obtained and the sd that would have been needed, so a
                    later reader knows whether the twin was close or
                    hopeless.

TWO BANDS, decided before the control pool had finished building and before
any twin number existed. The action thresholds (pays / hurts) stay at ±0.003,
the transfer floor, the smallest effect anyone would act on. The equivalence
band is ±0.005, the project's measured flatness band and the one E115c was
judged against. Reason: TIED needs |Δ| + 1.96·sd inside the band; E115c
resolved at sd 0.0013 with two thresholds on ONE pool and ONE selector fit,
and this twin changes both pool and fit, so its pairing is looser and its sd
will be larger. Under a ±0.003 equivalence band TIED would be declarable only
for |Δ| below about 0.0004 even at E115c's precision, i.e. never in practice,
and UNDERPOWERED would be the outcome for any Δ not almost exactly zero. The
split follows the E115c precedent (action on the floor, equivalence on the
flatness band). The single ±0.003 band was the earlier registration; it is
replaced here with this reasoning, not after seeing a number.

CONTROL PASS CRITERION, also fixed in advance. The tau 0 rebuild goes through
the same patched builder and must reproduce fused_pool_all_blob5 within a
declared tolerance, or the twin is VOID -- a builder that cannot reproduce its
own control cannot be trusted to isolate a one-flag change:
    candidate count   within ±5 of 9917
    builder positives exactly 4189 (the builder's own `y`, pool resolution)
    pooled exact PQ of the raw pool within 0.0005 of 0.3204
Exact equality is not demanded because clustering order, dict iteration and
float accumulation in the ring features can move a candidate or two; the
tolerance is the same 0.0005 the decomposition guard uses.

Selector refits add no stochastic noise: HistGradientBoosting with a pinned
random_state is deterministic given its data, so the two fits differ only as
a deterministic function of the two pools. Seeds are identical in both arms.

Reported as diagnostics only, not as the verdict: N = masks the rule rejected
at fusion (printed by build_fused_pool.py), pool sizes, kept-set sizes at the
policy, TP/FP/FN at the policy, and the same Δ at (0.40, 0.55).

Population caveat, as for E115b/E098c: this is the honest out-of-fold pool on
TRAIN images; the board is the all-fold selector on TEST images. It tests the
mechanism, it does not reproduce the board comparison.

Usage: python src/eda/e098d_pool_twin.py --a fused_pool_all_blob5_rebuild.parquet --b fused_pool_all_blob5_carve70.parquet
"""

import argparse
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
from two_stage_selector import label_pool
from e115c_threshold_ci import oof_scores, per_image, pq

OUT = ROOT / "outputs" / "eda"
SEL = ROOT / "outputs" / "selector"
BUNDLE = SEL / "selector_v5_shrunkblob5prodts.joblib"
FLOOR = 0.003        # action thresholds: pays / hurts (transfer floor)
EQUIV_BAND = 0.005   # equivalence band for TIED (measured flatness band; E115c precedent)
POLICIES = [(0.40, 0.55), (0.40, 0.60)]
CONTROL = dict(n=9917, n_tol=5, positives=4189, pq=0.3204, pq_tol=0.0005)
REPS = 4000


def control_check(pool_name, gt):
    """The tau 0 rebuild must reproduce the production pool within the declared
    tolerance; otherwise the twin is void."""
    raw = pd.read_parquet(SEL / pool_name)
    pred = raw[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
    g = gt[gt["stem"].isin(set(raw["stem"]))]
    r = score(g[["filament_id", "segmentation_rle"]], pred)
    ok_n = abs(len(raw) - CONTROL["n"]) <= CONTROL["n_tol"]
    ok_y = int(raw["y"].sum()) == CONTROL["positives"]
    ok_pq = abs(r["pq"] - CONTROL["pq"]) <= CONTROL["pq_tol"]
    log(f"  control: {len(raw)} candidates (target {CONTROL['n']}±{CONTROL['n_tol']}) {'OK' if ok_n else 'FAIL'}; "
        f"positives {int(raw['y'].sum())} (target {CONTROL['positives']}) {'OK' if ok_y else 'FAIL'}; "
        f"raw pooled PQ {r['pq']:.4f} (target {CONTROL['pq']}±{CONTROL['pq_tol']}) {'OK' if ok_pq else 'FAIL'}")
    return ok_n and ok_y and ok_pq


def log(*a):
    print(*a, flush=True)


def prepare(pool_name, feats, gt, fold_of):
    cache = OUT / f"e098d_{Path(pool_name).stem}_labelled.parquet"
    if cache.exists():
        cand = pd.read_parquet(cache)
        log(f"  {pool_name}: labelled pool from cache")
    else:
        cand, _, _ = label_pool(pool_name, 0.05, False)
        cand.to_parquet(cache, index=False)
    cand["fold"] = cand["stem"].map(fold_of).astype(int)
    log(f"  {pool_name}: {len(cand)} candidates, positives {int(cand['y'].sum())}; fitting OOF scores")
    cand["p_oof"] = oof_scores(cand, feats, cand["fold"].to_numpy())
    return cand


def exact_at(cand, gt, conf, pcut, fold=None):
    sub = cand[(cand["conf"] >= conf) & (cand["p_oof"] >= pcut)]
    g = gt
    if fold is not None:
        sub = sub[sub["fold"] == fold]
        g = gt[gt["fold"] == fold]
    pred = sub[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
    r = score(g[["filament_id", "segmentation_rle"]], pred)
    return r, len(sub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="tau 0 pool (control)")
    ap.add_argument("--b", required=True, help="tau 0.70 pool")
    a = ap.parse_args()
    import joblib
    feats = joblib.load(BUNDLE)["features"]
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    gt["fold"] = gt["stem"].map(fold_of)
    gt = gt[gt["fold"].notna()].copy()
    gt["fold"] = gt["fold"].astype(int)

    log("control pass criterion (pre-registered):")
    control_ok = control_check(a.a, gt)
    if not control_ok:
        log("  CONTROL FAILED -> the twin is VOID; numbers below are printed for the record only")
    pools = {"tau0": prepare(a.a, feats, gt, fold_of), "tau070": prepare(a.b, feats, gt, fold_of)}
    res = {"control_ok": bool(control_ok)}
    log(f"\n  {'policy':<14}{'pool':<8}{'kept':>7}{'PQ':>9}{'TP':>7}{'FP':>7}{'FN':>7}   per-fold PQ")
    for conf, pcut in POLICIES:
        for name, cand in pools.items():
            per = []
            for f in range(5):
                r, _ = exact_at(cand, gt, conf, pcut, fold=f)
                per.append(r["pq"])
            r, n = exact_at(cand, gt, conf, pcut)
            res[f"{name}_{conf}_{pcut}"] = dict(kept=n, pq_pooled=r["pq"], tp=r["tp"], fp=r["fp"], fn=r["fn"],
                                                per_fold=per, pq_mean=float(np.mean(per)))
            log(f"  ({conf:.2f},{pcut:.2f})   {name:<8}{n:>7}{r['pq']:>9.4f}{r['tp']:>7}{r['fp']:>7}{r['fn']:>7}   "
                + " ".join(f"{x:.4f}" for x in per))
        d = [res[f"tau070_{conf}_{pcut}"]["per_fold"][i] - res[f"tau0_{conf}_{pcut}"]["per_fold"][i] for i in range(5)]
        dm = float(np.mean(d))
        is_verdict = abs(pcut - 0.60) < 1e-9
        if dm >= FLOOR:
            verdict = "PAYS (>= +0.003)"
        elif dm <= -FLOOR:
            verdict = "HURTS (<= -0.003)"
        else:
            verdict = "point estimate inside ±0.003 -> needs the interval"
        tag = "  <- PRE-REGISTERED VERDICT" if is_verdict else "  (diagnostic)"
        log(f"    Δ tau0.70 - tau0 at ({conf:.2f},{pcut:.2f}): mean {dm:+.4f}, folds+ {sum(x > 0 for x in d)}/5, "
            f"per fold " + " ".join(f"{x:+.4f}" for x in d) + f"  -> {verdict}{tag}")
        res[f"delta_{conf}_{pcut}"] = dict(mean=dm, per_fold=d, verdict=verdict)

        if is_verdict:
            # paired image-level bootstrap on the verdict contrast: E115c's
            # machinery, guard at BOTH pools, images resampled whole, pooled PQ
            # recomputed from per-image parts on each resample
            stems = np.array(sorted(set(pools["tau0"]["stem"]) | set(pools["tau070"]["stem"])))
            ncopy = gt.groupby("stem")["entry_id"].nunique().reindex(stems).fillna(1).to_numpy()
            parts = {}
            for name, cand in pools.items():
                sub = cand[(cand["conf"] >= conf) & (cand["p_oof"] >= pcut)]
                st, r = per_image(sub, gt, stems, ncopy)
                tot = pq(st["tp"].sum(), st["fp"].sum(), st["fn"].sum(), st["iou"].sum())
                log(f"    guard {name}: decomposed {tot:.4f} vs exact {r['pq']:.4f} "
                    f"(FP {int(st['fp'].sum())}/{r['fp']})")
                if abs(tot - r["pq"]) > 5e-4:
                    log("    GUARD FAILED at this pool -> no interval, no equivalence claim")
                    parts = None
                    break
                parts[name] = st
            if parts:
                rng = np.random.default_rng(0)
                n = len(stems)
                boot = np.empty(REPS)
                for b in range(REPS):
                    idx = rng.integers(0, n, n)
                    v = [pq(s["tp"][idx].sum(), s["fp"][idx].sum(), s["fn"][idx].sum(), s["iou"][idx].sum())
                         for s in (parts["tau0"], parts["tau070"])]
                    boot[b] = v[1] - v[0]
                obs = (pq(*(parts["tau070"][k].sum() for k in ("tp", "fp", "fn", "iou")))
                       - pq(*(parts["tau0"][k].sum() for k in ("tp", "fp", "fn", "iou"))))
                lo, hi = np.percentile(boot, [2.5, 97.5])
                sd = float(boot.std())
                inside = lo > -EQUIV_BAND and hi < EQUIV_BAND
                overshoot = max(-EQUIV_BAND - lo, hi - EQUIV_BAND, 0.0)
                needed_sd = max(EQUIV_BAND - abs(obs), 0.0) / 1.96
                if dm >= FLOOR or dm <= -FLOOR:
                    final = verdict
                elif inside:
                    final = f"TIED: 95% interval inside ±{EQUIV_BAND} -> the sub025/sub026 ordering carries no information"
                else:
                    final = (f"UNDERPOWERED: interval reaches {overshoot:.4f} past ±{EQUIV_BAND}; "
                             f"sd obtained {sd:.4f}, sd needed {needed_sd:.4f} -> the twin does not settle it")
                log(f"    pooled Δ {obs:+.4f}; paired image bootstrap {REPS} reps over {n} images: "
                    f"95% CI [{lo:+.4f}, {hi:+.4f}], sd {sd:.4f}, P(Δ>0) {(boot > 0).mean():.3f}")
                log(f"    FINAL: {final}" + ("" if control_ok else "   [CONTROL FAILED -> VOID]"))
                res["interval"] = dict(observed=float(obs), ci=[float(lo), float(hi)], sd=sd,
                                       p_gt0=float((boot > 0).mean()), inside=bool(inside),
                                       overshoot=float(overshoot), needed_sd=float(needed_sd),
                                       equiv_band=EQUIV_BAND, final=final)
    json.dump(res, open(OUT / "e098d_pool_twin.json", "w"), indent=1, default=float)
    log(f"\nwrote {OUT/'e098d_pool_twin.json'}")


if __name__ == "__main__":
    main()
