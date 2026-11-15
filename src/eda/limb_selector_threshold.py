"""E119 -- a limb-specific selector threshold on the HONEST out-of-fold score.

Ledger §16: at r > 0.9, 36% of GT rows have a pool candidate that matches
at IoU > 0.5 but was not emitted, and at r 0.8-0.9 that share is 17.5%
against 11-13% on the disc. The selector, not the model, is the first-order
loss there. A radius-dependent threshold was tried once on the leaked
in-sample p_sel (−0.0003); on the honest p_oof (§11a), with the threshold
chosen fold-honestly, it has not been measured.

Policy: conf >= 0.40 everywhere; p_oof >= 0.60 for candidates with
r_norm <= R0; p_oof >= tau_limb for candidates with r_norm > R0. R0 and
tau_limb are chosen on the four training folds by exact PQ (a small grid, so
exact scoring is affordable), then applied to the held-out fold. The control
is the production policy on the same fold (tau_limb = 0.60).

PRE-REGISTERED: gate = mean Δ over folds >= +0.003 AND >= 4/5 folds positive,
plus the §11c interval machinery is NOT needed here because the claim being
tested is "pays", not "tied": a point estimate above the floor with 4/5 folds
is the same standard every other selector change met. Passing makes it a
single-variable submission candidate against sub025 (only the limb
threshold changes), to be put to Talha, never submitted unasked. Reported as
diagnostics: rows reached per radius band under the chosen policy, and the
FP:TP composition of what the looser limb threshold admits.

Usage: python src/eda/limb_selector_threshold.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament.pq import score

OUT = ROOT / "outputs" / "eda"
R0S = [0.75, 0.80, 0.85]
TAUS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
FLOOR = 0.003


def log(*a):
    print(*a, flush=True)


def exact(cand, gt, r0, tau, fold):
    sub = cand[(cand["fold"] == fold) & (cand["conf"] >= 0.40) &
               (((cand["r_norm"] <= r0) & (cand["p_oof"] >= 0.60)) |
                ((cand["r_norm"] > r0) & (cand["p_oof"] >= tau)))]
    g = gt[gt["fold"] == fold]
    pred = sub[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
    return score(g[["filament_id", "segmentation_rle"]], pred), len(sub)


def main():
    cand = pd.read_parquet(OUT / "e115b_labelled_pool.parquet")
    po = pd.read_parquet(OUT / "e115b_p_oof.parquet").set_index("cand_id")["p_oof"]
    cand["p_oof"] = cand["cand_id"].map(po)
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    cand["fold"] = cand["stem"].map(fold_of).astype(int)
    gt["fold"] = gt["stem"].map(fold_of)
    gt = gt[gt["fold"].notna()].copy()
    gt["fold"] = gt["fold"].astype(int)
    log(f"pool {len(cand)}, GT {len(gt)} (all folds_v1 stems); limb candidates r>0.8: "
        f"{int((cand['r_norm'] > 0.8).sum())}, of which emitted now {int(((cand['r_norm'] > 0.8) & (cand['conf'] >= 0.4) & (cand['p_oof'] >= 0.6)).sum())}")

    # exact PQ per fold for every grid point (cached), then fold-honest choice
    table = {}
    for f in range(5):
        for r0 in R0S:
            for tau in TAUS:
                r, n = exact(cand, gt, r0, tau, f)
                table[(f, r0, tau)] = (r["pq"], r["tp"], r["fp"], r["fn"], n)
        log(f"  fold {f} grid scored")
    res = {"per_fold": []}
    log(f"\n  {'fold':<5}{'chosen (R0, tau)':<18}{'base PQ':>9}{'limb PQ':>9}{'Δ':>9}{'TP':>10}{'FP':>10}")
    deltas = []
    for f in range(5):
        others = [k for k in range(5) if k != f]
        best = max(((r0, tau) for r0 in R0S for tau in TAUS),
                   key=lambda rt: np.mean([table[(k, rt[0], rt[1])][0] for k in others]))
        base = table[(f, 0.80, 0.60)]     # tau 0.60 everywhere == production policy (R0 irrelevant)
        pick = table[(f, best[0], best[1])]
        d = pick[0] - base[0]
        deltas.append(d)
        res["per_fold"].append(dict(fold=f, r0=best[0], tau=best[1], base=base[0], limb=pick[0], delta=d,
                                    tp=(base[1], pick[1]), fp=(base[2], pick[2])))
        log(f"  {f:<5}{str(best):<18}{base[0]:>9.4f}{pick[0]:>9.4f}{d:>+9.4f}{base[1]:>5}>{pick[1]:<4}{base[2]:>5}>{pick[2]:<4}")
    dm = float(np.mean(deltas))
    pos = sum(x > 0 for x in deltas)
    verdict = "GO: submission candidate vs sub025 (ask Talha)" if (dm >= FLOOR and pos >= 4) else "NO GO"
    log(f"\n  mean Δ {dm:+.4f}, folds positive {pos}/5 -> {verdict}")

    # what the fixed policy (0.80, 0.45) does per band, as a diagnostic
    log("\n  mean PQ over folds at fixed (R0, tau) grid (diagnostic):")
    for r0 in R0S:
        log(f"    R0 {r0}: " + " ".join(f"tau{t:.2f}={np.mean([table[(k, r0, t)][0] for k in range(5)]):.4f}" for t in TAUS))
    json.dump(dict(res=res, mean_delta=dm, folds_positive=pos, verdict=verdict,
                   grid={f"{k[0]}_{k[1]}_{k[2]}": v for k, v in table.items()}),
              open(OUT / "limb_selector_threshold.json", "w"), indent=1, default=float)
    log(f"\nwrote {OUT/'limb_selector_threshold.json'}")


if __name__ == "__main__":
    main()
