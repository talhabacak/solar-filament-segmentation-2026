"""E098c -- does the carve-retention rule have anything left to remove once the
selector threshold has run?

E098b killed both explanations offered for §12's board failure. Re-running the
greedy pass instead of filtering a frozen carving changes nothing (agreement to
the fourth decimal at every tau). And removing the GT filter from the base does
not shrink the gain, it triples it: on the unfiltered vote50 population tau=0.70
is +0.0270 against +0.0051 on the GT-matched one, 5/5 folds either way. So the
rule is a genuinely good false-positive filter -- on a pool that still contains
false positives for it to remove.

That points back at the one thing E098b held fixed: the pool. Its base has no
selector threshold at all. Production applies conf >= 0.40 and p_sel >= 0.60
first, and that threshold removes weak and overlapping candidates too. The
hypothesis this script tests is that the two mechanisms address ONE population,
so the rule's headroom falls as the threshold rises, and by p >= 0.60 there is
essentially nothing left for it to take -- which is what the board saw when
sub026 emitted three predictions MORE than sub025 and ranked below it.

Design: the same out-of-fold pool and honest p_sel as E115b/E115c, the
production greedy de-overlap ordered by confidence, and the retention rule
applied inside that pass. Measured at three thresholds so the trend is visible,
not just the endpoint, and reported as both PQ and "how many candidates does the
rule actually drop", because a rule that fires on nothing cannot help or hurt
much either way.

PRE-REGISTERED VERDICT, written before this script was run and before any of
its numbers existed. "Near zero" has to be a number or it gets interpreted
afterwards.

  Primary, headroom measured directly: the FALSE-POSITIVE UNITS the rule can
  possibly remove at p_sel >= 0.60 across the five folds. The pool there scores
  PQ 0.4361 with a denominator near 7346 (E115b: TP 4719, FP 1779, FN 3476).
  Removing one FP unit takes 0.5 off that denominator, so lifting PQ by the
  +0.003 transfer floor needs about 100 FP units removed. If the rule's total
  is below 100 it cannot reach the floor even if every unit it touches were a
  false positive -> NO HEADROOM, independent of the PQ column.
  The unit matters and the first draft of this criterion got it wrong. The
  scorer charges a prediction once per annotator entry it misses, so dropping
  one candidate removes n_copies FP units, not one; copies average about 1.6
  here, so a candidate-count threshold of 100 would have silently been a
  62-candidate threshold and would have called "no headroom" over a band where
  the rule could still clear the floor -- an error pointing at the hypothesis
  being tested. So the script does not convert with an average at all: it sums
  n_copies over exactly the candidates the rule fires on. That is exact and
  survives any skew in the fired-on set toward 1-copy or 3-copy images, which
  is plausible since consensus tier tracks candidate quality and the rule
  selects on quality. The raw candidate count is still reported as the
  interpretable headroom number.
  Secondary, PQ: mean Δ over folds at p_sel >= 0.60 below +0.003 -> no gain.
  Qualitative: the gain decays monotonically as the cut rises. If instead it
  survives at 0.60, the overlap hypothesis is dead too and §12 records that we
  have no explanation for the board result.

WHAT THIS CANNOT DO. It runs the honest out-of-fold pool on TRAIN images; the
board ran the deployed all-fold selector on TEST images. Those are different
populations, established while re-measuring E115. So this can confirm or refute
the mechanism, but it cannot reproduce the sub025/sub026 comparison, and §12
must not be written as though it did.

Usage: python src/eda/e098c_carve_on_production.py
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

from filament import IMG_H, IMG_W
from filament.pq import score
from filament.rle import decode_rle, encode_mask
from e115c_threshold_ci import oof_scores, CACHE

OUT = ROOT / "outputs" / "eda"
BUNDLE = ROOT / "outputs" / "selector" / "selector_v5_shrunkblob5prodts.joblib"
TAUS = [0.0, 0.70, 0.75]
PCUTS = [0.30, 0.45, 0.55, 0.60]


def log(*a):
    print(*a, flush=True)


def main():
    import joblib
    feats = joblib.load(BUNDLE)["features"]
    if not CACHE.exists():
        sys.exit(f"{CACHE} not there yet -- run src/eda/e115b_oof_threshold.py first")
    cand = pd.read_parquet(CACHE)
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt = gt[gt["stem"].isin(set(cand["stem"]))]
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    cand["fold"] = cand["stem"].map(fold_of).astype(int)
    gt["fold"] = gt["stem"].map(fold_of).astype(int)
    pcache = OUT / "e115b_p_oof.parquet"
    if pcache.exists():
        cand["p_oof"] = pd.read_parquet(pcache)["p_oof"].to_numpy()
        log("out-of-fold selector scores from cache")
    else:
        log("fitting out-of-fold selector scores:")
        cand["p_oof"] = oof_scores(cand, feats, cand["fold"].to_numpy())
        cand[["cand_id", "p_oof"]].to_parquet(pcache, index=False)
    rle_of = dict(zip(cand["cand_id"], cand["rle"]))

    # FP units per candidate: the scorer charges a prediction once per annotator
    # entry it misses, so a dropped candidate is worth n_copies units
    ncopy = gt.groupby("stem")["entry_id"].nunique()
    cand["n_copies"] = cand["stem"].map(ncopy).fillna(1).astype(int)
    copies_of = dict(zip(cand["cand_id"], cand["n_copies"]))

    res, gt_cache = {}, {}
    log(f"\n  {'p_sel':>6}{'tau':>6}{'preds':>8}{'dropped':>9}{'FPunits':>9}{'PQ':>9}{'Δ vs tau 0':>12}{'TP':>7}{'FP':>7}")
    for pcut in PCUTS:
        kept = cand[(cand["conf"] >= 0.40) & (cand["p_oof"] >= pcut)]
        base_pq = None
        for tau in TAUS:
            per_fold, dropped_tot, n_tot, units_tot = [], 0, 0, 0
            for f in range(5):
                kf = kept[kept["fold"] == f]
                gf = gt[gt["fold"] == f][["filament_id", "segmentation_rle"]]
                preds, dropped, units = [], 0, 0
                for stem, g in kf.groupby("stem"):
                    taken = np.zeros((IMG_H, IMG_W), bool)
                    k = 0
                    for r in g.sort_values("conf", ascending=False).itertuples():
                        m = decode_rle(rle_of[r.cand_id]).astype(bool)
                        a0 = int(m.sum())
                        mm = m & ~taken
                        a1 = int(mm.sum())
                        if a1 < 64:
                            continue
                        if a0 > 0 and a1 / a0 < tau:
                            dropped += 1
                            units += copies_of[r.cand_id]
                            continue
                        k += 1
                        preds.append((f"{stem}_{k}", encode_mask(mm.astype(np.uint8))))
                        taken |= mm
                r = score(gf, pd.DataFrame(preds, columns=["filament_id", "segmentation_rle"]), gt_cache=gt_cache)
                per_fold.append(r)
                dropped_tot += dropped
                units_tot += units
                n_tot += len(preds)
            pq = float(np.mean([x["pq"] for x in per_fold]))
            if tau == 0.0:
                base_pq = pq
            res[f"p{pcut}_tau{tau}"] = dict(pq=pq, n=n_tot, dropped=dropped_tot, fp_units=units_tot,
                                            tp=int(sum(x["tp"] for x in per_fold)),
                                            fp=int(sum(x["fp"] for x in per_fold)),
                                            per_fold=[x["pq"] for x in per_fold])
            d = [per_fold[i]["pq"] - res[f"p{pcut}_tau0.0"]["per_fold"][i] for i in range(5)]
            verdict = ""
            if tau > 0 and abs(pcut - 0.60) < 1e-9:
                verdict = ("   PRIMARY: FP units " + str(units_tot) +
                           (" < 100 -> NO HEADROOM" if units_tot < 100 else " >= 100 -> headroom exists"))
            log(f"  {pcut:>6.2f}{tau:>6.2f}{n_tot:>8}{dropped_tot:>9}{units_tot:>9}{pq:>9.4f}"
                f"{pq - base_pq:>+12.4f}{sum(x['tp'] for x in per_fold):>7}"
                f"{sum(x['fp'] for x in per_fold):>7}"
                + ("" if tau == 0.0 else f"   folds+ {sum(x > 0 for x in d)}/5") + verdict)
    json.dump(res, open(OUT / "e098c_carve_on_production.json", "w"), indent=1, default=float)
    log(f"\nwrote {OUT/'e098c_carve_on_production.json'}")


if __name__ == "__main__":
    main()
