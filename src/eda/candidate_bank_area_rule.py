"""Area-conditioned representation rule on the candidate bank
(_review_size_conditioned.md §3.2): the one form of the document's
"small -> shrink, large -> grow" hypothesis that can ADD pixels to a mask in
our pipeline (a threshold can only remove them; union adds).

Single hypothesis, fixed reject (vote50 IoU > 0.5, as in ledger §10b so the
number is comparable), legal pass, exact scorer, fold 0:
    R_area      vote50 area < q25 -> er_vote ; > q75 -> union ; else vote50
    R_area_rev  the reverse (small -> union, large -> er_vote), the control:
                if it scores the same, the signal is not in area
Quantiles are taken over the fold's vote50 areas.

PRE-REGISTERED GATE (from the review, unchanged): fold-0 Δ vs prod >= +0.003
AND >= 0.005 above the reversed rule; only then folds 1-4. Δ <= 0 closes the
axis including the document's §8/§10. Expectation stated before the run:
negative (R_thin −0.010, R_erode −0.021, R_union −0.058 in §10b); the one open
question is whether "which cluster wants erosion / union" correlates with area
at all, and the ceiling even if it does is ~+0.005 (65 winning clusters).

Usage: python src/eda/candidate_bank_area_rule.py --fold 0
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
from filament.rle import decode_rle
from candidate_bank_capture import legal_pass

OUT = ROOT / "outputs" / "eda"


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    a = ap.parse_args()
    vb = pd.read_parquet(OUT / f"variant_bank_fold{a.fold}.parquet")
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt = gt[gt["stem"].isin(set(vb["stem"]))][["filament_id", "segmentation_rle"]]
    rle_of = dict(zip(vb.index, vb["rle"]))

    def dec(i):
        return decode_rle(rle_of[i]).astype(bool)

    piv = {cid: g for cid, g in vb.groupby("cluster_id")}
    base_area = vb.loc[vb["variant"] == "vote50", "area"]
    q25, q75 = float(base_area.quantile(0.25)), float(base_area.quantile(0.75))
    log(f"fold {a.fold}: {len(piv)} clusters; vote50 area q25 {q25:.0f}, q75 {q75:.0f}")

    def pick(rule):
        chosen = []
        for cid, g in piv.items():
            base = g[g["variant"] == "vote50"]
            if base.empty or base.iloc[0]["iou"] <= 0.5:
                continue
            b = base.iloc[0]
            v = rule(b)
            row = g[g["variant"] == v]
            if row.empty:
                row = base
            r = row.iloc[0]
            chosen.append((r["stem"], float(r["conf"]), dec(row.index[0])))
        return chosen

    rules = {
        "prod":       lambda b: "vote50",
        "R_area":     lambda b: "er_vote" if b["area"] < q25 else ("union" if b["area"] > q75 else "vote50"),
        "R_area_rev": lambda b: "union" if b["area"] < q25 else ("er_vote" if b["area"] > q75 else "vote50"),
    }
    res = {}
    log(f"\n  {'arm':<12}{'preds':>7}{'PQ':>9}{'SQ':>9}{'RQ':>9}{'FN':>6}{'FP':>6}{'Δ':>9}")
    for name, rule in rules.items():
        pdf = legal_pass(pick(rule), None)
        r = score(gt, pdf)
        res[name] = dict(n=len(pdf), pq=r["pq"], sq=r["sq"], rq=r["rq"], fn=r["fn"], fp=r["fp"])
        d = r["pq"] - res["prod"]["pq"]
        log(f"  {name:<12}{len(pdf):>7}{r['pq']:>9.4f}{r['sq']:>9.4f}{r['rq']:>9.4f}{r['fn']:>6}{r['fp']:>6}{d:>+9.4f}")
    d = res["R_area"]["pq"] - res["prod"]["pq"]
    dr = res["R_area_rev"]["pq"] - res["prod"]["pq"]
    gate = d >= 0.003 and (d - dr) >= 0.005
    log(f"\n  gate: Δ {d:+.4f} (>= +0.003?) and Δ − Δ_rev {d - dr:+.4f} (>= 0.005?) -> "
        f"{'GO to folds 1-4' if gate else 'CLOSED (no fold 1-4)'}")
    json.dump(dict(fold=a.fold, q25=q25, q75=q75, res=res, delta=d, delta_rev=dr, gate=bool(gate)),
              open(OUT / f"candidate_bank_area_rule_fold{a.fold}.json", "w"), indent=1)
    log(f"wrote {OUT/f'candidate_bank_area_rule_fold{a.fold}.json'}")


if __name__ == "__main__":
    main()
