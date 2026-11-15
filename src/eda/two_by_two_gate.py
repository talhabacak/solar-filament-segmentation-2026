"""Two seeds x two folds: the only gate that can see a decoder change (ledger 8o).

A single fold-0, single-seed arm carries a ~40-row noise floor (the production
recipe's own second seed measured -42). So an architecture arm is judged here
against the SPREAD of baseline seeds, on two folds:

  per fold f:  baseline seeds  B_f = {production model, re-run}
               arm seeds       A_f = {s1, s2}
               net rows reached (gate_instance_quality.measure, IoU > 0.5 at
               the production threshold, both populations) for each
  statistic:   mean(A_f) - mean(B_f) per fold; the pooled seed spread
               (half the |B_f1 - B_f2| and |A_f1 - A_f2| ranges) is the noise
               scale; PASS if the arm beats the baseline mean on BOTH folds by
               more than the pooled spread, otherwise the arm is inside the
               noise and stays a committee-member candidate only.

Usage: python src/eda/two_by_two_gate.py --arm coordatt --folds 0,1
  fold-0 tags: base cldice, rerun cldice_rerun, arm coordatt_s1/coordatt_s2
  fold-1 tags: base cldice, rerun cldice_rerun, arm coordatt_s1/coordatt_s2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "eda"))
from gate_instance_quality import measure  # noqa: E402

OUT = ROOT / "outputs" / "eda"
SEM = ROOT / "outputs" / "semantic"


def log(*a):
    print(*a, flush=True)


def rows_for(fold):
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    st = set(folds.loc[folds["fold"] == fold, "file_name"].str.replace(".jpeg", "", regex=False))
    d = pd.read_parquet(OUT / "error_profile_fn_gt.parquet")
    d["matched"] = d["matched"].astype(bool)
    d["in_raw_pool"] = d["in_raw_pool"].astype(bool)
    d = d[d["stem"].isin(st)].copy()
    d["grp"] = np.where(d["matched"], "matched", np.where(d["in_raw_pool"], "in-pool", "never-proposed"))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="tag prefix of the arm maps, e.g. coordatt -> fold{k}_probs_coordatt_s{1,2}.npz")
    ap.add_argument("--folds", default="0,1")
    ap.add_argument("--base", default="cldice")
    ap.add_argument("--rerun", default="cldice_rerun")
    a = ap.parse_args()
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet").set_index("filament_id")["segmentation_rle"]
    res = {}
    for f in [int(x) for x in a.folds.split(",")]:
        d = rows_for(f)
        vals = {}
        for name, tag in (("base", a.base), ("rerun", a.rerun), ("s1", f"{a.arm}_s1"), ("s2", f"{a.arm}_s2")):
            p = SEM / f"fold{f}_probs_{tag}.npz"
            if not p.exists():
                log(f"fold {f}: {p.name} missing"); vals[name] = None; continue
            df, pdice = measure(p, d, gt)
            net = int((df["iou"] > 0.5).sum())
            pops = {g: int((df.loc[df.grp == g, "iou"] > 0.5).sum()) for g in ("matched", "never-proposed", "in-pool")}
            npd = float(df.loc[df.grp == "never-proposed", "dice"].mean())
            vals[name] = dict(net=net, pixel_dice=float(pdice), dice_all=float(df["dice"].mean()), pops=pops, np_dice=npd)
            log(f"fold {f} {name:6s} {tag:16s} net {net:5d} (m {pops['matched']} np {pops['never-proposed']} pool {pops['in-pool']})  "
                f"NP-Dice {npd:.4f}  pixel-Dice {pdice:.4f}  inst-Dice {df['dice'].mean():.4f}")
        res[f] = vals
    log("\nfold   base  rerun |   s1    s2 |  arm-base(mean)  seed spread")
    passes, deltas = [], []
    for f, v in res.items():
        if any(v[k] is None for k in ("base", "rerun", "s1", "s2")):
            log(f"{f:4d}   incomplete"); continue
        b = np.array([v["base"]["net"], v["rerun"]["net"]]); s = np.array([v["s1"]["net"], v["s2"]["net"]])
        spread = 0.5 * (abs(b[0] - b[1]) + abs(s[0] - s[1])) / 2
        d = s.mean() - b.mean()
        deltas.append(d); passes.append(d > spread)
        log(f"{f:4d} {b[0]:6d} {b[1]:6d} | {s[0]:5d} {s[1]:5d} |  {d:+8.1f}          {spread:5.1f}")
    if deltas:
        verdict = "PASS: beats the baseline seed mean on every fold by more than the seed spread" if all(passes) and len(passes) >= 2 \
            else "INSIDE THE NOISE: not a replacement; committee-member candidate only"
        log(f"\nmean delta over folds {np.mean(deltas):+.1f} rows -> {verdict}")
        res["verdict"] = verdict
    json.dump(res, open(OUT / f"two_by_two_{a.arm}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
