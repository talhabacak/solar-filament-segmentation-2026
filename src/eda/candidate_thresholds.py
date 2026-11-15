"""How much reachable GT are we leaving on the table at candidate generation?

Fold 0's raw candidates reach 0.8232 of the GT rows, and the fused pool the
selector sees reaches 0.7048 pooled. Everything downstream is bounded by that
raw number, and it was never swept — the detector floor (conf 0.05) and the
semantic cut (prob 0.3, area 256) were set early and never revisited.

Cheap to move, and there is a real reason to expect headroom: Unidentifiable is
37.5% of the GT, has the worst recall (0.482 against 0.66/0.62), makes up 61% of
never-found instances, and is SHORTER than the others (median spine 78 px vs
135) at the same width — so it is exactly the population a minimum-area cut and
a confidence floor would remove first.

Reports raw reachability per setting, not PQ: more candidates always costs
precision, and the selector's job is to pay that back. What matters here is
whether the GT becomes reachable at all.

Usage: python src/eda/candidate_thresholds.py --fold 0
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament.pq import score
from filament.rle import encode_mask
from semantic_candidates import prob_to_instances, R_MAX

OUT = ROOT / "outputs" / "eda"


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    a = ap.parse_args()

    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    stems = [f.rsplit(".", 1)[0]
             for f in sorted(folds.loc[folds["fold"] == a.fold, "file_name"])]
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt_f = gt[gt["file_name"].isin({s + ".jpeg" for s in stems})]
    N = len(gt_f)
    cat = dict(zip(gt_f["filament_id"], gt_f["category_id"]))
    n_u = sum(1 for v in cat.values() if v == 3)

    det_all = pd.read_parquet(ROOT / "outputs" / "oof_cache" / f"fold{a.fold}_preds_e013.parquet")
    if "view" in det_all.columns:
        det_all = det_all[det_all["view"] == "o"]
    det_all = det_all[det_all["r_norm"] <= R_MAX]
    z = np.load(ROOT / "outputs" / "semantic" / f"fold{a.fold}_probs_cldice.npz")
    probs = {s: z[s].astype(np.float32) / 255.0 for s in stems}

    gt_cache, res = {}, {}

    def reach(name, rows):
        df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
        r = score(gt_f[["filament_id", "segmentation_rle"]], df, gt_cache=gt_cache,
                  return_details=True)
        hits = set(r["matches"]["gt_filament_id"])
        u_hit = sum(1 for h in hits if cat.get(h) == 3)
        log(f"{name:34s} preds {len(df):6d}  reach {len(hits):4d}/{N}={len(hits)/N:.4f}"
            f"  U {u_hit:4d}/{n_u}={u_hit/max(n_u,1):.4f}")
        res[name] = dict(n_preds=len(df), reach=len(hits) / N, u_reach=u_hit / max(n_u, 1))
        return hits

    t0 = time.time()
    det_rows = {}
    for c in (0.05, 0.02, 0.01):
        d = det_all[det_all["conf"] >= c]
        det_rows[c] = [(f"{s}_d{i}", r) for i, (s, r) in
                       enumerate(zip(d["stem"], d["rle"]))]
        reach(f"detector conf>={c}", det_rows[c])

    sem_rows = {}
    for thr, area in ((0.3, 256), (0.2, 256), (0.2, 128), (0.15, 96), (0.1, 64)):
        rows = []
        for s in stems:
            for k, c in enumerate(prob_to_instances(probs[s], thr, area)):
                rows.append((f"{s}_s{k}", encode_mask(c["mask"].astype(np.uint8))))
        sem_rows[(thr, area)] = rows
        reach(f"semantic thr>={thr} area>={area}", rows)
    log(f"  ({time.time()-t0:.0f}s so far)")

    reach("UNION current (0.05 | 0.3/256)", det_rows[0.05] + sem_rows[(0.3, 256)])
    reach("UNION loose  (0.01 | 0.1/64)", det_rows[0.01] + sem_rows[(0.1, 64)])
    reach("UNION mid    (0.02 | 0.2/128)", det_rows[0.02] + sem_rows[(0.2, 128)])

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"candidate_thresholds_fold{a.fold}.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    log(f"saved -> {OUT / f'candidate_thresholds_fold{a.fold}.json'}")


if __name__ == "__main__":
    main()
