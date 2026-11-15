"""E209 -- the selector-opportunity ceiling, per high-frequency quartile.

The regime block read +0.0009 pooled with 72% of it coming from the target slice, but that says
nothing about how much was ON THE TABLE. This measures the ceiling: how much PQ would a PERFECT
per-candidate decision recover in each hf quartile, from the pool we already build?

Three policies, all scored with the exact scorer on the deployed pool:

  deployed     conf >= 0.40 and p >= 0.60
  admit-only   deployed, plus every REJECTED candidate whose realised value clears break-even
  full         keep exactly the candidates that clear break-even, dropping kept ones that do not

Break-even is the metric's own rule, not a label: a kept candidate pays iff v > 0.5*c*PQ, where
v = iou_sum (its realised IoU summed over the annotator copies it matches) and c = n_copies. It
is the same inequality as p*q > 0.5*PQ for c = 1; it uses the realised value, so it is an oracle.

Read per quartile because the mechanism is a quartile effect. If the Q4 ceiling is small the axis
is genuinely closed there; if it is large the current intervention captures a small part of a real
opportunity, which is a different conclusion and a different next step.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/eda", "src/models"):
    sys.path.insert(0, str(ROOT / p))
from e115c_threshold_ci import per_image, pq  # noqa: E402

OUT = ROOT / "outputs" / "eda"
GATE = (0.40, 0.60)


def main():
    pool = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    pool = pool.merge(pd.read_parquet(OUT / "bag5_oof_scores.parquet"), on="cand_id")
    q = pd.read_parquet(OUT / "image_quality.parquet")
    pool = pool.merge(q[q.split == "train"][["stem", "hf"]], on="stem", how="left")
    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt = gt[gt["stem"].isin(set(pool["stem"]))]
    stems = np.array(sorted(set(pool["stem"])))
    ncopy = gt.groupby("stem")["entry_id"].nunique().reindex(stems).fillna(1).to_numpy()

    kept = ((pool["conf"] >= GATE[0]) & (pool["p"] >= GATE[1])).to_numpy()
    base_parts = per_image(pool[kept][["cand_id", "rle", "stem"]], gt, stems, ncopy)[0]
    P0 = pq(base_parts["tp"].sum(), base_parts["fp"].sum(), base_parts["fn"].sum(),
            base_parts["iou"].sum())
    print(f"dagitilan PQ {P0:.4f}, tutulan {int(kept.sum())}\n")
    pool["net"] = pool["iou_sum"] - 0.5 * pool["n_copies"] * P0
    pays = (pool["net"] > 0).to_numpy()
    policies = {"dagitilan": kept,
                "admit-only": kept | (~kept & pays),
                "tam oracle": pays}
    hf_img = pool.drop_duplicates("stem").set_index("stem")["hf"].reindex(stems)
    qu = pd.qcut(hf_img, 4, labels=["Q1", "Q2", "Q3", "Q4"]).to_numpy()
    parts = {}
    for name, keep in policies.items():
        parts[name] = per_image(pool[keep][["cand_id", "rle", "stem"]], gt, stems, ncopy)[0]
        print(f"{name:12s} tutulan {int(keep.sum()):5d}  PQ "
              f"{pq(parts[name]['tp'].sum(), parts[name]['fp'].sum(), parts[name]['fn'].sum(), parts[name]['iou'].sum()):.4f}",
              flush=True)
    res = {}
    print(f"\n{'ceyrek':8s}{'n':>5s}{'dagitilan':>11s}{'admit-only':>12s}{'tavan':>9s}"
          f"{'tam oracle':>12s}{'tavan':>9s}{'atilan+':>9s}")
    for lab in ["Q1", "Q2", "Q3", "Q4"]:
        m = qu == lab
        v = {}
        for name in policies:
            pt = parts[name]
            v[name] = pq(pt["tp"][m].sum(), pt["fp"][m].sum(), pt["fn"][m].sum(), pt["iou"][m].sum())
        sl = pool[(~kept) & pays & pool["stem"].isin(stems[m])]
        print(f"{lab:8s}{int(m.sum()):>5d}{v['dagitilan']:>11.4f}{v['admit-only']:>12.4f}"
              f"{v['admit-only']-v['dagitilan']:>+9.4f}{v['tam oracle']:>12.4f}"
              f"{v['tam oracle']-v['dagitilan']:>+9.4f}{len(sl):>9d}")
        res[lab] = dict(n=int(m.sum()), deployed=float(v["dagitilan"]),
                        admit_only=float(v["admit-only"]), full=float(v["tam oracle"]),
                        ceiling_admit=float(v["admit-only"] - v["dagitilan"]),
                        ceiling_full=float(v["tam oracle"] - v["dagitilan"]),
                        n_rejected_paying=int(len(sl)))
    json.dump(res, open(OUT / "hf_oracle_ceiling.json", "w"), indent=1)
    print("\nokuma: 'tavan' = mukemmel aday-basi kararin o ceyrekte geri kazanacagi PQ.")
    print("regime blogu ayni ceyreklerde: Q1 -0.0002, Q2 +0.0018, Q3 -0.0006, Q4 +0.0026")


if __name__ == "__main__":
    main()
