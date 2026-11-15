"""E215b -- are shape-band misses fragments of a filament we did find?

R6 asks for GT 1 -> Pred N (split / fragmentation) to be measured on its own. In the error atlas a
copy is 'shape' when its best single pool candidate reaches IoU in (0.05, 0.5]. If two or more
KEPT candidates each lie mostly inside the copy and their UNION reaches IoU > 0.5, the copy is a
split: the object was found, the instance identity was not. That would be a grouping lever
(fragment linking), not a model-strength one.

For each shape-band copy (outcome == 'shape' in outputs/eda/error_atlas.parquet), with sub036's
out-of-fold keep set (conf >= 0.40, bag5 p >= 0.60):
  fragments  kept candidates with >= 50 % of their pixels inside the copy
  best1      best single kept IoU
  union      IoU of the union of the fragments
  class      split_recoverable  (>= 2 fragments, union IoU > 0.5)
             split_partial      (>= 2 fragments, union IoU <= 0.5)
             single_too_small   (1 fragment, its pixels mostly inside the copy)
             other              (no kept candidate mostly inside: offset, merged with a
                                 neighbour, or only rejected candidates)

Usage: python src/eda/e215b_shape_band_fragments.py
Output: outputs/eda/e215b_shape_band_fragments.json / .parquet
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from filament.rle import decode_rle  # noqa: E402

OUT = ROOT / "outputs" / "eda"


def main():
    atlas = pd.read_parquet(OUT / "error_atlas.parquet")
    shape = atlas[atlas["outcome"] == "shape"]
    a = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet",
                        columns=["stem", "cand_id", "rle", "conf"])
    a = a.merge(pd.read_parquet(OUT / "bag5_oof_scores.parquet"), on="cand_id")
    keep = a[(a["conf"] >= 0.40) & (a["p"] >= 0.60)]
    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet", columns=["filament_id", "segmentation_rle"])
    gt_rle = dict(zip(gt["filament_id"], gt["segmentation_rle"]))

    rows = []
    for stem, g in shape.groupby("stem"):
        kc = keep[keep["stem"] == stem]
        km = [decode_rle(r).astype(bool) for r in kc["rle"]]
        ka = [m.sum() for m in km]
        for fid in g["fid"]:
            G = decode_rle(gt_rle[fid]).astype(bool)
            gs = G.sum()
            inter = [int((m & G).sum()) for m in km]
            ious = [i / (ka[j] + gs - i) if i else 0.0 for j, i in enumerate(inter)]
            frag = [j for j, i in enumerate(inter) if ka[j] and i / ka[j] >= 0.5]
            best1 = max(ious, default=0.0)
            if frag:
                U = np.zeros_like(G)
                for j in frag:
                    U |= km[j]
                ui = (U & G).sum()
                union = ui / (U.sum() + gs - ui)
            else:
                union = 0.0
            if len(frag) >= 2:
                cls = "split_recoverable" if union > 0.5 else "split_partial"
            elif len(frag) == 1:
                cls = "single_too_small"
            else:
                cls = "other"
            rows.append(dict(stem=stem, fid=fid, n_frag=len(frag), best1=best1, union=union, cls=cls,
                             gt_area=int(gs)))
    r = pd.DataFrame(rows)
    r = r.merge(shape[["fid", "best_pool", "width", "n_copies", "n_entries"]], on="fid")
    summ = {c: dict(n=int((r.cls == c).sum()), share=float((r.cls == c).mean()),
                    median_best1=float(r[r.cls == c].best1.median()),
                    median_union=float(r[r.cls == c].union.median()),
                    drawn_by_all=float(((r.n_copies == r.n_entries) & (r.n_entries > 1))[r.cls == c].mean()))
            for c in ("split_recoverable", "split_partial", "single_too_small", "other")}
    near = r[(r.best_pool > 0.4)]
    summ["near_miss_0.4_0.5"] = near.cls.value_counts().to_dict()
    print(json.dumps(summ, indent=1))
    r.to_parquet(OUT / "e215b_shape_band_fragments.parquet", index=False)
    json.dump(summ, open(OUT / "e215b_shape_band_fragments.json", "w"), indent=1)


if __name__ == "__main__":
    main()
