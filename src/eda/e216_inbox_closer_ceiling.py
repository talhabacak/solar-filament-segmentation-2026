"""E216 -- what is the most a gap closer working INSIDE a kept ROI's bounding box could recover?

Question (Talha, 2026-10-04): add a gap-closing structure or model inside the bbox of each final
ROI? Any such method can only add pixels inside the box it is given. Its ceiling is therefore the
PERFECT in-box closer: for a shape-band GT copy whose best kept candidate is c, replace c by
c U (G n bbox(c)) -- every missing ground-truth pixel inside the box is filled, nothing outside it,
no background is added. A second ceiling widens the box by the candidate's own extent along its
major axis (box x1.5 about its centre), the most generous "ROI with context".

Population: shape-band copies (error atlas outcome 'shape') with at least one overlapping kept
candidate under sub036's out-of-fold selection (conf >= 0.40, bag5 p >= 0.60).
Reported: share of the missing pixels R = G minus c that lie inside the box, copies whose IoU would
cross 0.5, and the PQ that would be worth if every one converted (sum of the new IoUs over the
pooled denominator, which a conversion leaves unchanged: TP +1, FP -1, FN -1). Copy-wise filling
is an upper bound -- one mask cannot be filled toward two annotators at once.

Usage: python src/eda/e216_inbox_closer_ceiling.py
Output: outputs/eda/e216_inbox_closer_ceiling.json
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
D_POOLED = 4817 + 0.5 * 1729 + 0.5 * 3382        # sub036 OOF denominator (E215 guard arm)


def box_of(m, grow=1.0):
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    if grow != 1.0:
        cy, cx, hy, hx = (y0 + y1) / 2, (x0 + x1) / 2, (y1 - y0) * grow / 2, (x1 - x0) * grow / 2
        y0, y1 = int(max(0, cy - hy)), int(min(m.shape[0], cy + hy))
        x0, x1 = int(max(0, cx - hx)), int(min(m.shape[1], cx + hx))
    return y0, y1, x0, x1


def main():
    atlas = pd.read_parquet(OUT / "error_atlas.parquet")
    shape = atlas[atlas["outcome"] == "shape"]
    a = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet", columns=["stem", "cand_id", "rle", "conf"])
    a = a.merge(pd.read_parquet(OUT / "bag5_oof_scores.parquet"), on="cand_id")
    keep = a[(a["conf"] >= 0.40) & (a["p"] >= 0.60)]
    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet", columns=["filament_id", "segmentation_rle"])
    gt_rle = dict(zip(gt["filament_id"], gt["segmentation_rle"]))

    rows = []
    for stem, g in shape.groupby("stem"):
        kc = keep[keep["stem"] == stem]
        km = [decode_rle(r).astype(bool) for r in kc["rle"]]
        for fid in g["fid"]:
            G = decode_rle(gt_rle[fid]).astype(bool)
            inter = [int((m & G).sum()) for m in km]
            if not inter or max(inter) == 0:
                continue
            ious = [i / (m.sum() + G.sum() - i) for m, i in zip(km, inter)]
            c = km[int(np.argmax(ious))]
            R = G & ~c
            row = dict(stem=stem, fid=fid, iou=max(ious), r_px=int(R.sum()))
            for tag, grow in (("box", 1.0), ("box15", 1.5)):
                y0, y1, x0, x1 = box_of(c, grow)
                inbox = np.zeros_like(G)
                inbox[y0:y1, x0:x1] = True
                filled = c | (G & inbox)
                i2 = (filled & G).sum()
                row[f"r_inside_{tag}"] = float((R & inbox).sum() / max(R.sum(), 1))
                row[f"iou_{tag}"] = float(i2 / (filled.sum() + G.sum() - i2))
            rows.append(row)
    r = pd.DataFrame(rows)
    res = dict(n_copies=len(r), median_iou_now=float(r.iou.median()))
    for tag in ("box", "box15"):
        conv = r[f"iou_{tag}"] > 0.5
        res[tag] = dict(median_share_of_missing_inside=float(r[f"r_inside_{tag}"].median()),
                        mean_share_of_missing_inside=float(r[f"r_inside_{tag}"].mean()),
                        median_iou_after_perfect_fill=float(r[f"iou_{tag}"].median()),
                        copies_crossing_05=int(conv.sum()),
                        pq_ceiling=float(r.loc[conv, f"iou_{tag}"].sum() / D_POOLED))
    print(json.dumps(res, indent=1))
    r.to_parquet(OUT / "e216_inbox_closer_ceiling.parquet", index=False)
    json.dump(res, open(OUT / "e216_inbox_closer_ceiling.json", "w"), indent=1)


if __name__ == "__main__":
    main()
