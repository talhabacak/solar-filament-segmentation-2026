"""E216b -- does the in-box closer's ceiling TRANSFER across annotators?

E216's ceiling fills each copy toward ITS OWN annotator, which is an oracle on label noise as much
as on the filament (project rule: an oracle is valued by its transfer, docs/report §summary).
Here, on images with more than one annotator entry, the fill is decided by annotator A and scored
against annotator B's copy of the same filament (sibling: IoU(G_A, G_B) > 0.5 in another entry):

    before  IoU(c, G_B)
    after   IoU(c U (G_A n box), G_B)

A real, annotator-independent deficit (the filament is longer than our mask for everyone) shows
up as B crossing 0.5; an idiosyncratic one shows up as no gain or as harm (B's copies that were
already matched and fall below 0.5). Transfer = net B crossings per A crossing among the same
pairs.

Usage: python src/eda/e216b_inbox_closer_transfer.py
Output: outputs/eda/e216b_inbox_closer_transfer.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "eda"))
from filament.rle import decode_rle  # noqa: E402
from e216_inbox_closer_ceiling import box_of  # noqa: E402

OUT = ROOT / "outputs" / "eda"


def iou(a, b):
    i = (a & b).sum()
    u = a.sum() + b.sum() - i
    return i / u if u else 0.0


def main():
    atlas = pd.read_parquet(OUT / "error_atlas.parquet")
    shape = atlas[(atlas["outcome"] == "shape") & (atlas["n_entries"] > 1)]
    a = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet", columns=["stem", "cand_id", "rle", "conf"])
    a = a.merge(pd.read_parquet(OUT / "bag5_oof_scores.parquet"), on="cand_id")
    keep = a[(a["conf"] >= 0.40) & (a["p"] >= 0.60)]
    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet", columns=["filament_id", "segmentation_rle", "entry_id", "file_name"])
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)

    rows = []
    for stem, g in shape.groupby("stem"):
        kc = keep[keep["stem"] == stem]
        if kc.empty:
            continue
        km = [decode_rle(r).astype(bool) for r in kc["rle"]]
        gs = gt[gt["stem"] == stem]
        gm = {f: decode_rle(r).astype(bool) for f, r in zip(gs["filament_id"], gs["segmentation_rle"])}
        ent = dict(zip(gs["filament_id"], gs["entry_id"]))
        for fid in g["fid"]:
            GA = gm[fid]
            ious = [iou(m, GA) for m in km]
            if max(ious) == 0:
                continue
            c = km[int(np.argmax(ious))]
            sibs = [f for f in gm if ent[f] != ent[fid] and iou(gm[f], GA) > 0.5]
            for tag, grow in (("box", 1.0), ("box15", 1.5)):
                y0, y1, x0, x1 = box_of(c, grow)
                inbox = np.zeros_like(GA)
                inbox[y0:y1, x0:x1] = True
                filled = c | (GA & inbox)
                a_before, a_after = max(ious), iou(filled, GA)
                for sb in sibs:
                    GB = gm[sb]
                    rows.append(dict(stem=stem, fid=fid, sib=sb, tag=tag, a_before=a_before, a_after=a_after,
                                     b_before=iou(c, GB), b_after=iou(filled, GB)))
    r = pd.DataFrame(rows)
    res = {}
    for tag in ("box", "box15"):
        t = r[r.tag == tag]
        a_cross = int(((t.a_before <= 0.5) & (t.a_after > 0.5)).sum())
        b_cross = int(((t.b_before <= 0.5) & (t.b_after > 0.5)).sum())
        b_lost = int(((t.b_before > 0.5) & (t.b_after <= 0.5)).sum())
        res[tag] = dict(pairs=len(t), a_crossings=a_cross, b_crossings=b_cross, b_lost=b_lost,
                        transfer=(b_cross - b_lost) / a_cross if a_cross else None,
                        median_b_delta=float((t.b_after - t.b_before).median()),
                        mean_b_delta=float((t.b_after - t.b_before).mean()))
    print(json.dumps(res, indent=1))
    r.to_parquet(OUT / "e216b_inbox_closer_transfer.parquet", index=False)
    json.dump(res, open(OUT / "e216b_inbox_closer_transfer.json", "w"), indent=1)


if __name__ == "__main__":
    main()
