"""Does the query lane reach the GT that nothing else in our pool ever reached?

The M2F gate came back below the band standalone (0.3758 vs 0.42-0.44, SQ 0.632
-- quarter-resolution query masks cap boundary quality on 10-30px structures).
But the decision tree's real question was never absolute PQ: a lane earns a
place in the FUSION POOL by reaching ground truth the existing sources miss,
because the selector can then keep those candidates while ignoring the lane
elsewhere. The two-lane fusion earned LB +0.01 exactly this way.

The decisive population is fold-0's 440 never-proposed GT rows (no existing
candidate at IoU > 0.5, and the model-probability autopsy showed the semantic
map fires under them but never as a well-shaped component). For each, the best
IoU any M2F prediction achieves. A large reach count = a genuinely different
miss profile and the lane wires in as a third source; a small one closes the
family with one number.

Usage: python src/eda/m2f_miss_profile.py --weights outputs/runs_m2f/m2f_swint_f0_1024/last.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament.rle import decode_rle

CACHE = ROOT / "data" / "semantic_cache"
OUT = ROOT / "outputs" / "eda"
S = 1024
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--score", type=float, default=0.5,
                    help="loose gate: reach is a ceiling question, not a policy")
    a = ap.parse_args()

    d = pd.read_parquet(OUT / "error_profile_fn_gt.parquet")
    d["matched"] = d["matched"].astype(bool)
    d["in_raw_pool"] = d["in_raw_pool"].astype(bool)
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    d = d.merge(gt[["filament_id", "segmentation_rle"]], on="filament_id", how="left")
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    f0 = {f.rsplit(".", 1)[0] for f in folds.loc[folds["fold"] == 0, "file_name"]}
    never = d[d["stem"].isin(f0) & ~d["matched"] & ~d["in_raw_pool"]]
    log(f"fold-0 never-proposed GT rows: {len(never)}")

    from transformers import Mask2FormerForUniversalSegmentation
    ck = torch.load(a.weights, map_location="cpu", weights_only=False)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        ck["hf_name"], num_labels=1, ignore_mismatched_sizes=True)
    model.load_state_dict(ck["model"])
    model = model.cuda().eval()

    idx = json.loads((CACHE / f"index_{S}.json").read_text())
    imgs = np.load(CACHE / f"images_{S}.npy", mmap_mode="r")
    frow = {f: i for i, f in enumerate(idx["files"])}

    rows = []
    t0 = time.time()
    with torch.no_grad():
        for k, (stem, g) in enumerate(never.groupby("stem")):
            img = np.asarray(imgs[frow[stem + ".jpeg"]])
            x = (np.stack([img] * 3, -1).astype(np.float32) - MEAN) / STD
            x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].cuda()
            out = model(pixel_values=x)
            cls = out.class_queries_logits[0].softmax(-1)[:, 0].float().cpu().numpy()
            masks = out.masks_queries_logits[0].sigmoid()
            masks = torch.nn.functional.interpolate(
                masks[None], size=(2048, 2048), mode="bilinear")[0] > 0.5
            keep = [masks[i].cpu().numpy() for i in np.flatnonzero(cls >= a.score)]
            for _, r in g.iterrows():
                gm = decode_rle(r["segmentation_rle"]).astype(bool)
                ga = gm.sum()
                best = 0.0
                for m in keep:
                    inter = (m & gm).sum()
                    if inter:
                        best = max(best, inter / (m.sum() + ga - inter))
                rows.append(dict(filament_id=r["filament_id"], iou=best))
            if (k + 1) % 30 == 0:
                log(f"  {k+1} images ({time.time()-t0:.0f}s)")

    rf = pd.DataFrame(rows)
    log(f"\nof {len(rf)} never-proposed rows, M2F (score>={a.score}) reaches:")
    log(f"  IoU > 0.5 : {int((rf['iou'] > 0.5).sum())} "
        f"({100*(rf['iou'] > 0.5).mean():.1f}%)")
    log(f"  IoU 0.3-0.5: {int(((rf['iou'] > 0.3) & (rf['iou'] <= 0.5)).sum())}")
    log(f"  IoU <= 0.3 : {int((rf['iou'] <= 0.3).sum())}")
    log("\ncontext: seeded multithresh reached 3 (0.7%), pair-union 57. If the "
        "query lane clears well above those, its miss profile is genuinely "
        "different and it earns a pool source slot.")
    rf.to_parquet(OUT / "m2f_miss_profile.parquet", index=False)


if __name__ == "__main__":
    main()
