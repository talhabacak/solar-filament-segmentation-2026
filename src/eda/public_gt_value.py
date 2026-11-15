"""What would the PUBLIC MAGFiLO annotations score under the competition metric?

Why this is measured. The leaderboard is bimodal -- ~190 teams at 0.33-0.40,
nobody at 0.41-0.52, four teams at 0.53-0.55 -- and every modelling replication
of the leaders' visible recipe lands at 0.41-0.42. The competition's own source
dataset, MAGFiLO 1.0, is public and carries annotations for ALL 180 test stems.
The host has ruled external ground truth prohibited (discussion 729396:
"submissions will be disregarded"). If the public annotations score ~0.55 under
the metric, the leaders' cluster is explained by a prohibited lookup rather than
by a model, and chasing it with modelling is chasing a mirage.

Ethics and quarantine compliance, explicitly: this script scores the public
TRAIN-stem annotations against the competition TRAIN ground truth. It is an
ANALYSIS of two label sets' agreement. Nothing here trains a model, nothing is
predicted, no test-stem geometry is decoded, and no output feeds the pipeline.
The quarantine rule -- public label geometry never used for training or
predictions -- stands.

Method: public polygons -> masks per train stem, de-overlapped the way a
submission must be, scored with the organiser-parity scorer against the full
multi-annotator competition GT. Reported pooled, exactly like the leaderboard.

Usage: python src/eda/public_gt_value.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament import IMG_H, IMG_W
from filament.pq import score
from filament.rle import encode_mask

Q = ROOT / "outputs" / "external" / "magfilo_quarantine" / "magfilo_2024_v1.0.json"
OUT = ROOT / "outputs" / "eda"


def log(*a):
    print(*a, flush=True)


def main():
    d = json.load(open(Q, encoding="utf-8"))
    img_by_id = {im["id"]: Path(im["file_name"]).stem for im in d["images"]}
    train_stems = {p.stem for p in
                   (ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" /
                    "train_images").glob("*.jpeg")}

    by_stem = {}
    n_ann = 0
    for a in d["annotations"]:
        stem = img_by_id.get(a["image_id"])
        if stem not in train_stems:
            continue                    # test-stem geometry is never decoded
        by_stem.setdefault(stem, []).append(a)
        n_ann += 1
    log(f"public annotations on competition TRAIN stems: {n_ann} over "
        f"{len(by_stem)} images")

    preds = []
    for stem, anns in by_stem.items():
        masks = []
        for a in anns:
            seg = a.get("segmentation")
            m = np.zeros((IMG_H, IMG_W), np.uint8)
            if isinstance(seg, list):
                for poly in seg:
                    pts = np.array(poly, np.float64).reshape(-1, 2)
                    cv2.fillPoly(m, [pts.astype(np.int32)], 1)
            elif isinstance(seg, dict):
                from pycocotools import mask as mu
                m = mu.decode({"size": seg["size"],
                               "counts": seg["counts"].encode()
                               if isinstance(seg["counts"], str) else seg["counts"]})
            if m.sum() >= 5:
                masks.append(m.astype(bool))
        # a submission must be pixel-disjoint; keep larger instances first so a
        # small overlapping fragment cannot carve a hole in a big one
        masks.sort(key=lambda x: -int(x.sum()))
        acc = np.zeros((IMG_H, IMG_W), bool)
        for k, m in enumerate(masks):
            mm = m & ~acc
            acc |= m
            if mm.sum() >= 5:
                preds.append((f"{stem}_{k+1}", encode_mask(mm.astype(np.uint8))))

    pdf = pd.DataFrame(preds, columns=["filament_id", "segmentation_rle"])
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt = gt[gt["file_name"].str.replace(".jpeg", "", regex=False).isin(set(by_stem))]
    log(f"scoring {len(pdf)} public instances against {len(gt)} competition GT rows")
    s = score(gt[["filament_id", "segmentation_rle"]], pdf)
    log(f"\nPUBLIC MAGFiLO annotations vs competition GT (train, pooled):")
    log(f"  PQ {s['pq']:.4f}   SQ {s['sq']:.4f}   RQ {s['rq']:.4f}   "
        f"TP {s['tp']} FP {s['fp']} FN {s['fn']}")
    log(f"\n  leaderboard leaders: 0.53-0.55")
    log(f"  our best honest local: 0.4286 (~0.38 LB)")
    log("\nreading: if the number above sits in the leaders' band, the cluster is "
        "explained by the public lookup the host has prohibited, and the honest "
        "target is the 0.40-0.45 band -- not 0.55.")

    json.dump({k: float(s[k]) if not isinstance(s[k], int) else s[k]
               for k in ("pq", "sq", "rq", "tp", "fp", "fn")},
              open(OUT / "public_gt_value.json", "w"), indent=1)
    log(f"\nsaved -> {OUT/'public_gt_value.json'}")


if __name__ == "__main__":
    main()
