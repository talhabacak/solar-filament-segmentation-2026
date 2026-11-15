"""Does the seed-refined lane reach GT the other two lanes miss?

As a REPLACEMENT for the detector's masks the seed refiner is a wash: trained on
real detector boxes it scores 0.4122 against 0.4111 at conf 0.28, 0.3976 against
0.3961 at 0.20, 0.4164 against 0.4160 at 0.40. Tiny, consistent, not worth a
pipeline change on its own.

But the fused pool has never rewarded same-mechanism levers and has rewarded
different-mechanism ones: NMS tightening and TTA were absorbed entirely, while
the semantic lane was worth +0.01 on the leaderboard. The seed refiner segments
inside a scale-normalised crop from a box prior, which is a third mechanism, so
the question is not whether its masks are better but whether they are DIFFERENT
— whether they reach GT rows the detector and semantic lanes both miss.

Reports pairwise and union reachability, and the increment the refined lane adds
on top of the current two.

Usage: python src/eda/seed_lane_union.py --fold 0 --conf 0.05
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament.pq import score
from filament.rle import decode_rle, encode_mask
from semantic_candidates import prob_to_instances, R_MAX
from apply_seed_refiner import load, refine_image, bbox_of

DATA = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
OUT = ROOT / "outputs" / "eda"
SEM_THR, SEM_MIN_AREA = 0.3, 256


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--weights", default="outputs/runs_seed/seed_detbox_b3_512/best.pt")
    a = ap.parse_args()

    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    stems = [f.rsplit(".", 1)[0]
             for f in sorted(folds.loc[folds["fold"] == a.fold, "file_name"])]
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt_f = gt[gt["file_name"].isin({s + ".jpeg" for s in stems})]
    N = len(gt_f)
    cat = dict(zip(gt_f["filament_id"], gt_f["category_id"]))
    n_u = sum(1 for v in cat.values() if v == 3)

    det = pd.read_parquet(ROOT / "outputs" / "oof_cache" / f"fold{a.fold}_preds_e013.parquet")
    det = det[(det["conf"] >= a.conf) & (det["r_norm"] <= R_MAX)]
    if "view" in det.columns:
        det = det[det["view"] == "o"]
    z = np.load(ROOT / "outputs" / "semantic" / f"fold{a.fold}_probs_cldice.npz")
    model, S, pad, ctx = load(ROOT / a.weights)

    det_rows, sem_rows, ref_rows = [], [], []
    t0 = time.time()
    for n, stem in enumerate(stems):
        g = det[det["stem"] == stem]
        img = cv2.imread(str(DATA / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        masks = [decode_rle(r).astype(bool) for r in g["rle"]]
        for k, m in enumerate(masks):
            det_rows.append((f"{stem}_d{k}", encode_mask(m.astype(np.uint8))))
        for k, c in enumerate(prob_to_instances(z[stem].astype(np.float32) / 255.0,
                                                SEM_THR, SEM_MIN_AREA)):
            sem_rows.append((f"{stem}_s{k}", encode_mask(c["mask"].astype(np.uint8))))
        if img is not None and masks:
            boxes = [bbox_of(m) for m in masks]
            ok = [i for i, b in enumerate(boxes) if b is not None]
            for k, r in enumerate(refine_image(model, S, pad, img,
                                               [boxes[i] for i in ok],
                                               context=ctx)):
                if r is not None:
                    ref_rows.append((f"{stem}_r{k}", encode_mask(r.astype(np.uint8))))
        if (n + 1) % 40 == 0:
            log(f"  {n+1}/{len(stems)} images ({time.time()-t0:.0f}s)")

    gt_cache, res = {}, {}

    def reach(name, rows):
        df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
        r = score(gt_f[["filament_id", "segmentation_rle"]], df, gt_cache=gt_cache,
                  return_details=True)
        hits = set(r["matches"]["gt_filament_id"])
        u = sum(1 for h in hits if cat.get(h) == 3)
        log(f"{name:28s} preds {len(df):5d}  reach {len(hits):4d}/{N}={len(hits)/N:.4f}"
            f"  U {u:4d}/{n_u}={u/max(n_u,1):.4f}")
        res[name] = dict(n=len(df), reach=len(hits) / N, u_reach=u / max(n_u, 1))
        return hits

    h_d = reach("detector", det_rows)
    h_s = reach("semantic", sem_rows)
    h_r = reach("seed-refined", ref_rows)
    h_ds = reach("detector + semantic", det_rows + sem_rows)
    h_all = reach("all three", det_rows + sem_rows + ref_rows)

    log(f"\nseed-refined adds {len(h_all - h_ds)} GT rows the current two never reach "
        f"({len(h_all - h_ds)/N:+.4f})")
    log(f"  of those, U: {sum(1 for x in (h_all - h_ds) if cat.get(x) == 3)}")
    log(f"reference: the 1-band semantic model added 21 rows (+0.0139)")
    res["increment"] = dict(rows=len(h_all - h_ds), frac=len(h_all - h_ds) / N)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"seed_lane_union_fold{a.fold}.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    log(f"saved -> {OUT / f'seed_lane_union_fold{a.fold}.json'}")


if __name__ == "__main__":
    main()
