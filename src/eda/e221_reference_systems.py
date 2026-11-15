"""E221 -- reference systems on the same protocol: what does each stage of the pipeline add?

The organisers ask for comparison with baselines (discussion 742143). Every row is scored with the
exact scorer over all 707 training images (full folds_v1 scope: an image without kept masks
contributes all of its ground truth as FN), out of fold:

  detector only     the production YOLO11m-seg fold models on their held-out images (conf >= 0.05
                    cache, NMS 0.5), r_norm <= 0.97, greedy confidence-ordered de-overlap, keep
                    conf >= t; t in {0.3, 0.4, 0.5}
  semantic only     connected components of the out-of-fold blob-ensemble field at probability
                    >= 0.5, min area 256 px, r_norm <= 0.97
  fusion, no learned selector
                    the production candidate pool (fusion + de-overlap + ring erosion), keep
                    conf >= t; t in {0.4, 0.5, 0.6}
  full system       sub036 (selector), from outputs/eda/sub036_oof_per_image.parquet

Baseline thresholds are chosen in sample (the best of the grid), which favours the baselines.

Usage: python src/eda/e221_reference_systems.py
Output: outputs/eda/e221_reference_systems.json
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/eda"):
    sys.path.insert(0, str(ROOT / p))
from filament import IMG_H, IMG_W  # noqa: E402
from filament.rle import decode_rle, encode_mask  # noqa: E402
from e115c_threshold_ci import per_image, pq  # noqa: E402

OUT = ROOT / "outputs" / "eda"
CACHE = ROOT / "outputs" / "oof_cache"
SEM = ROOT / "outputs" / "semantic"
DISK_C, DISK_R, R_MAX = 1023.5, 900.0, 0.97


def score_rows(rows, gt, stems, ncopy):
    if not rows:
        return None
    st = per_image(pd.DataFrame(rows), gt, stems, ncopy)[0]
    A = np.stack([st["tp"], st["fp"], st["fn"], st["iou"]], 1).astype(float)
    t = A.sum(0)
    return dict(pq=float(pq(*t)), sq=float(t[3] / t[0]), rq=float(t[0] / (t[0] + 0.5 * t[1] + 0.5 * t[2])),
                tp=int(t[0]), fp=int(t[1]), fn=int(t[2]), kept=len(rows))


def main():
    folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
    folds["stem"] = folds["file_name"].str.replace(".jpeg", "", regex=False)
    stems = np.array(sorted(folds["stem"]))
    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    ncopy = gt.groupby("stem")["entry_id"].nunique().reindex(stems).fillna(1).to_numpy()
    res = {}

    # detector only
    det = pd.concat([pd.read_parquet(CACHE / f"fold{F}_preds_e013.parquet") for F in range(5)], ignore_index=True)
    det = det[det["r_norm"] <= R_MAX]
    carved = []
    for stem, g in det.groupby("stem"):
        taken = np.zeros((IMG_H, IMG_W), bool)
        for k, r in enumerate(g.sort_values("conf", ascending=False).itertuples()):
            m = decode_rle(r.rle).astype(bool) & ~taken
            if m.sum() < 32:
                continue
            taken |= m
            carved.append(dict(cand_id=f"{stem}_d{k}", stem=stem, conf=r.conf, rle=encode_mask(m.astype(np.uint8))))
    carved = pd.DataFrame(carved)
    for t in (0.3, 0.4, 0.5):
        res[f"detector_only_conf{t}"] = score_rows(carved[carved["conf"] >= t][["cand_id", "stem", "rle"]].to_dict("records"), gt, stems, ncopy)
        print(f"detector only conf>={t}: {res[f'detector_only_conf{t}']}", flush=True)

    # semantic only
    rows = []
    for F in range(5):
        z = np.load(SEM / f"fold{F}_probs_blob5ens.npz")
        for stem in folds.loc[folds["fold"] == F, "stem"]:
            if stem not in z.files:
                continue
            prob = z[stem].astype(np.float32) / 255.0
            n, lab = cv2.connectedComponents((prob >= 0.5).astype(np.uint8), connectivity=8)
            for k in range(1, n):
                m = lab == k
                a = int(m.sum())
                if a < 256:
                    continue
                ys, xs = np.nonzero(m)
                if np.hypot(xs.mean() - DISK_C, ys.mean() - DISK_C) / DISK_R > R_MAX:
                    continue
                rows.append(dict(cand_id=f"{stem}_s{k}", stem=stem, rle=encode_mask(m.astype(np.uint8))))
    res["semantic_only_p0.5"] = score_rows(rows, gt, stems, ncopy)
    print(f"semantic only: {res['semantic_only_p0.5']}", flush=True)

    # fusion without the learned selector
    pool = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet", columns=["stem", "cand_id", "rle", "conf"])
    for t in (0.4, 0.5, 0.6):
        res[f"fusion_conf{t}"] = score_rows(pool[pool["conf"] >= t][["cand_id", "stem", "rle"]].to_dict("records"), gt, stems, ncopy)
        print(f"fusion, conf>={t}: {res[f'fusion_conf{t}']}", flush=True)

    s = pd.read_parquet(OUT / "sub036_oof_per_image.parquet")
    t = s[["tp", "fp", "fn", "iou"]].sum().to_numpy(float)
    res["full_system_sub036"] = dict(pq=float(pq(*t)), sq=float(t[3] / t[0]), rq=float(t[0] / (t[0] + 0.5 * t[1] + 0.5 * t[2])),
                                     tp=int(t[0]), fp=int(t[1]), fn=int(t[2]))
    print(json.dumps(res, indent=1))
    json.dump(res, open(OUT / "e221_reference_systems.json", "w"), indent=1)


if __name__ == "__main__":
    main()
