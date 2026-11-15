"""E218 -- detector NMS tighter than 0.5 (0.3, 0.4) in today's fused pool.

Ledger: NMS 0.6/0.7 were measured in the fused pool (reach -24/-19 rows); tighter values were only
measured in week one on a single detector without fusion (iou 0.05: +0.0105, duplicates removed),
a gain the fused ensemble absorbed. Box NMS on long diagonal filaments can delete real neighbours,
so the direction is open in both senses (Talha, 2026-10-04).

Instead of one detector run per NMS value, the fold's detector is run ONCE with NMS effectively
off (iou 0.95, max_det 3000) and every raw detection is kept with its box; Ultralytics' NMS is
torchvision.ops.nms on those boxes followed by the max_det cut, so any threshold is then applied
offline exactly. Boxes are kept at 2048 scale; the 1792 -> 2048 scaling is uniform (square frame,
no letterbox padding), so box IoU is unchanged.

GUARD: offline NMS at 0.5 must reproduce the production cache fold{F}_preds_e013.parquet (count
per image and mask IoU in confidence order).

Masks are built the production way, verified in E217a: probability > 0.5 at 1792, uint8 resize to
2048, keep >= 0.5.

Steps:
  python src/eda/e218_nms_tighter.py infer --fold 0 --shard k --nshards n --threads t
  python src/eda/e218_nms_tighter.py build --fold 0          (writes fold0_preds_e013nms{03,04,05})
then the pool protocol for a candidate-distribution change (selector-free reach first):
  build_fused_pool.py --folds 0 --sem-tag _blob5ens --sem-thr 0.2 --det-tag _e013nmsXX ...
  outputs/eda/_label_one_pool.py, src/eda/pool_complementarity.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "eda"))
from filament import IMG_H, IMG_W  # noqa: E402
from filament.rle import decode_rle, encode_mask  # noqa: E402

IMG = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
CACHE = ROOT / "outputs" / "oof_cache"
DISK_C, DISK_R = 1023.5, 900.0


def infer(a):
    from ultralytics import YOLO
    from ultralytics.utils import ops
    from e217_yolo_soft_masks import soft_process_mask
    ops.process_mask = soft_process_mask
    if a.threads:
        torch.set_num_threads(a.threads)
    model = YOLO(str(ROOT / "outputs" / "runs" / f"e013_yolo11m_f{a.fold}_1792" / "weights" / "best.pt"))
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    files = sorted(folds.loc[folds["fold"] == a.fold, "file_name"])[a.shard::a.nshards]
    rows, t0 = [], time.time()
    for n, fn in enumerate(files):
        stem = fn.rsplit(".", 1)[0]
        src = cv2.cvtColor(cv2.imread(str(IMG / fn), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
        res = model.predict(src, imgsz=1792, conf=0.05, iou=0.95, max_det=3000, verbose=False, device="cpu")[0]
        if res.masks is not None:
            soft = res.masks.data.cpu().numpy()
            for m, c, b in zip(soft, res.boxes.conf.cpu().numpy(), res.boxes.xyxy.cpu().numpy()):
                mm = cv2.resize((m > 0.5).astype(np.uint8), (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR) >= 0.5
                ys, xs = np.nonzero(mm)
                rows.append(dict(stem=stem, conf=float(c), x0=float(b[0]), y0=float(b[1]), x1=float(b[2]),
                                 y1=float(b[3]), area=int(mm.sum()),
                                 r_norm=float(np.hypot(xs.mean() - DISK_C, ys.mean() - DISK_C) / DISK_R) if mm.any() else np.nan,
                                 rle=encode_mask(mm.astype(np.uint8)) if mm.any() else ""))
        print(f"  {n + 1}/{len(files)} {time.time() - t0:.0f}s raw {len(rows)}", flush=True)
    pd.DataFrame(rows).to_parquet(CACHE / f"raw95_e013_f{a.fold}_{a.shard}of{a.nshards}.parquet", index=False)


def build(a):
    import torchvision
    raw = pd.concat([pd.read_parquet(p) for p in sorted(CACHE.glob(f"raw95_e013_f{a.fold}_*of*.parquet"))],
                    ignore_index=True)
    prod = pd.read_parquet(CACHE / f"fold{a.fold}_preds_e013.parquet")
    print(f"raw detections {len(raw)} over {raw['stem'].nunique()} images; production {len(prod)}")
    out = {}
    for t in (0.3, 0.4, 0.5):
        keep = []
        for stem, g in raw.groupby("stem"):
            boxes = torch.tensor(g[["x0", "y0", "x1", "y1"]].to_numpy(), dtype=torch.float32)
            idx = torchvision.ops.nms(boxes, torch.tensor(g["conf"].to_numpy(), dtype=torch.float32), t)[:300]
            keep.append(g.iloc[idx.numpy()])
        d = pd.concat(keep)
        d = d[d["rle"] != ""]                    # the production cache drops empty masks
        tag = f"{int(round(t * 10)):02d}"
        d[["stem", "conf", "area", "r_norm", "rle"]].to_parquet(CACHE / f"fold{a.fold}_preds_e013nms{tag}.parquet", index=False)
        out[t] = dict(candidates=len(d), per_image=len(d) / prod["stem"].nunique())
        print(f"nms {t}: {len(d)} candidates")
    # guard: offline 0.5 against production
    d5 = pd.read_parquet(CACHE / f"fold{a.fold}_preds_e013nms05.parquet")
    g_rows = []
    for stem, gp in prod.groupby("stem"):
        gp = gp.sort_values("conf", ascending=False)
        gn = d5[d5["stem"] == stem].sort_values("conf", ascending=False)
        ious = []
        for (_, rp), (_, rn) in zip(gp.iterrows(), gn.iterrows()):
            p, q = decode_rle(rp["rle"]).astype(bool), decode_rle(rn["rle"]).astype(bool)
            ious.append((p & q).sum() / max((p | q).sum(), 1))
        g_rows.append(dict(stem=stem, n_prod=len(gp), n_new=len(gn), iou=float(np.mean(ious)) if ious else 1.0))
    g = pd.DataFrame(g_rows)
    guard = dict(count_match=float((g.n_prod == g.n_new).mean()), mean_iou=float(g.iou.mean()),
                 total_prod=int(g.n_prod.sum()), total_new=int(g.n_new.sum()))
    guard["passed"] = bool(guard["count_match"] >= 0.95 and guard["mean_iou"] >= 0.98)
    print("guard:", json.dumps(guard))
    json.dump(dict(nms=out, guard=guard), open(ROOT / "outputs" / "eda" / f"e218_nms_build_f{a.fold}.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["infer", "build"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    infer(a) if a.step == "infer" else build(a)


if __name__ == "__main__":
    main()
