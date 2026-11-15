"""Did the spine head change what the model can SHAPE, even though pixel-Dice fell?

The auxiliary arms came in below baseline on pixel-Dice (0.7006 and 0.6984 vs
0.7028), but that proxy answers the wrong question. The mechanism was never
"better average overlap"; it was "faint instances form components that clear
IoU 0.5", aimed at the 440 fold-0 ground-truth rows no candidate has ever
reached. This project has already been misled once by a proxy that inverted
against PQ (mask mAP50 ranked crack-yolov8x above yolo11m; PQ reversed it), so
the mechanism gets its own measurement before the axis closes.

For the baseline heavy model and each spine-aux model: connected components of
the probability map at the pool's own extraction settings, then the best IoU
each never-proposed GT row receives. Reach is a ceiling, not a score -- but if
the spine head shapes even a handful of these where the baseline shapes none,
the axis stays open as a pool source rather than a replacement.

Usage: python src/eda/spine_aux_reach.py
"""

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
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament.rle import decode_rle

CACHE = ROOT / "data" / "semantic_cache"
OUT = ROOT / "outputs" / "eda"
S = 768
SEM_THR, SEM_MIN_AREA = 0.3, 256

MODELS = {
    "heavy60 (baseline)": "outputs/runs_aug/aug_heavy_f0_768_60ep/best.pt",
    "spine-aux w0.3": "outputs/runs_aug/spineaux_f0_768_w0.3/best.pt",
    "spine-aux w0.1": "outputs/runs_aug/spineaux_f0_768_w0.1/best.pt",
}


def log(*a):
    print(*a, flush=True)


def load(path):
    import segmentation_models_pytorch as smp
    ck = torch.load(ROOT / path, map_location="cpu", weights_only=False)
    m = smp.Unet(encoder_name=ck["encoder"], encoder_weights=None, in_channels=3,
                 classes=1, decoder_attention_type="scse")
    nout = int(ck.get("out_channels", 1))
    if nout != 1:
        import torch.nn as nn
        h = m.segmentation_head[0]
        m.segmentation_head[0] = nn.Conv2d(h.in_channels, nout, h.kernel_size,
                                           padding=h.padding)
    m.load_state_dict(ck["model"])
    return m.cuda().eval()


def main():
    d = pd.read_parquet(OUT / "error_profile_fn_gt.parquet")
    d["matched"] = d["matched"].astype(bool)
    d["in_raw_pool"] = d["in_raw_pool"].astype(bool)
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    d = d.merge(gt[["filament_id", "segmentation_rle"]], on="filament_id", how="left")
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    f0 = {f.rsplit(".", 1)[0] for f in folds.loc[folds["fold"] == 0, "file_name"]}
    never = d[d["stem"].isin(f0) & ~d["matched"] & ~d["in_raw_pool"]]
    log(f"fold-0 never-proposed GT rows: {len(never)}")

    idx = json.loads((CACHE / f"index_{S}.json").read_text())
    imgs = np.load(CACHE / f"images_{S}.npy", mmap_mode="r")
    frow = {f: i for i, f in enumerate(idx["files"])}
    stems = sorted(never["stem"].unique())

    for label, path in MODELS.items():
        model = load(path)
        t0, reached, close = time.time(), 0, 0
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for stem in stems:
                img = np.asarray(imgs[frow[stem + ".jpeg"]]).astype(np.float32) / 255.0
                x = torch.from_numpy(np.stack([img] * 3))[None].cuda()
                p = torch.sigmoid(model(x)[:, :1])[0, 0].float().cpu().numpy()
                p = cv2.resize(p, (2048, 2048), interpolation=cv2.INTER_LINEAR)
                n, lab, stats, _ = cv2.connectedComponentsWithStats(
                    (p >= SEM_THR).astype(np.uint8), connectivity=8)
                comps = [(lab == i) for i in range(1, n)
                         if stats[i, cv2.CC_STAT_AREA] >= SEM_MIN_AREA]
                for _, r in never[never["stem"] == stem].iterrows():
                    gm = decode_rle(r["segmentation_rle"]).astype(bool)
                    ga = gm.sum()
                    best = 0.0
                    for m in comps:
                        inter = (m & gm).sum()
                        if inter:
                            best = max(best, inter / (m.sum() + ga - inter))
                    reached += best > 0.5
                    close += 0.3 < best <= 0.5
        log(f"{label:22s} reaches {reached:3d}/{len(never)} at IoU>0.5, "
            f"{close:3d} near-miss  ({time.time()-t0:.0f}s)")
        del model
        torch.cuda.empty_cache()

    log("\nreading: the spine head earns a pool source only if it shapes "
        "materially more of these than the baseline; equal or fewer closes the "
        "axis with the mechanism tested on its own terms.")


if __name__ == "__main__":
    main()
