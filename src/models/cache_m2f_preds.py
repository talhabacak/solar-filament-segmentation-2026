"""Cache M2F fold candidates in the oof_cache schema, so the fusion machinery
can consume them as a third source without any schema surgery.

The miss-profile measurement earned this step: the query lane reaches 121 of
fold-0's 440 never-proposed GT rows at IoU > 0.5 (27.5%), against 3 for
multithreshold extraction and 57 for pair-unions -- a genuinely different miss
profile, the one property that has ever added at the pipeline level here.

Schema matches sweep_oof's caches (stem, conf, area, r_norm, rle), one row per
query above a loose score gate, masks NOT de-overlapped (the pool build handles
that, same as the other sources). conf carries P(filament) so the selector's
confidence features stay meaningful.

Usage:
  python src/models/cache_m2f_preds.py --weights outputs/runs_m2f/m2f_swint_f0_1024/last.pt --fold 0
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

from filament.rle import encode_mask

CACHE = ROOT / "data" / "semantic_cache"
OOF = ROOT / "outputs" / "oof_cache"
S = 1024
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255
DISK_C, DISK_R = 1023.5, 900.0


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--score", type=float, default=0.30)
    # SI-M1/M2 (docs/32 section 3): the same checkpoint, the same raw query set (gate on the
    # class probability as before), but two inference details done the authors' way:
    #   _s1  conf = P(filament) * mean in-mask sigmoid   (Mask2Former's instance score)
    #   _s2  mask = bilinear(logits) > 0                 (resize BEFORE the sigmoid)
    #   _s12 both
    # written next to the unchanged recipe (_v) from ONE forward pass, so the four caches
    # differ only in what they claim to differ in.
    ap.add_argument("--variants", action="store_true")
    a = ap.parse_args()
    VARS = ["_v", "_s1", "_s2", "_s12"] if a.variants else [""]

    from transformers import Mask2FormerForUniversalSegmentation
    ck = torch.load(a.weights, map_location="cpu", weights_only=False)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        ck["hf_name"], num_labels=1, ignore_mismatched_sizes=True)
    model.load_state_dict(ck["model"])
    model = model.cuda().eval()

    idx = json.loads((CACHE / f"index_{S}.json").read_text())
    imgs = np.load(CACHE / f"images_{S}.npy", mmap_mode="r")
    frow = {f: i for i, f in enumerate(idx["files"])}
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    files = sorted(folds.loc[folds["fold"] == a.fold, "file_name"].unique())
    log(f"fold {a.fold}: {len(files)} images, score gate {a.score}")

    rows = {k: [] for k in VARS}
    t0 = time.time()
    F = torch.nn.functional
    with torch.no_grad():
        for n, fn in enumerate(files):
            img = np.asarray(imgs[frow[fn]])
            x = (np.stack([img] * 3, -1).astype(np.float32) - MEAN) / STD
            x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].cuda()
            out = model(pixel_values=x)
            cls = out.class_queries_logits[0].softmax(-1)[:, 0].float().cpu().numpy()
            stem = fn.rsplit(".", 1)[0]
            sel = np.flatnonzero(cls >= a.score)
            if len(sel) == 0:
                continue
            lg = out.masks_queries_logits[0][sel].float()
            P = F.interpolate(lg.sigmoid()[None], size=(2048, 2048), mode="bilinear")[0]
            mask_p = P > 0.5
            if a.variants:
                L = F.interpolate(lg[None], size=(2048, 2048), mode="bilinear")[0]
                mask_l, PL = L > 0, L.sigmoid()
            for j, qi in enumerate(sel):
                for key in VARS:
                    m = mask_l[j] if key in ("_s2", "_s12") else mask_p[j]
                    conf = float(cls[qi])
                    if key in ("_s1", "_s12"):
                        pr = PL[j] if key == "_s12" else P[j]
                        conf *= float((pr * m).sum() / (m.sum() + 1e-6))
                    mn = m.cpu().numpy()
                    area = int(mn.sum())
                    if area < 64:
                        continue
                    ys, xs = np.nonzero(mn)
                    rows[key].append(dict(
                        stem=stem, conf=conf, area=area,
                        r_norm=float(np.hypot(xs.mean() - DISK_C,
                                              ys.mean() - DISK_C) / DISK_R),
                        rle=encode_mask(mn.astype(np.uint8))))
            if (n + 1) % 40 == 0:
                log(f"  {n+1}/{len(files)} ({time.time()-t0:.0f}s)")

    for key in VARS:
        df = pd.DataFrame(rows[key])
        out = OOF / f"fold{a.fold}_preds_m2f{key}.parquet"
        df.to_parquet(out, index=False)
        log(f"{key or 'base':5s} {len(df)} candidates ({len(df)/len(files):.1f}/image) -> {out.name}")


if __name__ == "__main__":
    main()
