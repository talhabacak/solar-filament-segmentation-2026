"""Should a 1536-trained model be run at 2048 for inference, and who belongs in the ensemble?

Two decisions have to be made before spending a submission slot, and both are
cheap to measure on fold 0 -- where every candidate model was trained on folds
1-4 and has never seen these 142 images.

1. INPUT RESOLUTION. Talha's suggestion is to run inference at 2048 since the
   models were trained at 1536. The output is already resized back to 2048 by
   predict_prob, so the only question is what the network SEES. Feeding 2048 to a
   1536-trained CNN makes every filament 1.33x wider in pixels, and a
   convolutional stack is not scale-invariant: its filters were fitted to a
   particular apparent width. Training at 2048 was measured worse than 1536
   (PQ 0.3967 vs 0.4011), but that is a different question from testing at 2048,
   so it is measured here rather than assumed either way.

2. ENSEMBLE MEMBERSHIP. The three-model average already measured +0.0054 over the
   best single (0.7034 vs 0.6980), with pairwise correlations of 0.908-0.932 --
   real diversity, coming from recipe and resolution rather than seed or size.
   The open question is whether the GONG-pretrained model earns a place. It is
   the WORST single model we have (0.6915 at its own val, against 0.6981 for
   plain ImageNet), but ensemble value comes from making different mistakes, not
   from being good -- and a model pretrained on 10,001 archive frames has every
   reason to fail differently. This is the fair test of Talha's pretraining
   hypothesis: not "is it better alone", but "does it add anything".

Scored as pooled pixel-Dice at native 2048 against the annotator-union target,
which is the common ground the per-run val numbers do not share.

Usage: python src/eda/inference_size.py
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from semantic_candidates import load_model, predict_prob

OUT = ROOT / "outputs" / "eda"

# name -> (checkpoint, inference imgsz override or None for the trained size)
CANDIDATES = {
    "prod_1536@1536": ("outputs/runs_semantic/sem_b3_f0_1536_cldice/best.pt", None),
    "prod_1536@2048": ("outputs/runs_semantic/sem_b3_f0_1536_cldice/best.pt", 2048),
    "heavy_768@768": ("outputs/runs_aug/aug_heavy_f0_768_60ep/best.pt", None),
    "heavy_768@1024": ("outputs/runs_aug/aug_heavy_f0_768_60ep/best.pt", 1024),
    "heavy_1536@1536": ("outputs/runs_aug/aug_heavy_f0_1536_60ep/best.pt", None),
    "heavy_1536@2048": ("outputs/runs_aug/aug_heavy_f0_1536_60ep/best.pt", 2048),
    "gong50_768@768": ("outputs/runs_semantic/sem_b3_f0_768_std50_lr1e-4/best.pt", None),
}


def log(*a):
    print(*a, flush=True)


def main():
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    files = sorted(folds.loc[folds["fold"] == 0, "file_name"].unique())
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    from filament.rle import decode_rle
    tgt = {}
    for fn, g in gt[gt["file_name"].isin(files)].groupby("file_name"):
        m = None
        for r in g["segmentation_rle"]:
            d = decode_rle(r)
            m = d if m is None else np.maximum(m, d)
        tgt[fn] = m.astype(bool)
    log(f"fold 0: {len(files)} images\n")

    imdir = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
    probs = {}
    for name, (w, override) in CANDIDATES.items():
        model, trained, channels = load_model(str(ROOT / w))
        use = override or trained
        probs[name] = {}
        for fn in files:
            probs[name][fn] = predict_prob(model, use, fn.rsplit(".", 1)[0],
                                           tta=True, img_dir=imdir,
                                           channels=channels).astype(np.float32)
        del model
        torch.cuda.empty_cache()
        log(f"  {name}: trained {trained}, inferred at {use}")

    def pooled(keys):
        i = u = 0
        for fn in files:
            p = sum(probs[k][fn] for k in keys) / len(keys)
            b, t = p > 0.5, tgt[fn]
            i += (b & t).sum()
            u += b.sum() + t.sum()
        return 2 * i / max(u, 1)

    log("\n1. INPUT RESOLUTION -- single models, pooled Dice at native 2048:")
    single = {k: pooled([k]) for k in CANDIDATES}
    for base in ("prod_1536", "heavy_768", "heavy_1536"):
        ks = [k for k in single if k.startswith(base + "@")]
        if len(ks) < 2:
            continue
        a, b = sorted(ks, key=lambda k: int(k.split("@")[1]))
        log(f"   {base:14s} @{a.split('@')[1]:>4s} {single[a]:.4f}   "
            f"@{b.split('@')[1]:>4s} {single[b]:.4f}   "
            f"{single[b]-single[a]:+.4f} -> "
            f"{'bigger input helps' if single[b] > single[a] else 'bigger input HURTS'}")
    log(f"\n   {'gong50_768@768':22s} {single['gong50_768@768']:.4f}  "
        f"(the weakest single, as expected)")

    log("\n2. ENSEMBLE -- best combination at each size, native-size inference only:")
    native = [k for k in CANDIDATES if CANDIDATES[k][1] is None]
    best_of = {}
    for r in range(1, len(native) + 1):
        rows = [(c, pooled(c)) for c in itertools.combinations(native, r)]
        c, v = max(rows, key=lambda t: t[1])
        best_of[r] = (c, v)
        log(f"   {r} member(s): {v:.4f}  {' + '.join(c)}")
    top = max(best_of.values(), key=lambda t: t[1])
    log(f"\n   best overall {top[1]:.4f} with {len(top[0])} members: "
        f"{' + '.join(top[0])}")
    log(f"   gain over best single {top[1]-max(single[k] for k in native):+.4f}")

    gong_in = any("gong" in k for k in top[0])
    log(f"\n   does the GONG-pretrained model earn a place? "
        f"{'YES -- it makes different mistakes' if gong_in else 'no'}")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(dict(single={k: float(v) for k, v in single.items()},
                   best_by_size={r: ([*c], float(v)) for r, (c, v) in best_of.items()},
                   best=[list(top[0]), float(top[1])], gong_selected=bool(gong_in)),
              open(OUT / "inference_size.json", "w"), indent=1)
    log(f"\nsaved -> {OUT/'inference_size.json'}")


if __name__ == "__main__":
    main()
