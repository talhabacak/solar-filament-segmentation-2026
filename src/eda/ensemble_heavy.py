"""Does the heavy-recipe model add anything to the production semantic ensemble?

For TEST predictions any model trained on any training data is legitimate -- the
fold structure only constrains out-of-fold VALIDATION. So the production lane's
five 1536 models can be joined by the new heavy-augmentation model as a sixth
member without any leakage, and the question is only whether it helps.

The ledger's principle says it might, where more capacity would not: "what adds
value to an ensemble is not another model, it is a model that makes DIFFERENT
mistakes". Every previous diversity attempt failed that test because the members
differed only in seed or size (8-model diversity ensemble: 0.4880 vs 0.4879). The
heavy model differs in TRAINING RECIPE -- it saw a 1.5x-of-nature augmentation
distribution over 60 epochs -- which is a different kind of difference.

This is honestly measurable on fold 0: sem_b3_f0_1536_cldice and
aug_heavy_f0_768_60ep were both trained on folds 1-4, so neither has seen fold 0.
Their agreement, and whether averaging beats either alone, is a clean local
measurement.

Reports three things, because "the ensemble is better" is not enough to act on:

  correlation   how different ARE the two models' errors? Near-identical error
                maps mean averaging can only smooth noise.
  pixel-Dice    each alone, and the average, at the operating threshold.
  per-image     how often the ensemble beats the better single model, so a mean
                gain driven by a few images is not mistaken for a broad one.

Usage: python src/eda/ensemble_heavy.py
"""

import argparse
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
MEMBERS = {
    "prod_1536": "outputs/runs_semantic/sem_b3_f0_1536_cldice/best.pt",
    "heavy_768": "outputs/runs_aug/aug_heavy_f0_768_60ep/best.pt",
    "heavy_1536": "outputs/runs_aug/aug_heavy_f0_1536_60ep/best.pt",
}

# WHY every model is re-scored here instead of trusting the runs' own val numbers.
# Each run reports pixel-Dice at ITS OWN training resolution, against a mask
# rasterised at that resolution -- so aug_heavy_f0_768_60ep's 0.7028 and
# aug_heavy_f0_1536_60ep's 0.7024 are not the same measurement. A 768 mask is
# coarser and simply cannot see the boundary error a 1536 mask scores. The
# submission works at native 2048, so that is the only common ground, and the
# first 8-image probe already showed the ordering flipping there.
# This is the same incomparable-metric trap docs/15 records for seed_b3_512.


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all fold images")
    a = ap.parse_args()

    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    files = sorted(folds.loc[folds["fold"] == a.fold, "file_name"].unique())
    if a.limit:
        files = files[:a.limit]
    log(f"fold {a.fold}: {len(files)} images")

    # consensus target per image: a pixel is positive if any annotator drew it,
    # matching how the semantic lane is trained and scored elsewhere.
    # gt_full stores the bare COCO counts string, not a {"size","counts"} dict,
    # so the project's own decoder is what knows the frame size.
    from filament.rle import decode_rle
    tgt = {}
    for fn, g in gt[gt["file_name"].isin(files)].groupby("file_name"):
        m = None
        for r in g["segmentation_rle"]:
            d = decode_rle(r)
            m = d if m is None else np.maximum(m, d)
        tgt[fn] = m.astype(bool)

    # predict_prob reads the file itself, applies hflip TTA and returns the map at
    # native 2048 -- i.e. exactly the production inference path, so the two
    # members are compared the way the submission would use them.
    imdir = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
    probs = {k: {} for k in MEMBERS}
    for name, w in MEMBERS.items():
        model, imgsz, channels = load_model(ROOT / w)
        log(f"  {name}: imgsz={imgsz} channels={channels}")
        for fn in files:
            stem = fn.rsplit(".", 1)[0]
            probs[name][fn] = predict_prob(model, imgsz, stem, tta=True,
                                           img_dir=imdir,
                                           channels=channels).astype(np.float32)
        del model
        torch.cuda.empty_cache()

    def dice(p, t, thr=0.5):
        b = p > thr
        return 2 * (b & t).sum() / max(b.sum() + t.sum(), 1)

    names = list(MEMBERS)
    log("\npooled pixel-Dice at NATIVE 2048, the resolution the metric uses:")
    pooled = {}
    for n in names:
        i = u = 0
        for fn in files:
            b, t = probs[n][fn] > 0.5, tgt[fn]
            i += (b & t).sum()
            u += b.sum() + t.sum()
        pooled[n] = 2 * i / max(u, 1)
        log(f"  {n:12s} {pooled[n]:.4f}")

    log("\npairwise probability-map correlation (is the diversity real?):")
    import itertools
    corr = {}
    for a1, b1 in itertools.combinations(names, 2):
        c = float(np.mean([np.corrcoef(probs[a1][fn].ravel(),
                                       probs[b1][fn].ravel())[0, 1]
                           for fn in files]))
        corr[f"{a1}|{b1}"] = c
        log(f"  {a1:12s} vs {b1:12s} {c:.4f}")
    log("  (near 1.0 means the same errors, so averaging can only smooth noise)")

    log("\naveraged combinations:")
    combos = {}
    for k in range(2, len(names) + 1):
        for cmb in itertools.combinations(names, k):
            i = u = 0
            for fn in files:
                p = sum(probs[n][fn] for n in cmb) / len(cmb)
                b, t = p > 0.5, tgt[fn]
                i += (b & t).sum()
                u += b.sum() + t.sum()
            v = 2 * i / max(u, 1)
            combos["+".join(cmb)] = v
            best_member = max(pooled[n] for n in cmb)
            log(f"  {'+'.join(cmb):40s} {v:.4f}  "
                f"({v - best_member:+.4f} vs its best member)")
    overall_best_single = max(pooled.values())
    best_combo = max(combos.items(), key=lambda kv: kv[1])
    log(f"\n  best single {overall_best_single:.4f}   "
        f"best combination {best_combo[1]:.4f} ({best_combo[0]})   "
        f"gain {best_combo[1] - overall_best_single:+.4f}")

    df = pd.DataFrame([dict(file=fn, **{n: dice(probs[n][fn], tgt[fn]) for n in names})
                       for fn in files])

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "ensemble_heavy.parquet", index=False)
    json.dump(dict(pooled={k: float(v) for k, v in pooled.items()},
                   combos={k: float(v) for k, v in combos.items()},
                   corr=corr,
                   per_image={n: float(df[n].mean()) for n in names}),
              open(OUT / "ensemble_heavy.json", "w"), indent=1)
    log(f"\nsaved -> {OUT/'ensemble_heavy.json'}")


if __name__ == "__main__":
    main()
