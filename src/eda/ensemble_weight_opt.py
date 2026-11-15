"""E192 -- the one knob never optimised out of fold: the blob/clDice balance of the field.

Every experiment this week asked "does another member help", and the 1/k law bounds that answer.
The law says nothing about the weights of the members already there. The production field is

    0.50 x blob(5 models) + (2/6) x heavy(2 models) + (1/6) x clDice(5 folds)

and those numbers were set when the recipe was assembled. Variants of them (blob50, blob5_65,
blob5_pure) were submitted and read on the LEADERBOARD as whole recipes -- never measured out of
fold against the thing the field actually controls.

Scope. The two heavy arms were only ever trained on folds 0 and 1, so the sweep is over the one
weight every fold can carry: the blob share, with clDice taking the remainder. Folds 2-4 choose,
folds 0 and 1 confirm, and the confirmation is read once.

Criterion. Pool PQ per weight per fold would be hours of pool builds; the field controls REACH
(ground-truth copies whose best connected component at the production threshold 0.2 clears
IoU 0.5), and everything downstream is unchanged, so reach is the honest cheap criterion. A
weight that cannot move reach cannot move PQ except through the selector, which is not what a
blend weight is for.

Usage: python src/eda/ensemble_weight_opt.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/eda", "src/models"):
    sys.path.insert(0, str(ROOT / p))
from filament.rle import decode_rle  # noqa: E402
from filament.pq import _entry_iou_matrix  # noqa: E402
from semantic_candidates import prob_to_instances  # noqa: E402

OUT = ROOT / "outputs" / "eda"
SEM = ROOT / "outputs" / "semantic"
THR, MIN_AREA = 0.2, 256
BLOB, CLDICE = "blob1536", "cldice"
GRID = [0.35, 0.50, 0.65, 0.80, 1.00]


def log(*a):
    print(*a, flush=True)


def reach_for(field, G):
    comps = prob_to_instances(field, THR, MIN_AREA)
    if not comps:
        return 0
    P = np.stack([c["mask"].astype(np.uint8) for c in comps])
    return int((_entry_iou_matrix(G, P).max(axis=1) > 0.5).sum())


def main():
    folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
    folds["stem"] = folds["file_name"].str.replace(".jpeg", "", regex=False)
    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    log(f"grid: blob share {GRID}, clDice takes the remainder")
    per_fold = {}
    for F in (2, 3, 4, 1, 0):
        stems = sorted(set(folds.loc[folds["fold"] == F, "stem"]) & set(gt["stem"]))
        try:
            Zb = np.load(SEM / f"fold{F}_probs_{BLOB}.npz")
            Zc = np.load(SEM / f"fold{F}_probs_{CLDICE}.npz")
        except FileNotFoundError as e:
            log(f"fold {F}: missing map ({e}); skipped")
            continue
        counts = {b: 0 for b in GRID}
        tot = 0
        for stem in stems:
            gg = gt[gt["stem"] == stem]
            if not len(gg) or stem not in Zb.files or stem not in Zc.files:
                continue
            G = np.stack([decode_rle(r).astype(np.uint8) for r in gg["segmentation_rle"]])
            tot += len(G)
            mb = cv2.resize(Zb[stem], (2048, 2048), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            mc = cv2.resize(Zc[stem], (2048, 2048), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            for b in GRID:
                counts[b] += reach_for(b * mb + (1.0 - b) * mc, G)
        per_fold[F] = dict(total=tot, counts={str(b): v for b, v in counts.items()})
        log(f"fold {F}: {tot} copies | " + "  ".join(f"{b:.2f}:{counts[b]}" for b in GRID))
    choose = [k for k in (2, 3, 4) if k in per_fold]
    if choose:
        train = {b: sum(per_fold[k]["counts"][str(b)] for k in choose) for b in GRID}
        best = max(train, key=train.get)
        log("\nreach on the choosing folds " + str(choose) + ":")
        for b in GRID:
            log(f"  blob {b:.2f} / clDice {1 - b:.2f}: {train[b]}" + ("   <- chosen" if b == best else ""))
        for F in (0, 1):
            if F in per_fold:
                c = per_fold[F]["counts"]
                log(f"HELD-OUT fold {F} ({per_fold[F]['total']} copies): "
                    + "  ".join(f"{b:.2f}:{c[str(b)]}" for b in GRID)
                    + f"   | chosen {best:.2f} vs production 0.50: {c[str(best)] - c['0.5']:+d} rows")
        json.dump(dict(per_fold=per_fold, chosen=best), open(OUT / "ensemble_weight_opt.json", "w"), indent=1)
    log("WEIGHT_OPT_DONE")


if __name__ == "__main__":
    main()
