"""Precondition for spine-supervised training: do annotators agree on the
CENTRELINE better than on the MASK?

The organizer has now confirmed (discussion 736543) that the dataset's other
manual metadata -- spine, bbox, area, category_id -- may be used as training
supervision; only external ground truth is forbidden. That reopens an axis we
had recorded as a grey area and never used.

Why it could matter, from our own characterisation: the instances we miss are
faint and thin, the model's probability peaks 255/255 under them, and what
fails is SHAPE AGREEMENT with an unstable convention (recall drops from 0.74 at
3/3 consensus to 0.17 at 1-of-3, and the strongest predictor of matching an
instance is the annotators' own pairwise IoU, Cohen's d 0.85). A centreline is
plausibly a more stable annotation than a boundary: two people may disagree on
how wide a filament is while agreeing on where it runs. If so, a spine
auxiliary head gives the shared encoder a target whose supervision is cleaner
than the mask's, aimed exactly at the failing population.

If instead spine agreement is no better than mask agreement, the auxiliary task
is redundant and the idea dies here for the cost of one measurement.

Note this is NOT the previously-rejected spine-ribbon experiment, which used
the spine as the OUTPUT representation and failed because width is not
recoverable from a centreline (IoU 0.6572 vs 0.6666 even with oracle width).
Here the mask head stays exactly as it is; the spine is only an auxiliary
supervision signal.

Method: for each image with >= 2 annotator entries, match instances across
entries at mask IoU > 0.5 (the metric's own criterion), then for each matched
pair compare
  * mask agreement   : IoU
  * spine agreement  : symmetric Chamfer distance between the two polylines,
                       and the fraction of points within 2/5/10 px
both normalised by the filament's own width (area / spine length), since a
5 px disagreement means something different on a 10 px and a 40 px filament.

Usage: python src/eda/spine_agreement.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament.rle import decode_rle

ANN = (ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train"
       / "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
OUT = ROOT / "outputs" / "eda"


def log(*a):
    print(*a, flush=True)


def poly(flat):
    a = np.asarray(flat, np.float32)
    return a.reshape(-1, 2) if a.size >= 4 else np.zeros((0, 2), np.float32)


def chamfer(p, q):
    """Symmetric mean nearest-neighbour distance between two point sets."""
    if len(p) == 0 or len(q) == 0:
        return np.nan
    d = np.linalg.norm(p[:, None, :] - q[None, :, :], axis=-1)
    return 0.5 * (d.min(1).mean() + d.min(0).mean())


def frac_within(p, q, r):
    if len(p) == 0 or len(q) == 0:
        return np.nan
    d = np.linalg.norm(p[:, None, :] - q[None, :, :], axis=-1)
    return 0.5 * ((d.min(1) <= r).mean() + (d.min(0) <= r).mean())


def main():
    d = json.load(open(ANN, encoding="utf-8"))
    log(f"{len(d['annotations'])} annotations, {len(d['images'])} image records")

    # group annotations by physical image stem; entry = annotator copy
    by_stem = defaultdict(lambda: defaultdict(list))
    for a in d["annotations"]:
        entry = a["image_id"]                    # "<annotatorEntry>-<stem>"
        stem = entry.split("-", 1)[1]
        by_stem[stem][entry].append(a)
    multi = {s: e for s, e in by_stem.items() if len(e) >= 2}
    log(f"{len(multi)} images with >= 2 annotator entries")

    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    rle_by_id = dict(zip(gt["filament_id"], gt["segmentation_rle"]))
    # gt filament_id is "<entry>_<k>"; annotations carry their own uuid, so
    # rebuild the same key by position within the entry, matching gt_cache order
    key_of = {}
    for entry, anns in ((e, a) for s in by_stem for e, a in by_stem[s].items()):
        for k, a in enumerate(anns, 1):
            key_of[a["id"]] = f"{entry}_{k}"

    rows = []
    for si, (stem, entries) in enumerate(multi.items()):
        names = sorted(entries)
        decoded = {}
        for e in names:
            for a in entries[e]:
                kid = key_of[a["id"]]
                r = rle_by_id.get(kid)
                if r is not None:
                    decoded[a["id"]] = decode_rle(r).astype(bool)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                for a in entries[names[i]]:
                    ma = decoded.get(a["id"])
                    if ma is None or not ma.any():
                        continue
                    aa = ma.sum()
                    for b in entries[names[j]]:
                        mb = decoded.get(b["id"])
                        if mb is None or not mb.any():
                            continue
                        inter = (ma & mb).sum()
                        if inter == 0:
                            continue
                        iou = inter / (aa + mb.sum() - inter)
                        if iou <= 0.5:            # the metric's own criterion
                            continue
                        sa, sb = poly(a.get("spine", [])), poly(b.get("spine", []))
                        # width proxy from the pair's mean area and spine length
                        la = max(np.linalg.norm(np.diff(sa, axis=0), axis=1).sum(), 1)
                        lb = max(np.linalg.norm(np.diff(sb, axis=0), axis=1).sum(), 1)
                        width = 0.5 * (aa / la + mb.sum() / lb)
                        rows.append(dict(
                            stem=stem, iou=float(iou), width=float(width),
                            chamfer=chamfer(sa, sb),
                            w2=frac_within(sa, sb, 2.0),
                            w5=frac_within(sa, sb, 5.0),
                            w10=frac_within(sa, sb, 10.0),
                            spine_len=float(0.5 * (la + lb))))
                        break                      # one partner per instance
        if (si + 1) % 60 == 0:
            log(f"  {si+1}/{len(multi)} images, {len(rows)} matched pairs")

    df = pd.DataFrame(rows).dropna(subset=["chamfer"])
    log(f"\n{len(df)} cross-annotator matched pairs\n")
    log("MASK agreement (the metric's own currency)")
    log(f"  IoU: median {df['iou'].median():.4f}  mean {df['iou'].mean():.4f}")
    log("\nSPINE agreement")
    log(f"  Chamfer distance: median {df['chamfer'].median():.2f} px  "
        f"mean {df['chamfer'].mean():.2f} px")
    log(f"  filament width  : median {df['width'].median():.2f} px")
    log(f"  Chamfer / width : median {(df['chamfer']/df['width']).median():.3f}")
    log(f"  spine points within  2 px: {df['w2'].mean():.3f}")
    log(f"                       5 px: {df['w5'].mean():.3f}")
    log(f"                      10 px: {df['w10'].mean():.3f}")

    # the comparison that decides it: convert both to a common "agreement" scale.
    # A mask IoU of 0.84 means ~8% of each mask disagrees on each side; a spine
    # Chamfer of c px on a filament of width w means the centrelines differ by
    # c/w of a width. If c/w << (1-IoU), the centreline is the cleaner target.
    cw = (df["chamfer"] / df["width"]).median()
    mi = 1 - df["iou"].median()
    log(f"\nVERDICT INPUTS: mask disagreement {mi:.3f} of area, "
        f"spine disagreement {cw:.3f} of width")
    log("  centreline is the cleaner target iff spine disagreement is "
        "materially below mask disagreement")
    df.to_parquet(OUT / "spine_agreement.parquet", index=False)
    log(f"\nsaved -> {OUT/'spine_agreement.parquet'}")


if __name__ == "__main__":
    main()
