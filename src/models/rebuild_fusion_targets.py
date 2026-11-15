"""E191c -- fix the refiner's target: the CONSENSUS mask, not one annotator's copy.

The first run of the two-lane refiner made every mask worse (matched copies 0.687 -> 0.645,
boundary band 0.300 -> 0.243, net -45 crossings of IoU 0.5). Before reading that as a verdict
on the idea, one design error has to be removed, and it is mine: `build_fusion_crops.py` set the
target to the SINGLE ground-truth copy the candidate matched best.

The scorer does not work that way. It matches a prediction against every annotator entry, so a
mask is worth the SUM of its IoUs over the copies -- and ledger 6a measured that the majority
consensus sits closer to each annotator (0.706) than two annotators sit to each other (0.629).
Training toward one copy therefore teaches the refiner to imitate one annotator's convention,
which is measurably worse than the consensus the candidate was already near. The project's own
earlier refiner used the majority mask for exactly this reason.

This rebuilds only the target array (the five input channels are untouched): for each candidate,
every ground-truth copy overlapping it above IoU 0.05 votes, and the target is the pixel-wise
majority of those votes.

Usage: python src/models/rebuild_fusion_targets.py [--crop 384]
"""
import argparse, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/models"):
    sys.path.insert(0, str(ROOT / p))
from filament.rle import decode_rle
from filament.pq import _entry_iou_matrix
DEST = ROOT / "data" / "fusion_crops"
OUT = ROOT / "outputs" / "eda"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--crop", type=int, default=384)
    a = ap.parse_args(); C = a.crop
    md = pd.read_parquet(DEST / f"meta_{C}.parquet")
    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    pool = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet").set_index("cand_id")
    Y = np.load(DEST / f"y_{C}.npy", mmap_mode="r+")
    n_changed = 0
    for stem, g in md.groupby("stem"):
        gg = gt[gt["stem"] == stem]
        if not len(gg):
            continue
        G = np.stack([decode_rle(r).astype(np.uint8) for r in gg["segmentation_rle"]])
        for r in g.itertuples():
            m = decode_rle(pool.loc[r.cand_id, "rle"]).astype(np.uint8)
            iou = _entry_iou_matrix(G, m[None])[:, 0]
            hit = np.flatnonzero(iou > 0.05)
            sl = (slice(r.y0, r.y0 + C), slice(r.x0, r.x0 + C))
            if len(hit) == 0:
                Y[r.Index] = 0
                continue
            maj = (G[hit].sum(axis=0) * 2 >= len(hit)).astype(np.uint8)   # pixel-wise majority
            Y[r.Index] = maj[sl] * 255
            n_changed += 1
    Y.flush()
    print(f"rebuilt {n_changed} consensus targets of {len(md)} crops", flush=True)
    print("TARGETS_DONE")


if __name__ == "__main__":
    main()
