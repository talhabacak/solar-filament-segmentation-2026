"""E147 -- is E145's contour-level ceiling real, or is it the maximum of eight noisy draws?

E145 reported +0.0093 mean over five folds for letting each candidate pick its own
contour level, with a control that reproduced the deployed masks exactly. Two
attempts to learn that preference then failed badly (-0.0102 absolute target,
-0.0257 centred target), and E146b's diagnostic says why it might be unlearnable
in principle rather than in practice:

    IoU spread WITHIN a candidate across the eight levels   sd 0.0216
    IoU spread BETWEEN candidates                           sd 0.1925

Taking the maximum of eight draws with sd 0.0216 lifts the mean by roughly 1.4
standard deviations, about +0.030 in mean IoU, EVEN IF no level is genuinely
better for any candidate. An oracle that peeks at the answer collects that
selection bias in full. So the honest question is not how large the oracle is; it
is how much of the preference SURVIVES being measured on data the choice was not
made from.

The test uses a property of this dataset rather than a simulation. 296 of the 703
images carry more than one annotator's labels. For candidates in those images the
level can be chosen against the FIRST annotator's masks alone and then scored
against the REMAINING annotators' masks. A real preference -- this filament is
faint and wants a low cut -- transfers between annotators, because it is a fact
about the filament. Selection on noise does not transfer, because the noise is
redrawn.

Three numbers on the held-out annotator:

    at the deployed 0.20                    the floor
    at the level chosen on annotator A      what a perfect predictor could reach
    at the level chosen on annotator B      the within-annotator oracle, i.e. the
                                            same selection bias E145 measured

If the second sits at the first, the ceiling is an artefact and the axis is
closed. If it sits between them, the difference is the real recoverable budget.

Usage: python src/eda/contour_level_null.py [--fold 0]
"""

import argparse
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

OUT = ROOT / "outputs" / "eda"
SEM = ROOT / "outputs" / "semantic"
LEVELS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
GATE = (0.40, 0.60)
PAD = 40
DEFAULT = 0.20


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    F = ap.parse_args().fold
    a = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    a = a.merge(pd.read_parquet(OUT / "t020pox_oof_scores.parquet"), on="cand_id", how="left")
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    a["fold"] = a["stem"].map(fold_of).astype(int)
    a = a[a["fold"] == F]
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    ncopy = gt.groupby("stem")["entry_id"].nunique()
    multi = set(ncopy[ncopy >= 2].index)
    keep = a[(a["conf"] >= GATE[0]) & (a["p"] >= GATE[1]) & a["stem"].isin(multi)]
    log(f"fold {F}: {len(keep)} deployed candidates on {keep['stem'].nunique()} multi-annotator images")

    field = np.load(SEM / f"fold{F}_probs_blob5ens.npz")
    rows = []
    for stem, gg in keep.groupby("stem"):
        P = field[stem].astype(np.float32) / 255.0
        g = gt[gt["stem"] == stem]
        entries = sorted(g["entry_id"].unique())
        A, B = entries[:1], entries[1:]
        gA = [decode_rle(r).astype(bool) for r in g.loc[g["entry_id"].isin(A), "segmentation_rle"]]
        gB = [decode_rle(r).astype(bool) for r in g.loc[g["entry_id"].isin(B), "segmentation_rle"]]
        for cid, rle in zip(gg["cand_id"], gg["rle"]):
            m0 = decode_rle(rle).astype(bool)
            ys, xs = np.nonzero(m0)
            if len(ys) == 0:
                continue
            flat = int(np.argmax(np.where(m0, P, -1.0)))
            cy, cx = flat // P.shape[1], flat % P.shape[1]
            y0, y1 = max(ys.min() - PAD, 0), min(ys.max() + PAD + 1, P.shape[0])
            x0, x1 = max(xs.min() - PAD, 0), min(xs.max() + PAD + 1, P.shape[1])
            m0s = m0[y0:y1, x0:x1]
            wA = [m[y0:y1, x0:x1] for m in gA if (m[y0:y1, x0:x1] & m0s).any()]
            wB = [m[y0:y1, x0:x1] for m in gB if (m[y0:y1, x0:x1] & m0s).any()]
            if not wA or not wB:
                continue
            row = {"cand_id": cid}
            for L in LEVELS:
                _, lab = cv2.connectedComponents((P[y0:y1, x0:x1] >= L).astype(np.uint8))
                k = lab[cy - y0, cx - x0]
                mm = (lab == k) if k > 0 else np.zeros(lab.shape, bool)
                for tag, w in (("A", wA), ("B", wB)):
                    best = 0.0
                    for gsub in w:
                        u = (mm | gsub).sum()
                        if u:
                            best = max(best, (mm & gsub).sum() / u)
                    row[f"{tag}_{L:.2f}"] = best
            rows.append(row)
    d = pd.DataFrame(rows)
    log(f"{len(d)} candidates overlap both annotator groups\n")

    A = d[[f"A_{L:.2f}" for L in LEVELS]].to_numpy()
    B = d[[f"B_{L:.2f}" for L in LEVELS]].to_numpy()
    iD = LEVELS.index(DEFAULT)
    pickA, pickB = A.argmax(axis=1), B.argmax(axis=1)
    r = np.arange(len(d))
    floor = B[:, iD].mean()
    transfer = B[r, pickA].mean()
    within = B[r, pickB].mean()
    log("mean IoU measured on the HELD-OUT annotator group:")
    log(f"  at the deployed level 0.20                      {floor:.4f}")
    log(f"  at the level chosen on the other annotator      {transfer:.4f}  ({transfer - floor:+.4f})")
    log(f"  at the level chosen on the SAME annotator       {within:.4f}  ({within - floor:+.4f})")
    frac = (transfer - floor) / (within - floor) if within > floor else float("nan")
    log(f"\n  the oracle's apparent gain that actually transfers: {100 * frac:.0f}%")
    log(f"  agreement of the two chosen levels: {100 * (pickA == pickB).mean():.1f}% exact, "
        f"{100 * (np.abs(pickA - pickB) <= 1).mean():.1f}% within one step "
        f"(chance {100 / len(LEVELS):.1f}% and {100 * (3 * len(LEVELS) - 2) / len(LEVELS) ** 2:.1f}%)")
    log(f"  correlation of the chosen level index: {np.corrcoef(pickA, pickB)[0, 1]:+.3f}")
    json.dump(dict(fold=F, n=len(d), floor=float(floor), transfer=float(transfer),
                   within=float(within), transferred_fraction=float(frac),
                   exact_agreement=float((pickA == pickB).mean()),
                   corr=float(np.corrcoef(pickA, pickB)[0, 1])),
              open(OUT / f"contour_level_null_f{F}.json", "w"), indent=1, default=float)
    log(f"\nwrote {OUT / f'contour_level_null_f{F}.json'}")


if __name__ == "__main__":
    main()
