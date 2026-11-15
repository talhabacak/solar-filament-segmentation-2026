"""E145 -- how much is left in the CONTOUR LEVEL, one per candidate?

E143 showed the remaining headroom is not in selection: the true candidates the
gate drops are individually below the derived break-even (iou_sum > 0.5*c*PQ),
so admitting them cannot pay. A candidate's worth is its IoU, so the only way
across the line is a mask that FITS better. SQ is 0.678.

Every mask in the pipeline is cut at one global level, 0.2 on the probability
field. That is one number for every filament in every image, chosen once by a
sweep over the pooled score. But the level that best fits a faint, thin filament
is not the level that best fits a broad dark one, and nothing in the pipeline
lets them differ.

This measures the ceiling of letting them differ, before any of it is built.
For each deployed candidate the connected component through its centroid is
re-extracted at each level in a grid, its IoU against the ground-truth row it
already overlaps is computed, and the best level is kept. That is an ORACLE --
it uses the answer -- so it is an upper bound, not a method. It answers one
question: is there enough in per-candidate levels to be worth learning a
predictor for?

Reported alongside it: the best SINGLE global level on the same re-cut masks,
which is the part a predictor does not have to earn, and how far the oracle's
chosen levels spread. If they all land on one level the axis is closed on the
spot.

Usage: python src/eda/contour_level_oracle.py [--fold 0]
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
from filament.pq import score  # noqa: E402
from filament.rle import decode_rle, encode_mask  # noqa: E402
from mask_shrink import shrink_frac  # noqa: E402

OUT = ROOT / "outputs" / "eda"
SEM = ROOT / "outputs" / "semantic"
LEVELS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
GATE = (0.40, 0.60)
PAD = 40


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    a_ = ap.parse_args()
    F = a_.fold
    a = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    a = a.merge(pd.read_parquet(OUT / "t020pox_oof_scores.parquet"), on="cand_id", how="left")
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    a["fold"] = a["stem"].map(fold_of).astype(int)
    a = a[a["fold"] == F]
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    g = gt[gt["stem"].isin(set(a["stem"]))]
    keep = a[(a["conf"] >= GATE[0]) & (a["p"] >= GATE[1])].copy()
    log(f"fold {F}: {len(keep)} deployed candidates over {keep['stem'].nunique()} images")

    field = np.load(SEM / f"fold{F}_probs_blob5ens.npz")
    gt_by_stem = {s: list(zip(v["filament_id"], v["segmentation_rle"]))
                  for s, v in g.groupby("stem")[["filament_id", "segmentation_rle"]]}

    # the full-frame RLE encode is the whole cost, so only the two mask sets that
    # actually reach the exact scorer get encoded. Mean IoU needs no encoding, so
    # the best global level is chosen on that first.
    best_level, per_level_iou, per_cand = [], {L: [] for L in LEVELS}, []
    for stem, gg in keep.groupby("stem"):
        P = field[stem].astype(np.float32) / 255.0
        gm = [(fid, decode_rle(r).astype(bool)) for fid, r in gt_by_stem.get(stem, [])]
        for cid, rle in zip(gg["cand_id"], gg["rle"]):
            m0 = decode_rle(rle).astype(bool)
            ys, xs = np.nonzero(m0)
            if len(ys) == 0:
                continue
            flat = int(np.argmax(np.where(m0, P, -1.0)))
            cy, cx = flat // P.shape[1], flat % P.shape[1]
            y0, y1 = max(ys.min() - PAD, 0), min(ys.max() + PAD + 1, P.shape[0])
            x0, x1 = max(xs.min() - PAD, 0), min(xs.max() + PAD + 1, P.shape[1])
            sub = P[y0:y1, x0:x1]
            m0s = m0[y0:y1, x0:x1]
            cands = [gm_[y0:y1, x0:x1] for _, gm_ in gm if (gm_[y0:y1, x0:x1] & m0s).any()]
            masks, ious = {}, {}
            for L in LEVELS:
                _, lab = cv2.connectedComponents((sub >= L).astype(np.uint8))
                k = lab[cy - y0, cx - x0]
                mm = (lab == k) if k > 0 else np.zeros(lab.shape, bool)
                iou = 0.0
                for gsub in cands:
                    u = (mm | gsub).sum()
                    if u:
                        iou = max(iou, (mm & gsub).sum() / u)
                masks[L], ious[L] = mm, iou
                per_level_iou[L].append(iou)
            bl = max(LEVELS, key=lambda L: ious[L])
            best_level.append(bl)
            per_cand.append((cid, (y0, y1, x0, x1), masks, bl))

    def encode_at(chooser):
        rows = []
        for cid, (y0, y1, x0, x1), masks, bl in per_cand:
            full = np.zeros(field[keep["stem"].iloc[0]].shape, bool)
            full[y0:y1, x0:x1] = masks[chooser(bl)]
            full = shrink_frac(full, 0.05)
            rows.append((cid, encode_mask(full.astype(np.uint8))))
        return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])

    def sc(df):
        return score(g[["filament_id", "segmentation_rle"]], df)

    base = score(g[["filament_id", "segmentation_rle"]],
                 keep[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"}))
    log(f"\ndeployed (global 0.20, then shrunk 0.05)   PQ {base['pq']:.4f}  SQ {base['sq']:.4f}  "
        f"TP {base['tp']} FP {base['fp']} FN {base['fn']}")
    res = {"deployed": dict(pq=base["pq"], sq=base["sq"])}

    log("\nmean IoU of the re-cut mask at each single global level (no shrink):")
    for L in LEVELS:
        log(f"  level {L:.2f}   mean IoU {np.mean(per_level_iou[L]):.4f}")
    ctl = sc(encode_at(lambda bl: 0.20))
    log(f"\nCONTROL, same procedure at the deployed level 0.20 + shrink: PQ {ctl['pq']:.4f} "
        f"TP {ctl['tp']} FP {ctl['fp']} (deployed {base['pq']:.4f}, TP {base['tp']} FP {base['fp']})")
    if abs(ctl["pq"] - base["pq"]) > 0.005:
        log("  control does NOT reproduce the deployed masks; the oracle below is not "
            "comparable to deployment and only the level-to-level differences are readable")
    res["control_020"] = dict(pq=ctl["pq"], sq=ctl["sq"])
    bestg = max(LEVELS, key=lambda L: float(np.mean(per_level_iou[L])))
    rg, ro = sc(encode_at(lambda bl: bestg)), sc(encode_at(lambda bl: bl))
    res[f"global_{bestg}"] = dict(pq=rg["pq"], sq=rg["sq"])
    res["oracle"] = dict(pq=ro["pq"], sq=ro["sq"])
    log(f"\nbest single global level {bestg:.2f}   PQ {rg['pq']:.4f}  SQ {rg['sq']:.4f}  "
        f"TP {rg['tp']} FP {rg['fp']}")
    log(f"ORACLE per-candidate level    PQ {ro['pq']:.4f}  SQ {ro['sq']:.4f}  "
        f"TP {ro['tp']} FP {ro['fp']}")
    log(f"  -> {ro['pq'] - rg['pq']:+.4f} over the best global level, "
        f"{ro['pq'] - base['pq']:+.4f} over deployed")

    bl = pd.Series(best_level)
    log("\nwhere the oracle puts each candidate's level:")
    for L in LEVELS:
        n = int((bl == L).sum())
        log(f"  {L:.2f}: {n:5d} ({n / max(len(bl), 1):5.1%})")
    log(f"  {bl.nunique()} distinct levels used, median {bl.median():.2f}")
    res["level_hist"] = {str(L): int((bl == L).sum()) for L in LEVELS}
    # the per-candidate table is what a predictor is trained on: the level each
    # candidate wants, and what it would score at every other level
    pd.DataFrame([dict(cand_id=cid, best_level=b,
                       **{f"iou_{L:.2f}": per_level_iou[L][i] for L in LEVELS})
                  for i, (cid, _, _, b) in enumerate(per_cand)]
                 ).to_parquet(OUT / f"contour_levels_f{F}.parquet", index=False)
    log(f"wrote {OUT / f'contour_levels_f{F}.parquet'} ({len(per_cand)} candidates)")
    json.dump(res, open(OUT / f"contour_level_oracle_f{F}.json", "w"), indent=1, default=float)
    log(f"\nwrote {OUT / f'contour_level_oracle_f{F}.json'}")


if __name__ == "__main__":
    main()
