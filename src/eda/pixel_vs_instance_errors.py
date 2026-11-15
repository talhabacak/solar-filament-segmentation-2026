"""Pixel-level and instance-level FP/FN side by side, and how the two connect.

Everything in this project is judged instance-side, because that is what PQ
counts. But the loss is optimised pixel-side, and the two can disagree
completely: a mask can be 90% correct by pixels and still be a false positive
and a false negative at once, because it fell short of IoU 0.5.

Three things are measured, and the third is the one that should shape what we
train next:

  1. PIXEL confusion over the val fold at the deployed extraction threshold.
     This is what the loss sees.
  2. INSTANCE confusion at the deployed operating point, scored the way the
     competition scores. This is what we are paid for.
  3. THE CROSSING. For each instance outcome -- matched, missed-by-under-coverage,
     missed-by-over-coverage, hallucinated -- the pixel composition inside it.
     An instance that fails IoU 0.5 fails for one of two opposite reasons, and
     they call for opposite loss adjustments:

         under-coverage   the prediction sits inside the truth and is too small.
                          FN pixels dominate. A recall-leaning term helps.
         over-coverage    the prediction swallows the truth and more. FP pixels
                          dominate. A precision-leaning term helps.

     Applying the wrong one costs twice, and this project has already paid that
     twice: Skeleton Recall pushed recall globally and inflated the foreground to
     1.09x ground truth (-0.018), while the first blob-loss arm pushed precision
     and cut coverage from 0.743 to 0.632.

The split also gives the honest denominator for any "add a loss term" proposal:
if 80% of the failing instances fail one way, a term aimed at the other way
cannot pay no matter how well it is tuned.

Usage: python src/eda/pixel_vs_instance_errors.py
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament.rle import decode_rle

OUT = ROOT / "outputs" / "eda"
SEL = ROOT / "outputs" / "selector"
SEM = ROOT / "outputs" / "semantic"
PROBS = "fold0_probs_cldice.npz"
SEM_THR, MIN_AREA = 0.3, 256
PSEL, CONF = 0.35, 0.40


def log(*a):
    print(*a, flush=True)


def boxed(m):
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return None
    return ((int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1),
            m[ys.min():ys.max() + 1, xs.min():xs.max() + 1], int(m.sum()))


def iou_and_parts(pred, gt):
    """IoU plus the two one-sided errors, all in pixels."""
    inter = int(np.logical_and(pred, gt).sum())
    p, g = int(pred.sum()), int(gt.sum())
    union = p + g - inter
    return (inter / union if union else 0.0), inter, p - inter, g - inter


def main():
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    f0 = set(folds.loc[folds["fold"] == 0, "file_name"]
             .str.replace(".jpeg", "", regex=False))
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt = gt[gt["stem"].isin(f0)]
    cand = pd.read_parquet(SEL / "fused_pool_all_scored.parquet")
    cand = cand[cand["stem"].isin(f0)].copy()
    cand["kept"] = (cand["p_BASE+SRC+CROSS"] >= PSEL) & (cand["conf"] >= CONF)
    kept = cand[cand["kept"]]
    z = np.load(SEM / PROBS)
    log(f"fold 0: {len(gt)} ground-truth rows, {len(cand)} candidates, "
        f"{len(kept)} kept by the deployed policy")

    # ---------------- 1. pixel confusion, what the loss sees ----------------
    tp_px = fp_px = fn_px = 0
    t0 = time.time()
    for stem, g in gt.groupby("stem"):
        if stem not in z:
            continue
        prob = np.asarray(z[stem]).astype(np.float32) / 255.0
        pred = prob >= SEM_THR
        truth = np.zeros_like(pred)
        for r in g.itertuples():
            truth |= decode_rle(r.segmentation_rle).astype(bool)
        tp_px += int((pred & truth).sum())
        fp_px += int((pred & ~truth).sum())
        fn_px += int((~pred & truth).sum())
    p_prec = tp_px / max(tp_px + fp_px, 1)
    p_rec = tp_px / max(tp_px + fn_px, 1)
    log(f"\n1. PIXEL LEVEL (semantic field at {SEM_THR}, what the loss optimises) "
        f"[{time.time()-t0:.0f}s]")
    log(f"   TP {tp_px:9d}   FP {fp_px:9d}   FN {fn_px:9d}")
    log(f"   precision {p_prec:.4f}   recall {p_rec:.4f}   "
        f"Dice {2*tp_px/max(2*tp_px+fp_px+fn_px,1):.4f}")
    log(f"   FP/FN ratio {fp_px/max(fn_px,1):.2f}  "
        f"-> {'over' if fp_px > fn_px else 'under'}-predicting by pixels")

    # ---------------- 2 and 3. instance confusion and its pixel make-up -----
    rows, t0 = [], time.time()
    for stem, g in gt.groupby("stem"):
        k = kept[kept["stem"] == stem]
        preds = [(r.cand_id, decode_rle(r.rle).astype(bool)) for r in k.itertuples()]
        used = set()
        for r in g.itertuples():
            truth = decode_rle(r.segmentation_rle).astype(bool)
            best, bid, bparts = 0.0, None, (0, 0, 0)
            for cid, pm in preds:
                v, i, fp, fn = iou_and_parts(pm, truth)
                if v > best:
                    best, bid, bparts = v, cid, (i, fp, fn)
            inter, fp, fn = bparts
            if best > 0.5:
                out = "matched"
                used.add(bid)
            elif best == 0.0:
                out = "no overlapping prediction"
            elif fn > fp:
                out = "missed, UNDER-covered"
            else:
                out = "missed, OVER-covered"
            rows.append(dict(kind="gt", stem=stem, id=r.filament_id, outcome=out,
                             iou=best, inter=inter, fp_px=fp, fn_px=fn,
                             gt_area=int(truth.sum())))
        for cid, pm in preds:
            if cid in used:
                continue
            best = 0.0
            for r in g.itertuples():
                v, _, _, _ = iou_and_parts(pm, decode_rle(r.segmentation_rle).astype(bool))
                best = max(best, v)
            rows.append(dict(kind="pred", stem=stem, id=cid,
                             outcome=("hallucination, no overlap" if best == 0
                                      else "unmatched prediction"),
                             iou=best, inter=0, fp_px=int(pm.sum()), fn_px=0,
                             gt_area=0))
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "pixel_vs_instance_errors.parquet", index=False)

    g_ = df[df.kind == "gt"]
    p_ = df[df.kind == "pred"]
    n_tp = int((g_["outcome"] == "matched").sum())
    n_fn = len(g_) - n_tp
    n_fp = len(p_)
    log(f"\n2. INSTANCE LEVEL (deployed policy, what PQ counts) "
        f"[{time.time()-t0:.0f}s]")
    log(f"   TP {n_tp:5d}   FP {n_fp:5d}   FN {n_fn:5d}")
    log(f"   precision {n_tp/max(n_tp+n_fp,1):.4f}   "
        f"recall {n_tp/max(n_tp+n_fn,1):.4f}")
    log(f"   FP/FN ratio {n_fp/max(n_fn,1):.2f}  "
        f"-> {'over' if n_fp > n_fn else 'under'}-predicting by instances")

    log("\n3. THE CROSSING -- why each missed ground-truth row was missed")
    t = g_.groupby("outcome").agg(n=("id", "size"), median_iou=("iou", "median"),
                                  median_fp=("fp_px", "median"),
                                  median_fn=("fn_px", "median"),
                                  median_gt_area=("gt_area", "median"))
    t["share"] = 100 * t["n"] / len(g_)
    log(t.to_string(float_format=lambda v: f"{v:.1f}"))
    miss = g_[g_["outcome"].str.startswith("missed")]
    if len(miss):
        u = int((miss["outcome"] == "missed, UNDER-covered").sum())
        log(f"\n   of the {len(miss)} rows that had an overlapping prediction but "
            f"missed the threshold:")
        log(f"     UNDER-covered (prediction too small): {u} "
            f"({100*u/len(miss):.1f}%)")
        log(f"     OVER-covered  (prediction too large): {len(miss)-u} "
            f"({100*(len(miss)-u)/len(miss):.1f}%)")
        log("   a lopsided split names the loss adjustment; a balanced one says")
        log("   no global term can help, because half the population needs the")
        log("   opposite of what the other half needs.")
    log(f"\n   unmatched predictions: {len(p_)}, of which "
        f"{int((p_['outcome'] == 'hallucination, no overlap').sum())} overlap no "
        f"ground truth at all")

    json.dump(dict(pixel=dict(tp=tp_px, fp=fp_px, fn=fn_px,
                              precision=p_prec, recall=p_rec),
                   instance=dict(tp=n_tp, fp=n_fp, fn=n_fn),
                   outcomes={k: int(v) for k, v in
                             g_["outcome"].value_counts().items()}),
              open(OUT / "pixel_vs_instance_errors.json", "w"), indent=1)
    log(f"\nwrote {OUT/'pixel_vs_instance_errors.json'}")


if __name__ == "__main__":
    main()
