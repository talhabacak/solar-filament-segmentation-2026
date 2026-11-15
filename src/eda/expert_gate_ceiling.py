"""Would a per-instance gate beat the fixed blend, and could anything drive it?

The blob field already pays as an ensemble MEMBER: averaged with the production
clDice field at equal weight it gains +0.0039 on the fold-0 harness with a paired
standard error of 0.0012. The mixture-of-experts proposal is that a GATE -- send
each region to whichever expert is better for it -- would beat that fixed average.

Two questions, in the order that can kill the idea cheaply:

  1. CEILING. Give an oracle the per-instance choice between the two experts,
     knowing the ground truth. If that barely beats the fixed 50/50 blend, no
     gate can help and the feature question never arises.
  2. DRIVABILITY. If the ceiling is real, is the winner predictable from things
     visible at INFERENCE? The oracle picks per ground-truth instance; a
     deployed gate sees only the prediction. So the features tested are ones a
     candidate carries on its own -- area, elongation, width, its own predicted
     confidence -- not ground-truth morphology.

The second question is where the per-instance mask-offset lane died: its oracle
was worth +0.0320 and the predictor reached a correlation of 0.2177 with it,
which cost 0.0284 to deploy. Any gate faces the same test.

Experts compared, all extracted at the production threshold:
  A  production clDice 1536      fold0_probs_cldice.npz
  B  instance-balanced blob 1536 fold0_probs_blob1536.npz
  C  their equal-weight average  fold0_probs_blobens.npz

Usage: python src/eda/expert_gate_ceiling.py
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

from filament.rle import decode_rle

OUT = ROOT / "outputs" / "eda"
SEM = ROOT / "outputs" / "semantic"
FIELDS = {"cldice": "fold0_probs_cldice.npz",
          "blob": "fold0_probs_blob1536.npz",
          "blend": "fold0_probs_blobens.npz"}
THR, MIN_AREA = 0.3, 256


def log(*a):
    print(*a, flush=True)


def boxed(m):
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return None
    return ((int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1),
            m[ys.min():ys.max() + 1, xs.min():xs.max() + 1], int(m.sum()))


def iou(a, b):
    if a is None or b is None:
        return 0.0
    (ay0, ay1, ax0, ax1), asub, asum = a
    (by0, by1, bx0, bx1), bsub, bsum = b
    y0, y1 = max(ay0, by0), min(ay1, by1)
    x0, x1 = max(ax0, bx0), min(ax1, bx1)
    if y0 >= y1 or x0 >= x1:
        return 0.0
    i = int(np.logical_and(asub[y0 - ay0:y1 - ay0, x0 - ax0:x1 - ax0],
                           bsub[y0 - by0:y1 - by0, x0 - bx0:x1 - bx0]).sum())
    u = asum + bsum - i
    return i / u if u else 0.0


def comps(prob):
    b = (prob >= THR).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA:
            out.append(lab == i)
    return out


def shape(m):
    """Features a deployed gate could read off the PREDICTION alone."""
    ys, xs = np.nonzero(m)
    if len(ys) < 8:
        return None
    cy, cx = ys.mean(), xs.mean()
    y, x = ys - cy, xs - cx
    cov = np.array([[(x * x).mean(), (x * y).mean()],
                    [(x * y).mean(), (y * y).mean()]])
    w, _ = np.linalg.eigh(cov)
    elong = float(np.sqrt(max(w[1], 1e-9) / max(w[0], 1e-9)))
    area = int(m.sum())
    return dict(pred_area=area, pred_elong=elong,
                pred_width=area / max(np.sqrt(max(w[1], 1e-9)) * 4, 1e-6))


def main():
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    f0 = set(folds.loc[folds["fold"] == 0, "file_name"]
             .str.replace(".jpeg", "", regex=False))
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt = gt[gt["stem"].isin(f0)]
    z = {k: np.load(SEM / v) for k, v in FIELDS.items()}
    log(f"fold 0: {len(gt)} ground-truth rows, experts "
        f"{', '.join(FIELDS)}")

    rows, t0 = [], time.time()
    for stem, g in gt.groupby("stem"):
        if any(stem not in z[k] for k in FIELDS):
            continue
        cs = {}
        for k in FIELDS:
            prob = np.asarray(z[k][stem]).astype(np.float32) / 255.0
            cs[k] = [(boxed(c), shape(c)) for c in comps(prob)]
            cs[k] = [(b, s) for b, s in cs[k] if b is not None]
        for r in g.itertuples():
            gm = boxed(decode_rle(r.segmentation_rle).astype(bool))
            if gm is None:
                continue
            rec = dict(filament_id=r.filament_id, stem=stem)
            for k in FIELDS:
                best, bs = 0.0, None
                for b, s in cs[k]:
                    v = iou(gm, b)
                    if v > best:
                        best, bs = v, s
                rec[k] = best
                if k == "blend" and bs:
                    rec.update(bs)
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "expert_gate_ceiling.parquet", index=False)
    log(f"scored {len(df)} rows ({time.time()-t0:.0f}s)\n")

    df["oracle_gate"] = df[["cldice", "blob"]].max(axis=1)
    log("1. CEILING -- rows reached at IoU > 0.5, and mean IoU")
    log(f"{'expert':26s} {'rows > 0.5':>11s} {'mean IoU':>10s}")
    for k in ["cldice", "blob", "blend", "oracle_gate"]:
        log(f"  {k:24s} {int((df[k] > 0.5).sum()):11d} {df[k].mean():10.4f}")
    gain = int((df["oracle_gate"] > 0.5).sum()) - int((df["blend"] > 0.5).sum())
    log(f"\n  a PERFECT per-instance gate beats the fixed blend by {gain:+d} rows "
        f"and {df['oracle_gate'].mean() - df['blend'].mean():+.4f} mean IoU")
    log("  (the oracle picks knowing the ground truth; the blend does not)")

    # 2. drivability -- only among rows where the two experts actually differ
    d = df[(df["cldice"] > 0.02) | (df["blob"] > 0.02)].copy()
    d["delta"] = d["blob"] - d["cldice"]
    d["blob_wins"] = (d["delta"] > 0).astype(int)
    sep = d[abs(d["delta"]) > 0.05]
    log(f"\n2. DRIVABILITY")
    log(f"   rows where either expert fires: {len(d)}")
    log(f"   blob strictly better: {int((d['delta'] > 0).sum())}, "
        f"clDice better: {int((d['delta'] < 0).sum())}, "
        f"tied: {int((d['delta'] == 0).sum())}")
    log(f"   rows where they differ by more than 0.05 IoU: {len(sep)}")

    feats = [c for c in ["pred_area", "pred_elong", "pred_width"] if c in d.columns]
    sub = sep.dropna(subset=feats)
    if len(sub) > 60 and feats:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import GroupKFold
        from sklearn.metrics import roc_auc_score
        X = sub[feats].to_numpy(np.float64)
        y = sub["blob_wins"].to_numpy()
        grp = sub["stem"].to_numpy()
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, grp):
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06)
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        auc = roc_auc_score(y, oof)
        log(f"   grouped-CV AUC predicting WHICH expert wins, from prediction-side "
            f"features only: {auc:.4f}")
        log(f"   base rate (blob wins): {y.mean():.3f}")
        log("   an AUC near 0.5 means the winner is not visible at inference and")
        log("   a gate would be guessing -- which is how the per-instance offset")
        log("   lane died, at a correlation of 0.2177 with its own oracle.")
        json_auc = float(auc)
    else:
        json_auc = None
        log("   too few separated rows to fit a gate")

    json.dump(dict(rows=len(df),
                   reached={k: int((df[k] > 0.5).sum())
                            for k in ["cldice", "blob", "blend", "oracle_gate"]},
                   mean_iou={k: float(df[k].mean())
                             for k in ["cldice", "blob", "blend", "oracle_gate"]},
                   gate_auc=json_auc),
              open(OUT / "expert_gate_ceiling.json", "w"), indent=1)
    log(f"\nwrote {OUT/'expert_gate_ceiling.json'}")


if __name__ == "__main__":
    main()
