"""Rubric arm D: the organizer's many-to-many plot on OUR output, plus the
unannotated-real-filament gallery.

Both exist because the organizer said so explicitly. On judging (735256): the
committee "will also look at the distribution of many-to-many relations", and
the Self-Evaluation notebook's plot is named as the reference -- its definition
is per annotator-image entry, an IoU matrix between GT and predictions, a link
being ANY overlap (IoU > 0), and the histogram of per-GT and per-prediction link
degrees. Degree 1 both ways is the ideal; GT-side degree >= 2 is fragmentation,
prediction-side degree >= 2 is merging. This reproduces those counts for the
production submission path (shrunk masks, production policy, OOF selector), so
the report can show the distribution the judges will draw, before they draw it.

The gallery answers the other statement: "we are interested in algorithms which
detect all filaments whether they were annotated by human or not." Our scored
false positives at high selector confidence are exactly the place where such
detections live -- dark, elongated, filament-shaped regions no annotator drew.
Showing the strongest of them, with their images, makes the case that part of
our FP count is unannotated recall, which the metric charges for and the
committee says it values.

Usage: python src/eda/m2n_report.py
"""

import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament.rle import decode_rle

sys.path.insert(0, str(ROOT / "src" / "eda"))
from setlevel_pq_separability import oof_selector_scores  # noqa: E402

OUT = ROOT / "outputs" / "eda"
GAL = ROOT / "outputs" / "gallery"
IMDIR = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"

# 2026-09-14: refreshed to the SHIPPED configuration. The first version of this script read the
# August pool (threshold 0.3), the 32-feature selector and a 0.35 gate, so the distribution it
# reported was three axes away from the submission the judges will score. The pool below is the
# production one with masks already shrunk 5 %, and reproducing PQ 0.4421 from it is the check
# that the path is the deployed one.
POOL = ROOT / "outputs" / "eda" / "fused_pool_all_blob5_t020_labelled.parquet"
BUNDLE = ROOT / "outputs" / "selector" / "selector_v5_shrunk_t020pox.joblib"
POLICY = (0.40, 0.60)
EXPECT_PQ = 0.4421


def log(*a):
    print(*a, flush=True)


def main():
    import joblib
    from filament.pq import score
    from extra_features import EXTRA

    t0 = time.time()
    cand = pd.read_parquet(POOL)
    extra = pd.read_parquet(OUT / "extra_features_t020.parquet")
    cand = cand.merge(extra[["cand_id"] + [c for c in EXTRA if c in extra.columns]],
                      on="cand_id", how="left")
    folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    cand["fold"] = cand["stem"].map(fold_of).astype(int)
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    log(f"{len(cand)} candidates, {len(gt)} GT rows")

    feats = list(joblib.load(BUNDLE)["features"])
    scored = oof_selector_scores(cand, feats)
    scored["p_sel"] = scored["p"]
    kept = scored[(scored["conf"] >= POLICY[0]) & (scored["p_sel"] >= POLICY[1])].copy()
    log(f"kept {len(kept)} at production policy ({time.time()-t0:.0f}s)")

    # the path is only the deployed one if it returns the deployed score
    got = score(gt[["filament_id", "segmentation_rle"]],
                kept[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id",
                                                         "rle": "segmentation_rle"}))["pq"]
    log(f"reproduced PQ {got:.4f} against the shipped {EXPECT_PQ:.4f}")
    assert abs(got - EXPECT_PQ) < 0.001, f"this is not the production path: PQ {got:.4f}"

    # ---- many-to-one / one-to-many degrees, organizer's definition ---------
    gt["stem"] = gt["file_name"].str.rsplit(".", n=1).str[0]
    gt["entry"] = gt["filament_id"].str.split("_", n=1).str[0]
    # The official get_overlap_df builds ONE MATRIX PER ANNOTATOR-IMAGE ENTRY,
    # so a prediction's degree is counted within each entry separately and the
    # prediction contributes one count per entry it is scored against. A first
    # version accumulated across entries, which inflates prediction degrees by
    # the annotator multiplicity (a prediction rightly overlapping the same
    # filament's 3 copies is degree 1 three times, not degree 3).
    gt_deg, pr_deg = [], []
    for stem, kg in kept.groupby("stem"):
        pmasks = [decode_rle(r).astype(bool) for r in kg["rle"]]
        gsub = gt[gt["stem"] == stem]
        for entry, eg in gsub.groupby("entry"):
            gmasks = [decode_rle(r).astype(bool) for r in eg["segmentation_rle"]]
            pred_hits = np.zeros(len(pmasks), int)
            for gm in gmasks:
                deg = 0
                for j, pm in enumerate(pmasks):
                    if (gm & pm).any():
                        deg += 1
                        pred_hits[j] += 1
                gt_deg.append(deg)
            pr_deg.extend(pred_hits.tolist())
    # stems with no kept predictions: every GT there has degree 0
    empty = set(gt["stem"]) - set(kept["stem"])
    gt_deg += [0] * int(gt[gt["stem"].isin(empty)].shape[0])

    from collections import Counter
    gd, pdg = Counter(gt_deg), Counter(pr_deg)
    log(f"GT degrees:   {dict(sorted(gd.items()))}")
    log(f"pred degrees: {dict(sorted(pdg.items()))}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    degs = sorted(set(gd) | set(pdg))
    ax.barh([d + 0.2 for d in degs], [gd.get(d, 0) for d in degs], height=0.38,
            label="GT -> predictions (1:n)", color="#2c7fb8")
    ax.barh([d - 0.2 for d in degs], [-pdg.get(d, 0) for d in degs], height=0.38,
            label="prediction -> GTs (n:1)", color="#d95f0e")
    ax.axhline(1.5, ls="--", c="gray", lw=0.8)
    ax.set_yticks(degs)
    ax.set_ylabel("link degree (IoU > 0 partners)")
    ax.set_xlabel("count   (left: predictions, right: ground truth)")
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{abs(int(t))}" for t in ticks])
    ax.legend()
    ax.set_title("Many-to-one / one-to-many relations, production OOF output")
    fig.tight_layout()
    GAL.mkdir(parents=True, exist_ok=True)
    fig.savefig(GAL / "m2n_distribution.png", dpi=150)
    log(f"plot -> {GAL/'m2n_distribution.png'}")

    frag = sum(c for d, c in gd.items() if d >= 2)
    merg = sum(c for d, c in pdg.items() if d >= 2)
    json.dump(dict(gt_degrees=dict(gd), pred_degrees=dict(pdg),
                   gt_fragmented=frag, pred_merging=merg,
                   gt_rows=len(gt_deg), preds=len(pr_deg)),
              open(OUT / "m2n_degrees.json", "w"), indent=1)

    # ---- gallery: high-confidence scored-FPs that look like filaments ------
    fp = kept[kept["weight"] == 0].sort_values("p_sel", ascending=False).head(16)
    log(f"\ntop scored-FP candidates by selector confidence: {len(fp)}")
    tiles = []
    for r in fp.itertuples():
        m = decode_rle(r.rle).astype(np.uint8)
        ys, xs = np.nonzero(m)
        cy, cx = int(ys.mean()), int(xs.mean())
        h = max(256, int(max(np.ptp(ys), np.ptp(xs)) * 1.6))
        y0, x0 = np.clip(cy - h // 2, 0, 2048 - h), np.clip(cx - h // 2, 0, 2048 - h)
        img = cv2.imread(str(next(IMDIR.rglob(r.stem + ".jpeg"))),
                         cv2.IMREAD_GRAYSCALE)
        crop = img[y0:y0 + h, x0:x0 + h]
        lo, hi = np.percentile(crop, 2), np.percentile(crop, 98)
        vis = np.clip((crop.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1) * 255
        vis = cv2.cvtColor(vis.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cnt, _ = cv2.findContours(m[y0:y0 + h, x0:x0 + h],
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnt, -1, (0, 200, 255), 1)
        vis = cv2.resize(vis, (256, 256), interpolation=cv2.INTER_AREA)
        cv2.putText(vis, f"p={r.p_sel:.2f}", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
        tiles.append(vis)
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    cv2.imwrite(str(GAL / "fp_unannotated.png"), np.vstack(rows))
    log(f"gallery -> {GAL/'fp_unannotated.png'}")


if __name__ == "__main__":
    main()
