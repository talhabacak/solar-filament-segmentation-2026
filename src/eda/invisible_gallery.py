"""What do the consensus filaments our model cannot see actually look like?

The chain that leads here: the gap to the leaders is recall (budget: ceiling
0.6516 vs our 0.4105); three quarters of the missed consensus instances were
never proposed by any candidate; and multi-threshold extraction reaches only
0.7% of them -- at threshold 0.15 the probability map holds nothing filament-
shaped at those locations. The model does not see them AT ALL.

Yet on fold 0 alone, 60 of them were drawn by ALL THREE annotators (median area
2954 px) and 71 by both of two. Real, visible, consensus objects, each worth
marginal 1.412 PQ contribution -- invisible to a network that matches humans on
everything else.

Numbers cannot say what these look like; the fat-mask diagnosis came from a
gallery, so this renders one. Three panels per instance:

  crop        the image around the GT, contrast-stretched, GT contour drawn
  raw         the same crop unstretched, as the network receives it
  probability the model's OOF probability map there (same window)

If they are diffuse smudges, the model may be information-limited and the GT
partly conventional. If they are visibly filament-like, the failure is the
model's and something about the training regime suppresses faint positives --
which the loss's treatment of the 37.5% Unidentifiable class could plausibly do.

Usage: python src/eda/invisible_gallery.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament.rle import decode_rle

OUT = ROOT / "outputs" / "gallery"
SEM = ROOT / "outputs" / "semantic"
IMDIR = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
PANEL = 288
N_SHOW = 10


def log(*a):
    print(*a, flush=True)


def stretch(x):
    lo, hi = np.percentile(x, 2), np.percentile(x, 98)
    return np.clip((x.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1) * 255


def main():
    rf = pd.read_parquet(ROOT / "outputs" / "eda" / "multithresh_reach.parquet")
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    rf = rf.merge(gt[["filament_id", "segmentation_rle", "category_id"]],
                  on="filament_id", how="left")
    z = np.load(SEM / "fold0_probs_cldice.npz")
    paths = {p.stem: p for p in IMDIR.rglob("*.jpeg")}

    for tier_want, fname in (("3/3", "invisible_33.png"), ("2/2", "invisible_22.png")):
        sub = rf[(rf["tier"] == tier_want) & (rf["iou_new"] <= 0.3)]
        sub = sub.sort_values("area", ascending=False).head(N_SHOW)
        log(f"tier {tier_want}: rendering {len(sub)} largest never-proposed")
        rows = []
        for _, r in sub.iterrows():
            g = decode_rle(r["segmentation_rle"]).astype(np.uint8)
            ys, xs = np.nonzero(g)
            cy, cx = int(ys.mean()), int(xs.mean())
            h = max(PANEL, int(max(np.ptp(ys), np.ptp(xs)) * 1.6))
            y0 = np.clip(cy - h // 2, 0, 2048 - h)
            x0 = np.clip(cx - h // 2, 0, 2048 - h)
            img = cv2.imread(str(paths[r["stem"]]), cv2.IMREAD_GRAYSCALE)
            crop = img[y0:y0 + h, x0:x0 + h]
            gcrop = g[y0:y0 + h, x0:x0 + h]
            prob = z[r["stem"]][y0:y0 + h, x0:x0 + h]

            p1 = cv2.cvtColor(stretch(crop).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            cnt, _ = cv2.findContours(gcrop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(p1, cnt, -1, (0, 200, 255), 1)
            p2 = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            p3 = cv2.applyColorMap(prob, cv2.COLORMAP_INFERNO)
            cv2.drawContours(p3, cnt, -1, (0, 200, 255), 1)
            panels = [cv2.resize(p, (PANEL, PANEL), interpolation=cv2.INTER_AREA)
                      for p in (p1, p2, p3)]
            lab = f"{r['stem'][:13]} a{int(r['area'])} c{r['contrast']:.3f} pmax{prob.max()}"
            cv2.putText(panels[0], lab, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                        (0, 255, 0), 1, cv2.LINE_AA)
            rows.append(np.hstack(panels))
        sheet = np.vstack(rows)
        OUT.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(OUT / fname), sheet)
        log(f"  saved -> {OUT/fname}  (cols: stretched+GT | raw | prob-map+GT)")

    # summary stat: peak model probability under each never-proposed GT
    stats = []
    for _, r in rf[rf["iou_new"] <= 0.3].iterrows():
        g = decode_rle(r["segmentation_rle"]).astype(bool)
        stats.append(dict(tier=r["tier"], pmax=int(z[r["stem"]][g].max()),
                          pmean=float(z[r["stem"]][g].mean())))
    sf = pd.DataFrame(stats)
    log("\npeak model probability under the never-proposed GT (uint8 0-255):")
    log(sf.groupby("tier")["pmax"].describe()[["count", "50%", "75%", "max"]]
        .to_string(float_format=lambda v: f"{v:.0f}"))


if __name__ == "__main__":
    main()
