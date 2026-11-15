"""E133 -- is the limb failure GEOMETRIC (foreshortening) or PHOTOMETRIC (signal)?

Talha's proposal: put a small physics-specific model beyond r 0.8. Before
spending GPU on one, the question is which physics is actually failing there,
because the two candidates imply different fixes:

  geometric   near the limb the line of sight compresses the RADIAL direction by
              mu = sqrt(1 - r^2) (0.60 at r 0.8, 0.44 at 0.9, 0.31 at 0.95) while
              leaving the tangential direction untouched. A filament lying along
              the radius therefore looks shorter and thinner than an identical
              one on the disc; a filament lying tangentially does not. The fix is
              a deprojection (stretch by 1/mu radially, run the model, map back),
              which is the one limb idea never measured here.

  photometric the structure is simply fainter or the background differs, in which
              case deprojection changes nothing and only signal strength helps.

The two hypotheses make opposite predictions, and the prediction is testable
without training anything: if the loss is geometric, missed ground-truth rows at
the limb should be concentrated among RADIALLY oriented instances; if it is
photometric, the miss rate should not depend on orientation once contrast is
controlled.

For every ground-truth row this computes the angle between its principal axis
and the local radial direction, its mu, its contrast against a local ring, and
whether the deployed pipeline reached it (IoU > 0.5). It then reports the miss
rate by (radial band x orientation) and, inside the limb bands, by contrast
quartile.

Usage: python src/eda/limb_geometry_test.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from filament.rle import decode_rle  # noqa: E402

OUT = ROOT / "outputs" / "eda"
IMG = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
CX = CY = 1023.5
R = 900.0
RING = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))


def log(*a):
    print(*a, flush=True)


def axis_angle(m):
    ys, xs = np.nonzero(m)
    if len(xs) < 5:
        return None, None, None
    cy, cx = ys.mean(), xs.mean()
    c = np.cov(np.vstack([xs, ys]).astype(np.float64))
    v = np.linalg.eigh(c)[1]
    ax = np.degrees(np.arctan2(v[1, -1], v[0, -1])) % 180.0
    rad = np.degrees(np.arctan2(cy - 1023.5, cx - 1023.5)) % 180.0
    d = abs(ax - rad) % 180.0
    return min(d, 180.0 - d), cy, cx


def main():
    prof = pd.read_parquet(OUT / "error_profile_fn_gt.parquet")
    prof["matched"] = prof["matched"].astype(bool)
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet").set_index("filament_id")
    rows = []
    stems = sorted(prof["stem"].unique())
    for n, stem in enumerate(stems):
        g = prof[prof["stem"] == stem]
        img = cv2.imread(str(IMG / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        for r in g.itertuples():
            m = decode_rle(gt.loc[r.filament_id, "segmentation_rle"]).astype(bool)
            if not m.any():
                continue
            ang, cy, cx = axis_angle(m)
            if ang is None:
                continue
            rr = float(np.sqrt((cx - 1023.5) ** 2 + (cy - 1023.5) ** 2) / 900.0)
            ring = cv2.dilate(m.astype(np.uint8), RING).astype(bool) & ~m
            bg = float(np.median(img[ring])) if ring.any() else float(img[m].mean())
            rows.append(dict(stem=stem, filament_id=r.filament_id, matched=bool(r.matched),
                             r_norm=rr, mu=float(np.sqrt(max(1 - min(rr, 1.0) ** 2, 0))),
                             axis_to_radial=ang, area=int(m.sum()),
                             contrast=(bg - float(img[m].mean())) / max(bg, 1.0)))
        if n % 100 == 0:
            log(f"  {n}/{len(stems)} images")
    d = pd.DataFrame(rows)
    d.to_parquet(OUT / "limb_geometry.parquet", index=False)
    d["band"] = pd.cut(d["r_norm"], [0, 0.5, 0.8, 0.9, 1.01], labels=["<0.5", "0.5-0.8", "0.8-0.9", ">0.9"])
    d["orient"] = pd.cut(d["axis_to_radial"], [-0.1, 30, 60, 90.1], labels=["radial", "oblique", "tangential"])
    log(f"\n{len(d)} ground-truth rows with a defined axis\n")
    log("MISS RATE by radial band x orientation (miss = not reached at IoU > 0.5)")
    t = d.pivot_table(index="band", columns="orient", values="matched", aggfunc=lambda s: 1 - s.mean(), observed=True)
    n = d.pivot_table(index="band", columns="orient", values="matched", aggfunc="size", observed=True)
    log(t.round(3).to_string()); log("\ncounts:"); log(n.to_string())
    log("\nif the loss were geometric, the 'radial' column would be worst at the limb")
    limb = d[d["r_norm"] > 0.8].copy()
    limb["cq"] = pd.qcut(limb["contrast"], 4, labels=["faintest", "q2", "q3", "brightest"])
    log("\nLIMB rows only (r > 0.8): miss rate by contrast quartile x orientation")
    t2 = limb.pivot_table(index="cq", columns="orient", values="matched", aggfunc=lambda s: 1 - s.mean(), observed=True)
    log(t2.round(3).to_string())
    log("\nmedian contrast: disc rows {:.3f}, limb rows {:.3f}".format(
        d.loc[d["r_norm"] <= 0.8, "contrast"].median(), limb["contrast"].median()))
    log("median area: disc {:.0f}, limb {:.0f}".format(
        d.loc[d["r_norm"] <= 0.8, "area"].median(), limb["area"].median()))
    json.dump(dict(miss_by_band_orient=t.round(4).to_dict(), counts=n.to_dict(),
                   limb_miss_by_contrast_orient=t2.round(4).to_dict()),
              open(OUT / "limb_geometry.json", "w"), indent=1, default=str)
    log(f"\nwrote {OUT / 'limb_geometry.json'}")


if __name__ == "__main__":
    main()
