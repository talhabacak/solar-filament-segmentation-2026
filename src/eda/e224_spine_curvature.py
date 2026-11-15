"""E224 -- how far can a filament's continuation be predicted from its own shape?

Question (Talha, 2026-10-04): the filaments are wavy, curved and turning; can that structure guide
an axial extender (E223) through the faint part? E223's first design follows a STRAIGHT ray from the
mask's end; if filaments curve within tens of pixels, a straight ray leaves the filament and every
evidence along it reads "stopped", which would reject the idea for the wrong reason.

Measured on the manually drawn spines (organiser-sanctioned metadata, discussion 736543). Each spine
is resampled at 1 px of arc length. From every point with 24 px of spine behind it, the
continuation at distances d = 10, 20, 40, 80 px is predicted by
  straight   the tangent of the last 12 px
  arc        a constant-curvature arc through the last 24 px (turning rate from the last two
             12-px chords)
and the error is the distance from the predicted point to the true spine ahead. An error below the
filament's half-width keeps the prediction inside the filament.

Also reported: sinuosity (arc length / chord) and the distribution of curvature radius.

Usage: python src/eda/e224_spine_curvature.py
Output: outputs/eda/e224_spine_curvature.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
OUT = ROOT / "outputs" / "eda"
DS = [10, 20, 40, 80]


def resample(p, step=1.0):
    seg = np.hypot(*np.diff(p, axis=0).T)
    keep = np.r_[True, seg > 1e-6]
    p = p[keep]
    if len(p) < 2:
        return None
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    t = np.arange(0, s[-1], step)
    return np.c_[np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])]


def unit(v):
    n = np.hypot(*v)
    return v / n if n > 0 else v


def main():
    d = json.load(open(ANN, encoding="utf-8"))
    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet", columns=["filament_id", "ann_id"])
    fid_of = dict(zip(gt["ann_id"], gt["filament_id"]))
    atlas = pd.read_parquet(OUT / "error_atlas.parquet").set_index("fid")
    rows, shape_rows = [], []
    for a in d["annotations"]:
        p = np.asarray(a.get("spine", []), float).reshape(-1, 2)
        if len(p) < 2:
            continue
        q = resample(p)
        if q is None or len(q) < 40:
            continue
        fid = fid_of.get(a["id"])
        hw = float(atlas.loc[fid, "width"]) / 2 if fid in atlas.index else np.nan
        outcome = atlas.loc[fid, "outcome"] if fid in atlas.index else None
        L = len(q)
        chord = np.hypot(*(q[-1] - q[0]))
        # signed turning per unit length from consecutive 12-px chords
        shape_rows.append(dict(length=L, sinuosity=L / max(chord, 1), half_width=hw, outcome=outcome))
        for i in range(24, L - 10, 6):
            t1 = unit(q[i - 12] - q[i - 24])
            t2 = unit(q[i] - q[i - 12])
            dtheta = np.arctan2(t1[0] * t2[1] - t1[1] * t2[0], t1 @ t2)   # turning over 12 px
            kappa = dtheta / 12.0
            for dd in DS:
                if i + dd >= L:
                    continue
                truth = q[i:min(L, i + 2 * dd)]
                ps = q[i] + t2 * dd
                # constant-curvature arc: integrate the heading
                ang = np.arctan2(t2[1], t2[0])
                steps = np.arange(1, dd + 1)
                heading = ang + kappa * steps
                pa = q[i] + np.c_[np.cos(heading), np.sin(heading)].sum(0)
                es = np.min(np.hypot(*(truth - ps).T))
                ea = np.min(np.hypot(*(truth - pa).T))
                rows.append(dict(d=dd, err_straight=es, err_arc=ea, half_width=hw, outcome=outcome,
                                 radius=abs(1 / kappa) if kappa != 0 else np.inf))
    R = pd.DataFrame(rows)
    S = pd.DataFrame(shape_rows)
    res = {"spines": len(S), "sinuosity_median": float(S.sinuosity.median()),
           "sinuosity_p90": float(S.sinuosity.quantile(0.9)),
           "length_median_px": float(S.length.median()),
           "curvature_radius_median_px": float(R.radius.replace(np.inf, np.nan).median()),
           "by_distance": []}
    for dd, g in R.groupby("d"):
        ok = g.dropna(subset=["half_width"])
        res["by_distance"].append(dict(
            d=int(dd), n=len(g),
            straight_median_err=float(g.err_straight.median()), arc_median_err=float(g.err_arc.median()),
            straight_inside=float((ok.err_straight <= ok.half_width).mean()),
            arc_inside=float((ok.err_arc <= ok.half_width).mean()),
            arc_better=float((g.err_arc < g.err_straight).mean())))
    print(json.dumps(res, indent=1))
    json.dump(res, open(OUT / "e224_spine_curvature.json", "w"), indent=1)


if __name__ == "__main__":
    main()
