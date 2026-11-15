"""E195 -- does a candidate's NEIGHBOURHOOD GEOMETRY carry signal the selector does not have?

SFG-Net's CAFE (docs/v3_suggestion/solar_filament_sfgnet_adaptation.md §8-12, §41-42) says a
fragmented target should be judged with its neighbours, not alone. The document's own first step
is the cheap one: give the existing tabular selector relational features and measure; only fund a
graph network if they pay.

The pool already carries three neighbourhood columns -- nn_dist, n_within_200px, img_cand_density
-- and E172 gave every candidate its image's SCORE distribution (+0.0001, null). What has never
been computed is the GEOMETRY between candidates: does my neighbour continue my spine, does its
endpoint face mine, is it aligned with me. That is the question a human asks when deciding whether
two dark fragments are one filament, and it is exactly the population our error atlas names: the
166 unambiguous missed copies sit at 3.0 neighbours within 200 px against 1.0 for matched ones,
and 34 of them are merged into a neighbour while 17 are fragmented (ledger 29a).

Per candidate, from the pool's own masks (no ground truth, so it deploys):

  rel_nn1_orient      |axis angle difference| with the nearest candidate, degrees
  rel_nn1_collinear   |cos| between my axis and the direction to it (1 = it lies along my spine)
  rel_nn1_area_ratio  its area / mine
  rel_nn1_conf_diff   its conf minus mine
  rel_nn1_sem_diff    its sem_prob_mean minus mine
  rel_nn1_rdiff       |r_norm difference|
  rel_nn2_dist        distance to the second nearest centroid
  rel_end_gap         smallest skeleton endpoint-to-endpoint distance to any candidate
  rel_end_tangent     tangent compatibility at that closest endpoint pair, in [0,1]
  rel_cont_best       best continuation score over neighbours:
                      exp(-end_gap/60) * |cos(orient diff)| * tangent compatibility
  rel_n_collinear     neighbours within 200 px that are aligned (|cos| > 0.9, orient < 30 deg)
  rel_conf_rank_200   my conf rank among candidates within 200 px, scaled to [0,1]
  rel_iso             1 / (1 + neighbours within 200 px)         (isolation, the bagging arm's
                      strongest bin -- see the 2026-09-16 board reading)

Written for both pools, keyed by cand_id, so the selector harness can merge them:
  outputs/eda/relational_features_t020.parquet        (training pool, 10042 rows)
  outputs/eda/relational_features_test_t020.parquet   (test pool, 2892 rows)

Usage: python src/eda/relational_features.py [--test] [--jobs 8]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/models"):
    sys.path.insert(0, str(ROOT / p))
from filament.rle import decode_rle  # noqa: E402
from maskops import bbox, nonzero_boxed  # noqa: E402

OUT = ROOT / "outputs" / "eda"
REL = ["rel_nn1_orient", "rel_nn1_collinear", "rel_nn1_area_ratio", "rel_nn1_conf_diff",
       "rel_nn1_sem_diff", "rel_nn1_rdiff", "rel_nn2_dist", "rel_end_gap", "rel_end_tangent",
       "rel_cont_best", "rel_n_collinear", "rel_conf_rank_200", "rel_iso"]


def axis_and_endpoints(mask):
    """Principal axis angle (degrees, mod 180) and the skeleton's endpoints with their inward
    tangents. Everything on the bounding-box crop; coordinates returned in frame space."""
    from skimage.morphology import skeletonize
    box = bbox(mask)
    if box is None:
        return 0.0, np.zeros((0, 2)), np.zeros((0, 2))
    y0, y1, x0, x1 = box
    crop = mask[y0:y1, x0:x1]
    ys, xs = nonzero_boxed(mask, box)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    pts -= pts.mean(0)
    if len(pts) >= 3:
        v = np.linalg.eigh(pts.T @ pts / len(pts))[1][:, -1]
        axis = float(np.degrees(np.arctan2(v[1], v[0])) % 180.0)
    else:
        axis = 0.0
    sk = skeletonize(crop)
    sy, sx = np.nonzero(sk)
    if len(sy) == 0:
        return axis, np.zeros((0, 2)), np.zeros((0, 2))
    pad = np.pad(sk, 1)
    deg = sum(pad[1 + dy:1 + dy + sk.shape[0], 1 + dx:1 + dx + sk.shape[1]]
              for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0))
    ends = np.argwhere(sk & (deg == 1))
    if len(ends) == 0:                        # a loop: take the two farthest skeleton pixels
        sp = np.stack([sy, sx], 1)
        d = ((sp[:, None, :] - sp[None, :, :]) ** 2).sum(-1)
        i, j = np.unravel_index(d.argmax(), d.shape)
        ends = sp[[i, j]]
    sp = np.stack([sy, sx], 1).astype(np.float64)
    tang = []
    for e in ends:
        d = np.hypot(sp[:, 0] - e[0], sp[:, 1] - e[1])
        far = sp[d >= min(8.0, d.max())]      # a point ~8 px inward along the skeleton
        base = far[np.argmin(np.hypot(far[:, 0] - e[0], far[:, 1] - e[1]))] if len(far) else sp.mean(0)
        v = np.array([e[1] - base[1], e[0] - base[0]], float)   # (dx, dy), pointing outward
        n = np.hypot(*v)
        tang.append(v / n if n > 1e-6 else np.array([1.0, 0.0]))
    endpoints = np.stack([ends[:, 1] + x0, ends[:, 0] + y0], 1).astype(np.float64)  # (x, y)
    return axis, endpoints, np.stack(tang)


def angle_diff(a, b):
    d = abs(a - b) % 180.0
    return float(min(d, 180.0 - d))


def image_rows(g, rle_col):
    """Relational features for one image's candidates."""
    n = len(g)
    cx = g["cx"].to_numpy(float); cy = g["cy"].to_numpy(float)
    conf = g["conf"].to_numpy(float); sem = g["sem_prob_mean"].to_numpy(float)
    area = g["area"].to_numpy(float); rn = g["r_norm"].to_numpy(float)
    axes, ends, tangs = [], [], []
    for r in g[rle_col]:
        a, e, t = axis_and_endpoints(decode_rle(r).astype(bool))
        axes.append(a); ends.append(e); tangs.append(t)
    D = np.hypot(cx[:, None] - cx[None, :], cy[:, None] - cy[None, :])
    np.fill_diagonal(D, np.inf)
    rows = []
    for i in range(n):
        near200 = np.flatnonzero(D[i] < 200)
        d = dict(cand_id=g["cand_id"].iloc[i],
                 rel_iso=1.0 / (1.0 + len(near200)),
                 rel_conf_rank_200=(float((conf[near200] < conf[i]).mean()) if len(near200) else 1.0),
                 rel_nn2_dist=float(np.sort(D[i])[1]) if n > 2 else 2048.0)
        if n == 1:
            d |= dict(rel_nn1_orient=90.0, rel_nn1_collinear=0.0, rel_nn1_area_ratio=1.0,
                      rel_nn1_conf_diff=0.0, rel_nn1_sem_diff=0.0, rel_nn1_rdiff=0.0,
                      rel_end_gap=2048.0, rel_end_tangent=0.0, rel_cont_best=0.0,
                      rel_n_collinear=0)
            rows.append(d)
            continue
        j = int(np.argmin(D[i]))
        v = np.array([cx[j] - cx[i], cy[j] - cy[i]])
        v = v / max(np.hypot(*v), 1e-6)
        ax = np.array([np.cos(np.radians(axes[i])), np.sin(np.radians(axes[i]))])
        d |= dict(rel_nn1_orient=angle_diff(axes[i], axes[j]),
                  rel_nn1_collinear=float(abs(v @ ax)),
                  rel_nn1_area_ratio=float(area[j] / max(area[i], 1.0)),
                  rel_nn1_conf_diff=float(conf[j] - conf[i]),
                  rel_nn1_sem_diff=float(sem[j] - sem[i]),
                  rel_nn1_rdiff=float(abs(rn[j] - rn[i])))
        # endpoint geometry against every other candidate in the image
        best_gap, best_tan, best_cont = 2048.0, 0.0, 0.0
        for k in range(n):
            if k == i or len(ends[i]) == 0 or len(ends[k]) == 0:
                continue
            dd = np.hypot(ends[i][:, None, 0] - ends[k][None, :, 0],
                          ends[i][:, None, 1] - ends[k][None, :, 1])
            a_i, b_k = np.unravel_index(dd.argmin(), dd.shape)
            gap = float(dd[a_i, b_k])
            # my outward tangent against the direction to its endpoint, and its tangent back
            w = ends[k][b_k] - ends[i][a_i]
            w = w / max(np.hypot(*w), 1e-6)
            tan = 0.5 * (float(tangs[i][a_i] @ w) + float(-tangs[k][b_k] @ w))
            tan = max(0.0, tan)
            cont = float(np.exp(-gap / 60.0) * abs(np.cos(np.radians(angle_diff(axes[i], axes[k])))) * tan)
            if gap < best_gap:
                best_gap, best_tan = gap, tan
            best_cont = max(best_cont, cont)
        coll = 0
        for k in near200:
            vk = np.array([cx[k] - cx[i], cy[k] - cy[i]])
            vk = vk / max(np.hypot(*vk), 1e-6)
            if abs(vk @ ax) > 0.9 and angle_diff(axes[i], axes[k]) < 30.0:
                coll += 1
        d |= dict(rel_end_gap=best_gap, rel_end_tangent=best_tan, rel_cont_best=best_cont,
                  rel_n_collinear=coll)
        rows.append(d)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="the test pool instead of the training pool")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    if a.test:
        pool = pd.read_parquet(OUT / "test_pool_t020pox.parquet")
        rle_col, out = "rle_shrunk", OUT / "relational_features_test_t020.parquet"
    else:
        pool = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
        rle_col, out = "rle", OUT / "relational_features_t020.parquet"
    groups = [g for _, g in pool.groupby("stem", sort=True)]
    t0 = time.time()
    if a.jobs > 1:
        from joblib import Parallel, delayed
        parts = Parallel(n_jobs=a.jobs, backend="loky")(
            delayed(image_rows)(g, rle_col) for g in groups)
    else:
        parts = [image_rows(g, rle_col) for g in groups]
    df = pd.DataFrame([r for part in parts for r in part])
    assert len(df) == len(pool) and df["cand_id"].is_unique
    df.to_parquet(out, index=False)
    print(f"{len(df)} candidates over {len(groups)} images ({time.time()-t0:.0f}s) -> {out.name}")
    print(df[REL].describe().round(3).to_string())


if __name__ == "__main__":
    main()
