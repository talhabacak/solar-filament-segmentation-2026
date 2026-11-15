"""E223 -- can any evidence we have tell where a filament CONTINUES beyond our mask's end?

Ledger §39a/§39b: near-miss masks have the right width and ~70 % of the length; on the missing
length the ensemble's field is ~0 (median 0.153), an isotropic flood fill bloats masks by a median
0.75, and "axial hysteresis" (grow along the filament's tangent only) was named as the last CPU arm
and never run. E216b (2026-10-04): the only part of a perfect fill that transfers across annotators
is beyond the box, along the filament (25 %), ceiling about +0.014.

E224 (same day, on the annotators' spines): a straight continuation stays inside the filament for
92 % / 71 % / 44 % / 22 % of points at 10 / 20 / 40 / 80 px, and extrapolating the past curvature is
no better (the waviness is not persistent). So the path is followed in two modes:
  straight  the endpoint tangent
  track     every 2 px, turn by -10..+10 deg (5 deg steps) towards the darkest ridge 6 px ahead,
            with a small turning penalty: the filament's waviness is followed from the image, not
            extrapolated from its past shape

This is the precondition for an axial extender, on fold 0 (the only fold with cached detector soft
masks, E217a). For every kept out-of-fold mask (sub036 keep set) that is the best match of a GT
copy, skeleton endpoints and tangents are found and the path is followed for up to 200 px. At each
2-px step s along the path:

  label     the path point lies inside the GT copy (does the filament continue here?)
  evidence  sem   semantic out-of-fold field (max in a 2-px disc), 15-px look-ahead mean
            det   detector soft field (E217a cache), 15-px look-ahead mean
            line  image darkness across the path: mean of the flanks at +/-(h + 4) px minus the
                  centre, divided by the frame's disc sd, 15-px look-ahead mean -- a matched line
                  filter that integrates evidence along the filament direction

Reported per mode: AUC of each evidence for the label, pooled and by distance band, for copies
drawn independently by two annotators (n_copies >= 2) and for the rest. A stop rule (extend while
the look-ahead evidence stays above tau) is simulated with a tube of the mask's own half-width
along the path: IoU change against the annotator who drew the copy, against a sibling annotator
(transfer), and on matched copies (harm).

PRE-REGISTERED GATE: proceed to the five-fold, selector-fixed protocol (E215 harness) only if some
evidence reaches AUC >= 0.75 on agreed copies IN THE 20-60 PX BAND (tightened before the full run,
after a four-image smoke test showed the pooled AUC is inflated by distance: points next to the
end are both inside the filament and high in every evidence) AND the best stop rule crosses more agreed copies
above IoU 0.5 against the sibling annotator than it pushes matched copies below 0.5.

Usage: python src/eda/e223_axial_continuation.py [N]   (N = smoke test on the first N images)
Output: outputs/eda/e223_axial_continuation.json
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/eda"):
    sys.path.insert(0, str(ROOT / p))
from filament import IMG_H, IMG_W  # noqa: E402
from filament.rle import decode_rle  # noqa: E402

OUT = ROOT / "outputs" / "eda"
CACHE = ROOT / "outputs" / "oof_cache"
IMG = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
STEP, LMAX, LOOK = 2, 200, 15
MODES = ("straight", "track")
TAUS = {"sem": [0.05, 0.1, 0.2], "det": [0.1, 0.2, 0.3], "line": [0.3, 0.6, 1.0]}
TURNS = np.deg2rad([-10, -5, 0, 5, 10])


def endpoints_and_tangents(m):
    from skimage.morphology import skeletonize
    ys, xs = np.nonzero(m)
    y0, x0 = ys.min(), xs.min()
    sk = skeletonize(m[y0:ys.max() + 1, x0:xs.max() + 1])
    pts = set(zip(*np.nonzero(sk)))
    out = []
    for (y, x) in pts:
        nb = [(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy or dx) and (y + dy, x + dx) in pts]
        if len(nb) != 1:
            continue
        cur, path = nb[0], [(y, x)]
        for _ in range(12):
            path.append(cur)
            nxt = [(cur[0] + dy, cur[1] + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                   if (dy or dx) and (cur[0] + dy, cur[1] + dx) in pts and (cur[0] + dy, cur[1] + dx) not in path]
            if not nxt:
                break
            cur = nxt[0]
        if len(path) < 6:
            continue
        v = np.array(path[0], float) - np.array(path[-1], float)
        n = np.hypot(*v)
        if n == 0:
            continue
        out.append(((y + y0, x + x0), v / n))
    return out


def inside(py, px):
    return 0 <= py < IMG_H and 0 <= px < IMG_W


def pix(img, py, px):
    return float(img[int(round(py)), int(round(px))]) if inside(py, px) else np.nan


def contrast(img, py, px, d, off):
    """flanks minus centre across direction d (dark ridge -> positive)."""
    nrm = np.array([-d[1], d[0]])
    fl = [pix(img, py + nrm[0] * k * off, px + nrm[1] * k * off) for k in (-1, 1)]
    c = pix(img, py, px)
    return np.nanmean(fl) - c if not np.isnan(c) else np.nan


def trace(img, ey, ex, v, h, mode):
    """Path points (s, y, x, dy, dx) from the endpoint for up to LMAX px."""
    out, py, px, d = [], float(ey), float(ex), v.copy()
    for s in range(STEP, LMAX + 1, STEP):
        if mode == "track":
            best, bd = -np.inf, d
            for a in TURNS:
                ca, sa = np.cos(a), np.sin(a)
                dd = np.array([d[0] * ca - d[1] * sa, d[0] * sa + d[1] * ca])
                e = contrast(img, py + dd[0] * 6, px + dd[1] * 6, dd, h + 4)
                if np.isnan(e):
                    continue
                e -= 0.15 * abs(np.rad2deg(a)) / 10.0      # turning penalty (grey levels)
                if e > best:
                    best, bd = e, dd
            d = bd / max(np.hypot(*bd), 1e-9)
        py, px = py + d[0] * STEP, px + d[1] * STEP
        if not inside(py, px):
            break
        out.append((s, py, px, d[0], d[1]))
    return out


def sample(arr, y, x, r=2):
    y, x = int(round(y)), int(round(x))
    if not inside(y, x):
        return np.nan
    return float(arr[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].max())


def auc(score, label):
    from sklearn.metrics import roc_auc_score
    ok = ~np.isnan(score)
    if ok.sum() < 50 or len(np.unique(label[ok])) < 2:
        return None
    return float(roc_auc_score(label[ok], score[ok]))


def lookahead(x, k):
    return pd.Series(x)[::-1].rolling(k, min_periods=1).mean()[::-1].to_numpy()


def main():
    folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
    stems = sorted(folds.loc[folds["fold"] == 0, "file_name"].str.replace(".jpeg", "", regex=False))
    smoke = len(sys.argv) > 1
    if smoke:
        stems = stems[:int(sys.argv[1])]
    pool = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet", columns=["stem", "cand_id", "rle", "conf"])
    pool = pool.merge(pd.read_parquet(OUT / "bag5_oof_scores.parquet"), on="cand_id")
    keep = pool[(pool["conf"] >= 0.40) & (pool["p"] >= 0.60) & pool["stem"].isin(stems)]
    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt = gt[gt["stem"].isin(stems)]
    atlas = pd.read_parquet(OUT / "error_atlas.parquet").set_index("fid")
    sem_z = np.load(ROOT / "outputs" / "semantic" / "fold0_probs_blob5ens.npz")
    q = pd.read_parquet(OUT / "image_quality.parquet").set_index("stem")
    k = max(1, LOOK // STEP)

    pts_rows, grow_rows = [], []
    for n, stem in enumerate(stems):
        kc = keep[keep["stem"] == stem]
        gg = gt[gt["stem"] == stem]
        if kc.empty or gg.empty:
            continue
        img = cv2.GaussianBlur(cv2.imread(str(IMG / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE).astype(np.float32), (0, 0), 1.0)
        dsd = float(q.loc[stem, "disk_sd"]) if stem in q.index else float(img.std())
        sem = sem_z[stem].astype(np.float32) / 255.0
        z = np.load(CACHE / "soft_e013_f0" / f"{stem}.npz")
        det = np.zeros((IMG_H, IMG_W), np.float32)
        for i, (y0, y1, x0, x1) in enumerate(z["box"]):
            det[y0:y1, x0:x1] = np.maximum(det[y0:y1, x0:x1], z[f"m{i}"].astype(np.float32) / 255.0)
        km = [decode_rle(r).astype(bool) for r in kc["rle"]]
        owned = np.zeros((IMG_H, IMG_W), bool)
        for m in km:
            owned |= m
        gm = {f: decode_rle(r).astype(bool) for f, r in zip(gg["filament_id"], gg["segmentation_rle"])}
        ent = dict(zip(gg["filament_id"], gg["entry_id"]))
        for f, G in gm.items():
            if f not in atlas.index:
                continue
            ious = [(m & G).sum() / max((m | G).sum(), 1) for m in km]
            j = int(np.argmax(ious))
            if ious[j] < 0.2:
                continue
            c = km[j]
            outcome = atlas.loc[f, "outcome"]
            agreed = bool(atlas.loc[f, "n_copies"] >= 2)
            sib = [s for s in gm if ent[s] != ent[f] and (gm[s] & G).sum() / max((gm[s] | G).sum(), 1) > 0.5]
            dt = cv2.distanceTransform(c.astype(np.uint8), cv2.DIST_L2, 5)
            h = max(1.5, float(np.median(dt[dt > 0])) * 1.5) if (dt > 0).any() else 2.0
            ends = endpoints_and_tangents(c)
            for mode in MODES:
                paths = []
                for (ey, ex), v in ends:
                    tr = trace(img, ey, ex, v, h, mode)
                    if not tr:
                        continue
                    s_, ys_, xs_, dy_, dx_ = map(np.array, zip(*tr))
                    line = np.array([contrast(img, y, x, np.array([a, b]), h + 4) for y, x, a, b in zip(ys_, xs_, dy_, dx_)]) / dsd
                    P = pd.DataFrame(dict(s=s_, y=ys_, x=xs_,
                                          inG=[bool(G[int(round(y)), int(round(x))]) for y, x in zip(ys_, xs_)],
                                          sem=lookahead([sample(sem, y, x) for y, x in zip(ys_, xs_)], k),
                                          det=lookahead([sample(det, y, x) for y, x in zip(ys_, xs_)], k),
                                          line=lookahead(line, k)))
                    for r in P.itertuples():
                        pts_rows.append(dict(mode=mode, agreed=agreed, outcome=outcome, s=r.s, inG=r.inG,
                                             sem=r.sem, det=r.det, line=r.line))
                    paths.append(((ey, ex), P))
                for ev, taus in TAUS.items():
                    for tau in taus:
                        grown = c.copy()
                        for (ey, ex), P in paths:
                            ok = P[ev].to_numpy() >= tau
                            stop = np.flatnonzero(~ok)
                            L = stop[0] if len(stop) else len(P)
                            if L == 0:
                                continue
                            pts = np.r_[[[ex, ey]], P[["x", "y"]].to_numpy()[:L]].round().astype(np.int32)
                            tube = np.zeros((IMG_H, IMG_W), np.uint8)
                            cv2.polylines(tube, [pts], False, 1, thickness=max(1, int(round(2 * h))))
                            grown |= tube.astype(bool) & ~owned
                        rec = dict(fid=f, mode=mode, agreed=agreed, outcome=outcome, ev=ev, tau=tau,
                                   iou_before=float(ious[j]),
                                   iou_after=float((grown & G).sum() / max((grown | G).sum(), 1)),
                                   added=int((grown & ~c).sum()))
                        if sib:
                            B = gm[sib[0]]
                            rec["sib_before"] = float((c & B).sum() / max((c | B).sum(), 1))
                            rec["sib_after"] = float((grown & B).sum() / max((grown | B).sum(), 1))
                        grow_rows.append(rec)
        if n % 20 == 0:
            print(f"  {n + 1}/{len(stems)}", flush=True)

    T = pd.DataFrame(pts_rows)
    Gw = pd.DataFrame(grow_rows)
    res = {"auc": {}, "growth": []}
    for mode in MODES:
        for name, sub in (("agreed", T[(T["mode"] == mode) & T.agreed]), ("other", T[(T["mode"] == mode) & ~T.agreed])):
            key = f"{mode}/{name}"
            res["auc"][key] = {ev: auc(sub[ev].to_numpy(float), sub["inG"].to_numpy(int)) for ev in ("sem", "det", "line")}
            res["auc"][key]["n_points"] = len(sub)
            res["auc"][key]["continue_rate"] = float(sub["inG"].mean()) if len(sub) else None
            for lo, hi in ((0, 20), (20, 60), (60, 200)):
                b = sub[(sub.s > lo) & (sub.s <= hi)]
                res["auc"][key][f"s{lo}-{hi}"] = {ev: auc(b[ev].to_numpy(float), b["inG"].to_numpy(int)) for ev in ("sem", "det", "line")}
                res["auc"][key][f"s{lo}-{hi}"]["continue_rate"] = float(b["inG"].mean()) if len(b) else None
    for (mode, ev, tau), g in Gw.groupby(["mode", "ev", "tau"]):
        ag = g[(g.outcome == "shape") & g.agreed]
        mt = g[g.outcome == "matched"]
        sb = ag.dropna(subset=["sib_before"])
        res["growth"].append(dict(
            mode=mode, ev=ev, tau=tau,
            agreed_shape_n=len(ag), agreed_cross_own=int(((ag.iou_before <= 0.5) & (ag.iou_after > 0.5)).sum()),
            agreed_cross_sibling=int(((sb.sib_before <= 0.5) & (sb.sib_after > 0.5)).sum()),
            matched_n=len(mt), matched_lost=int(((mt.iou_before > 0.5) & (mt.iou_after <= 0.5)).sum()),
            matched_mean_delta=float((mt.iou_after - mt.iou_before).mean()),
            added_px_median=float(g.added.median())))
    best = max(res["growth"], key=lambda r: r["agreed_cross_sibling"] - r["matched_lost"])
    aucs = [res["auc"][f"{m}/agreed"]["s20-60"][e] for m in MODES for e in ("sem", "det", "line")]
    best_auc = max(a for a in aucs if a is not None)
    res["gate"] = dict(best_auc_agreed=best_auc, best_rule=best,
                       passed=bool(best_auc >= 0.75 and best["agreed_cross_sibling"] > best["matched_lost"]))
    print(json.dumps(res, indent=1))
    if not smoke:
        json.dump(res, open(OUT / "e223_axial_continuation.json", "w"), indent=1)


if __name__ == "__main__":
    main()
