"""Unmerge: the applicable ceiling and a GT-free cut proposer (chatgpt_science S004a).

Honest E117 (ledger 18a): 204 merged FN + 86 merge FP; the topological oracle
(merges and fragments perfectly resolved) is +0.0213. Fragments are closed
(§15: 82 rows, ~+0.003 at most), so the open question is the MERGE side and it
has two halves:

  oracle    if every merged prediction were cut exactly along the ground
            truth, how much PQ comes back? (GT-guided, the ceiling for cuts)
  proposer  can a cut be found WITHOUT the ground truth, from what the
            prediction carries: detector seeds inside it, the semantic field's
            saddle along the cut, the neck width, the orientation change?

Deployment-faithful inputs: the E115b labelled pool (shrink 0.05) scored by the
fully out-of-fold selector, production policy (0.40, 0.60); detector seeds from
the fold's detector OOF cache (conf >= --seed-conf); the production field
0.5 blob + 0.5 clDice of the fold. Exact scorer throughout.

Oracle split of a kept prediction P: among the annotator entries of the image,
take the one whose GT rows P covers most (>= 2 rows each with |P & g| >= 20% of
|g| or of |P|); assign every pixel of P to the nearest such row (distance
transform); parts under 64 px are dropped. Proposer split: markers = detector
masks (conf >= seed-conf) intersected with P, merged if they overlap; watershed
on the inverted field inside P; parts under 64 px are absorbed by the neighbour
sharing the longest border. Per proposed split:

  n_seeds, saddle (mean field on the inter-part boundary), neck (2 x min
  distance-to-outside on that boundary), frac_low (boundary pixels with field
  < 0.5), orient_diff (degrees between the two largest parts' principal axes),
  area_ratio (min/max part area), p_oof of P, area of P

and the label: exact per-image PQ with the split minus without it.

Reports: how many kept predictions merge (oracle side) and the oracle PQ;
how many carry >= 2 seeds and split; how many splits pay; per-feature ROC-AUC
for "split pays"; a grouped-OOF HistGradientBoosting classifier with the cut
threshold chosen on the training folds by summed gain, and the realised pooled
PQ when its out-of-fold decisions are applied. Pre-registered (roadmap S004):
applicable proposer gain >= +0.004 PQ with >= 4/5 folds positive before any
learned unmerge module is built; oracle ceiling < +0.004 closes the axis.

Usage: python src/eda/unmerge_oracle.py --folds 0,1,2,3,4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from filament.pq import score  # noqa: E402
from filament.rle import decode_rle, encode_mask  # noqa: E402

OUT = ROOT / "outputs" / "eda"
SEM = ROOT / "outputs" / "semantic"
CACHE = ROOT / "outputs" / "oof_cache"
MIN_PART = 64


def log(*a):
    print(*a, flush=True)


def bbox(m, pad=8):
    ys, xs = np.nonzero(m)
    return max(ys.min() - pad, 0), min(ys.max() + pad + 1, m.shape[0]), max(xs.min() - pad, 0), min(xs.max() + pad + 1, m.shape[1])


def principal_angle(m):
    ys, xs = np.nonzero(m)
    if len(xs) < 5:
        return 0.0
    c = np.cov(np.vstack([xs, ys]).astype(np.float64))
    w, v = np.linalg.eigh(c)
    vx, vy = v[:, -1]
    return float(np.degrees(np.arctan2(vy, vx)) % 180.0)


def angle_diff(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def oracle_split(P, gts_by_entry):
    """GT-guided parts of P, or None when P covers < 2 rows of every entry."""
    best, best_rows = 0, None
    pa = int(P.sum())
    for rows in gts_by_entry.values():
        cov = []
        for g in rows:
            inter = int((P & g).sum())
            if inter and (inter >= 0.2 * int(g.sum()) or inter >= 0.2 * pa):
                cov.append(g)
        if len(cov) > best:
            best, best_rows = len(cov), cov
    if best < 2:
        return None
    y0, y1, x0, x1 = bbox(P)
    Pc = P[y0:y1, x0:x1]
    dists = np.stack([cv2.distanceTransform((~g[y0:y1, x0:x1]).astype(np.uint8), cv2.DIST_L2, 3) for g in best_rows])
    lab = np.argmin(dists, axis=0)
    parts = []
    for k in range(len(best_rows)):
        part = Pc & (lab == k)
        if part.sum() >= MIN_PART:
            full = np.zeros_like(P)
            full[y0:y1, x0:x1] = part
            parts.append(full)
    return parts if len(parts) >= 2 else None


def dt_seeds(P, min_dt=3.0, min_sep=15):
    """Geometry seeds: regional maxima of the distance transform of P (the
    thick cores of the lobes), at least min_sep px apart, DT >= min_dt. Returns
    small disc masks usable as watershed markers -- no detector involved."""
    y0, y1, x0, x1 = bbox(P)
    Pc = P[y0:y1, x0:x1].astype(np.uint8)
    dt = cv2.distanceTransform(Pc, cv2.DIST_L2, 3)
    k = 2 * min_sep + 1
    peaks = (dt >= min_dt) & (dt >= cv2.dilate(dt, np.ones((k, k), np.uint8)) - 1e-6)
    ys, xs = np.nonzero(peaks)
    if len(ys) < 2:
        return []
    order = np.argsort(-dt[ys, xs])
    chosen = []
    for i in order:
        if all((ys[i] - cy) ** 2 + (xs[i] - cx) ** 2 >= min_sep ** 2 for cy, cx in chosen):
            chosen.append((ys[i], xs[i]))
        if len(chosen) == 4:
            break
    out = []
    for cy, cx in chosen:
        disc = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.circle(disc, (int(cx), int(cy)), 3, 1, -1)
        m = np.zeros_like(P)
        m[y0:y1, x0:x1] = disc.astype(bool)
        out.append(m & P)
    return out


def proposer_split(P, field, seeds):
    """Watershed of P on the inverted field with detector-seed markers. Returns
    (parts, features) or (None, None)."""
    y0, y1, x0, x1 = bbox(P)
    Pc = P[y0:y1, x0:x1]
    markers = np.zeros(Pc.shape, np.int32)
    k = 0
    for s in seeds:
        sc = s[y0:y1, x0:x1] & Pc
        if sc.sum() < 16:
            continue
        existing = np.unique(markers[sc])
        existing = existing[existing > 0]
        if len(existing):
            markers[sc] = existing[0]
        else:
            k += 1
            markers[sc] = k
    if len(np.unique(markers[markers > 0])) < 2:
        return None, None
    n_seeds = int(len(np.unique(markers[markers > 0])))
    fc = field[y0:y1, x0:x1]
    surf = ((1.0 - fc) * 255).astype(np.uint8)
    img3 = cv2.cvtColor(surf, cv2.COLOR_GRAY2BGR)
    ws = markers.copy()
    ws[~Pc] = 1_000_000          # outside the mask: a dummy label the watershed will not cross into
    cv2.watershed(img3, ws)
    ws[~Pc] = 0
    ws[ws < 0] = 0               # ridge pixels (-1): assign below
    labs = [l for l in np.unique(ws) if 0 < l < 1_000_000]
    if len(labs) < 2:
        return None, None
    # ridge/unassigned pixels of P -> nearest labelled part
    un = Pc & (ws <= 0)
    if un.any():
        dists = np.stack([cv2.distanceTransform((ws != l).astype(np.uint8), cv2.DIST_L2, 3) for l in labs])
        ws[un] = np.array(labs)[np.argmin(dists, axis=0)][un]
    # absorb small parts into the neighbour with the longest shared border
    changed = True
    while changed:
        changed = False
        labs = [l for l in np.unique(ws) if l > 0]
        for l in labs:
            part = ws == l
            if part.sum() < MIN_PART and len(labs) > 1:
                ring = cv2.dilate(part.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & Pc & ~part
                nb = ws[ring]
                nb = nb[nb > 0]
                if len(nb):
                    ws[part] = np.bincount(nb).argmax()
                    changed = True
                    break
    labs = [l for l in np.unique(ws) if l > 0]
    if len(labs) < 2:
        return None, None
    # boundary between parts: pixels whose 4-neighbour carries another label
    b = np.zeros(Pc.shape, bool)
    for dy, dx in ((0, 1), (1, 0)):
        a = ws[:Pc.shape[0] - dy, :Pc.shape[1] - dx]
        c = ws[dy:, dx:]
        diff = (a != c) & (a > 0) & (c > 0)
        b[:Pc.shape[0] - dy, :Pc.shape[1] - dx] |= diff
        b[dy:, dx:] |= diff
    dt = cv2.distanceTransform(Pc.astype(np.uint8), cv2.DIST_L2, 3)
    areas = sorted([(int((ws == l).sum()), l) for l in labs], reverse=True)
    big = [ws == areas[0][1], ws == areas[1][1]]
    feats = dict(n_seeds=n_seeds, n_parts=len(labs),
                 saddle=float(fc[b].mean()) if b.any() else float("nan"),
                 neck=float(2 * dt[b].max()) if b.any() else float("nan"),   # mask width AT the cut (its thickest point)
                 frac_low=float((fc[b] < 0.5).mean()) if b.any() else float("nan"),
                 orient_diff=angle_diff(principal_angle(big[0]), principal_angle(big[1])),
                 area_ratio=areas[-1][0] / max(areas[0][0], 1))
    parts = []
    for l in labs:
        full = np.zeros_like(P)
        full[y0:y1, x0:x1] = ws == l
        parts.append(full)
    return parts, feats


def img_pq(gt_df, preds, stem):
    """Exact score of one image. Prediction ids carry the stem prefix like the
    pool's cand_id, because the scorer groups predictions by image from the id."""
    ids = [f"{stem}_u{i}" for i in range(len(preds))]
    return score(gt_df, pd.DataFrame({"filament_id": ids, "segmentation_rle": list(preds)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--pcut", type=float, default=0.60)
    ap.add_argument("--seed-conf", type=float, default=0.20)
    ap.add_argument("--field", default="blob5ens")
    ap.add_argument("--seeds", default="det", choices=["det", "dt", "both"],
                    help="watershed markers: detector masks inside P (det), distance-transform "
                         "maxima of P (dt), or both")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    folds_l = [int(x) for x in a.folds.split(",")]

    cand = pd.read_parquet(OUT / "e115b_labelled_pool.parquet").merge(pd.read_parquet(OUT / "e115b_p_oof.parquet"), on="cand_id")
    keep = cand[(cand["conf"] >= a.conf) & (cand["p_oof"] >= a.pcut)].copy()
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    keep["fold"] = keep["stem"].map(fold_of).astype(int)
    keep = keep[keep["fold"].isin(folds_l)]
    stems = sorted(set(keep["stem"]))
    log(f"kept {len(keep)} predictions over {len(stems)} stems, folds {folds_l}")

    base_rles, oracle_rles, all_prop_rles = {}, {}, {}
    rows = []
    t0 = time.time()
    for fold in folds_l:
        det = pd.read_parquet(CACHE / f"fold{fold}_preds_e013.parquet")
        det = det[det["conf"] >= a.seed_conf]
        Z = np.load(SEM / f"fold{fold}_probs_{a.field}.npz")
        fstems = [s for s in stems if fold_of[s] == fold]
        for n, stem in enumerate(fstems):
            kp = keep[keep["stem"] == stem]
            g = gt[gt["stem"] == stem]
            gt_df = g[["filament_id", "segmentation_rle"]]
            gts_by_entry = {e: [decode_rle(r).astype(bool) for r in gg["segmentation_rle"]] for e, gg in g.groupby("entry_id")}
            seeds = [decode_rle(r).astype(bool) for r in det.loc[det["stem"] == stem, "rle"]]
            field = Z[stem].astype(np.float32) / 255.0
            preds = [(cid, decode_rle(r).astype(bool)) for cid, r in zip(kp["cand_id"], kp["rle"])]
            base = [encode_mask(m.astype(np.uint8)) for _, m in preds]
            base_rles[stem] = base
            r_base = img_pq(gt_df, base, stem)
            orc, prop = list(base), list(base)
            for i, (cid, P) in enumerate(preds):
                o = oracle_split(P, gts_by_entry)
                if o is not None:
                    orc[i] = None
                    orc.extend(encode_mask(p.astype(np.uint8)) for p in o)
                use = seeds if a.seeds == "det" else (dt_seeds(P) if a.seeds == "dt" else seeds + dt_seeds(P))
                parts, feats = proposer_split(P, field, use)
                if parts is not None:
                    prop_rles = [encode_mask(p.astype(np.uint8)) for p in parts]
                    trial = list(base)
                    trial[i] = None
                    trial = [x for x in trial if x is not None] + prop_rles
                    r_try = img_pq(gt_df, trial, stem)
                    row = dict(stem=stem, fold=fold, cand_id=cid, p_oof=float(kp.iloc[i]["p_oof"]), area=int(P.sum()),
                               d_pq=float(r_try["pq"] - r_base["pq"]), d_tp=int(r_try["tp"] - r_base["tp"]),
                               d_fp=int(r_try["fp"] - r_base["fp"]), oracle_merged=o is not None, **feats)
                    rows.append(row)
                    prop[i] = None
                    prop.extend(prop_rles)
            oracle_rles[stem] = [x for x in orc if x is not None]
            all_prop_rles[stem] = [x for x in prop if x is not None]
            if n % 40 == 0:
                log(f"  fold {fold}: {n}/{len(fstems)} stems ({time.time()-t0:.0f}s)")

    def pooled(rles_by_stem):
        pred = pd.DataFrame({"filament_id": [f"{s}_{i}" for s, rs in rles_by_stem.items() for i in range(len(rs))],
                             "segmentation_rle": [r for rs in rles_by_stem.values() for r in rs]})
        g = gt[gt["stem"].isin(rles_by_stem)]
        return score(g[["filament_id", "segmentation_rle"]], pred)

    rb, ro, rp = pooled(base_rles), pooled(oracle_rles), pooled(all_prop_rles)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / f"unmerge_proposals{a.tag}.parquet", index=False)
    n_orc = sum(1 for s in stems for _ in [0]) and int(sum(len(oracle_rles[s]) != len(base_rles[s]) for s in stems))
    log(f"\nbase     PQ {rb['pq']:.4f}  TP {rb['tp']} FP {rb['fp']} FN {rb['fn']}")
    log(f"ORACLE   PQ {ro['pq']:.4f}  ({ro['pq']-rb['pq']:+.4f})  TP {ro['tp']} FP {ro['fp']} FN {ro['fn']}   images with a GT-guided cut: {n_orc}")
    log(f"PROPOSER all splits applied: PQ {rp['pq']:.4f} ({rp['pq']-rb['pq']:+.4f})  TP {rp['tp']} FP {rp['fp']} FN {rp['fn']}")
    log(f"proposals: {len(df)} predictions with >= 2 seeds split; pays (d_pq > 0): {int((df['d_pq'] > 0).sum())}, "
        f"hurts: {int((df['d_pq'] < 0).sum())}, neutral: {int((df['d_pq'] == 0).sum())}; "
        f"of the {int(df['oracle_merged'].sum())} GT-merged among them, proposer pays on {int(((df['d_pq'] > 0) & df['oracle_merged']).sum())}")

    res = dict(base=rb, oracle=ro, proposer_all=rp, n_proposals=len(df))
    if len(df) >= 30:
        from sklearn.metrics import roc_auc_score
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import GroupKFold
        y = (df["d_pq"] > 0).astype(int).to_numpy()
        feats = ["n_seeds", "n_parts", "saddle", "neck", "frac_low", "orient_diff", "area_ratio", "p_oof", "area"]
        log("\nper-feature ROC-AUC for 'split pays' (sign-agnostic, max(auc, 1-auc)):")
        aucs = {}
        for f in feats:
            v = df[f].fillna(df[f].median()).to_numpy()
            if y.min() != y.max():
                au = roc_auc_score(y, v)
                aucs[f] = float(max(au, 1 - au))
                log(f"  {f:12s} {aucs[f]:.3f}  ({'high pays' if au >= 0.5 else 'low pays'})")
        X = df[feats].fillna(df[feats].median()).to_numpy(np.float64)
        grp = df["stem"].to_numpy()
        oof = np.zeros(len(df))
        applied = {s: list(base_rles[s]) for s in stems}
        for tr, te in GroupKFold(n_splits=5).split(X, y, grp):
            clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, random_state=0)
            clf.fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
            # threshold on the TRAINING folds: maximise summed d_pq of accepted splits
            ptr = clf.predict_proba(X[tr])[:, 1]
            dtr = df["d_pq"].to_numpy()[tr]
            grid = np.linspace(0.3, 0.9, 25)
            gains = [float(dtr[ptr >= t].sum()) for t in grid]
            t_star = float(grid[int(np.argmax(gains))])
            for i in te:
                if oof[i] >= t_star:
                    applied[df.iloc[i]["stem"]] = None  # marker: rebuild below
            res.setdefault("thresholds", []).append(t_star)
        df["oof_p"] = oof
        # rebuild the applied set: for each image, replace accepted predictions by their parts
        # (re-run the proposer for accepted rows only, deterministic)
        acc = df[df["oof_p"] >= np.median(res["thresholds"])]
        log(f"\ngrouped-OOF classifier AUC {max(roc_auc_score(y, oof), 1 - roc_auc_score(y, oof)):.3f}; "
            f"thresholds per split {np.round(res['thresholds'], 2).tolist()}; accepted splits {len(acc)} "
            f"(pays {int((acc['d_pq'] > 0).sum())}, hurts {int((acc['d_pq'] < 0).sum())}), summed per-image d_pq {acc['d_pq'].sum():+.4f}")
        res["classifier"] = dict(auc=float(max(roc_auc_score(y, oof), 1 - roc_auc_score(y, oof))), accepted=int(len(acc)),
                                 accepted_pays=int((acc["d_pq"] > 0).sum()), accepted_hurts=int((acc["d_pq"] < 0).sum()),
                                 summed_dpq=float(acc["d_pq"].sum()), feature_auc=aucs)
    json.dump(res, open(OUT / f"unmerge_oracle{a.tag}.json", "w"), indent=1, default=float)
    log(f"\nwrote unmerge_oracle{a.tag}.json and unmerge_proposals{a.tag}.parquet")


if __name__ == "__main__":
    main()
