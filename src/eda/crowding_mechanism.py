"""E173 -- why do we miss the UNAMBIGUOUS copies? The atlas says: they are crowded.

E170 (error atlas) found a described, addressable population: of the 1563 ground-truth copies
that every annotator of the image drew and agreed on (pairwise IoU >= 0.6), we miss 166 --
103 in the shape band, 63 dropped by the selector -- worth about +0.0158 PQ. Their profile
against matched copies: shorter (0.75x skeleton), fainter (0.83x contrast), slightly thinner
(0.92x), and above all CROWDED: 3 neighbouring copies within 200 px against 1 for matched.

Crowding can fail in four distinct ways, and they need different fixes, so this classifies each
unambiguous miss by mechanism:

  merged_into   the best pool candidate covers this copy AND another copy of the same annotator
                entry (>=0.2 of each): our pipeline made one object where the annotator made two
  carved        a pool candidate BEFORE de-overlap reaches IoU > 0.5, but the candidate that
                survives de-overlap does not: a neighbour with higher confidence ate its pixels
  split         two or more pool candidates each cover part of this copy, and their union would
                reach IoU > 0.5: we fragmented one annotator object
  weak          none of the above: the semantic field and the detector simply did not shape it

For the 63 selector-dropped ones the mechanism is by definition "a good candidate existed", so
they are reported separately with the selector's score, which says whether a threshold or a
feature is the obstacle.

Usage: python src/eda/crowding_mechanism.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/eda", "src/models"):
    sys.path.insert(0, str(ROOT / p))
from filament.rle import decode_rle  # noqa: E402
from filament.pq import _entry_iou_matrix  # noqa: E402
from e115c_threshold_ci import make_cls, make_reg  # noqa: E402
from extra_features import EXTRA  # noqa: E402

OUT = ROOT / "outputs" / "eda"


def log(*a):
    print(*a, flush=True)


def main():
    atlas = pd.read_parquet(OUT / "error_atlas.parquet")
    una = atlas[(atlas["n_copies"] == atlas["n_entries"]) & (atlas["n_entries"] >= 2)
                & (atlas["agree_iou"] >= 0.6)]
    miss = una[una["outcome"] != "matched"]
    log(f"{len(una)} unambiguous copies, {len(miss)} missed "
        f"({(miss['outcome'] == 'shape').sum()} shape, {(miss['outcome'] == 'selector').sum()} selector)")

    feats = list(joblib.load(ROOT / "outputs/selector/selector_v5_shrunkblob5prodts.joblib")["features"]) + list(EXTRA)
    folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    # the deployed pool (after de-overlap) and the SAME pool without de-overlap, to tell
    # "a neighbour carved it" from "nothing ever shaped it"
    pool = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    pool = pool.merge(pd.read_parquet(OUT / "extra_features_t020.parquet")[["cand_id"] + list(EXTRA)],
                      on="cand_id", how="left")
    pool["fold"] = pool["stem"].map(fold_of).astype(int)
    p = np.zeros(len(pool))
    for F in range(5):
        trn = pool[pool["fold"] != F]
        te = (pool["fold"] == F).to_numpy()
        X, y, v = trn[feats].to_numpy(np.float64), trn["y"].to_numpy(), trn["iou_sum"].to_numpy(np.float64)
        inner = np.zeros(len(trn))
        for g in [k for k in range(5) if k != F]:
            itr, msk = (trn["fold"] != g).to_numpy(), (trn["fold"] == g).to_numpy()
            c = make_cls().fit(X[itr], y[itr])
            r = make_reg().fit(X[itr][y[itr] == 1], v[itr][y[itr] == 1])
            inner[msk] = c.predict_proba(X[msk])[:, 1] * np.clip(r.predict(X[msk]), 0, None)
        ref = np.sort(inner)
        cls = make_cls().fit(X, y)
        reg = make_reg().fit(X[y == 1], v[y == 1])
        Z = pool.loc[te, feats].to_numpy(np.float64)
        p[te] = np.searchsorted(ref, cls.predict_proba(Z)[:, 1] * np.clip(reg.predict(Z), 0, None),
                                side="right") / len(ref)
    pool["p"] = p
    nodeov = OUT / "fused_pool_all_nodeov_t020_labelled.parquet"
    nd = pd.read_parquet(nodeov) if nodeov.exists() else None
    log(f"de-overlap-free pool: {'present' if nd is not None else 'absent (carved/weak not separable)'}")

    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt["entry"] = gt["entry_id"].astype(str)
    gtx = gt.set_index("filament_id")

    rows = []
    for stem, mg in miss.groupby("stem"):
        gg = gt[gt["stem"] == stem]
        pc = pool[pool["stem"] == stem]
        if not len(pc):
            for fid in mg["fid"]:
                rows.append(dict(fid=fid, mech="weak", best=0.0, p=0.0))
            continue
        G = np.stack([decode_rle(r).astype(np.uint8) for r in gg["segmentation_rle"]])
        P = np.stack([decode_rle(r).astype(np.uint8) for r in pc["rle"]])
        iou = _entry_iou_matrix(G, P)
        gid = list(gg["filament_id"])
        entry = gg["entry"].to_numpy()
        areas_g = G.reshape(len(G), -1).sum(1)
        inter = (G.reshape(len(G), -1).astype(np.float32) @ P.reshape(len(P), -1).T.astype(np.float32))
        cov_g = inter / np.maximum(areas_g[:, None], 1)               # share of each GT covered
        areas_p = P.reshape(len(P), -1).sum(1)
        cov_p = inter / np.maximum(areas_p[None, :], 1)               # share of each candidate used
        ndg = nd[nd["stem"] == stem] if nd is not None else None
        if ndg is not None and len(ndg):
            ND = np.stack([decode_rle(r).astype(np.uint8) for r in ndg["rle"]])
            iou_nd = _entry_iou_matrix(G, ND)
        else:
            iou_nd = None
        for fid in mg["fid"]:
            gi = gid.index(fid)
            j = int(np.argmax(iou[gi]))
            best = float(iou[gi, j])
            pj = float(pc["p"].iloc[j])
            same_entry = (entry == entry[gi])
            same_entry[gi] = False
            # merged: the best candidate also covers >=20% of another copy of the SAME entry
            merged = bool(((cov_g[:, j] >= 0.2) & same_entry).any())
            # split: several candidates each cover a part, union clears 0.5
            part = np.flatnonzero((cov_g[gi] >= 0.15) & (cov_p[gi] >= 0.3))
            union_iou = 0.0
            if len(part) >= 2:
                u = (P[part].sum(0) > 0).astype(np.uint8)
                union_iou = float(_entry_iou_matrix(G[gi:gi + 1], u[None])[0, 0])
            carved = bool(iou_nd is not None and iou_nd[gi].max() > 0.5 and best <= 0.5)
            mech = ("selector_drop" if best > 0.5 else
                    "carved" if carved else
                    "merged_into" if merged else
                    "split" if union_iou > 0.5 else "weak")
            rows.append(dict(fid=fid, mech=mech, best=best, p=pj, union_iou=union_iou,
                             crowding=int(miss.loc[miss["fid"] == fid, "crowding"].iloc[0])))
    d = pd.DataFrame(rows).merge(miss[["fid", "outcome", "log_area", "width", "contrast", "r_norm", "crowding"]],
                                 on="fid", how="left", suffixes=("", "_a"))
    d.to_parquet(OUT / "crowding_mechanism.parquet", index=False)
    log("\nmechanism of the unambiguous misses:")
    log(d["mech"].value_counts().to_string())
    log("\nmedian properties by mechanism:")
    log(d.groupby("mech")[["best", "p", "log_area", "width", "contrast", "crowding"]].median().round(3).to_string())
    sd = d[d["mech"] == "selector_drop"]
    if len(sd):
        log(f"\nthe {len(sd)} selector drops: best candidate IoU median {sd['best'].median():.3f}, "
            f"selector score p median {sd['p'].median():.3f} (gate 0.60); "
            f"{int((sd['p'] >= 0.4).sum())} of them score >= 0.40")
    # how much is each mechanism worth?
    tp = int((atlas["outcome"] == "matched").sum())
    fn = int((atlas["outcome"] != "matched").sum())
    den = tp + 0.5 * fn + 0.5 * 1733      # FP from the production read
    log("\nvalue of fixing each mechanism (at IoU 0.7 per recovered copy):")
    for mech, n in d["mech"].value_counts().items():
        log(f"  {mech:14s} {n:4d} copies -> {0.7 * n / den:+.4f} PQ")
    json.dump(dict(n_unambiguous_missed=int(len(d)), mech=d["mech"].value_counts().to_dict(),
                   pq_per_mech={k: float(0.7 * v / den) for k, v in d["mech"].value_counts().items()}),
              open(OUT / "crowding_mechanism.json", "w"), indent=1)
    log("\nCROWD_MECH_DONE")


if __name__ == "__main__":
    main()
