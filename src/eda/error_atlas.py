"""E170 -- the error atlas: which of our misses are OUR fault, and what do they look like?

Talha, 2026-09-12: "we have a lot of errors on the disc; we should improve them in both the
selector and the models -- and we may need something that finds what we miss and what we get
wrong better than we do now."

Every diagnostic so far counted outcomes per band (r, size, contrast) one axis at a time. This
one fits the joint structure and, crucially, separates the part of the loss that is label noise
from the part that is ours:

  outcome per GT copy, from the deployed 5-fold out-of-fold selection
    matched        an accepted prediction reaches IoU > 0.5 on this copy
    selector       a pool candidate reaches IoU > 0.5 but was not accepted
    shape          the best pool candidate is in (0.05, 0.5]
    never          the best pool candidate is <= 0.05

  measurable properties of the copy
    log area, skeleton length, mean width (area / skeleton length), contrast against a 15 px
    ring, r_norm, |latitude|, crowding (GT copies of the same entry within 200 px),
    elongation, chirality class, site letter, year

  AGREEMENT, the label-noise axis
    n_copies      how many annotator entries of this image drew a filament matching this copy
                  at IoU > 0.5 (1 = only this annotator saw it)
    agree_iou     mean IoU with those sibling copies (how much they agree on its shape)

Two numbers decide where to spend next:

  1. AUC of "will this copy be missed" with and without the agreement pair. If agreement
     carries most of the signal, the misses are the annotators' disagreement, not our failure,
     and the population is not addressable. The gap is the addressable part.
  2. The UNAMBIGUOUS miss set: copies with n_copies == the image's annotator count (every
     annotator drew it) and high agree_iou, that we still miss. Those are ours by construction.
     The atlas prints their size/contrast/width/crowding profile and the PQ that recovering
     them would be worth, so the next model change can be aimed at a described population
     instead of a guess.

Usage: python src/eda/error_atlas.py            (writes outputs/eda/error_atlas.json)
"""
import json
import sys
from pathlib import Path

import cv2
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
IMG = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
C, R = 1023.5, 904.1


def log(*a):
    print(*a, flush=True)


def deployed_selection():
    """The production 5-fold pool at the deployed gate, selector fitted out of fold."""
    feats = list(joblib.load(ROOT / "outputs/selector/selector_v5_shrunkblob5prodts.joblib")["features"]) + list(EXTRA)
    folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    pool = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    pool = pool.merge(pd.read_parquet(OUT / "extra_features_t020.parquet")[["cand_id"] + list(EXTRA)],
                      on="cand_id", how="left")
    pool["fold"] = pool["stem"].map(fold_of).astype(int)
    keep = []
    for F in range(5):
        trn, tst = pool[pool["fold"] != F], pool[pool["fold"] == F].copy()
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
        Z = tst[feats].to_numpy(np.float64)
        tst["p"] = np.searchsorted(ref, cls.predict_proba(Z)[:, 1] * np.clip(reg.predict(Z), 0, None),
                                   side="right") / len(ref)
        keep.append(tst)
    pool = pd.concat(keep)
    pool["accepted"] = (pool["conf"] >= 0.40) & (pool["p"] >= 0.60)
    return pool, fold_of


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    pool, fold_of = deployed_selection()
    log(f"pool {len(pool)} candidates, accepted {int(pool['accepted'].sum())}")
    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt["entry"] = gt["entry_id"].astype(str)

    rows = []
    K15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    for n, (stem, gg) in enumerate(gt.groupby("stem")):
        pc = pool[pool["stem"] == stem]
        img = cv2.imread(str(IMG / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        imf = img.astype(np.float32)
        G = np.stack([decode_rle(r).astype(np.uint8) for r in gg["segmentation_rle"]])
        entries = gg["entry"].to_numpy()
        n_entries = len(set(entries))
        # candidate IoUs: all pool candidates, and the accepted subset
        if len(pc):
            P = np.stack([decode_rle(r).astype(np.uint8) for r in pc["rle"]])
            iou_pool = _entry_iou_matrix(G, P)
            acc = pc["accepted"].to_numpy()
        else:
            iou_pool = np.zeros((len(G), 1))
            acc = np.zeros(1, bool)
        # sibling agreement: best IoU against copies from OTHER entries
        iou_gg = _entry_iou_matrix(G, G)
        cent = []
        for gi in range(len(G)):
            ys, xs = np.nonzero(G[gi])
            cent.append((xs.mean(), ys.mean()))
        cent = np.array(cent)
        for gi in range(len(G)):
            best_pool = float(iou_pool[gi].max()) if iou_pool.size else 0.0
            best_acc = float(iou_pool[gi][acc].max()) if acc.any() else 0.0
            outcome = ("matched" if best_acc > 0.5 else
                       "selector" if best_pool > 0.5 else
                       "shape" if best_pool > 0.05 else "never")
            other = entries != entries[gi]
            sib = iou_gg[gi][other]
            n_copies = 1 + int((sib > 0.5).sum())
            agree_iou = float(sib[sib > 0.5].mean()) if (sib > 0.5).any() else 0.0
            m = G[gi].astype(bool)
            area = int(m.sum())
            ys, xs = np.nonzero(m)
            skel = cv2.ximgproc.thinning(G[gi] * 255).sum() / 255 if hasattr(cv2, "ximgproc") else None
            if skel is None or skel < 1:
                # perimeter/2 is a robust length proxy for a thin ribbon
                cnts, _ = cv2.findContours(G[gi], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                skel = max(sum(cv2.arcLength(c, True) for c in cnts) / 2.0, 1.0)
            ring = (cv2.dilate(G[gi], K15) > 0) & ~m
            fg = float(imf[m].mean())
            bg = float(imf[ring].mean()) if ring.sum() > 10 else fg
            d = np.hypot(cent[:, 0] - cent[gi, 0], cent[:, 1] - cent[gi, 1])
            X_ = (xs.mean() - C) / R
            Y_ = (C - ys.mean()) / R
            lat = np.degrees(np.arcsin(np.clip(Y_, -1, 1)))
            rows.append(dict(
                stem=stem, fid=gg["filament_id"].iloc[gi], outcome=outcome, best_pool=best_pool,
                log_area=float(np.log10(max(area, 1))), skel_len=float(skel),
                width=float(area / max(skel, 1.0)),
                contrast=float((bg - fg) / max(bg, 1.0)),
                r_norm=float(np.hypot(X_, Y_)), abs_lat=float(abs(lat)),
                crowding=int(((d > 1e-6) & (d < 200)).sum()),
                elong=float(skel ** 2 / max(area, 1)),
                n_copies=n_copies, n_entries=n_entries, agree_iou=agree_iou,
                site=stem[-2], year=int(stem[:4])))
        if (n + 1) % 100 == 0:
            log(f"  {n + 1} images, {len(rows)} GT copies")
    d = pd.DataFrame(rows)
    d.to_parquet(OUT / "error_atlas.parquet", index=False)
    log(f"\n{len(d)} GT copies")
    log(d["outcome"].value_counts(normalize=True).round(4).to_string())

    # 1. how much of "missed" is the annotators' disagreement?
    d["missed"] = (d["outcome"] != "matched").astype(int)
    PHYS = ["log_area", "skel_len", "width", "contrast", "r_norm", "abs_lat", "crowding", "elong"]
    AGREE = ["n_copies", "agree_iou"]
    folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
    fo = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    d["fold"] = d["stem"].map(fo).astype(int)

    def auc(cols):
        oof = np.zeros(len(d))
        for F in range(5):
            tr, te = (d["fold"] != F).to_numpy(), (d["fold"] == F).to_numpy()
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                               min_samples_leaf=25, l2_regularization=1.0, random_state=0)
            m.fit(d.loc[tr, cols], d.loc[tr, "missed"])
            oof[te] = m.predict_proba(d.loc[te, cols])[:, 1]
        return roc_auc_score(d["missed"], oof), oof
    a_phys, _ = auc(PHYS)
    a_all, oof_all = auc(PHYS + AGREE)
    a_agree, _ = auc(AGREE)
    log(f"\npredicting 'we will miss this copy':")
    log(f"  physical properties only      AUC {a_phys:.4f}")
    log(f"  agreement only (n_copies, agree_iou) AUC {a_agree:.4f}")
    log(f"  both                          AUC {a_all:.4f}")

    # 2. the unambiguous misses: every annotator of the image drew it, and they agree on it
    una = d[(d["n_copies"] == d["n_entries"]) & (d["n_entries"] >= 2) & (d["agree_iou"] >= 0.6)]
    log(f"\nunambiguous copies (all {una['n_entries'].min()}+ annotators drew it, agree_iou >= 0.6): {len(una)}")
    log(una["outcome"].value_counts().to_string())
    miss_una = una[una["outcome"] != "matched"]
    tp = int((d["outcome"] == "matched").sum())
    fp = int(pool["accepted"].sum() * 1.635 - tp)
    fn = int((d["outcome"] != "matched").sum())
    den = tp + 0.5 * max(fp, 0) + 0.5 * fn
    log(f"  of these we miss {len(miss_una)} ({len(miss_una) / max(len(una), 1):.1%}); "
        f"recovering them at IoU 0.7 would be worth about {0.7 * len(miss_una) / den:+.4f} PQ")
    if len(miss_una):
        prof = miss_una[PHYS].median()
        ref = d.loc[d["outcome"] == "matched", PHYS].median()
        log("\n  profile of the unambiguous misses (median) against matched copies:")
        for c in PHYS:
            log(f"    {c:10s} {prof[c]:10.3f}   matched {ref[c]:10.3f}   ratio {prof[c] / (ref[c] + 1e-9):6.2f}")
        log("\n  their outcome split: " + miss_una["outcome"].value_counts().to_dict().__str__())
        for col, bins in (("log_area", [0, 2.5, 3, 3.5, 4, 9]), ("contrast", [-1, 0.05, 0.1, 0.15, 1]),
                          ("width", [0, 8, 14, 22, 999]), ("r_norm", [0, 0.5, 0.7, 0.8, 0.9, 2])):
            g = d.assign(b=pd.cut(d[col], bins)).groupby("b", observed=True).apply(
                lambda x: pd.Series(dict(n=len(x), miss=(x["outcome"] != "matched").mean(),
                                         una_n=int(((x["n_copies"] == x["n_entries"]) & (x["n_entries"] >= 2) & (x["agree_iou"] >= 0.6)).sum()),
                                         una_miss=float(((x["n_copies"] == x["n_entries"]) & (x["n_entries"] >= 2) & (x["agree_iou"] >= 0.6) & (x["outcome"] != "matched")).sum()))), include_groups=False)
            log(f"\n  by {col}:\n" + g.round(3).to_string())
    json.dump(dict(n=len(d), outcome=d["outcome"].value_counts().to_dict(),
                   auc_phys=float(a_phys), auc_agree=float(a_agree), auc_both=float(a_all),
                   n_unambiguous=int(len(una)), n_unambiguous_missed=int(len(miss_una)),
                   headroom_pq=float(0.7 * len(miss_una) / den)),
              open(OUT / "error_atlas.json", "w"), indent=1)
    log("\nATLAS_DONE")


if __name__ == "__main__":
    main()
