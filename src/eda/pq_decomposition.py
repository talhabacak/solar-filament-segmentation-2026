"""E182 -- where does the loss come from on the EASY disc (r 0.2-0.7), and is it RQ or SQ?

Talha, 2026-09-13: "for the relatively easy ones at radius 0.2-0.7, where do the errors come
from, and for which filament types is the loss from RQ or from SQ -- and is it IoU falling
just short or far short?"

PQ = sum IoU(TP) / (TP + FP/2 + FN/2) = SQ x RQ, with SQ = mean IoU of matched pairs and
RQ = TP / (TP + FP/2 + FN/2). That factorisation makes the counterfactuals exact:

    perfect shapes, same matching      PQ -> RQ          so the SQ loss is RQ x (1 - SQ)
    perfect matching, same shapes      PQ -> SQ          so the RQ loss is SQ x (1 - RQ)

Both are computed per stratum, on the deployed five-fold out-of-fold selection, with the
metric's own per-entry accounting: a prediction that matches in one annotator entry and not in
another is a true positive in the first and a false positive in the second.

For the RQ half the question "how far short" is answered by the best IoU any POOL candidate
reaches on the unmatched copy: 0.4-0.5 is a shape a small improvement would recover, under 0.05
means nothing ever saw it.

Strata: radius band first (the band Talha named against the rest), then inside 0.2-0.7 by
width, contrast, area, crowding and chirality class.

Usage: python src/eda/pq_decomposition.py
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


def deployed():
    feats = list(joblib.load(ROOT / "outputs/selector/selector_v5_shrunkblob5prodts.joblib")["features"]) + list(EXTRA)
    folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
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
    pool["accepted"] = (pool["conf"] >= 0.40) & (pool["p"] >= 0.60)
    return pool


def stats(d, fp):
    """PQ, SQ, RQ and the two counterfactual losses for a set of GT copies + their FP share."""
    tp = int((d["iou_acc"] > 0.5).sum())
    fn = len(d) - tp
    sq = float(d.loc[d["iou_acc"] > 0.5, "iou_acc"].mean()) if tp else 0.0
    den = tp + 0.5 * fp + 0.5 * fn
    rq = tp / den if den else 0.0
    pq = sq * rq
    return dict(n=len(d), tp=tp, fn=fn, fp=fp, SQ=round(sq, 4), RQ=round(rq, 4), PQ=round(pq, 4),
                loss_RQ=round(sq * (1 - rq), 4), loss_SQ=round(rq * (1 - sq), 4))


def main():
    pool = deployed()
    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt["entry"] = gt["entry_id"].astype(str)
    cls_col = next((c for c in ("category_id", "category", "chirality") if c in gt.columns), None)
    K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))

    rows, fp_rows = [], []
    for n, (stem, gg) in enumerate(gt.groupby("stem")):
        pc = pool[pool["stem"] == stem]
        img = cv2.imread(str(IMG / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        imf = img.astype(np.float32)
        G = np.stack([decode_rle(r).astype(np.uint8) for r in gg["segmentation_rle"]])
        acc = pc["accepted"].to_numpy()
        if len(pc):
            P = np.stack([decode_rle(r).astype(np.uint8) for r in pc["rle"]])
            iou = _entry_iou_matrix(G, P)
        else:
            P, iou = None, np.zeros((len(G), 1))
        entries = gg["entry"].to_numpy()
        n_entries = len(set(entries))
        # GT side
        for gi in range(len(G)):
            best_pool = float(iou[gi].max()) if iou.size else 0.0
            iou_acc = float(iou[gi][acc].max()) if acc.any() else 0.0
            m = G[gi].astype(bool)
            area = int(m.sum())
            ys, xs = np.nonzero(m)
            cnts, _ = cv2.findContours(G[gi], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            skel = max(sum(cv2.arcLength(c, True) for c in cnts) / 2.0, 1.0)
            ring = (cv2.dilate(G[gi], K) > 0) & ~m
            fg = float(imf[m].mean())
            bg = float(imf[ring].mean()) if ring.sum() > 10 else fg
            X_ = (xs.mean() - C) / R
            Y_ = (C - ys.mean()) / R
            rec = dict(stem=stem, entry=entries[gi], iou_acc=iou_acc, best_pool=best_pool,
                       r_norm=float(np.hypot(X_, Y_)), log_area=float(np.log10(max(area, 1))),
                       width=float(area / skel), contrast=float((bg - fg) / max(bg, 1.0)),
                       skel_len=float(skel))
            if cls_col:
                rec["cls"] = str(gg[cls_col].iloc[gi])
            rows.append(rec)
        # FP side, with the metric's per-entry accounting
        if P is not None and acc.any():
            sel = np.flatnonzero(acc)
            for j in sel:
                ys, xs = np.nonzero(P[j])
                rp = float(np.hypot(xs.mean() - C, ys.mean() - C) / R)
                for e in set(entries):
                    gi_e = np.flatnonzero(entries == e)
                    matched = bool((iou[gi_e, j] > 0.5).any())
                    if not matched:
                        fp_rows.append(dict(stem=stem, r_norm=rp))
        if (n + 1) % 150 == 0:
            log(f"  {n + 1} images, {len(rows)} copies")
    d = pd.DataFrame(rows)
    fpd = pd.DataFrame(fp_rows)
    d.to_parquet(OUT / "pq_decomposition.parquet", index=False)
    log(f"\n{len(d)} GT copies, {len(fpd)} false positives (prediction x entry)")

    def band_fp(lo, hi):
        return int(((fpd["r_norm"] >= lo) & (fpd["r_norm"] < hi)).sum()) if len(fpd) else 0

    log("\n=== by radius band (the whole metric, split by where the GT copy sits)")
    out = {}
    for name, lo, hi in (("centre r<0.2", 0.0, 0.2), ("EASY r 0.2-0.7", 0.2, 0.7),
                         ("outer r 0.7-0.8", 0.7, 0.8), ("limb r 0.8-0.9", 0.8, 0.9),
                         ("far limb r>0.9", 0.9, 9.9)):
        sub = d[(d["r_norm"] >= lo) & (d["r_norm"] < hi)]
        if len(sub):
            out[name] = stats(sub, band_fp(lo, hi))
    t = pd.DataFrame(out).T
    log(t.to_string())
    log("\n  loss_RQ = SQ x (1-RQ): what perfect MATCHING would add;  loss_SQ = RQ x (1-SQ): what perfect SHAPES would add")

    easy = d[(d["r_norm"] >= 0.2) & (d["r_norm"] < 0.7)].copy()
    fp_easy = band_fp(0.2, 0.7)
    log(f"\n=== inside the easy band: {len(easy)} copies, {fp_easy} false positives")
    for col, bins, label in (("width", [0, 6, 8, 11, 14, 999], "drawn width (px)"),
                             ("contrast", [-1, 0.08, 0.12, 0.16, 1], "contrast"),
                             ("log_area", [0, 2.7, 3.0, 3.3, 9], "log10 area"),
                             ("skel_len", [0, 80, 140, 220, 9999], "skeleton length (px)")):
        easy["b"] = pd.cut(easy[col], bins)
        rowsx = {}
        for b, sub in easy.groupby("b", observed=True):
            share = len(sub) / max(len(easy), 1)
            rowsx[str(b)] = stats(sub, int(round(fp_easy * share)))
        log(f"\n  by {label} (FP shared pro rata):")
        log(pd.DataFrame(rowsx).T.to_string())
    if "cls" in easy.columns:
        rowsx = {}
        for b, sub in easy.groupby("cls", observed=True):
            rowsx[str(b)] = stats(sub, int(round(fp_easy * len(sub) / max(len(easy), 1))))
        log("\n  by chirality class (1=Left, 2=Right, 3=Unidentifiable):")
        log(pd.DataFrame(rowsx).T.to_string())

    miss = easy[easy["iou_acc"] <= 0.5]
    log(f"\n=== how far short are the {len(miss)} unmatched copies in the easy band?")
    h = pd.cut(miss["best_pool"], [-0.01, 0.05, 0.2, 0.3, 0.4, 0.45, 0.5, 1.0]).value_counts(sort=False)
    log((h / len(miss)).round(3).to_string())
    near = miss[(miss["best_pool"] > 0.4)]
    log(f"  within one small step of the threshold (best pool IoU > 0.4): {len(near)} "
        f"({len(near) / max(len(miss), 1):.1%} of the misses)")
    tp_all = int((d["iou_acc"] > 0.5).sum())
    den_all = tp_all + 0.5 * len(fpd) + 0.5 * int((d["iou_acc"] <= 0.5).sum())
    log(f"  recovering exactly those at IoU 0.55 would be worth {0.55 * len(near) / den_all:+.4f} PQ overall")
    json.dump(dict(bands={k: v for k, v in out.items()},
                   easy_n=int(len(easy)), easy_fp=int(fp_easy),
                   near_miss=int(len(near)), near_miss_share=float(len(near) / max(len(miss), 1))),
              open(OUT / "pq_decomposition.json", "w"), indent=1, default=float)
    log("\nPQ_DECOMP_DONE")


if __name__ == "__main__":
    main()
