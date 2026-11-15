"""E217b -- does the detector's soft field carry the length our masks miss, and does using it pay?

Reads the soft masks cached by e217_yolo_soft_masks.py (fold F, out of fold, guard passed) and
sub036's out-of-fold keep set (conf >= 0.40, bag5 p >= 0.60), selection held FIXED as in E215.

For a kept candidate c, its detector field F_c is the pixelwise max of the soft masks of the
detections that are the same object (binary detection mask with >= 30 % of its area inside c, or
covering >= 30 % of c).

PART 1, mechanism (shape-band GT copies of the fold with an overlapping kept candidate; R = G \\ c):
  coverage(t)   share of R with F_c >= t, t in 0.1 ... 0.5; t = 0.5 is the binary we already
                fuse, so coverage(t) - coverage(0.5) is what the soft field ADDS
  precision(t)  of the added pixels E_t = {F_c >= t} \\ c: |E_t n G| / |E_t|
  reported for all shape-band copies and for the agreed ones (n_copies >= 2, drawn independently
  by two annotators). Comparator, ledger §39b: the semantic field >= 0.2 covers 11-16 % of R.

PART 2, the arm (every kept candidate of the fold, exact scorer, all fold images):
  full_t     c U ({F_c >= t} minus pixels owned by other kept masks)
  ring_t     the same, but {F_c >= t} first loses its one-pixel boundary ring, the operation every
             production mask goes through (worth +0.0117, ledger §54) -- so the arm adds length
             without undoing the erosion laterally
  Restored pixels are claimed in selector-score order; masks stay disjoint.

PRE-REGISTERED GATES (written before any result was seen, 2026-10-04):
  P  mechanism: on agreed copies, coverage(0.3) - coverage(0.5) >= 0.10 AND precision(0.3) >= 0.5.
  A  arm: some arm beats the base by > +0.002 PQ on this fold with the paired image-bootstrap CI
     excluding zero. One fold never funds a decision (project rule): A only buys folds 1-4,
     after which the threshold is chosen on four folds and read on the fifth.
  Both fail -> the soft-field cell is closed. P fails but A passes -> read as gain of a different
  mechanism than the length hypothesis, still only a ticket to folds 1-4.

Usage: python src/eda/e217b_soft_extension.py --fold 0
Output: outputs/eda/e217b_soft_extension_f{F}.json
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/eda", "src/models"):
    sys.path.insert(0, str(ROOT / p))
from filament import IMG_H, IMG_W  # noqa: E402
from filament.rle import decode_rle, encode_mask  # noqa: E402
from e115c_threshold_ci import per_image, pq  # noqa: E402

OUT = ROOT / "outputs" / "eda"
CACHE = ROOT / "outputs" / "oof_cache"
TS = [0.1, 0.2, 0.3, 0.4, 0.5]
ARM_TS = [0.2, 0.3, 0.4]
CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


def load_soft(fold, stem):
    z = np.load(CACHE / f"soft_e013_f{fold}" / f"{stem}.npz")
    out = []
    for i, (y0, y1, x0, x1) in enumerate(z["box"]):
        crop = z[f"m{i}"].astype(np.float32) / 255.0
        out.append((float(z["conf"][i]), (int(y0), int(y1), int(x0), int(x1)), crop))
    return out


def field_for(c, dets):
    """Pixelwise max of the soft masks of the detections that are the same object as c."""
    F = np.zeros((IMG_H, IMG_W), np.float32)
    ca = c.sum()
    for _, (y0, y1, x0, x1), crop in dets:
        b = crop >= 0.5
        ba = b.sum()
        if not ba:
            continue
        inter = (b & c[y0:y1, x0:x1]).sum()
        if inter >= 0.3 * ba or inter >= 0.3 * ca:
            F[y0:y1, x0:x1] = np.maximum(F[y0:y1, x0:x1], crop)
    return F


def ring_off(m):
    """Remove the one-pixel boundary ring (4-adjacent to background), as production does."""
    u = m.astype(np.uint8)
    return cv2.erode(u, CROSS).astype(bool) if u.any() else m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--smoke", action="store_true", help="only images whose soft cache exists (test run)")
    a_ = ap.parse_args()
    F_ = a_.fold

    folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
    stems = np.array(sorted(folds.loc[folds["fold"] == F_, "file_name"].str.replace(".jpeg", "", regex=False)))
    if a_.smoke:
        stems = np.array([s for s in stems if (CACHE / f"soft_e013_f{F_}" / f"{s}.npz").exists()])
    pool = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet", columns=["stem", "cand_id", "rle", "conf"])
    pool = pool.merge(pd.read_parquet(OUT / "bag5_oof_scores.parquet"), on="cand_id")
    keep = pool[(pool["conf"] >= 0.40) & (pool["p"] >= 0.60) & pool["stem"].isin(set(stems))]
    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt = gt[gt["stem"].isin(set(stems))]
    ncopy = gt.groupby("stem")["entry_id"].nunique().reindex(stems).fillna(1).to_numpy()
    gmask = dict(zip(gt["filament_id"], gt["segmentation_rle"]))
    atlas = pd.read_parquet(OUT / "error_atlas.parquet")
    shape = atlas[(atlas["outcome"] == "shape") & atlas["stem"].isin(set(stems))]

    mech, arms = [], {f"{k}_{t}": [] for k in ("full", "ring") for t in ARM_TS}
    base_rows = []
    for n, stem in enumerate(stems):
        kc = keep[keep["stem"] == stem].sort_values("p", ascending=False).reset_index(drop=True)
        if kc.empty:
            continue
        dets = load_soft(F_, stem)
        km = [decode_rle(r).astype(bool) for r in kc["rle"]]
        owned = np.zeros((IMG_H, IMG_W), bool)
        for m in km:
            owned |= m
        fields = [field_for(m, dets) for m in km]
        base_rows += [dict(cand_id=c, stem=stem, rle=r) for c, r in zip(kc["cand_id"], kc["rle"])]
        # PART 2: grown masks per arm, claimed in selector order
        for kind in ("full", "ring"):
            for t in ARM_TS:
                claimed = np.zeros_like(owned)
                for i, m in enumerate(km):
                    s = fields[i] >= t
                    if kind == "ring":
                        s = ring_off(s)
                    add = s & ~owned & ~claimed
                    claimed |= add
                    arms[f"{kind}_{t}"].append(dict(cand_id=kc.at[i, "cand_id"], stem=stem,
                                                    rle=encode_mask((m | add).astype(np.uint8))))
        # PART 1: mechanism on shape-band copies of this image
        for _, row in shape[shape["stem"] == stem].iterrows():
            G = decode_rle(gmask[row["fid"]]).astype(bool)
            ious = [(m & G).sum() / max((m | G).sum(), 1) for m in km]
            j = int(np.argmax(ious))
            if ious[j] == 0:
                continue
            c, Fc = km[j], fields[j]
            R = G & ~c
            rec = dict(fid=row["fid"], agreed=bool(row["n_copies"] >= 2), r_px=int(R.sum()), iou=float(ious[j]))
            for t in TS:
                E = (Fc >= t) & ~c
                rec[f"cov_{t}"] = float((R & (Fc >= t)).sum() / max(R.sum(), 1))
                rec[f"prec_{t}"] = float((E & G).sum() / E.sum()) if E.any() else np.nan
                rec[f"ext_px_{t}"] = int(E.sum())
            mech.append(rec)
        if n % 20 == 0:
            print(f"  {n + 1}/{len(stems)}", flush=True)

    m = pd.DataFrame(mech)
    res = {"mechanism": {}}
    for name, sub in (("all", m), ("agreed", m[m.agreed])):
        res["mechanism"][name] = dict(n=len(sub), **{f"cov_{t}": float(sub[f"cov_{t}"].mean()) for t in TS},
                                      **{f"prec_{t}": float(sub[f"prec_{t}"].mean()) for t in TS})
    ag = res["mechanism"]["agreed"]
    res["gate_P"] = dict(added_coverage_03=ag["cov_0.3"] - ag["cov_0.5"], precision_03=ag["prec_0.3"],
                         passed=bool(ag["cov_0.3"] - ag["cov_0.5"] >= 0.10 and ag["prec_0.3"] >= 0.5))

    def stats(rows):
        st = per_image(pd.DataFrame(rows), gt, stems, ncopy)[0]
        return np.stack([st["tp"], st["fp"], st["fn"], st["iou"]], 1).astype(float)

    base = stats(base_rows)
    res["base_pq"] = float(pq(*base.sum(0)))
    rng = np.random.default_rng(0)
    idx = [rng.choice(len(stems), len(stems), replace=True) for _ in range(a_.reps)]
    res["arms"] = {}
    best = None
    for k, rows in arms.items():
        A = stats(rows)
        d = float(pq(*A.sum(0)) - res["base_pq"])
        boots = np.array([pq(*A[i].sum(0)) - pq(*base[i].sum(0)) for i in idx])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        tot = A.sum(0)
        res["arms"][k] = dict(pq=float(pq(*tot)), delta=d, ci95=[float(lo), float(hi)],
                              tp=int(tot[0]), fp=int(tot[1]), fn=int(tot[2]), sq=float(tot[3] / tot[0]))
        if best is None or d > res["arms"][best]["delta"]:
            best = k
        print(f"  {k:9s} PQ {pq(*tot):.4f}  delta {d:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
              f"TP {int(tot[0])} FP {int(tot[1])} FN {int(tot[2])}  SQ {tot[3] / tot[0]:.4f}", flush=True)
    b = res["arms"][best]
    res["gate_A"] = dict(best=best, delta=b["delta"], ci95=b["ci95"],
                         passed=bool(b["delta"] > 0.002 and b["ci95"][0] > 0))
    print(json.dumps({k: res[k] for k in ("mechanism", "gate_P", "base_pq", "gate_A")}, indent=1))
    if not a_.smoke:
        m.to_parquet(OUT / f"e217b_mechanism_f{F_}.parquet", index=False)
    if not a_.smoke:
        json.dump(res, open(OUT / f"e217b_soft_extension_f{F_}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
