"""What ARE the hallucination false positives? (chatgpt_science S001, honest version)

E117 counted 476 hallucination FPs (no overlap with any ground-truth copy) among
the 1359 FPs of the production policy, with an in-sample selector. The
literature (Diercke 2024, SWEFil 2026, Qu 2005) names sunspots / penumbral
fibrils, diffuse dark clouds and limb artefacts as the classic confusers, and
SWEFil reports that a separate sunspot class raised precision. Our selector has
no compactness, dark-core or annular-gradient feature, so a sunspot-shaped FP
population would be invisible to it.

This script takes the deployment-faithful labelled pool (E115b: shrink 0.05)
with the fully out-of-fold selector scores (E115b p_oof), applies the production
policy (conf >= 0.40, p_oof >= 0.60) and, for every kept candidate:

  max_iou      best IoU against any GT copy of the image (any annotator)
  class        TP (max_iou > 0.5) / partial_pair (0.05 < max_iou <= 0.5) /
               hallucination (max_iou <= 0.05)
  shape        solidity (area / convex-hull area), compactness P^2/(4 pi A),
               bbox_aspect, eccentricity, area, skel_len, width_proxy
  photometry   from the JPEG: fg median, fg p05, ring median (15 px annulus),
               contrast (ring - fg)/ring, dark_core = fg_p05 / ring_median
  position     r_norm, abs_lat
  provenance   from_det, from_sem, sem_prob_mean, p_oof

and writes:
  outputs/eda/fp_taxonomy_honest.parquet          one row per kept candidate
  outputs/eda/fp_taxonomy_honest_summary.txt      medians per class, r-band and
                                                  compactness bins; the fraction
                                                  of hallucinations that are
                                                  sunspot-shaped by a reported,
                                                  not tuned, definition
  outputs/eda/fp_gallery_darkcompact.png          36 hallucinations sorted by
                                                  solidity x (1 - dark_core)
  outputs/eda/fp_gallery_random.png               36 random hallucinations
  outputs/eda/fp_gallery_partial.png              36 random partial pairs

No decision is taken here; the gate for a sunspot feature/class is prevalence
(>= 15% of hallucinations sunspot-shaped, per the roadmap) read from the table.

Usage: python src/eda/fp_hallucination_taxonomy.py
"""

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
CONF, PCUT = 0.40, 0.60
RING = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))


def log(*a):
    print(*a, flush=True)


def shape_feats(m):
    m8 = m.astype(np.uint8)
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return dict(solidity=np.nan, compactness=np.nan)
    c = max(cnts, key=cv2.contourArea)
    area = float(m.sum())
    hull = cv2.convexHull(c)
    ha = max(cv2.contourArea(hull), 1.0)
    per = max(cv2.arcLength(c, True), 1.0)
    return dict(solidity=area / ha, compactness=per * per / (4 * np.pi * max(area, 1.0)))


def photometry(img, m):
    fg = img[m]
    ring = cv2.dilate(m.astype(np.uint8), RING).astype(bool) & ~m
    rg = img[ring]
    if len(fg) == 0 or len(rg) == 0:
        return dict(fg_med=np.nan, fg_p05=np.nan, ring_med=np.nan, contrast=np.nan, dark_core=np.nan)
    fm, rm, p05 = float(np.median(fg)), float(np.median(rg)), float(np.percentile(fg, 5))
    return dict(fg_med=fm, fg_p05=p05, ring_med=rm, contrast=(rm - fm) / max(rm, 1.0), dark_core=p05 / max(rm, 1.0))


def main():
    cand = pd.read_parquet(OUT / "e115b_labelled_pool.parquet")
    poof = pd.read_parquet(OUT / "e115b_p_oof.parquet")
    cand = cand.merge(poof, on="cand_id", how="inner")
    keep = cand[(cand["conf"] >= CONF) & (cand["p_oof"] >= PCUT)].copy()
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    log(f"kept {len(keep)} candidates over {keep['stem'].nunique()} images at ({CONF}, {PCUT})")

    rows, crops = [], {}
    for i, (stem, g) in enumerate(keep.groupby("stem")):
        img = cv2.imread(str(IMG / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        gts = [decode_rle(r).astype(bool) for r in gt.loc[gt["stem"] == stem, "segmentation_rle"]]
        for r in g.itertuples():
            m = decode_rle(r.rle).astype(bool)
            best = 0.0
            for gm in gts:
                inter = int((m & gm).sum())
                if inter:
                    best = max(best, inter / int((m | gm).sum()))
            cls = "TP" if best > 0.5 else ("partial_pair" if best > 0.05 else "hallucination")
            d = dict(stem=stem, cand_id=r.cand_id, cls=cls, max_iou=best, p_oof=r.p_oof, conf=r.conf,
                     area=r.area, bbox_aspect=r.bbox_aspect, eccentricity=r.eccentricity,
                     skel_len=r.skel_len, width_proxy=r.width_proxy, boundary_complexity=r.boundary_complexity,
                     r_norm=r.r_norm, abs_lat=r.abs_lat, from_det=r.from_det, from_sem=r.from_sem,
                     sem_prob_mean=r.sem_prob_mean, sem_contrast=r.sem_contrast, cy=r.cy, cx=r.cx)
            d.update(shape_feats(m))
            d.update(photometry(img, m))
            rows.append(d)
            if cls != "TP":
                ys, xs = np.nonzero(m)
                y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
                cy_, cx_ = (y0 + y1) // 2, (x0 + x1) // 2
                h = max(64, (max(y1 - y0, x1 - x0) // 2) + 24)
                ya, yb, xa, xb = max(cy_ - h, 0), min(cy_ + h, 2048), max(cx_ - h, 0), min(cx_ + h, 2048)
                crop = cv2.cvtColor(img[ya:yb, xa:xb], cv2.COLOR_GRAY2BGR)
                cn, _ = cv2.findContours(m[ya:yb, xa:xb].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(crop, cn, -1, (0, 0, 255), 1)
                crops[r.cand_id] = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)
        if i % 100 == 0:
            log(f"  {i}/{keep['stem'].nunique()} images")
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "fp_taxonomy_honest.parquet", index=False)

    lines = [f"kept {len(df)}: " + ", ".join(f"{k} {v}" for k, v in df["cls"].value_counts().items())]
    feats = ["area", "bbox_aspect", "eccentricity", "solidity", "compactness", "skel_len", "width_proxy",
             "contrast", "dark_core", "fg_p05", "sem_prob_mean", "p_oof", "r_norm", "abs_lat", "from_det", "from_sem"]
    lines.append("\nmedians per class")
    lines.append(df.groupby("cls")[feats].median().round(3).T.to_string())
    h = df[df["cls"] == "hallucination"].copy()
    h["r_band"] = pd.cut(h["r_norm"], [0, 0.5, 0.8, 0.9, 1.01], labels=["<0.5", "0.5-0.8", "0.8-0.9", ">0.9"])
    lines.append("\nhallucinations by r band: " + ", ".join(f"{k} {v}" for k, v in h["r_band"].value_counts().sort_index().items()))
    # sunspot-shaped by a reported definition: compact (solidity >= 0.8 and bbox_aspect <= 2)
    # AND darker core than any typical filament (dark_core <= the TP 10th percentile)
    tp = df[df["cls"] == "TP"]
    dc10 = float(tp["dark_core"].quantile(0.10))
    sun = h[(h["solidity"] >= 0.8) & (h["bbox_aspect"] <= 2.0) & (h["dark_core"] <= dc10)]
    comp = h[(h["solidity"] >= 0.8) & (h["bbox_aspect"] <= 2.0)]
    lines.append(f"\nTP dark_core p10 = {dc10:.3f}")
    lines.append(f"hallucinations compact (solidity>=0.8 & aspect<=2): {len(comp)}/{len(h)} = {len(comp)/max(len(h),1):.1%}")
    lines.append(f"hallucinations compact AND darker than TP p10: {len(sun)}/{len(h)} = {len(sun)/max(len(h),1):.1%}  <- sunspot-shaped proxy")
    lines.append(f"TPs that would trip the same proxy: {int(((tp['solidity'] >= 0.8) & (tp['bbox_aspect'] <= 2.0) & (tp['dark_core'] <= dc10)).sum())}/{len(tp)}")
    lines.append("\nhallucinations by solidity bin: " + ", ".join(
        f"{k} {v}" for k, v in pd.cut(h["solidity"], [0, 0.5, 0.65, 0.8, 0.9, 1.0]).value_counts().sort_index().items()))
    lines.append("hallucinations by dark_core bin: " + ", ".join(
        f"{k} {v}" for k, v in pd.cut(h["dark_core"], [0, 0.5, 0.65, 0.8, 0.9, 2.0]).value_counts().sort_index().items()))
    lines.append("hallucinations by provenance: " + ", ".join(
        f"{k} {v}" for k, v in h.groupby(["from_det", "from_sem"]).size().items()))
    txt = "\n".join(lines)
    (OUT / "fp_taxonomy_honest_summary.txt").write_text(txt, encoding="utf-8")
    log(txt)

    def gallery(ids, name, title):
        tiles = []
        for cid in ids[:36]:
            t = crops[cid].copy()
            r = df[df["cand_id"] == cid].iloc[0]
            cv2.putText(t, f"r{r.r_norm:.2f} s{r.solidity:.2f} d{r.dark_core:.2f}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
            cv2.putText(t, f"p{r.p_oof:.2f} a{r.bbox_aspect:.1f}", (2, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
            tiles.append(t)
        while len(tiles) < 36:
            tiles.append(np.zeros((128, 128, 3), np.uint8))
        grid = np.vstack([np.hstack(tiles[i * 6:(i + 1) * 6]) for i in range(6)])
        canvas = np.zeros((grid.shape[0] + 24, grid.shape[1], 3), np.uint8)
        canvas[24:] = grid
        cv2.putText(canvas, title, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(str(OUT / name), canvas)

    h["score"] = h["solidity"] * (1 - h["dark_core"].clip(0, 1))
    gallery(h.sort_values("score", ascending=False)["cand_id"].tolist(), "fp_gallery_darkcompact.png",
            "hallucination FPs, most compact & darkest first  (r, solidity, dark_core / p_oof, aspect)")
    rng = np.random.default_rng(0)
    gallery(list(rng.permutation(h["cand_id"].tolist())), "fp_gallery_random.png", "hallucination FPs, random 36")
    pp = df[df["cls"] == "partial_pair"]
    gallery(list(rng.permutation(pp["cand_id"].tolist())), "fp_gallery_partial.png", "partial-pair FPs (overlap GT, IoU <= 0.5), random 36")
    log("galleries written")


if __name__ == "__main__":
    main()
