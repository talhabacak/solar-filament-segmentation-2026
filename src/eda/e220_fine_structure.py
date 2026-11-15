"""E220 -- fine-scale structure: how much of the annotated lateral structure (barbs, side branches)
do our masks recover, compared with a second annotator drawing the same filament?

The organisers list fine-scale structure as the first challenge of the task (Overview, "What are
the main challenges"; repeated in discussion 744765). This measures it on the annotators' own terms
using the manually drawn spines (organiser-sanctioned metadata, discussion 736543).

Lateral structure of a ground-truth copy G with spine polyline S:
  d(p)      distance from pixel p to the rasterised spine
  h         body half-width = 2 * median(d) over G (for a ribbon of half-width h around a line, d is
            uniform on [0, h], so its median is h/2; barbs are a minority and barely move it)
  L         {p in G : d(p) > h + 2 px}, excluding end caps (pixels whose nearest spine pixel lies
            within 3 px of a spine endpoint) -- protrusions beyond the body, i.e. barbs and side
            branches, not the drawn width and not extension past the spine ends
  body      G minus L
Recovered share of L and of the body by
  ours      the kept out-of-fold candidate (sub036: conf >= 0.40, bag5 p >= 0.60) with the highest
            IoU to G, reported for matched copies (IoU > 0.5)
  human     a sibling copy drawn by another annotator of the same image (IoU(G, G_B) > 0.5)
Only copies with at least 20 lateral pixels enter the recall statistics.

Also writes docs/report/figures/fig4_fine_structure.png: barb-rich matched copies with the image,
the annotator's outline, our outline and the lateral pixels.

Usage: python src/eda/e220_fine_structure.py
Output: outputs/eda/e220_fine_structure.json, outputs/eda/e220_fine_structure.parquet
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from filament.rle import decode_rle  # noqa: E402

OUT = ROOT / "outputs" / "eda"
IMG = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
ANN = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
PAD = 6


def lateral(G, spine):
    """Lateral (barb) pixels of mask G given its spine polyline, computed on G's padded box."""
    ys, xs = np.nonzero(G)
    y0, y1 = max(0, ys.min() - PAD), min(G.shape[0], ys.max() + PAD + 1)
    x0, x1 = max(0, xs.min() - PAD), min(G.shape[1], xs.max() + PAD + 1)
    g = G[y0:y1, x0:x1]
    line = np.full(g.shape, 255, np.uint8)
    pts = np.round(spine - [x0, y0]).astype(np.int32)
    cv2.polylines(line, [pts], False, 0, thickness=1)
    if (line == 0).sum() == 0:
        return None
    dist, lab = cv2.distanceTransformWithLabels(line, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    zy, zx = np.nonzero(line == 0)
    # label k (1-based) -> coordinates of the k-th zero pixel in scan order
    order = np.zeros(lab.max() + 1, np.int64)
    order[lab[zy, zx]] = np.arange(len(zy))
    ends = np.array([pts[0][::-1], pts[-1][::-1]])
    near_y, near_x = zy[order[lab]], zx[order[lab]]
    cap = np.zeros(g.shape, bool)
    for ey, ex in ends:
        cap |= np.hypot(near_y - ey, near_x - ex) <= 3
    d = dist[g]
    h = 2.0 * np.median(d)
    L = g & (dist > h + 2) & ~cap
    return (y0, y1, x0, x1), g, L, h, dist


def thin_protrusions(g):
    """Barbs as the parts of the mask thinner than its body: the residue of a morphological opening
    with a disc of radius 0.35 x the body width (body width = 2 x median distance-transform value on
    the mask's own skeleton), minus end caps (within 1.5 body widths of a skeleton endpoint), so the
    tapering ends are not counted. Independent of where the annotator placed the spine."""
    from skimage.morphology import skeletonize
    u = g.astype(np.uint8)
    dt = cv2.distanceTransform(u, cv2.DIST_L2, 5)
    sk = skeletonize(g)
    if sk.sum() < 3:
        return None
    w = 2.0 * float(np.median(dt[sk]))
    rho = max(2, int(round(0.35 * w)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rho + 1, 2 * rho + 1))
    opened = cv2.morphologyEx(u, cv2.MORPH_OPEN, k).astype(bool)
    L = g & ~opened
    ys, xs = np.nonzero(sk)
    deg = cv2.filter2D(sk.astype(np.uint8), -1, np.ones((3, 3), np.float32)) - 1
    ends = [(y, x) for y, x in zip(ys, xs) if deg[y, x] == 1]
    if ends:
        yy, xx = np.mgrid[0:g.shape[0], 0:g.shape[1]]
        for ey, ex in ends:
            L &= np.hypot(yy - ey, xx - ex) > 1.5 * w
    return L


def main():
    d = json.load(open(ANN, encoding="utf-8"))
    spine_of = {a["id"]: np.asarray(a.get("spine", []), np.float32).reshape(-1, 2) for a in d["annotations"]}
    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    pool = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet", columns=["stem", "cand_id", "rle", "conf"])
    pool = pool.merge(pd.read_parquet(OUT / "bag5_oof_scores.parquet"), on="cand_id")
    keep = pool[(pool["conf"] >= 0.40) & (pool["p"] >= 0.60)]

    rows, examples = [], []
    for n, (stem, gg) in enumerate(gt.groupby("stem")):
        kc = keep[keep["stem"] == stem]
        km = [decode_rle(r).astype(bool) for r in kc["rle"]]
        gm = {f: decode_rle(r).astype(bool) for f, r in zip(gg["filament_id"], gg["segmentation_rle"])}

        def bb(m):
            yy, xx = np.nonzero(m)
            return (yy.min(), yy.max() + 1, xx.min(), xx.max() + 1) if len(yy) else None
        kb = [bb(m) for m in km]
        gb = {f: bb(m) for f, m in gm.items()}

        def overl(a, b):
            return a is not None and b is not None and a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]

        def iou_box(A, B, a, b):
            y0_, y1_, x0_, x1_ = min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3])
            Aa, Bb = A[y0_:y1_, x0_:x1_], B[y0_:y1_, x0_:x1_]
            i = (Aa & Bb).sum()
            return i / max((Aa | Bb).sum(), 1)
        ent = dict(zip(gg["filament_id"], gg["entry_id"]))
        for f, aid in zip(gg["filament_id"], gg["ann_id"]):
            sp = spine_of.get(aid)
            G = gm[f]
            if sp is None or len(sp) < 2 or not G.any():
                continue
            r = lateral(G, sp)
            if r is None:
                continue
            (y0, y1, x0, x1), g, L_spine, h, dist = r
            Lt = thin_protrusions(g)
            if Lt is None:
                continue
            # a barb is thinner than the body AND leaves the spine by more than half a body width:
            # the opening alone also removes thin necks of the body itself, which the spine runs
            # through; the spine distance alone counts wide body parts when the spine is drawn
            # off-centre. The intersection keeps only side protrusions.
            L = Lt & (dist > h + 2)
            body = g & ~L
            rec = dict(stem=stem, fid=f, area=int(g.sum()), lateral_px=int(L.sum()), spine_lateral_px=int(L_spine.sum()),
                       half_width=float(h))
            # ours: best kept candidate
            best, bi = 0.0, -1
            for i, m in enumerate(km):
                if not overl(kb[i], gb[f]):
                    continue
                iou = iou_box(m, G, kb[i], gb[f])
                if iou > best:
                    best, bi = iou, i
            rec["ours_iou"] = float(best)
            if bi >= 0:
                P = km[bi][y0:y1, x0:x1]
                rec["ours_lat"] = float((L & P).sum() / max(L.sum(), 1))
                rec["ours_body"] = float((body & P).sum() / max(body.sum(), 1))
            # human: sibling copies of other annotators
            sib = [s for s in gm if ent[s] != ent[f] and overl(gb[s], gb[f]) and iou_box(gm[s], G, gb[s], gb[f]) > 0.5]
            if sib:
                hl, hb = [], []
                for s in sib:
                    B = gm[s][y0:y1, x0:x1]
                    hl.append((L & B).sum() / max(L.sum(), 1))
                    hb.append((body & B).sum() / max(body.sum(), 1))
                rec["human_lat"], rec["human_body"] = float(np.mean(hl)), float(np.mean(hb))
            rows.append(rec)
            if best > 0.5 and L.sum() >= 150:
                examples.append((int(L.sum()), stem, f, (y0, y1, x0, x1), g, L, km[bi][y0:y1, x0:x1], sp))
        if n % 100 == 0:
            print(f"  {n}/707", flush=True)

    r = pd.DataFrame(rows)
    big = r[r["lateral_px"] >= 20]
    m = big[big["ours_iou"] > 0.5]
    both = big[(big["ours_iou"] > 0.5) & big["human_lat"].notna()]
    res = dict(
        definition=("barbs: pixels both thinner than the body (residue of an opening at 0.35 x body "
                    "width, end caps excluded) and farther from the annotated spine than the body "
                    "half-width + 2 px"),
        spine_based_lateral_share=float(r["spine_lateral_px"].sum() / r["area"].sum()),
        copies=len(r), copies_with_lateral=len(big),
        lateral_share_of_area=float(r["lateral_px"].sum() / r["area"].sum()),
        lateral_share_median_per_copy=float((r["lateral_px"] / r["area"]).median()),
        matched=dict(n=len(m), ours_lateral=float(m["ours_lat"].mean()), ours_body=float(m["ours_body"].mean())),
        paired=dict(n=len(both), ours_lateral=float(both["ours_lat"].mean()), ours_body=float(both["ours_body"].mean()),
                    human_lateral=float(both["human_lat"].mean()), human_body=float(both["human_body"].mean())),
        all_with_sibling=dict(n=int(big["human_lat"].notna().sum()), human_lateral=float(big["human_lat"].mean()),
                              human_body=float(big["human_body"].mean())),
    )
    print(json.dumps(res, indent=1))
    r.to_parquet(OUT / "e220_fine_structure.parquet", index=False)
    json.dump(res, open(OUT / "e220_fine_structure.json", "w"), indent=1)

    # figure: six barb-rich matched copies
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    examples.sort(key=lambda e: -e[0])
    picked, seen = [], set()
    for e in examples:
        if e[1] in seen:
            continue
        picked.append(e)
        seen.add(e[1])
        if len(picked) == 6:
            break
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 7.0))
    for ax, (_, stem, f, (y0, y1, x0, x1), g, L, P, sp) in zip(axes.ravel(), picked):
        im = cv2.imread(str(IMG / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)[y0:y1, x0:x1]
        ax.imshow(im, cmap="gray")
        ov = np.zeros((*g.shape, 4))
        ov[L] = (0.10, 0.75, 0.95, 0.55)
        ax.imshow(ov)
        ax.contour(g, levels=[0.5], colors="#2ca02c", linewidths=1.2)
        ax.contour(P, levels=[0.5], colors="#d62728", linewidths=1.2, linestyles="--")
        ax.plot(sp[:, 0] - x0, sp[:, 1] - y0, color="#f1c40f", lw=0.9)
        lat_rec = (L & P).sum() / max(L.sum(), 1)
        ax.set_title(f"{stem}  barbs recovered {lat_rec:.0%}", fontsize=7.5)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Barbs and side branches (shaded: thinner than the body and away from the spine): "
                 "annotator outline (solid), our mask (dashed), spine (thin line)", fontsize=9)
    fig.tight_layout()
    out = ROOT / "docs" / "report" / "figures" / "fig4_fine_structure.png"
    fig.savefig(out, dpi=200)
    print("figure ->", out)


if __name__ == "__main__":
    main()
