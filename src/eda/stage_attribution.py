"""E171 -- which encoder stage carries which filament, and does it depend on type or position?

Talha, 2026-09-12: "which stages is the model learning from most, and is there a relation
between the stages and filament types and positions?"

The production U-Net is EfficientNet-B3 at 1536 with six encoder outputs, so each skip carries
a different spatial scale:

  stage 0  1536 px  the input itself (identity skip)
  stage 1   768 px   40 ch
  stage 2   384 px   32 ch
  stage 3   192 px   48 ch
  stage 4    96 px  136 ch
  stage 5    48 px  384 ch   (the bottleneck the decoder starts from)

The attribution is an intervention, not a saliency heuristic: run the model normally, then run
it again with ONE skip replaced by zeros, and measure what the prediction loses. Zeroing the
bottleneck (stage 5) removes the decoder's input entirely, so it is reported but not
comparable; stages 1-4 are the readable ones.

Read per GROUND-TRUTH COPY rather than per pixel, because the metric is instance-level: for
each copy, the best IoU a connected component of the thresholded field reaches, before and
after the ablation. Copies are then stratified by the properties Talha asked about -- size,
width, contrast, radius, latitude -- and by chirality class, so "stage 2 carries the thin ones"
becomes a number instead of an intuition.

Usage: python src/eda/stage_attribution.py [--fold 0] [--limit 60]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/eda", "src/models"):
    sys.path.insert(0, str(ROOT / p))
from filament.rle import decode_rle  # noqa: E402
from filament.pq import _entry_iou_matrix  # noqa: E402

OUT = ROOT / "outputs" / "eda"
IMG = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
C, R = 1023.5, 904.1
THR = 0.2          # the production extraction threshold
MIN_AREA = 256


def log(*a):
    print(*a, flush=True)


def field(model, img_small, zero_stage=None, size=1536):
    """One forward pass at the training size, optionally with skip `zero_stage` zeroed."""
    x = torch.from_numpy(np.stack([img_small] * 3)).float()[None].cuda() / 255.0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        feats = model.encoder(x)
        if zero_stage is not None:
            feats = list(feats)
            feats[zero_stage] = torch.zeros_like(feats[zero_stage])
        out = model.decoder(feats)
        out = model.segmentation_head(out)
    p = torch.sigmoid(out.float())[0, 0].cpu().numpy()
    return cv2.resize(p, (2048, 2048), interpolation=cv2.INTER_LINEAR)


def components(p):
    m = (p >= THR).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    return [(lab == k).astype(np.uint8) for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= MIN_AREA]


def best_ious(G, comps):
    if not comps:
        return np.zeros(len(G))
    P = np.stack(comps)
    return _entry_iou_matrix(G, P).max(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--limit", type=int, default=60, help="images (each costs 6 forward passes)")
    ap.add_argument("--weights", default="outputs/runs_semantic/sem_b3_f0_1536_cldice/best.pt")
    a = ap.parse_args()
    from semantic_candidates import load_model
    model, imgsz, channels = load_model(str(ROOT / a.weights))
    model = model.cuda().eval()
    n_stages = len(model.encoder(torch.zeros(1, 3, imgsz, imgsz).cuda()))
    log(f"{a.weights}: {n_stages} encoder outputs at {imgsz}")

    folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
    folds["stem"] = folds["file_name"].str.replace(".jpeg", "", regex=False)
    stems = sorted(folds.loc[folds["fold"] == a.fold, "stem"].unique())
    step = max(len(stems) // a.limit, 1)
    stems = stems[::step][: a.limit]
    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    cls_col = next((c for c in ("category_id", "category", "chirality") if c in gt.columns), None)
    K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))

    rows = []
    for n, stem in enumerate(stems):
        gg = gt[gt["stem"] == stem]
        if not len(gg):
            continue
        img = cv2.imread(str(IMG / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        small = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
        imf = img.astype(np.float32)
        G = np.stack([decode_rle(r).astype(np.uint8) for r in gg["segmentation_rle"]])
        base = best_ious(G, components(field(model, small, None, imgsz)))
        abl = {}
        for s in range(n_stages):
            abl[s] = best_ious(G, components(field(model, small, s, imgsz)))
        for gi in range(len(G)):
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
            rec = dict(stem=stem, fid=gg["filament_id"].iloc[gi], base=float(base[gi]),
                       log_area=float(np.log10(max(area, 1))), width=float(area / skel),
                       skel_len=float(skel), contrast=float((bg - fg) / max(bg, 1.0)),
                       r_norm=float(np.hypot(X_, Y_)),
                       abs_lat=float(abs(np.degrees(np.arcsin(np.clip(Y_, -1, 1))))))
            if cls_col:
                rec["cls"] = str(gg[cls_col].iloc[gi])
            for s in range(n_stages):
                rec[f"d{s}"] = float(abl[s][gi] - base[gi])
            rows.append(rec)
        if (n + 1) % 10 == 0:
            log(f"  {n + 1}/{len(stems)} images, {len(rows)} copies")
    d = pd.DataFrame(rows)
    d.to_parquet(OUT / f"stage_attribution_f{a.fold}.parquet", index=False)
    dcols = [f"d{s}" for s in range(n_stages)]
    log(f"\n{len(d)} GT copies, mean best IoU {d['base'].mean():.4f}")
    log("\nmean IoU lost when a stage's skip is zeroed (more negative = the model needs it):")
    log(d[dcols].mean().round(4).to_string())
    log("\nshare of copies that fall below IoU 0.5 when the stage is zeroed (they were above):")
    was = d["base"] > 0.5
    for s in range(n_stages):
        lost = was & ((d["base"] + d[f"d{s}"]) <= 0.5)
        log(f"  stage {s}: {int(lost.sum()):4d} / {int(was.sum())} ({lost.sum() / max(was.sum(), 1):.1%})")
    for col, bins, name in ((("width"), [0, 8, 14, 22, 999], "drawn width (px)"),
                            (("log_area"), [0, 2.5, 3, 3.5, 9], "log10 area"),
                            (("contrast"), [-1, 0.05, 0.1, 0.15, 1], "contrast"),
                            (("r_norm"), [0, 0.5, 0.7, 0.8, 2], "radius"),
                            (("abs_lat"), [0, 15, 30, 45, 90], "|latitude|")):
        g = d.assign(b=pd.cut(d[col], bins)).groupby("b", observed=True)[dcols].mean()
        cnt = d.assign(b=pd.cut(d[col], bins)).groupby("b", observed=True).size()
        log(f"\nby {name} (mean IoU lost per stage; n on the right):")
        log(pd.concat([g.round(3), cnt.rename("n")], axis=1).to_string())
    if "cls" in d.columns:
        g = d.groupby("cls")[dcols].mean()
        log("\nby chirality class:")
        log(pd.concat([g.round(3), d.groupby("cls").size().rename("n")], axis=1).to_string())
    json.dump(dict(n=len(d), base_mean=float(d["base"].mean()),
                   mean_loss={f"stage{s}": float(d[f"d{s}"].mean()) for s in range(n_stages)}),
              open(OUT / f"stage_attribution_f{a.fold}.json", "w"), indent=1)
    log("\nSTAGE_ATTR_DONE")


if __name__ == "__main__":
    main()
