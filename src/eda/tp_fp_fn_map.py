"""TP / FN / FP map in disc coordinates -- the production policy on the honest
out-of-fold pool, drawn like outputs/eda/morph_centroid_heatmap.png.

Panels (all in (x - cx)/R, (y - cy)/R with the dashed unit circle, N up):
  TP   centroids of GT rows matched at IoU > 0.5 by an emitted candidate
  FN   centroids of GT rows not matched
  FP   centroids of emitted candidates that match no GT row in any copy
  miss-rate  FN / (TP + FN) per spatial bin (bins with >= 4 rows), diverging
             colormap centred on the global miss rate -- where the misses
             concentrate, independent of where filaments are dense
plus radial profiles: TP/FN/FP counts per radius band, miss rate and FP share.

Population: GT over all folds_v1 stems (fixed scope, ledger §12b rule);
candidates = fused_pool_all_blob5 with the E115b honest p_oof; policy
conf >= 0.40, p_oof >= 0.60 (sub025's). Centroids from the masks themselves.

Usage: python src/eda/tp_fp_fn_map.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from filament.pq import score
from filament.rle import decode_rle

OUT = ROOT / "outputs" / "eda"
DISK_C, DISK_R = 1023.5, 900.0
NB = 60


def log(*a):
    print(*a, flush=True)


def centroids(rles):
    xs, ys = [], []
    for r in rles:
        m = decode_rle(r)
        yy, xx = np.nonzero(m)
        if len(xx):
            xs.append((xx.mean() - DISK_C) / DISK_R)
            ys.append((yy.mean() - DISK_C) / DISK_R)
        else:
            xs.append(np.nan); ys.append(np.nan)
    return np.array(xs), np.array(ys)


def main():
    cand = pd.read_parquet(OUT / "e115b_labelled_pool.parquet")
    po = pd.read_parquet(OUT / "e115b_p_oof.parquet").set_index("cand_id")["p_oof"]
    cand["p_oof"] = cand["cand_id"].map(po)
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    gt = gt[gt["stem"].isin(set(folds["file_name"].str.replace(".jpeg", "", regex=False)))].copy()

    emitted = cand[(cand["conf"] >= 0.40) & (cand["p_oof"] >= 0.60)]
    pred = emitted[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
    res = score(gt[["filament_id", "segmentation_rle"]], pred, return_details=True)
    m = res["matches"]
    log(f"production policy: PQ {res['pq']:.4f} TP {res['tp']} FP {res['fp']} FN {res['fn']}; emitted {len(emitted)}")
    hit_gt = set(m["gt_filament_id"])
    hit_pred = set(m["pred_filament_id"])
    tp_rows = gt[gt["filament_id"].isin(hit_gt)]
    fn_rows = gt[~gt["filament_id"].isin(hit_gt)]
    fp_cands = emitted[~emitted["cand_id"].isin(hit_pred)]
    log(f"rows: TP {len(tp_rows)}, FN {len(fn_rows)}; FP candidates (no match in any copy) {len(fp_cands)}")

    log("centroids ...")
    tx, ty = centroids(tp_rows["segmentation_rle"])
    fx, fy = centroids(fn_rows["segmentation_rle"])
    px, py = centroids(fp_cands["rle"])
    edges = np.linspace(-1.02, 1.02, NB + 1)
    H = lambda x, y: np.histogram2d(y, x, bins=[edges, edges])[0]
    Htp, Hfn, Hfp = H(tx, ty), H(fx, fy), H(px, py)
    tot = Htp + Hfn
    miss = np.where(tot >= 4, Hfn / np.maximum(tot, 1), np.nan)
    gmiss = len(fn_rows) / (len(tp_rows) + len(fn_rows))

    fig = plt.figure(figsize=(16, 13), dpi=140)
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.55], hspace=0.28, wspace=0.25)
    panels = [("TP: matched GT rows", Htp, "Blues", None), ("FN: missed GT rows", Hfn, "Oranges", None),
              ("FP: emitted, no match in any copy", Hfp, "Reds", None),
              (f"miss rate FN/(TP+FN), bins with >= 4 rows (global {gmiss:.2f})", miss, "RdBu_r", (0.0, 1.0))]
    for i, (title, Hh, cmap, vr) in enumerate(panels):
        ax = fig.add_subplot(gs[i // 2, (i % 2) * 2:(i % 2) * 2 + 2])
        im = ax.imshow(Hh, origin="lower", extent=[-1.02, 1.02, -1.02, 1.02], cmap=cmap,
                       vmin=(vr[0] if vr else 0), vmax=(vr[1] if vr else None), interpolation="nearest")
        th = np.linspace(0, 2 * np.pi, 400)
        ax.plot(np.cos(th), np.sin(th), "k--", lw=1)
        for rr in (0.8, 0.9):
            ax.plot(rr * np.cos(th), rr * np.sin(th), "k:", lw=0.6, alpha=0.6)
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(1.05, -1.05)     # image coords, N up after invert
        ax.set_xlabel("(x - disk_cx) / R"); ax.set_ylabel("(y - disk_cy) / R")
        n = int(np.nansum(Hh)) if vr is None else int(tot[tot >= 4].sum())
        ax.set_title(f"{title}  (n={n})", fontsize=11)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("per bin" if vr is None else "miss rate")
        ax.set_aspect("equal")

    # radial profiles
    bands = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    rt, rf, rp = np.hypot(tx, ty), np.hypot(fx, fy), np.hypot(px, py)
    labels = [f"({a},{b}]" for a, b in bands]
    cnt = {k: [] for k in ("TP", "FN", "FP")}
    for a, b in bands:
        cnt["TP"].append(int(((rt > a) & (rt <= b)).sum()))
        cnt["FN"].append(int(((rf > a) & (rf <= b)).sum()))
        cnt["FP"].append(int(((rp > a) & (rp <= b)).sum()))
    ax = fig.add_subplot(gs[2, 0:2])
    xg = np.arange(len(bands)); w = 0.27
    ax.bar(xg - w, cnt["TP"], w, color="#3b7dd8", label="TP")
    ax.bar(xg, cnt["FN"], w, color="#f28e2b", label="FN")
    ax.bar(xg + w, cnt["FP"], w, color="#d62728", label="FP")
    ax.set_xticks(xg); ax.set_xticklabels(labels, fontsize=8); ax.set_ylabel("rows / candidates")
    ax.set_title("counts by radius band", fontsize=11); ax.legend(fontsize=8)
    ax2 = fig.add_subplot(gs[2, 2:4])
    mr = [f / max(t + f, 1) for t, f in zip(cnt["TP"], cnt["FN"])]
    fs = [p / max(t + p, 1) for t, p in zip(cnt["TP"], cnt["FP"])]
    ax2.plot(xg, mr, "o-", color="#f28e2b", label="miss rate FN/(TP+FN)")
    ax2.plot(xg, fs, "s-", color="#d62728", label="FP share FP/(TP+FP)")
    ax2.set_xticks(xg); ax2.set_xticklabels(labels, fontsize=8); ax2.set_ylim(0, 1); ax2.grid(alpha=0.3)
    ax2.set_title("miss rate and FP share by radius band", fontsize=11); ax2.legend(fontsize=8)
    for x_, a_, b_ in zip(xg, mr, fs):
        ax2.annotate(f"{a_:.2f}", (x_, a_), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
        ax2.annotate(f"{b_:.2f}", (x_, b_), textcoords="offset points", xytext=(0, -11), ha="center", fontsize=7)

    fig.suptitle(f"sub025 policy on the honest out-of-fold pool -- PQ {res['pq']:.4f}, TP {res['tp']} FP {res['fp']} FN {res['fn']}  "
                 f"(GT rows {len(gt)}, emitted {len(emitted)})", fontsize=12)
    out = OUT / "tp_fp_fn_map.png"
    fig.savefig(out, bbox_inches="tight")
    pd.DataFrame(dict(band=labels, **cnt, miss_rate=mr, fp_share=fs)).to_csv(OUT / "tp_fp_fn_map_bands.csv", index=False)
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
