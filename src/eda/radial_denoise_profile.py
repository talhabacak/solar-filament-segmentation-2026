"""Where, by radius, does the denoised model beat the baseline -- and is the
model's size error systematic?

Two questions from Talha, both answerable from maps already on disk
(fold0_probs_cldice = production, fold0_probs_dn = bilateral-denoised input).

1. The denoised model raised coverage of never-proposed rows +0.0226 but lost
   net reach (-21) because it blurred sharp boundaries. If faint rows live near
   the limb (limb darkening lowers contrast) and sharp ones near the centre,
   the two effects may be spatially separable: use the denoised map OUTSIDE a
   radius r* and the baseline inside. This measures the per-instance gate
   quantities as a function of r_norm for both maps, then writes radially
   gated maps for a grid of r* so the real gate can score each.

2. "Too big in some places, too small in others" -- is the size error
   systematic in radius or in filament width? For every GT instance with a
   matching component, log2(pred_area / gt_area) is recorded and binned.

Usage: python src/eda/radial_denoise_profile.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "eda"))
from filament.rle import decode_rle
from gate_instance_quality import cc, THR_PROD, HI

OUT = ROOT / "outputs" / "eda"
SEM = ROOT / "outputs" / "semantic"
DISK_C, DISK_R = 1023.5, 900.0
R_GRID = (0.5, 0.6, 0.7, 0.8)


def log(*a):
    print(*a, flush=True)


def best_match(gt, lab, keep):
    """(iou, pred_area) of the component that best overlaps the GT instance."""
    if not keep:
        return 0.0, 0
    ids, counts = np.unique(lab[gt], return_counts=True)
    best, barea, g, ks = 0.0, 0, int(gt.sum()), set(keep)
    for i, c in zip(ids, counts):
        if i == 0 or int(i) not in ks:
            continue
        a = int((lab == i).sum())
        iou = int(c) / (a + g - int(c))
        if iou > best:
            best, barea = iou, a
    return best, barea


def width_px(m):
    try:
        sk = cv2.ximgproc.thinning((m * 255).astype(np.uint8))
        L = max(int((sk > 0).sum()), 1)
    except Exception:
        c, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        L = max(int(sum(len(x) for x in c)) // 2, 1)
    return m.sum() / L


def main():
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    f0 = set(folds.loc[folds["fold"] == 0, "file_name"].str.replace(".jpeg", "", regex=False))
    d = pd.read_parquet(OUT / "error_profile_fn_gt.parquet")
    d = d[d["stem"].isin(f0)].copy()
    d["grp"] = np.where(d["matched"].astype(bool), "matched",
                        np.where(d["in_raw_pool"].astype(bool), "in-pool", "never-proposed"))
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet").set_index("filament_id")["segmentation_rle"]
    zb, zd = np.load(SEM / "fold0_probs_cldice.npz"), np.load(SEM / "fold0_probs_dn.npz")

    yy, xx = np.mgrid[0:2048, 0:2048]
    rmap = (np.hypot(xx - DISK_C, yy - DISK_C) / DISK_R).astype(np.float32)

    rows = []
    gated = {r: {} for r in R_GRID}
    for n, (stem, g) in enumerate(d.groupby("stem")):
        if stem not in zb or stem not in zd:
            continue
        pb = zb[stem].astype(np.float32) / 255.0
        pdn = zd[stem].astype(np.float32) / 255.0
        labb, kb = cc(pb >= THR_PROD)
        labd, kd = cc(pdn >= THR_PROD)
        for r in g.itertuples():
            m = decode_rle(gt.loc[r.filament_id]).astype(bool)
            if not m.any():
                continue
            ys, xs = np.nonzero(m)
            rn = float(np.hypot(xs.mean() - DISK_C, ys.mean() - DISK_C) / DISK_R)
            ib, ab = best_match(m, labb, kb)
            idn, adn = best_match(m, labd, kd)
            ga = int(m.sum())
            rows.append(dict(stem=stem, fid=r.filament_id, grp=r.grp, r=rn,
                             gt_area=ga, width=width_px(m),
                             iou_b=ib, iou_d=idn,
                             cov_b=float((pb[m] >= HI).mean()), cov_d=float((pdn[m] >= HI).mean()),
                             ratio_b=(ab / ga) if ab else np.nan,
                             ratio_d=(adn / ga) if adn else np.nan))
        # radially gated maps: baseline inside r*, denoised outside, 0.05 soft band
        for rs in R_GRID:
            w = np.clip((rmap - rs) / 0.05 + 0.5, 0, 1)
            gated[rs][stem] = np.clip(((1 - w) * pb + w * pdn) * 255, 0, 255).astype(np.uint8)
        if (n + 1) % 40 == 0:
            log(f"  {n+1} images")

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "radial_denoise_profile.parquet", index=False)
    for rs in R_GRID:
        np.savez_compressed(SEM / f"fold0_probs_dn_r{int(rs*100)}.npz", **gated[rs])
    log(f"\n{len(df)} GT instances; gated maps written for r* in {R_GRID}\n")

    df["reach_b"], df["reach_d"] = df.iou_b > 0.5, df.iou_d > 0.5
    df["rbin"] = pd.cut(df.r, [0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    log("1. DENOISE vs BASELINE by radius (all fold-0 GT instances)")
    log(f"  {'r band':>12}{'n':>5}{'cov_b':>8}{'cov_d':>8}{'d cov':>8}"
        f"{'reach_b':>9}{'reach_d':>9}{'d reach':>9}{'dice_b':>8}{'d dice':>8}")
    for b_, g in df.groupby("rbin", observed=True):
        db = 2 * g.iou_b / (1 + g.iou_b)
        dd = 2 * g.iou_d / (1 + g.iou_d)
        log(f"  {str(b_):>12}{len(g):>5}{g.cov_b.mean():>8.3f}{g.cov_d.mean():>8.3f}"
            f"{g.cov_d.mean()-g.cov_b.mean():>+8.3f}{int(g.reach_b.sum()):>9}"
            f"{int(g.reach_d.sum()):>9}{int(g.reach_d.sum()-g.reach_b.sum()):>+9}"
            f"{db.mean():>8.3f}{dd.mean()-db.mean():>+8.3f}")
    np_ = df[df.grp == "never-proposed"]
    log("\n   never-proposed rows only, by radius: coverage gain / reach gain")
    for b_, g in np_.groupby("rbin", observed=True):
        log(f"  {str(b_):>12}{len(g):>5}{g.cov_d.mean()-g.cov_b.mean():>+8.3f}"
            f"{int(g.reach_d.sum()-g.reach_b.sum()):>+9}")
    log("\n   where do the populations live?")
    for grp, g in df.groupby("grp"):
        log(f"  {grp:<16} n={len(g):>4}  median r {g.r.median():.3f}  "
            f"share r>0.7 {100*(g.r>0.7).mean():.0f}%")

    log("\n2. SIZE ERROR log2(pred_area/gt_area), baseline map, matched components")
    ok = df.dropna(subset=["ratio_b"]).copy()
    ok["l2"] = np.log2(ok.ratio_b)
    log(f"  overall: median {ok.l2.median():+.3f}  (ratio {2**ok.l2.median():.3f}), "
        f"n={len(ok)}; >0 means prediction bigger than GT")
    log(f"  {'r band':>12}{'n':>5}{'median log2':>12}{'ratio':>7}{'IQR':>12}")
    for b_, g in ok.groupby("rbin", observed=True):
        q1, q3 = g.l2.quantile([0.25, 0.75])
        log(f"  {str(b_):>12}{len(g):>5}{g.l2.median():>+12.3f}{2**g.l2.median():>7.3f}"
            f"{q1:>+6.2f}..{q3:>+5.2f}")
    ok["wbin"] = pd.qcut(ok.width, 5, duplicates="drop")
    log(f"\n  {'GT width px':>12}{'n':>5}{'median log2':>12}{'ratio':>7}")
    for b_, g in ok.groupby("wbin", observed=True):
        log(f"  {g.width.median():>12.1f}{len(g):>5}{g.l2.median():>+12.3f}{2**g.l2.median():>7.3f}")
    from scipy.stats import spearmanr
    log(f"\n  Spearman(log2 ratio, r)     {spearmanr(ok.l2, ok.r).statistic:+.3f}")
    log(f"  Spearman(log2 ratio, width) {spearmanr(ok.l2, ok.width).statistic:+.3f}")
    log(f"\nwrote {OUT/'radial_denoise_profile.parquet'}")


if __name__ == "__main__":
    main()
