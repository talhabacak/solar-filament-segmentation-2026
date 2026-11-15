"""Resolution degradation study for MAGFiLO filament masks.

Quantifies what label downsampling (2048 -> 1024/768/512, cv2.INTER_AREA + 0.5
threshold) destroys, per annotation instance:
  - survival (any foreground pixel left)
  - fragmentation (1 connected component at 2048 -> N>1 at scale, 8-connectivity)
  - skeleton length retention (skimage skeletonize; scaled length * (2048/s) / base length)
  - local width at 2048 (2 * distance-transform value on skeleton pixels)
  - total foreground pixel loss (entry-level union mask, label-downsampled)
  - inter-instance merge risk: pairs of DIFFERENT instances of the same image
    entry that become 8-connected-touching at a scale but were not at 2048.

Outputs (all under outputs/eda/):
  resolution_study.json               aggregate numbers
  resolution_study.md                 human-readable report
  resolution_study.png                figure: vanish / fragmentation / retention / merge risk vs scale
  resolution_instance_metrics.parquet per-instance records for downstream reuse

Re-runnable: paths derived from repo layout (this file lives in <repo>/src/eda/).

Usage:
  python src/eda/resolution_study.py                 # all 1154 entries
  python src/eda/resolution_study.py --subsample 300 # stratified by (site, year)
  python src/eda/resolution_study.py --limit 20      # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as maskUtils
from skimage.morphology import skeletonize
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
ANN_JSON = (
    REPO / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train"
    / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
)
OUT_DIR = REPO / "outputs" / "eda"

BASE = 2048
SCALES = (1024, 768, 512)
K3 = np.ones((3, 3), np.uint8)  # 8-connectivity structuring element


# ----------------------------------------------------------------------------- helpers

def skel_length(sk: np.ndarray) -> float:
    """Polyline-ish length of a boolean skeleton: orthogonal links = 1, diagonal = sqrt(2).
    A single isolated pixel counts as length 1."""
    n = int(sk.sum())
    if n == 0:
        return 0.0
    orth = int((sk[:, 1:] & sk[:, :-1]).sum()) + int((sk[1:, :] & sk[:-1, :]).sum())
    diag = int((sk[1:, 1:] & sk[:-1, :-1]).sum()) + int((sk[1:, :-1] & sk[:-1, 1:]).sum())
    return max(orth + math.sqrt(2.0) * diag, 1.0)


def crop_pad(m: np.ndarray, bbox, pad: int = 2) -> np.ndarray:
    """Crop mask to inclusive bbox (y0,y1,x0,x1) with a zero border of `pad`.
    The bbox is tight, so zero-padding reproduces the full-frame neighbourhood
    exactly (valid for distance transform / skeleton / component counting)."""
    y0, y1, x0, x1 = bbox
    c = np.zeros((y1 - y0 + 1 + 2 * pad, x1 - x0 + 1 + 2 * pad), np.uint8)
    c[pad:-pad, pad:-pad] = m[y0 : y1 + 1, x0 : x1 + 1]
    return c


def tight_bbox(m: np.ndarray):
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def downscale_mask(m255: np.ndarray, s: int) -> np.ndarray:
    """Label downsample: INTER_AREA average then 0.5 threshold. m255 is uint8 {0,255}."""
    r = cv2.resize(m255, (s, s), interpolation=cv2.INTER_AREA)
    return r >= 128  # 0.5 * 255 = 127.5


def touching_pairs(masks, bboxes):
    """Set of index pairs (i<j) whose masks overlap or are 8-adjacent.
    masks: list of bool full-frame arrays (or None); bboxes inclusive (y0,y1,x0,x1) or None."""
    pairs = set()
    n = len(masks)
    for i in range(n):
        if bboxes[i] is None:
            continue
        ay0, ay1, ax0, ax1 = bboxes[i]
        for j in range(i + 1, n):
            if bboxes[j] is None:
                continue
            by0, by1, bx0, bx1 = bboxes[j]
            if ay0 > by1 + 1 or by0 > ay1 + 1 or ax0 > bx1 + 1 or bx0 > ax1 + 1:
                continue  # dilated bboxes disjoint
            y0, y1 = min(ay0, by0), max(ay1, by1)
            x0, x1 = min(ax0, bx0), max(ax1, bx1)
            a = masks[i][y0 : y1 + 1, x0 : x1 + 1].astype(np.uint8)
            b = masks[j][y0 : y1 + 1, x0 : x1 + 1]
            if (cv2.dilate(a, K3).astype(bool) & b).any():
                pairs.add((i, j))
    return pairs


def site_year(image_id: str):
    stem = image_id.split("-", 1)[1]  # e.g. 20140609195854Bh
    return stem[14], stem[:4]


def stratified_subsample(entry_ids, id2sy, n_target, seed):
    """Proportional (largest-remainder) allocation over (site, year) strata, >=1 per stratum."""
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for eid in entry_ids:
        groups[id2sy[eid]].append(eid)
    keys = sorted(groups)
    total = len(entry_ids)
    raw = {k: n_target * len(groups[k]) / total for k in keys}
    alloc = {k: max(1, int(raw[k])) for k in keys}
    # largest remainder to hit target
    while sum(alloc.values()) < n_target:
        k = max(keys, key=lambda k: (raw[k] - alloc[k], k))
        alloc[k] += 1
    while sum(alloc.values()) > n_target:
        k = min((k for k in keys if alloc[k] > 1), key=lambda k: (raw[k] - alloc[k], k))
        alloc[k] -= 1
    chosen = []
    for k in keys:
        g = sorted(groups[k])
        take = min(alloc[k], len(g))
        chosen.extend(rng.choice(g, size=take, replace=False).tolist())
    return sorted(chosen)


# ----------------------------------------------------------------------------- main study

def run(args):
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    coco = json.loads(ANN_JSON.read_text())
    images = {im["id"]: im for im in coco["images"]}
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    entry_ids = sorted(anns_by_img)
    id2sy = {eid: site_year(eid) for eid in entry_ids}

    sampling_note = f"all {len(entry_ids)} entries"
    if args.subsample and args.subsample < len(entry_ids):
        entry_ids = stratified_subsample(entry_ids, id2sy, args.subsample, args.seed)
        sampling_note = (
            f"stratified subsample of {len(entry_ids)} entries over (site, year) strata, "
            f"seed={args.seed}"
        )
    if args.limit:
        entry_ids = entry_ids[: args.limit]
        sampling_note += f" [LIMIT {args.limit} — smoke test]"

    rows = []
    skipped_bad_seg = 0
    zero_area_at_base = 0
    union_base_fg = 0
    union_scale_fg = {s: 0.0 for s in SCALES}  # upscaled-equivalent px
    contact = {
        s: {"touch_pairs": 0, "new_pairs": 0, "entries_with_new": 0}
        for s in SCALES
    }
    base_touch_pairs_total = 0
    pairs_checked_total = 0
    n_entries_done = 0

    for eid in tqdm(entry_ids, desc="entries", unit="img"):
        anns = anns_by_img[eid]
        h, w = images[eid]["height"], images[eid]["width"]
        inst = []  # (ann, mask_base uint8, m255, bbox_base)
        for a in anns:
            try:
                rles = maskUtils.frPyObjects(a["segmentation"], h, w)
                m = maskUtils.decode(maskUtils.merge(rles))
            except Exception:
                skipped_bad_seg += 1
                continue
            bb = tight_bbox(m)
            if bb is None:
                zero_area_at_base += 1
                continue
            inst.append((a, m, m * np.uint8(255), bb))
        if not inst:
            continue
        n_entries_done += 1

        scale_masks = {s: [] for s in SCALES}
        scale_bboxes = {s: [] for s in SCALES}

        for a, m, m255, bb in inst:
            area = int(m.sum())
            crop = crop_pad(m, bb)
            ncomp_base = cv2.connectedComponents(crop, connectivity=8)[0] - 1
            sk = skeletonize(crop.astype(bool))
            slen_base = skel_length(sk)
            dist = cv2.distanceTransform(crop, cv2.DIST_L2, 5)
            wvals = 2.0 * dist[sk]
            mean_w = float(wvals.mean()) if wvals.size else 0.0
            min_w = float(wvals.min()) if wvals.size else 0.0

            rec = {
                "entry_id": eid,
                "ann_id": a["id"],
                "category_id": a["category_id"],
                "site": id2sy[eid][0],
                "year": int(id2sy[eid][1]),
                "area2048": area,
                "ncomp2048": ncomp_base,
                "skel_len2048": slen_base,
                "mean_width": mean_w,
                "min_width": min_w,
            }
            for s in SCALES:
                ms = downscale_mask(m255, s)
                bbs = tight_bbox(ms)
                scale_masks[s].append(ms if bbs is not None else None)
                scale_bboxes[s].append(bbs)
                if bbs is None:
                    rec[f"surv_{s}"] = False
                    rec[f"ncomp_{s}"] = 0
                    rec[f"area_{s}"] = 0
                    rec[f"skelret_{s}"] = 0.0
                    continue
                crop_s = crop_pad(ms.astype(np.uint8), bbs)
                ncomp_s = cv2.connectedComponents(crop_s, connectivity=8)[0] - 1
                slen_s = skel_length(skeletonize(crop_s.astype(bool))) * (BASE / s)
                rec[f"surv_{s}"] = True
                rec[f"ncomp_{s}"] = ncomp_s
                rec[f"area_{s}"] = int(ms.sum())
                rec[f"skelret_{s}"] = slen_s / slen_base if slen_base > 0 else np.nan
            rows.append(rec)

        # entry-level union foreground (semantic label downsample)
        union = np.zeros((h, w), np.uint8)
        for _, m, _, _ in inst:
            np.maximum(union, m, out=union)
        union_base_fg += int(union.sum())
        u255 = union * np.uint8(255)
        for s in SCALES:
            union_scale_fg[s] += float(downscale_mask(u255, s).sum()) * (BASE / s) ** 2

        # inter-instance contact / merge risk
        base_masks = [m.astype(bool) for _, m, _, _ in inst]
        base_bbs = [bb for _, _, _, bb in inst]
        base_pairs = touching_pairs(base_masks, base_bbs)
        base_touch_pairs_total += len(base_pairs)
        n = len(inst)
        pairs_checked_total += n * (n - 1) // 2
        for s in SCALES:
            sp = touching_pairs(scale_masks[s], scale_bboxes[s])
            new = sp - base_pairs
            contact[s]["touch_pairs"] += len(sp)
            contact[s]["new_pairs"] += len(new)
            if new:
                contact[s]["entries_with_new"] += 1

    df = pd.DataFrame(rows)
    df["area_q"] = pd.qcut(df["area2048"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    df["width_q"] = pd.qcut(df["mean_width"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    df.to_parquet(OUT_DIR / "resolution_instance_metrics.parquet", index=False)

    # ---------------------------------------------------------------- aggregates
    def per_scale_stats(sub: pd.DataFrame) -> dict:
        out = {}
        for s in SCALES:
            surv = sub[f"surv_{s}"]
            single = sub["ncomp2048"] == 1
            frag_pool = sub[single & surv]
            out[s] = {
                "n": int(len(sub)),
                "pct_vanished": float(100.0 * (~surv).mean()) if len(sub) else np.nan,
                "pct_fragmented": (
                    float(100.0 * (frag_pool[f"ncomp_{s}"] > 1).mean())
                    if len(frag_pool)
                    else np.nan
                ),
                "median_skel_retention": (
                    float(sub.loc[surv, f"skelret_{s}"].median()) if surv.any() else np.nan
                ),
            }
        return out

    overall = per_scale_stats(df)
    by_area_q = {q: per_scale_stats(df[df["area_q"] == q]) for q in ["Q1", "Q2", "Q3", "Q4"]}
    by_width_q = {q: per_scale_stats(df[df["width_q"] == q]) for q in ["Q1", "Q2", "Q3", "Q4"]}

    fg_loss = {
        s: float(1.0 - union_scale_fg[s] / union_base_fg) for s in SCALES
    }
    inst_fg_loss = {
        s: float(
            1.0
            - (df[f"area_{s}"].sum() * (BASE / s) ** 2) / df["area2048"].sum()
        )
        for s in SCALES
    }

    contact_out = {
        "pairs_checked_total": int(pairs_checked_total),
        "touch_pairs_at_2048": int(base_touch_pairs_total),
        "per_scale": {
            s: {
                "touch_pairs": int(contact[s]["touch_pairs"]),
                "newly_touching_pairs": int(contact[s]["new_pairs"]),
                "entries_with_new_touch": int(contact[s]["entries_with_new"]),
                "pct_entries_with_new_touch": float(
                    100.0 * contact[s]["entries_with_new"] / n_entries_done
                ),
            }
            for s in SCALES
        },
    }

    quart_edges = {
        "area2048": [float(x) for x in df["area2048"].quantile([0, 0.25, 0.5, 0.75, 1]).tolist()],
        "mean_width": [float(x) for x in df["mean_width"].quantile([0, 0.25, 0.5, 0.75, 1]).tolist()],
    }

    result = {
        "config": {
            "sampling": sampling_note,
            "n_entries_analyzed": n_entries_done,
            "n_instances_analyzed": int(len(df)),
            "skipped_bad_segmentation": skipped_bad_seg,
            "zero_area_at_2048": zero_area_at_base,
            "scales": list(SCALES),
            "method": "cv2.resize INTER_AREA on binary mask, threshold 0.5; "
            "8-connectivity components; skimage skeletonize; width = 2*L2 dist on skeleton",
            "runtime_sec": round(time.time() - t0, 1),
        },
        "pct_instances_multicomponent_at_2048": float(100.0 * (df["ncomp2048"] > 1).mean()),
        "width_stats_2048_px": {
            "mean_width_median": float(df["mean_width"].median()),
            "mean_width_p10": float(df["mean_width"].quantile(0.10)),
            "min_width_median": float(df["min_width"].median()),
        },
        "overall_per_scale": overall,
        "by_area_quartile": by_area_q,
        "by_width_quartile": by_width_q,
        "quartile_edges": quart_edges,
        "total_foreground_loss_union": fg_loss,
        "total_foreground_loss_instance_sum": inst_fg_loss,
        "inter_instance_contact": contact_out,
    }

    (OUT_DIR / "resolution_study.json").write_text(json.dumps(result, indent=2))
    write_markdown(result)
    make_figure(result)
    print(json.dumps(result["overall_per_scale"], indent=2))
    print("fg loss (union):", fg_loss)
    print("contact:", contact_out["per_scale"])
    print(f"done in {result['config']['runtime_sec']}s -> {OUT_DIR}")
    return result


# ----------------------------------------------------------------------------- report

def write_markdown(r: dict):
    o = r["overall_per_scale"]
    c = r["inter_instance_contact"]
    L = []
    L.append("# Resolution degradation study — MAGFiLO 2048px masks\n")
    L.append(f"*Generated {time.strftime('%Y-%m-%d %H:%M')} by `src/eda/resolution_study.py`*\n")
    cfg = r["config"]
    L.append(
        f"- Sampling: {cfg['sampling']}\n"
        f"- Entries analyzed: {cfg['n_entries_analyzed']}, instances: {cfg['n_instances_analyzed']} "
        f"(skipped bad segmentation: {cfg['skipped_bad_segmentation']}, "
        f"zero-area at 2048: {cfg['zero_area_at_2048']})\n"
        f"- Method: {cfg['method']}\n"
    )
    L.append(
        "Note: threshold is inclusive (coverage >= 0.5 -> foreground), so exactly-half "
        "covered pixels survive; total foreground can be preserved or even slightly "
        "inflated while thin structure is still destroyed (see skeleton/fragmentation).\n"
    )
    L.append("## Per-scale instance damage (label downsample, INTER_AREA + 0.5 threshold)\n")
    L.append("| scale | % vanished | % fragmented (1→N comp, among survivors) | median skeleton retention | total fg px lost (union) |")
    L.append("|---|---|---|---|---|")
    for s in [str(x) for x in cfg["scales"]]:
        si = o[s] if s in o else o[int(s)]
        L.append(
            f"| {s} | {si['pct_vanished']:.2f}% | {si['pct_fragmented']:.2f}% | "
            f"{si['median_skel_retention']:.3f} | {100*r['total_foreground_loss_union'][s if s in r['total_foreground_loss_union'] else int(s)]:.2f}% |"
        )
    L.append("")
    L.append(
        f"- Instances with >1 component already at 2048: "
        f"{r['pct_instances_multicomponent_at_2048']:.2f}%\n"
        f"- Local width at 2048 (px): median of per-instance mean width = "
        f"{r['width_stats_2048_px']['mean_width_median']:.2f}, p10 = "
        f"{r['width_stats_2048_px']['mean_width_p10']:.2f}, median min-width = "
        f"{r['width_stats_2048_px']['min_width_median']:.2f}\n"
    )

    def strat_table(title, d, edges_key, edges):
        L.append(f"## {title}\n")
        e = [f"{x:.0f}" for x in edges]
        L.append(
            f"Quartile edges ({edges_key}): "
            + " / ".join(f"[{e[i]}–{e[i+1]}]" for i in range(4))
            + "\n"
        )
        L.append("| quartile | metric | " + " | ".join(str(s) for s in r["config"]["scales"]) + " |")
        L.append("|---|---|" + "---|" * len(r["config"]["scales"]))
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            qs = d[q]
            for met, fmt in [
                ("pct_vanished", "{:.2f}%"),
                ("pct_fragmented", "{:.2f}%"),
                ("median_skel_retention", "{:.3f}"),
            ]:
                vals = []
                for s in r["config"]["scales"]:
                    v = qs[s if s in qs else str(s)][met]
                    vals.append("–" if v != v else fmt.format(v))
                L.append(f"| {q} | {met} | " + " | ".join(vals) + " |")
        L.append("")

    strat_table("By instance area quartile (area at 2048)", r["by_area_quartile"],
                "area px", r["quartile_edges"]["area2048"])
    strat_table("By instance width quartile (mean local width at 2048)", r["by_width_quartile"],
                "width px", r["quartile_edges"]["mean_width"])

    L.append("## Inter-instance merge risk (same-entry instance pairs, 8-connected touch)\n")
    L.append(
        f"Pairs checked: {c['pairs_checked_total']}; already touching at 2048: "
        f"{c['touch_pairs_at_2048']}\n"
    )
    L.append("| scale | touching pairs | NEWLY touching pairs | entries affected | % of entries |")
    L.append("|---|---|---|---|---|")
    for s in r["config"]["scales"]:
        cs = c["per_scale"][s if s in c["per_scale"] else str(s)]
        L.append(
            f"| {s} | {cs['touch_pairs']} | {cs['newly_touching_pairs']} | "
            f"{cs['entries_with_new_touch']} | {cs['pct_entries_with_new_touch']:.2f}% |"
        )
    L.append("")
    L.append("![figure](resolution_study.png)\n")
    (OUT_DIR / "resolution_study.md").write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------------------- figure

def make_figure(r: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # validated reference palette (light mode)
    SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
    SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # slots 1-4, fixed order

    scales_x = [512, 768, 1024, 2048]  # 2048 = lossless baseline
    sc = r["config"]["scales"]

    def series_vals(d, met, baseline):
        get = lambda s: d[s if s in d else str(s)][met]
        return [get(s) for s in sorted(sc)] + [baseline]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), facecolor=SURFACE)
    for ax in axes.flat:
        ax.set_facecolor(SURFACE)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color(GRID)
        ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_xticks(scales_x)
        ax.set_xticklabels(["512", "768", "1024", "2048\n(native)"])
        ax.set_xlabel("mask resolution (px)", color=INK2, fontsize=9.5)

    def quartile_panel(ax, strat, met, baseline, title, qlabel):
        for i, q in enumerate(["Q1", "Q2", "Q3", "Q4"]):
            y = series_vals(strat[q], met, baseline)
            ax.plot(scales_x, y, color=SERIES[i], linewidth=2, marker="o",
                    markersize=5, label=f"{q} {qlabel}")
            ax.annotate(q, (scales_x[0], y[0]), textcoords="offset points",
                        xytext=(-14, -3), color=SERIES[i], fontsize=9, fontweight="bold")
        ax.set_title(title, color=INK, fontsize=11, loc="left", fontweight="bold")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")

    quartile_panel(
        axes[0, 0], r["by_width_quartile"], "pct_vanished", 0.0,
        "Instances vanished (%) — by width quartile", "width",
    )
    axes[0, 0].set_ylabel("% vanished", color=INK2, fontsize=9.5)
    v_all = [r["by_width_quartile"][q][s if s in r["by_width_quartile"][q] else str(s)]
             ["pct_vanished"] for q in ["Q1", "Q2", "Q3", "Q4"] for s in sc]
    if max(v_all) == 0.0:
        axes[0, 0].set_ylim(-0.05, 1.0)
        axes[0, 0].annotate("0% at every scale — no instance vanishes",
                            (0.03, 0.55), xycoords="axes fraction",
                            color=INK2, fontsize=10)

    quartile_panel(
        axes[0, 1], r["by_area_quartile"], "pct_fragmented", 0.0,
        "Instances fragmented 1→N (%) — by area quartile", "area",
    )
    axes[0, 1].set_ylabel("% fragmented (survivors)", color=INK2, fontsize=9.5)

    quartile_panel(
        axes[1, 0], r["by_width_quartile"], "median_skel_retention", 1.0,
        "Median skeleton-length retention — by width quartile", "width",
    )
    axes[1, 0].set_ylabel("retention ratio", color=INK2, fontsize=9.5)
    axes[1, 0].axhline(1.0, color=MUTED, linewidth=1, linestyle="--")

    ax = axes[1, 1]
    c = r["inter_instance_contact"]["per_scale"]
    y = series_vals(c, "pct_entries_with_new_touch", 0.0)
    pairs = series_vals(c, "newly_touching_pairs", 0)
    ax.plot(scales_x, y, color=SERIES[0], linewidth=2, marker="o", markersize=5)
    for xx, yy, pp in zip(scales_x, y, pairs):
        ax.annotate(f"{pp} pairs", (xx, yy), textcoords="offset points",
                    xytext=(0, 8), ha="center", color=INK2, fontsize=8.5)
    ax.set_title("Merge risk: entries with newly touching instance pairs (%)",
                 color=INK, fontsize=11, loc="left", fontweight="bold")
    ax.set_ylabel("% of entries", color=INK2, fontsize=9.5)

    fig.suptitle(
        "Label downsampling damage vs resolution (INTER_AREA + 0.5 threshold, "
        f"{r['config']['n_instances_analyzed']} instances)",
        color=INK, fontsize=12.5, x=0.02, ha="left", fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_DIR / "resolution_study.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------- cli

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--subsample", type=int, default=0,
                   help="if >0, stratified subsample of this many entries over (site, year)")
    p.add_argument("--limit", type=int, default=0,
                   help="hard cap on entries (smoke test only)")
    p.add_argument("--seed", type=int, default=42)
    run(p.parse_args())


if __name__ == "__main__":
    main()
