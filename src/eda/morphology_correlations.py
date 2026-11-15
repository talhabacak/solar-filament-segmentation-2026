"""Morphology correlation study for MAGFiLO Kaggle 2026.

Does filament skeleton/shape structure correlate with anything exploitable
(chirality label, position, annotator agreement, detectability)?

Starts from outputs/eda/morphology_instances.parquet, RECOMPUTES positional
features with the FIXED disk geometry (center (1023.5, 1023.5), R=900 px for
all images), and adds boundary/spine features from the COCO json:

1. Boundary complexity P^2/(4*pi*A) per annotation; tests whether
   Unidentifiable have lower complexity than Left/Right at matched size
   (within area quartiles). Also vertex density (polygon points / perimeter).
2. Spine skeleton stats: tortuosity (= arclength / endpoint distance), mean
   absolute turning angle, net signed turning (canonical left-to-right
   traversal; positive = spine bows toward solar north) and its sign vs
   chirality and hemisphere.
3. Orientation vs latitude: median tilt + fraction East-West aligned per
   |lat| bin; signed tilt by hemisphere (Joy's-law-like trend test).
4. Category (L/R/U) proportions vs |lat| bins, r_norm bins, and size.
5. Spearman correlation matrix across morphology + position + is_U +
   (optional) annotator-disagreement proxy / cluster support.
6. Detectability: fold-0 val GT instances matched (IoU>0.5) by the E010
   model at conf 0.4 / min-area 128 / r<=0.97 -> which morphology features
   separate matched vs missed (Cohen's d + logistic CV AUC).

Outputs (under outputs/eda/):
    morphology_correlations.parquet   per-annotation enriched feature table
    morphology_correlations.md        report with all measured numbers
    morphcorr_stats.json              machine-readable headline stats
    morphcorr_complexity.png, morphcorr_spine.png, morphcorr_tilt.png,
    morphcorr_category_position.png, morphcorr_spearman.png,
    morphcorr_detectability.png

RAM-lean: detectability decodes masks one image stem at a time.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DATA_DIR = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"
ANN_JSON = DATA_DIR / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
OUT_DIR = ROOT / "outputs" / "eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# CONFIRMED fixed disk geometry (GONG remaps) — identical for all 887 images.
CX, CY, R = 1023.5, 1023.5, 900.0

CAT_COLORS = {"Left": "#0072B2", "Right": "#E69F00", "Unidentifiable": "#009E73"}
EW_ALIGNED_DEG = 30.0  # |tilt from E-W axis| <= this => "East-West aligned"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.grid": True,
    "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10,
})


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# Feature extraction from COCO json (perimeter, vertex density, spine stats)
# ---------------------------------------------------------------------------

def ring_perimeter(pts):
    """Closed-polygon perimeter of an (n,2) vertex array."""
    if len(pts) < 3:
        return 0.0
    d = np.diff(np.vstack([pts, pts[:1]]), axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def spine_stats(flat):
    """Return (endpoint_dist, tortuosity, mean_abs_turn_deg, net_turn_deg).

    Canonical traversal: reorder so the first point has the smaller x (tie:
    smaller y).  In image coords (y down), positive net turning then means the
    spine bows toward smaller y, i.e. toward solar NORTH.
    """
    pts = np.asarray(flat, dtype=float).reshape(-1, 2)
    if len(pts) < 2:
        return np.nan, np.nan, np.nan, np.nan
    p0, p1 = pts[0], pts[-1]
    if (p1[0], p1[1]) < (p0[0], p0[1]):
        pts = pts[::-1]
    seg = np.diff(pts, axis=0)
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    keep = seglen > 1e-9
    seg = seg[keep]
    if len(seg) == 0:
        return 0.0, np.nan, np.nan, np.nan
    arclen = float(np.hypot(seg[:, 0], seg[:, 1]).sum())
    end_dist = float(np.hypot(*(pts[-1] - pts[0])))
    tort = arclen / max(end_dist, 1e-9)
    if len(seg) < 2:
        return end_dist, tort, np.nan, np.nan
    v1, v2 = seg[:-1], seg[1:]
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    dot = (v1 * v2).sum(axis=1)
    ang = np.degrees(np.arctan2(cross, dot))     # signed turn at each vertex
    # image y is down: cross>0 = bow toward smaller y = toward north; keep sign
    return end_dist, tort, float(np.abs(ang).mean()), float(ang.sum())


def build_features():
    log("[1/7] loading COCO json + computing boundary/spine features ...")
    with open(ANN_JSON) as f:
        coco = json.load(f)
    rows = []
    for ann in coco["annotations"]:
        perim = 0.0
        for part in (ann.get("segmentation") or []):
            perim += ring_perimeter(np.asarray(part, dtype=float).reshape(-1, 2))
        ed, tort, mat, net = spine_stats(ann.get("spine") or [])
        rows.append((ann["id"], perim, ed, tort, mat, net))
    feat = pd.DataFrame(rows, columns=[
        "ann_id", "perimeter", "spine_end_dist", "tortuosity",
        "mean_abs_turn_deg", "net_turn_deg"])

    morph = pd.read_parquet(OUT_DIR / "morphology_instances.parquet")
    morph = morph.drop(columns=["disk_cx", "disk_cy", "disk_r",
                                "dx_norm", "dy_norm", "r_norm"])
    df = morph.merge(feat, on="ann_id", how="left")

    # FIXED disk geometry positions (recomputed; ignore per-image fits)
    df["dx_norm"] = (df["centroid_x"] - CX) / R
    df["dy_norm"] = (CY - df["centroid_y"]) / R          # north = up = positive
    df["r_norm"] = np.hypot(df["dx_norm"], df["dy_norm"])
    dy = df["dy_norm"].clip(-1, 1)
    df["lat_deg"] = np.degrees(np.arcsin(dy))
    coslat = np.cos(np.radians(df["lat_deg"])).clip(1e-6)
    df["lon_deg"] = np.degrees(np.arcsin((df["dx_norm"] / coslat).clip(-1, 1)))
    df["abs_lat_deg"] = df["lat_deg"].abs()
    df["hemisphere"] = np.where(df["lat_deg"] >= 0, "N", "S")

    # derived shape features
    df["boundary_complexity"] = df["perimeter"] ** 2 / (
        4 * math.pi * df["poly_area"].clip(lower=1e-9))
    df["vertex_density"] = df["n_poly_points"] / df["perimeter"].clip(lower=1e-9)
    df["tilt_solar_deg"] = -df["orientation_deg"]        # y-down -> y-up flip
    df["is_U"] = (df["category_id"] == 3).astype(int)
    df["log_area"] = np.log10(df["poly_area"].clip(lower=1))

    # annotator count-disagreement proxy from manifest (multi-copy files)
    man = pd.read_parquet(OUT_DIR / "manifest_entries.parquet")
    g = man.groupby("file_name")["num_annotations"]
    dis = pd.DataFrame({
        "n_copies": g.size(),
        "cnt_range": g.max() - g.min(),
        "cnt_mean": g.mean(),
    })
    dis["file_disagree_norm"] = np.where(
        dis["n_copies"] >= 2, dis["cnt_range"] / dis["cnt_mean"].clip(lower=1), np.nan)
    df = df.merge(dis[["n_copies", "file_disagree_norm"]].reset_index(),
                  on="file_name", how="left")
    log(f"    features for {len(df)} annotations "
        f"({df['perimeter'].isna().sum()} missing perimeter, "
        f"{df['tortuosity'].isna().sum()} missing tortuosity)")
    return df


# ---------------------------------------------------------------------------
# 1. Boundary complexity by category within area quartiles
# ---------------------------------------------------------------------------

def analysis_complexity(df, report, statsd):
    log("[2/7] boundary complexity vs category (size-controlled) ...")
    d = df[np.isfinite(df["boundary_complexity"]) & (df["poly_area"] > 0)].copy()
    edges = d["poly_area"].quantile([0, .25, .5, .75, 1]).values
    d["area_q"] = pd.cut(d["poly_area"], edges, labels=["Q1", "Q2", "Q3", "Q4"],
                         include_lowest=True)
    lines = ["## 1. Boundary complexity P^2/(4piA) by category, size-controlled",
             "",
             "Hypothesis: Unidentifiable (U) drawn with less boundary detail than L/R at matched size.",
             "",
             "| area quartile | med(L) | med(R) | med(U) | med(L+R) | U vs L+R Mann-Whitney p | U lower? |",
             "|---|---|---|---|---|---|---|"]
    qstats = {}
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        sub = d[d["area_q"] == q]
        med = sub.groupby("category")["boundary_complexity"].median()
        u = sub.loc[sub["is_U"] == 1, "boundary_complexity"]
        lr = sub.loc[sub["is_U"] == 0, "boundary_complexity"]
        mw = sps.mannwhitneyu(u, lr, alternative="less")
        lines.append(f"| {q} (n={len(sub)}) | {med.get('Left', np.nan):.3f} | "
                     f"{med.get('Right', np.nan):.3f} | {med.get('Unidentifiable', np.nan):.3f} | "
                     f"{lr.median():.3f} | {mw.pvalue:.2e} | "
                     f"{'YES' if u.median() < lr.median() else 'no'} |")
        qstats[q] = dict(n=int(len(sub)), med_U=float(u.median()),
                         med_LR=float(lr.median()), p_U_less=float(mw.pvalue))
    med_all = d.groupby("category")["boundary_complexity"].median().to_dict()
    vd = d.groupby("category")["vertex_density"].median().to_dict()
    u_all = d.loc[d["is_U"] == 1, "boundary_complexity"]
    lr_all = d.loc[d["is_U"] == 0, "boundary_complexity"]
    mw_all = sps.mannwhitneyu(u_all, lr_all, alternative="less")
    lines += ["",
              f"Overall medians: L={med_all.get('Left'):.3f} R={med_all.get('Right'):.3f} "
              f"U={med_all.get('Unidentifiable'):.3f}; U vs L+R one-sided p={mw_all.pvalue:.2e}.",
              "",
              "Vertex density (polygon points per px of perimeter), median: "
              + ", ".join(f"{k}={v:.4f}" for k, v in sorted(vd.items())), ""]
    statsd["complexity"] = dict(
        per_quartile=qstats, overall_med=dict((k, float(v)) for k, v in med_all.items()),
        overall_p_U_less=float(mw_all.pvalue),
        vertex_density_med=dict((k, float(v)) for k, v in vd.items()))
    report.extend(lines)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    cats = ["Left", "Right", "Unidentifiable"]
    width = 0.26
    x = np.arange(4)
    for i, c in enumerate(cats):
        meds = [d[(d["area_q"] == q) & (d["category"] == c)]["boundary_complexity"].median()
                for q in ["Q1", "Q2", "Q3", "Q4"]]
        axes[0].bar(x + (i - 1) * width, meds, width, label=c, color=CAT_COLORS[c])
    axes[0].set_xticks(x, ["Q1", "Q2", "Q3", "Q4"])
    axes[0].set_xlabel("Area quartile")
    axes[0].set_ylabel("Median boundary complexity $P^2/4\\pi A$")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Complexity by category within size bins")
    data = [d.loc[d["category"] == c, "vertex_density"].dropna() for c in cats]
    bp = axes[1].boxplot(data, tick_labels=cats, showfliers=False, patch_artist=True,
                         medianprops=dict(color="#222"))
    for patch, c in zip(bp["boxes"], cats):
        patch.set_facecolor(CAT_COLORS[c]); patch.set_alpha(.75)
    axes[1].set_ylabel("Vertex density (pts/px perimeter)")
    axes[1].set_title("Polygon vertex density")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "morphcorr_complexity.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Spine stats: tortuosity, turning, net-curvature sign vs chirality/hemis
# ---------------------------------------------------------------------------

def analysis_spine(df, report, statsd):
    log("[3/7] spine skeleton stats + curvature sign vs chirality/hemisphere ...")
    d = df[np.isfinite(df["net_turn_deg"])].copy()
    d["bow_north"] = (d["net_turn_deg"] > 0).astype(int)
    lines = ["## 2. Spine skeleton statistics", "",
             "Tortuosity = arclength/endpoint distance; net turning signed with "
             "canonical W->E traversal (positive = bows toward solar north).", ""]
    med = d.groupby("category")[["tortuosity", "mean_abs_turn_deg"]].median()
    lines.append(med.round(3).to_markdown())
    lines.append("")

    # bow-north fraction by chirality (L vs R), by hemisphere, and interaction
    lr = d[d["category_id"].isin([1, 2])]
    tab_ch = pd.crosstab(lr["category"], lr["bow_north"])
    chi_ch = sps.chi2_contingency(tab_ch)
    tab_h = pd.crosstab(d["hemisphere"], d["bow_north"])
    chi_h = sps.chi2_contingency(tab_h)
    frac_ch = lr.groupby("category")["bow_north"].mean()
    frac_h = d.groupby("hemisphere")["bow_north"].mean()
    inter = lr.groupby(["hemisphere", "category"])["bow_north"].agg(["mean", "size"])

    # hemispheric chirality rule (sanity physics): N->Right(dextral), S->Left
    tab_rule = pd.crosstab(lr["hemisphere"], lr["category"])
    chi_rule = sps.chi2_contingency(tab_rule)
    fr_R_N = tab_rule.loc["N", "Right"] / tab_rule.loc["N"].sum()
    fr_L_S = tab_rule.loc["S", "Left"] / tab_rule.loc["S"].sum()
    # within-hemisphere chirality->bow (deconfounded from hemisphere)
    wh = {}
    for h in ["N", "S"]:
        s = lr[lr["hemisphere"] == h]
        t = pd.crosstab(s["category"], s["bow_north"])
        wh[h] = float(sps.chi2_contingency(t).pvalue)
    lines += [
        f"Hemispheric chirality rule: Right fraction in N = {fr_R_N:.3f} "
        f"(n={tab_rule.loc['N'].sum()}), Left fraction in S = {fr_L_S:.3f} "
        f"(n={tab_rule.loc['S'].sum()}); chi2 p={chi_rule.pvalue:.3e}.",
        "",
        f"Within-hemisphere chirality vs bow-north chi2 p: N={wh['N']:.3e}, S={wh['S']:.3e}.",
        "",
        f"Bow-north fraction: Left={frac_ch.get('Left'):.3f} (n={tab_ch.loc['Left'].sum()}), "
        f"Right={frac_ch.get('Right'):.3f} (n={tab_ch.loc['Right'].sum()}); "
        f"chi2 p={chi_ch.pvalue:.3e}.",
        "",
        f"Bow-north fraction: N-hemisphere={frac_h.get('N'):.3f} "
        f"(n={tab_h.loc['N'].sum()}), S={frac_h.get('S'):.3f} (n={tab_h.loc['S'].sum()}); "
        f"chi2 p={chi_h.pvalue:.3e}.",
        "",
        "Hemisphere x chirality bow-north fractions:",
        inter.round(3).to_markdown(), ""]
    statsd["spine"] = dict(
        med_tortuosity=dict((k, float(v)) for k, v in med["tortuosity"].items()),
        med_abs_turn=dict((k, float(v)) for k, v in med["mean_abs_turn_deg"].items()),
        bow_north_frac_L=float(frac_ch.get("Left")), bow_north_frac_R=float(frac_ch.get("Right")),
        chirality_chi2_p=float(chi_ch.pvalue),
        bow_north_frac_N=float(frac_h.get("N")), bow_north_frac_S=float(frac_h.get("S")),
        hemisphere_chi2_p=float(chi_h.pvalue),
        chirality_rule_frac_R_in_N=float(fr_R_N), chirality_rule_frac_L_in_S=float(fr_L_S),
        chirality_rule_chi2_p=float(chi_rule.pvalue),
        within_hemi_bow_chi2_p=wh)
    report.extend(lines)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    cats = ["Left", "Right", "Unidentifiable"]
    data = [d.loc[d["category"] == c, "tortuosity"].clip(upper=3) for c in cats]
    bp = axes[0].boxplot(data, tick_labels=cats, showfliers=False, patch_artist=True,
                         medianprops=dict(color="#222"))
    for patch, c in zip(bp["boxes"], cats):
        patch.set_facecolor(CAT_COLORS[c]); patch.set_alpha(.75)
    axes[0].set_ylabel("Tortuosity (clipped at 3)")
    axes[0].set_title("Spine tortuosity")
    axes[1].bar(["Left", "Right"], [frac_ch.get("Left"), frac_ch.get("Right")],
                color=[CAT_COLORS["Left"], CAT_COLORS["Right"]])
    axes[1].axhline(0.5, color="#333", ls="--", lw=1)
    axes[1].set_ylim(0, 1); axes[1].set_ylabel("Fraction bowing north")
    axes[1].set_title(f"Bow direction vs chirality (p={chi_ch.pvalue:.1e})")
    labels, vals = [], []
    for (h, c), row in inter.iterrows():
        labels.append(f"{h}-{c[0]}"); vals.append(row["mean"])
    axes[2].bar(labels, vals, color="#0072B2")
    axes[2].axhline(0.5, color="#333", ls="--", lw=1)
    axes[2].set_ylim(0, 1); axes[2].set_ylabel("Fraction bowing north")
    axes[2].set_title("Hemisphere x chirality")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "morphcorr_spine.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Orientation (tilt) vs latitude
# ---------------------------------------------------------------------------

def analysis_tilt(df, report, statsd):
    log("[4/7] orientation vs latitude ...")
    d = df[np.isfinite(df["tilt_solar_deg"]) & np.isfinite(df["lat_deg"])].copy()
    elong = d[d["eccentricity"] >= 0.85].copy()   # orientation only meaningful if elongated
    bins = [0, 15, 30, 45, 90]
    labels = ["0-15", "15-30", "30-45", "45+"]
    for frame in (d, elong):
        frame["lat_bin"] = pd.cut(frame["abs_lat_deg"], bins, labels=labels,
                                  include_lowest=True)
    lines = ["## 3. Orientation vs latitude (tilt)", "",
             f"tilt_solar_deg = -PCA orientation (y-up convention); E-W aligned = |tilt| <= {EW_ALIGNED_DEG:.0f} deg.",
             f"Elongated subset: eccentricity >= 0.85 (n={len(elong)} of {len(d)}).", "",
             "| |lat| bin | n | median |tilt| (deg) | frac E-W aligned | n elong | frac E-W (elong) |",
             "|---|---|---|---|---|---|"]
    binstats = {}
    for lb in labels:
        sub = d[d["lat_bin"] == lb]
        sube = elong[elong["lat_bin"] == lb]
        few = (sub["tilt_solar_deg"].abs() <= EW_ALIGNED_DEG).mean()
        fewe = (sube["tilt_solar_deg"].abs() <= EW_ALIGNED_DEG).mean()
        lines.append(f"| {lb} | {len(sub)} | {sub['tilt_solar_deg'].abs().median():.1f} | "
                     f"{few:.3f} | {len(sube)} | {fewe:.3f} |")
        binstats[lb] = dict(n=int(len(sub)),
                            med_abs_tilt=float(sub["tilt_solar_deg"].abs().median()),
                            frac_EW=float(few), frac_EW_elong=float(fewe))
    rho_all, p_all = sps.spearmanr(d["abs_lat_deg"], d["tilt_solar_deg"].abs())
    lines += ["", f"Spearman |tilt| vs |lat| (all): rho={rho_all:.3f} p={p_all:.2e}.", "",
              "Signed tilt by hemisphere (elongated subset):", "",
              "| |lat| bin | med tilt N | n N | med tilt S | n S |", "|---|---|---|---|---|"]
    hemi = {}
    for lb in labels:
        sn = elong[(elong["lat_bin"] == lb) & (elong["hemisphere"] == "N")]["tilt_solar_deg"]
        ss = elong[(elong["lat_bin"] == lb) & (elong["hemisphere"] == "S")]["tilt_solar_deg"]
        lines.append(f"| {lb} | {sn.median():.1f} | {len(sn)} | {ss.median():.1f} | {len(ss)} |")
        hemi[lb] = dict(med_N=float(sn.median()) if len(sn) else None, n_N=int(len(sn)),
                        med_S=float(ss.median()) if len(ss) else None, n_S=int(len(ss)))
    rho_j, p_j = sps.spearmanr(elong["lat_deg"], elong["tilt_solar_deg"])
    lines += ["", f"Spearman signed tilt vs signed lat (elongated): rho={rho_j:.3f} p={p_j:.2e} "
              "(Joy's-law-like antisymmetry would give rho != 0).", ""]
    statsd["tilt"] = dict(bins=binstats, hemi_signed=hemi,
                          spearman_abs_tilt_abs_lat=[float(rho_all), float(p_all)],
                          spearman_signed_tilt_signed_lat=[float(rho_j), float(p_j)])
    report.extend(lines)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(labels))
    axes[0].bar(x - 0.18, [binstats[l]["frac_EW"] for l in labels], 0.36,
                label="all", color="#0072B2")
    axes[0].bar(x + 0.18, [binstats[l]["frac_EW_elong"] for l in labels], 0.36,
                label="elongated", color="#E69F00")
    axes[0].set_xticks(x, labels); axes[0].set_xlabel("|latitude| bin (deg)")
    axes[0].set_ylabel(f"Fraction |tilt|<={EW_ALIGNED_DEG:.0f} deg")
    axes[0].legend(fontsize=8); axes[0].set_title("E-W alignment vs latitude")
    for h, c in [("N", "#0072B2"), ("S", "#E69F00")]:
        meds = [hemi[l][f"med_{h}"] for l in labels]
        axes[1].plot(x, meds, marker="o", label=f"{h} hemisphere", color=c)
    axes[1].axhline(0, color="#333", lw=1, ls="--")
    axes[1].set_xticks(x, labels); axes[1].set_xlabel("|latitude| bin (deg)")
    axes[1].set_ylabel("Median signed tilt (deg)")
    axes[1].legend(fontsize=8); axes[1].set_title("Signed tilt by hemisphere (elongated)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "morphcorr_tilt.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Category vs position and size
# ---------------------------------------------------------------------------

def analysis_category_position(df, report, statsd):
    log("[5/7] category proportions vs position/size ...")
    d = df[np.isfinite(df["lat_deg"])].copy()
    lat_bins = [0, 15, 30, 45, 90]
    lat_labels = ["0-15", "15-30", "30-45", "45+"]
    d["lat_bin"] = pd.cut(d["abs_lat_deg"], lat_bins, labels=lat_labels, include_lowest=True)
    r_edges = [0, .3, .5, .7, .85, 1.01]
    r_labels = ["0-.3", ".3-.5", ".5-.7", ".7-.85", ".85+"]
    d["r_bin"] = pd.cut(d["r_norm"], r_edges, labels=r_labels, include_lowest=True)

    prop_lat = pd.crosstab(d["lat_bin"], d["category"], normalize="index")
    prop_r = pd.crosstab(d["r_bin"], d["category"], normalize="index")
    lines = ["## 4. Category (L/R/U) vs position and size", "",
             "Proportions per |lat| bin:", prop_lat.round(3).to_markdown(), "",
             "Proportions per r_norm bin:", prop_r.round(3).to_markdown(), ""]

    # U fraction vs size deciles
    d["area_dec"] = pd.qcut(d["poly_area"], 10, labels=False, duplicates="drop")
    ufrac = d.groupby("area_dec")["is_U"].mean()
    amed = d.groupby("area_dec")["poly_area"].median()
    rho_u, p_u = sps.spearmanr(d["log_area"], d["is_U"])
    lines += ["U fraction by area decile (median area px^2 -> frac U):", ""]
    lines.append(" | ".join(f"{amed[i]:.0f}:{ufrac[i]:.2f}" for i in ufrac.index))
    lines += ["", f"Spearman is_U vs log area: rho={rho_u:.3f} p={p_u:.2e}.",
              f"Median area: U={d.loc[d.is_U==1,'poly_area'].median():.0f} px^2, "
              f"L/R={d.loc[d.is_U==0,'poly_area'].median():.0f} px^2.", ""]
    rho_ur, p_ur = sps.spearmanr(d["r_norm"], d["is_U"])
    rho_ul, p_ul = sps.spearmanr(d["abs_lat_deg"], d["is_U"])
    lines += [f"Spearman is_U vs r_norm: rho={rho_ur:.3f} p={p_ur:.2e}; "
              f"is_U vs |lat|: rho={rho_ul:.3f} p={p_ul:.2e}.", ""]
    statsd["category_position"] = dict(
        prop_lat=prop_lat.round(4).to_dict(), prop_r=prop_r.round(4).to_dict(),
        u_frac_by_area_decile={int(k): float(v) for k, v in ufrac.items()},
        spearman_isU_logarea=[float(rho_u), float(p_u)],
        spearman_isU_rnorm=[float(rho_ur), float(p_ur)],
        spearman_isU_abslat=[float(rho_ul), float(p_ul)],
        med_area_U=float(d.loc[d.is_U == 1, "poly_area"].median()),
        med_area_LR=float(d.loc[d.is_U == 0, "poly_area"].median()))
    report.extend(lines)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, prop, xl in [(axes[0], prop_lat, "|lat| bin"), (axes[1], prop_r, "r_norm bin")]:
        bottom = np.zeros(len(prop))
        for c in ["Left", "Right", "Unidentifiable"]:
            ax.bar(prop.index.astype(str), prop[c], bottom=bottom, label=c,
                   color=CAT_COLORS[c])
            bottom += prop[c].values
        ax.set_xlabel(xl); ax.set_ylabel("Proportion"); ax.legend(fontsize=7)
    axes[2].plot(amed.values, ufrac.values, marker="o", color="#009E73")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Median area of decile (px$^2$)")
    axes[2].set_ylabel("Fraction Unidentifiable")
    axes[2].set_title(f"U fraction vs size (rho={rho_u:.2f})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "morphcorr_category_position.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Spearman correlation matrix
# ---------------------------------------------------------------------------

SPEAR_COLS = ["log_area", "width_proxy", "spine_len", "tortuosity",
              "mean_abs_turn_deg", "boundary_complexity", "eccentricity",
              "r_norm", "abs_lat_deg", "is_U", "file_disagree_norm"]


def analysis_spearman(df, report, statsd, extra_cols=()):
    log("[6/7] Spearman correlation matrix ...")
    cols = SPEAR_COLS + [c for c in extra_cols if c in df.columns]
    sub = df[cols]
    mat = sub.corr(method="spearman")
    lines = ["## 5. Spearman correlation matrix", "",
             "(file_disagree_norm = per-file annotator count-disagreement proxy, "
             "multi-copy files only; pairwise-complete.)", "",
             mat.round(3).to_markdown(), ""]
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            v = mat.loc[a, b]
            if np.isfinite(v):
                pairs.append((a, b, float(v)))
    pairs.sort(key=lambda t: -abs(t[2]))
    lines.append("Strongest pairs: " + "; ".join(
        f"{a}~{b}={v:+.3f}" for a, b, v in pairs[:10]))
    lines.append("")
    statsd["spearman"] = {f"{a}|{b}": v for a, b, v in pairs}
    report.extend(lines)

    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(mat.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)), cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = mat.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v) > 0.55 else "#222")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Spearman correlations (8199 annotations)")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "morphcorr_spearman.png", dpi=150)
    plt.close(fig)
    return pairs


# ---------------------------------------------------------------------------
# 6. Detectability: matched vs missed fold-0 GT under E010 model
# ---------------------------------------------------------------------------

def analysis_detectability(df, report, statsd, conf=0.4, min_area=128, r_max=0.97):
    log("[7/7] detectability: matching fold-0 GT vs E010 preds ...")
    preds_path = ROOT / "outputs" / "oof_cache" / "fold0_preds_e010.parquet"
    if not preds_path.exists():
        report.append("## 6. Detectability\n\nfold0_preds_e010.parquet not found - skipped.\n")
        return None
    cache = OUT_DIR / "morphcorr_fold0_matched.parquet"
    if cache.exists():
        det = pd.read_parquet(cache)
        log(f"    reusing cached match table ({len(det)} rows)")
        return _detect_report(df, det, report, statsd, conf, min_area)
    from filament import IMG_H, IMG_W
    from filament.pq import _entry_iou_matrix
    from filament.rle import decode_rle, rles_to_layers

    preds = pd.read_parquet(preds_path)
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    val_files = set(folds.loc[folds["fold"] == 0, "file_name"])
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt_val = gt[gt["file_name"].isin(val_files)]
    log(f"    fold0: {len(val_files)} files, {len(gt_val)} GT instances, "
        f"{len(preds)} raw preds")

    rows = []
    for i, (fn, sub_f) in enumerate(gt_val.groupby("file_name")):
        stem = fn.rsplit(".", 1)[0]
        g = preds[preds["stem"] == stem]
        g = g[(g["conf"] >= conf) & (g["r_norm"] <= r_max)].sort_values(
            "conf", ascending=False)
        taken = np.zeros((IMG_H, IMG_W), dtype=bool)
        kept = []
        for r in g.itertuples():
            m = decode_rle(r.rle).astype(bool) & ~taken
            if m.sum() < min_area:
                continue
            taken |= m
            kept.append(m)
        pred_layers = (np.stack(kept).astype(np.uint8) if kept
                       else np.zeros((0, IMG_H, IMG_W), np.uint8))
        for entry_id, sub in sub_f.groupby("entry_id"):
            gt_layers = rles_to_layers(sub["segmentation_rle"].tolist())
            iou = _entry_iou_matrix(gt_layers, pred_layers)
            matched = (iou > 0.5).any(axis=1) if iou.size else np.zeros(len(sub), bool)
            best = iou.max(axis=1) if iou.size else np.zeros(len(sub))
            for aid, mt, bi in zip(sub["ann_id"], matched, best):
                rows.append((aid, bool(mt), float(bi)))
        del pred_layers, kept, taken
        if (i + 1) % 30 == 0:
            log(f"    ... {i + 1} files done")
    det = pd.DataFrame(rows, columns=["ann_id", "matched", "best_iou"])
    det.to_parquet(OUT_DIR / "morphcorr_fold0_matched.parquet", index=False)
    return _detect_report(df, det, report, statsd, conf, min_area)


def _detect_report(df, det, report, statsd, conf, min_area):
    d = df.merge(det, on="ann_id", how="inner")
    mrate = d["matched"].mean()
    log(f"    matched {d['matched'].sum()}/{len(d)} GT ({mrate:.3f})")

    feat_cols = ["log_area", "width_proxy", "spine_len", "tortuosity",
                 "mean_abs_turn_deg", "boundary_complexity", "eccentricity",
                 "r_norm", "abs_lat_deg", "is_U"]
    lines = ["## 6. Detectability: fold-0 GT matched vs missed (E010, conf=0.4, "
             f"min_area=128, r<=0.97)", "",
             f"Matched {int(d['matched'].sum())} / {len(d)} GT instances "
             f"(recall proxy {mrate:.3f}).", "",
             "| feature | mean(matched) | mean(missed) | Cohen's d | Spearman(feat, matched) |",
             "|---|---|---|---|---|"]
    cohend = {}
    for c in feat_cols:
        a = d.loc[d["matched"], c].dropna()
        b = d.loc[~d["matched"], c].dropna()
        sp = math.sqrt(((len(a) - 1) * a.std() ** 2 + (len(b) - 1) * b.std() ** 2)
                       / max(len(a) + len(b) - 2, 1))
        cd = (a.mean() - b.mean()) / sp if sp > 0 else np.nan
        rho, _ = sps.spearmanr(d[c], d["matched"].astype(int), nan_policy="omit")
        cohend[c] = dict(d=float(cd), rho=float(rho))
        lines.append(f"| {c} | {a.mean():.3f} | {b.mean():.3f} | {cd:+.3f} | {rho:+.3f} |")

    # logistic AUC (5-fold CV, standardized)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    X = d[feat_cols].replace([np.inf, -np.inf], np.nan)
    y = d["matched"].astype(int).values
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    aucs = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    pipe.fit(X, y)
    coefs = dict(zip(feat_cols, pipe[-1].coef_[0]))
    lines += ["", f"Logistic 5-fold CV AUC = {aucs.mean():.3f} +/- {aucs.std():.3f}.",
              "Standardized logit coefficients: " + ", ".join(
                  f"{k}={v:+.2f}" for k, v in sorted(coefs.items(),
                                                     key=lambda t: -abs(t[1]))), ""]
    statsd["detectability"] = dict(
        n=int(len(d)), matched=int(d["matched"].sum()), match_rate=float(mrate),
        cohens_d=cohend, cv_auc_mean=float(aucs.mean()), cv_auc_std=float(aucs.std()),
        logit_coefs={k: float(v) for k, v in coefs.items()})
    report.extend(lines)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ks = sorted(cohend, key=lambda k: cohend[k]["d"])
    vals = [cohend[k]["d"] for k in ks]
    ax.barh(ks, vals, color=["#D55E00" if v < 0 else "#0072B2" for v in vals])
    ax.axvline(0, color="#333", lw=1)
    ax.set_xlabel("Cohen's d (matched - missed)")
    ax.set_title(f"What separates detected vs missed GT (AUC={aucs.mean():.3f})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "morphcorr_detectability.png", dpi=150)
    plt.close(fig)
    return det


# ---------------------------------------------------------------------------

def main():
    df = build_features()
    df.to_parquet(OUT_DIR / "morphology_correlations.parquet", index=False)
    log("    saved morphology_correlations.parquet (incremental)")

    report = ["# Morphology correlation study", "",
              f"n = {len(df)} annotations across {df['file_name'].nunique()} files; "
              f"fixed disk geometry center=({CX},{CY}) R={R}.", ""]
    statsd = {}

    analysis_complexity(df, report, statsd)
    _save(report, statsd)
    analysis_spine(df, report, statsd)
    _save(report, statsd)
    analysis_tilt(df, report, statsd)
    _save(report, statsd)
    analysis_category_position(df, report, statsd)
    _save(report, statsd)

    # optional best-effort: annotator cluster support from parallel agent
    extra = []
    clus_path = OUT_DIR / "annotator_clusters.parquet"
    if clus_path.exists():
        try:
            clus = pd.read_parquet(clus_path)
            log(f"    annotator_clusters.parquet found: cols={list(clus.columns)}")
            clus["cluster_support"] = clus["support"] / clus["n_copies_file"]
            sub = clus[["member_ann_ids", "cluster_support", "mean_pair_iou"]].copy()
            sub["ann_id"] = sub["member_ann_ids"].str.split(",")
            sub = sub.explode("ann_id")[["ann_id", "cluster_support", "mean_pair_iou"]]
            sub = sub.drop_duplicates("ann_id")
            df = df.merge(sub, on="ann_id", how="left")
            extra = [c for c in ["cluster_support", "mean_pair_iou"] if c in df.columns]
            log(f"    joined cluster support for "
                f"{df['cluster_support'].notna().sum()} / {len(df)} annotations")
        except Exception as e:
            log(f"    cluster join failed (non-fatal): {e}")
    else:
        log("    annotator_clusters.parquet absent -> using file_disagree_norm proxy only")

    analysis_spearman(df, report, statsd, extra_cols=extra)
    _save(report, statsd)

    det = analysis_detectability(df, report, statsd)
    if det is not None:
        df = df.merge(det, on="ann_id", how="left")
    df.to_parquet(OUT_DIR / "morphology_correlations.parquet", index=False)
    _save(report, statsd)
    log("DONE. Report -> outputs/eda/morphology_correlations.md")


def _save(report, statsd):
    (OUT_DIR / "morphology_correlations.md").write_text("\n".join(report),
                                                        encoding="utf-8")
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            o = o.item()
        if isinstance(o, float) and not math.isfinite(o):
            return None
        return o
    with open(OUT_DIR / "morphcorr_stats.json", "w") as f:
        json.dump(clean(statsd), f, indent=1)


if __name__ == "__main__":
    main()
