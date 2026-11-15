"""Figures for the final report, generated from committed measurement artefacts.

Each figure answers one question the report makes a claim about, and each is built from a
parquet or json this repository produced, so a reviewer can regenerate it:

  fig1_rq_sq.png        where the loss is, and what kind it is
                        [outputs/eda/pq_decomposition.parquet, src/eda/pq_decomposition.py]
  fig2_atlas.png        what we miss that we cannot excuse, and what it looks like
                        [outputs/eda/error_atlas.parquet, src/eda/error_atlas.py]
  fig3_capture.png      the project's central pattern: intermediate gains do not convert
                        [values from docs/15_ABLATION_LEDGER.md, each cited in the caption]

Usage: python src/eda/report_figures.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs" / "eda"
FIG = ROOT / "outputs" / "figures"


def log(*a):
    print(*a, flush=True)


def stats(d, fp):
    tp = int((d["iou_acc"] > 0.5).sum())
    fn = len(d) - tp
    sq = float(d.loc[d["iou_acc"] > 0.5, "iou_acc"].mean()) if tp else 0.0
    den = tp + 0.5 * fp + 0.5 * fn
    rq = tp / den if den else 0.0
    return sq, rq, sq * rq


def fig1(plt):
    d = pd.read_parquet(OUT / "pq_decomposition.parquet")
    j = json.load(open(OUT / "pq_decomposition.json"))
    bands = [("centre\nr<0.2", 0.0, 0.2), ("disc\n0.2-0.7", 0.2, 0.7), ("outer\n0.7-0.8", 0.7, 0.8),
             ("limb\n0.8-0.9", 0.8, 0.9), ("far limb\nr>0.9", 0.9, 9.9)]
    keys = ["centre r<0.2", "EASY r 0.2-0.7", "outer r 0.7-0.8", "limb r 0.8-0.9", "far limb r>0.9"]
    sq = [j["bands"][k]["SQ"] for k in keys]
    rq = [j["bands"][k]["RQ"] for k in keys]
    pq = [j["bands"][k]["PQ"] for k in keys]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    x = np.arange(len(bands))
    ax[0].plot(x, sq, "o-", color="#1b6ca8", label="SQ (mask quality of matched pairs)")
    ax[0].plot(x, rq, "s-", color="#c0392b", label="RQ (recognition)")
    ax[0].plot(x, pq, "^--", color="#444444", label="PQ")
    ax[0].axhline(0.629, color="#888888", ls=":", lw=1)
    ax[0].text(0.05, 0.515, "two annotators agree\nwith each other at 0.629",
               fontsize=8, color="#666666")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([b[0] for b in bands], fontsize=8)
    ax[0].set_ylim(0, 1.0)
    ax[0].set_ylabel("score")
    ax[0].legend(fontsize=8, loc="upper right")
    ax[0].set_title("Mask quality is flat everywhere; only recognition falls", fontsize=10)

    # contrast panel, inside the easy band
    easy = d[(d["r_norm"] >= 0.2) & (d["r_norm"] < 0.7)].copy()
    fp_easy = j["easy_fp"]
    bins = [(-1, 0.08, "faint\n<=0.08"), (0.08, 0.12, "0.08-0.12"), (0.12, 0.16, "0.12-0.16"),
            (0.16, 1.0, "bright\n>0.16")]
    ls, rs, ns = [], [], []
    for lo, hi, _ in bins:
        sub = easy[(easy["contrast"] > lo) & (easy["contrast"] <= hi)]
        s_, r_, _ = stats(sub, int(round(fp_easy * len(sub) / max(len(easy), 1))))
        ls.append(s_ * (1 - r_))     # loss from matching
        rs.append(r_ * (1 - s_))     # loss from shape
        ns.append(len(sub))
    x = np.arange(len(bins))
    ax[1].bar(x - 0.18, ls, 0.36, color="#c0392b", label="loss from recognition")
    ax[1].bar(x + 0.18, rs, 0.36, color="#1b6ca8", label="loss from shape")
    for i, n in enumerate(ns):
        ax[1].text(i, max(ls[i], rs[i]) + 0.012, f"n={n}", ha="center", fontsize=7, color="#555555")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([b[2] for b in bins], fontsize=8)
    ax[1].set_ylabel("PQ that a perfect stage would return")
    ax[1].legend(fontsize=8)
    ax[1].set_title("Inside the disc the loss inverts with contrast", fontsize=10)
    fig.suptitle("Where the score is lost (deployment-faithful five-fold out-of-fold, 8199 annotator copies)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_rq_sq.png", dpi=160)
    plt.close(fig)
    log("fig1_rq_sq.png")


def fig2(plt):
    a = pd.read_parquet(OUT / "error_atlas.parquet")
    mech = pd.read_parquet(OUT / "crowding_mechanism.parquet")[["fid", "mech"]]
    una = a[(a["n_copies"] == a["n_entries"]) & (a["n_entries"] >= 2) & (a["agree_iou"] >= 0.6)]
    miss = una[una["outcome"] != "matched"].merge(mech, on="fid", how="left")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))

    order = ["matched", "shape", "selector", "never"]
    labels = ["matched", "shape band\n(0.05<IoU<=0.5)", "selector\ndropped", "never\nproposed"]
    share = [float((a["outcome"] == k).mean()) for k in order]
    ax[0].bar(range(4), share, color=["#2e7d32", "#e67e22", "#c0392b", "#7f8c8d"])
    for i, s in enumerate(share):
        ax[0].text(i, s + 0.012, f"{s:.1%}", ha="center", fontsize=9)
    ax[0].set_xticks(range(4))
    ax[0].set_xticklabels(labels, fontsize=8)
    ax[0].set_ylim(0, 0.72)
    ax[0].set_ylabel("share of 8199 annotator copies")
    ax[0].set_title("Outcome of every ground-truth copy", fontsize=10)

    props = ["skel_len", "contrast", "width", "crowding"]
    names = ["skeleton\nlength", "contrast", "drawn\nwidth", "crowding\n(<200 px)"]
    ratio = [float(miss[p].median() / a.loc[a["outcome"] == "matched", p].median()) for p in props]
    cols = ["#e67e22" if r > 1.5 or r < 0.8 else "#95a5a6" for r in ratio]
    ax[1].barh(range(4), ratio, color=cols)
    ax[1].axvline(1.0, color="#333333", lw=1)
    for i, r in enumerate(ratio):
        ax[1].text(r + 0.05, i, f"{r:.2f}x", va="center", fontsize=9)
    ax[1].set_yticks(range(4))
    ax[1].set_yticklabels(names, fontsize=8)
    ax[1].set_xlim(0, 3.5)
    ax[1].set_xlabel("median, unambiguous miss / matched")
    ax[1].set_title(f"The {len(miss)} misses we cannot excuse", fontsize=10)

    # the denominator the metric actually uses at the deployed operating point
    j = json.load(open(OUT / "pq_decomposition.json"))
    tp = sum(j["bands"][k]["tp"] for k in j["bands"])
    fp = sum(j["bands"][k]["fp"] for k in j["bands"])
    fn = sum(j["bands"][k]["fn"] for k in j["bands"])
    den = tp + 0.5 * fp + 0.5 * fn
    mc = miss["mech"].value_counts()
    names2 = {"selector_drop": "selector drop", "weak": "weak (best IoU 0.44)",
              "merged_into": "merged into a neighbour", "split": "fragmented"}
    ks = [k for k in ("selector_drop", "weak", "merged_into", "split") if k in mc.index]
    ax[2].bar(range(len(ks)), [mc[k] for k in ks], color="#34495e")
    for i, k in enumerate(ks):
        ax[2].text(i, mc[k] + 1.2, f"{mc[k]}\n{0.7 * mc[k] / den:+.4f}", ha="center", fontsize=8)
    ax[2].set_xticks(range(len(ks)))
    ax[2].set_xticklabels([names2[k] for k in ks], fontsize=7, rotation=18, ha="right")
    ax[2].set_ylim(0, max(mc) * 1.35)
    ax[2].set_ylabel("copies")
    ax[2].set_title("Mechanism, and the PQ each would return", fontsize=10)
    fig.suptitle("The addressable population: described, priced, and signed by crowding", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_atlas.png", dpi=160)
    plt.close(fig)
    log("fig2_atlas.png")


def fig3(plt):
    # every row is a measurement in docs/15_ABLATION_LEDGER.md; the intermediate quantity is
    # rescaled to a common axis only for display, the realised value is PQ as measured
    rows = [
        ("multi-threshold extraction", 0.0318, -0.0161, "reachability"),
        ("de-overlap moved to the end", 0.0124, 0.0000, "oracle"),
        ("learned cluster representative", 0.0225, -0.0025, "oracle"),
        # ledger §4: oracle +0.024; realised +0.0022 in-sample but -0.0011 cross-fitted, and the
        # figure reports the deployment-faithful (cross-fitted) reading for every row
        ("selection-time de-overlap", 0.0240, -0.0011, "oracle"),
        ("metric-derived value rule", 0.0054, 0.0000, "oracle"),
        ("query lane (Mask2Former)", 0.0275, 0.0020, "unreached GT shaped"),
        ("lane identity for the selector", 0.0030, 0.0003, "one-fold pilot"),
        ("magnetogram as teacher", 0.0146, 0.0015, "pixel AUC"),
        ("transformer member", 0.0550, -0.0052, "error decorrelation"),
        ("two-lane refiner", 0.0300, -0.0150, "crop-level fit"),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    y = np.arange(len(rows))
    ax.barh(y - 0.19, [r[1] for r in rows], 0.38, color="#7fb3d5", label="intermediate quantity gained")
    ax.barh(y + 0.19, [r[2] for r in rows], 0.38, color="#c0392b", label="realised PQ")
    ax.axvline(0, color="#333333", lw=1)
    ax.axvspan(-0.003, 0.003, color="#eeeeee", zorder=0)
    ax.text(0.0032, len(rows) - 0.6, "±0.003 measurement floor", fontsize=7.5, color="#666666")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r[0]}\n({r[3]})" for r in rows], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("gain")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Intermediate gains do not convert: ten of the eighteen measured mechanisms", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_capture.png", dpi=160)
    plt.close(fig)
    log("fig3_capture.png")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.grid": True, "grid.alpha": 0.25,
                         "axes.spines.top": False, "axes.spines.right": False})
    FIG.mkdir(parents=True, exist_ok=True)
    fig1(plt)
    fig2(plt)
    fig3(plt)
    log("FIGURES_DONE")


if __name__ == "__main__":
    main()
