"""E159 -- do the filaments we never propose sit on polarity inversion lines?

The chirality re-measurement (2026-09-09) found the population the pipeline loses
most: filaments whose chirality the annotators could not identify. They are 37.5%
of the ground truth, and against Left/Right they are lost at both stages:

    category         in pool   deployed   dropped by selector   never proposed
    Left               73.6%     64.9%           8.6%               26.4%
    Right              75.1%     66.9%           8.2%               24.9%
    Unidentifiable     65.8%     46.4%          19.4%               34.2%

and it is not size: in the largest area quartile U is missed at 52.3% against 28%.

The physical reading is that chirality is "unidentifiable" exactly when the
magnetic structure is not legible in H-alpha -- and the magnetogram shows that
structure directly. If that is right, the never-proposed U filaments should lie on
PILs as strongly as the ones we find, and a magnetogram-aware candidate generator
could reach them. If instead they sit OFF the PILs, or on weak ones, the
magnetogram cannot find what H-alpha cannot, and the hypothesis is closed before
any model is built.

So this measures, for every ground-truth row on an image with a registered
magnetogram, its magnetic features (distance to the nearest PIL, fraction of the
mask within the PIL band, field gradient on the PIL, |B| in the mask) and compares
them across outcome (found / dropped / never proposed) and category.

The control that decides whether the features mean anything at all: ground-truth
filaments against RANDOM disk locations of the same size. If real filaments do not
sit closer to PILs than random patches, the registration or the PIL extraction is
wrong and nothing downstream can be trusted.

Usage: python src/eda/pil_hypothesis.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/eda", "src/models"):
    sys.path.insert(0, str(ROOT / p))
from filament.pq import score  # noqa: E402
from filament.rle import decode_rle  # noqa: E402
from magnetogram_features import load_registered, pil_maps, mask_features, PIL_FEATS  # noqa: E402

OUT = ROOT / "outputs" / "eda"
GATE = (0.40, 0.60)
NAMES = {1: "Left", 2: "Right", 3: "Unidentifiable", 4: "Ambiguous"}


def log(*a):
    print(*a, flush=True)


def main():
    a = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    a = a.merge(pd.read_parquet(OUT / "t020pox_oof_scores.parquet"), on="cand_id", how="left")
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt = gt[gt["stem"].isin(set(a["stem"]))]

    def hits(cand):
        r = score(gt[["filament_id", "segmentation_rle"]],
                  cand[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id",
                                                           "rle": "segmentation_rle"}),
                  return_details=True)
        return set(r["matches"]["gt_filament_id"])
    dep, pool = hits(a[(a["conf"] >= GATE[0]) & (a["p"] >= GATE[1])]), hits(a)
    gt["outcome"] = np.where(gt["filament_id"].isin(dep), "found",
                    np.where(gt["filament_id"].isin(pool), "dropped", "never proposed"))

    have = {p.stem.replace("reg_", "") for p in (ROOT / "outputs" / "magnetograms").glob("reg_*.npy")}
    g = gt[gt["stem"].isin(have)].copy()
    log(f"{len(g)} ground-truth rows on {g['stem'].nunique()} images with a registered magnetogram")

    rng = np.random.default_rng(0)
    rows, ctrl = [], []
    for stem, gg in g.groupby("stem"):
        B = load_registered(stem)
        maps = pil_maps(B) if B is not None else None
        if maps is None:
            continue
        for r in gg.itertuples():
            m = decode_rle(r.segmentation_rle).astype(bool)
            f = mask_features(m, maps)
            f.update(filament_id=r.filament_id, category=NAMES.get(r.category_id, r.category_id),
                     outcome=r.outcome, area=int(m.sum()))
            rows.append(f)
            # the control: the same mask translated to a random on-disk position
            ys, xs = np.nonzero(m)
            for _ in range(2):
                dy, dx = rng.integers(-600, 600, 2)
                y2, x2 = ys + dy, xs + dx
                ok = (y2 >= 0) & (y2 < 2048) & (x2 >= 0) & (x2 < 2048)
                if ok.mean() < 0.9:
                    continue
                rr = np.hypot(x2[ok] - 1023.5, y2[ok] - 1023.5) / 904.1
                if rr.mean() > 0.85:
                    continue
                m2 = np.zeros_like(m); m2[y2[ok], x2[ok]] = True
                fc = mask_features(m2, maps)
                fc.update(category="RANDOM", outcome="random", area=int(m2.sum()))
                ctrl.append(fc)
    d = pd.DataFrame(rows + ctrl)
    d.to_parquet(OUT / "pil_hypothesis.parquet", index=False)

    cols = ["pil_dist_mean", "pil_frac_near", "pil_strength", "b_abs_mean", "b_grad_mean"]
    log("\nCONTROL -- real filaments against random same-size patches (medians):")
    for c in cols:
        real = d.loc[d["outcome"] != "random", c].median()
        rand = d.loc[d["outcome"] == "random", c].median()
        log(f"  {c:14s} real {real:8.3f}   random {rand:8.3f}")
    ctl_ok = (d.loc[d["outcome"] != "random", "pil_dist_mean"].median()
              < d.loc[d["outcome"] == "random", "pil_dist_mean"].median())
    log("  -> " + ("filaments sit closer to PILs than chance: the geometry is meaningful"
                   if ctl_ok else "NO closer than chance: registration or PIL extraction is wrong"))

    log("\nby outcome (medians; 'never proposed' is the open population):")
    t = d[d["outcome"] != "random"].groupby("outcome")[cols].median()
    log(t.round(3).to_string())
    log("\nby category x outcome, PIL distance and PIL strength:")
    t2 = d[d["outcome"] != "random"].pivot_table(index="category", columns="outcome",
                                                  values=["pil_dist_mean", "pil_strength"], aggfunc="median")
    log(t2.round(3).to_string())
    u_np = d[(d["category"] == "Unidentifiable") & (d["outcome"] == "never proposed")]
    u_f = d[(d["category"] == "Unidentifiable") & (d["outcome"] == "found")]
    lr_np = d[(d["category"].isin(["Left", "Right"])) & (d["outcome"] == "never proposed")]
    log(f"\nUnidentifiable, never proposed (n={len(u_np)}): PIL distance {u_np['pil_dist_mean'].median():.2f}, "
        f"strength {u_np['pil_strength'].median():.3f}")
    log(f"Unidentifiable, found          (n={len(u_f)}): PIL distance {u_f['pil_dist_mean'].median():.2f}, "
        f"strength {u_f['pil_strength'].median():.3f}")
    log(f"Left/Right, never proposed     (n={len(lr_np)}): PIL distance {lr_np['pil_dist_mean'].median():.2f}, "
        f"strength {lr_np['pil_strength'].median():.3f}")
    log(f"random patches                 (n={(d['outcome'] == 'random').sum()}): PIL distance "
        f"{d.loc[d['outcome'] == 'random', 'pil_dist_mean'].median():.2f}")
    json.dump(dict(n_rows=int((d["outcome"] != "random").sum()), n_images=int(g["stem"].nunique()),
                   control_ok=bool(ctl_ok),
                   medians={k: v for k, v in t.to_dict().items()},
                   by_cat={"|".join(map(str, k)): v for k, v in t2.to_dict().items()}),
              open(OUT / "pil_hypothesis.json", "w"), indent=1, default=str)
    log(f"\nwrote {OUT / 'pil_hypothesis.json'}")


if __name__ == "__main__":
    main()
