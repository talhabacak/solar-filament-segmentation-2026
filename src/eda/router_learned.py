"""E153b -- can a router actually be learned? The first oracle in this project whose transferable part is positive.

E153 measured the mixture-of-experts idea from arXiv 2606.18123v2 the way every
ceiling now gets measured, with the annotator transfer test:

    production, clDice alone                      0.4590
    equal-weight mean of eight (the committee)    0.4646   +0.0056
    ORACLE top-1 router, chosen on the SAME       0.4994   +0.0348 over the mean
    top-1 router chosen on the OTHER annotator    0.4724   +0.0079 over the mean

23% transfers. That is the first positive transferable number since the annotator
ceiling closed the selection stage, and the reason is structural: choosing which
MEMBER handles an image is a judgement about the IMAGE -- how noisy it is, how much
the members disagree, which site took it -- and an image is equally noisy for every
annotator. Choosing which CANDIDATE an annotator will like is not, which is why
E152's ranking oracle transferred at -25%.

So the ceiling for an image-level router is about +0.0079 over the committee. This
asks whether any of it is reachable without the answer, using only quantities
computable at inference time:

    image        disk-mean intensity, contrast, gradient energy, a noise proxy,
                 measured disk radius, observatory site, year
    consensus    across the eight member fields on that image: mean, spread,
                 mean pairwise correlation, area above 0.2, disagreement mass
    detector     candidate count and confidence spread from the detector lane

Following E146's lesson, the model does not classify the winning arm. It predicts
each arm's per-image score with the arm identity as an input, so all arms share the
fit, and the decision is the argmax of the prediction.

Honest twice over: the score is fitted on ONE annotator and evaluated on ANOTHER,
and the router is cross-validated over images so no image informs its own choice.

Usage: python src/eda/router_learned.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/eda", "src/models"):
    sys.path.insert(0, str(ROOT / p))
from filament.pq import score  # noqa: E402

OUT = ROOT / "outputs" / "eda"
SEM = ROOT / "outputs" / "semantic"
IMG = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
ARMS = ["cldice", "1536wdec4832", "2048b2", "upernet1536", "b5_1536", "saaf1536", "gn8_1536", "rdn08"]
GATE = (0.40, 0.60)
FOLD = 0


def log(*a):
    print(*a, flush=True)


def image_features(stem, fields, geo):
    g = cv2.imread(str(IMG / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    yy, xx = np.mgrid[0:g.shape[0], 0:g.shape[1]]
    e = geo.get(stem, dict(cx=1023.5, cy=1023.5, radius=904.1))
    rr = np.hypot(xx - e["cx"], yy - e["cy"]) / e["radius"]
    on = rr < 0.95
    d = g[on]
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)[on]
    # a noise proxy that does not respond to real structure: the median absolute
    # difference between a pixel and its 3x3 median
    med = cv2.medianBlur(g.astype(np.uint8), 3).astype(np.float32)
    f = dict(img_mean=float(d.mean()), img_std=float(d.std()),
             img_p05=float(np.percentile(d, 5)), img_p95=float(np.percentile(d, 95)),
             grad_mean=float(grad.mean()), grad_p95=float(np.percentile(grad, 95)),
             noise=float(np.median(np.abs(g - med)[on])),
             disk_radius=float(e["radius"]), disk_off=float(np.hypot(e["cx"] - 1023.5, e["cy"] - 1023.5)),
             year=float(int(stem[:4])), site=float(ord(stem[-2])))
    P = np.stack([fields[t][stem].astype(np.float32) / 255.0 for t in ARMS])
    small = P[:, ::4, ::4].reshape(len(ARMS), -1)
    c = np.corrcoef(small)
    f.update(field_mean=float(P.mean()), field_spread=float(P.std(axis=0).mean()),
             field_area20=float((P.mean(axis=0) > 0.2).mean()),
             member_corr=float(c[np.triu_indices(len(ARMS), 1)].mean()),
             member_corr_min=float(c[np.triu_indices(len(ARMS), 1)].min()),
             disagree=float((P.std(axis=0) > 0.15).mean()))
    return f


def main():
    from sklearn.ensemble import HistGradientBoostingRegressor
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    g0 = gt[gt["stem"].map(fold_of) == FOLD]
    ncop = gt.groupby("stem")["entry_id"].nunique()
    multi = [s for s in sorted(set(g0["stem"])) if ncop.get(s, 1) >= 2]
    log(f"{len(multi)} multi-annotator images in fold 0")
    first = g0.groupby("stem")["entry_id"].transform("min")
    gA, gB = g0[g0["entry_id"] == first], g0[g0["entry_id"] != first]

    keeps = {}
    for t in ARMS + ["MEAN8"]:
        p = (OUT / f"fused_pool_f{FOLD}_df_ref8_t020_labelled.parquet" if t == "MEAN8"
             else OUT / f"fused_pool_f{FOLD}_sm_{t}_t020_labelled.parquet")
        a = pd.read_parquet(p)
        s = pd.read_parquet(OUT / f"router_scores_{t}.parquet") if (OUT / f"router_scores_{t}.parquet").exists() else None
        keeps[t] = a
    # the per-arm keep sets were already scored by E153; reuse its selector output
    sel = json.load(open(OUT / "router_oracle.json"))
    log(f"reusing E153's arm scores (production {sel['production']:.4f}, mean8 {sel['mean8']:.4f})")

    fields = {t: np.load(SEM / f"fold{FOLD}_probs_{t}.npz") for t in ARMS}
    geo = {}
    gp = OUT / "disk_geometry.parquet"
    if gp.exists():
        d = pd.read_parquet(gp)
        d = d[d["r_iqr"] < 5]
        geo = d.set_index("stem")[["cx", "cy", "radius"]].to_dict("index")
    log(f"{len(geo)} images with a clean disk fit")

    feats = pd.DataFrame([dict(stem=s, **image_features(s, fields, geo)) for s in multi])
    fcols = [c for c in feats.columns if c != "stem"]
    log(f"image features: {len(fcols)} over {len(feats)} images")

    # per-image, per-arm score on each annotator group, from the E153 keep sets
    def per_image_pq(df, gtr, stem):
        d = df[df["stem"] == stem]
        gg = gtr[gtr["stem"] == stem]
        if not len(gg):
            return np.nan
        return score(gg[["filament_id", "segmentation_rle"]],
                     d[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id",
                                                           "rle": "segmentation_rle"}))["pq"]

    kept = {}
    for t in ARMS + ["MEAN8"]:
        f = OUT / f"router_keep_{t}.parquet"
        kept[t] = pd.read_parquet(f) if f.exists() else None
    if any(v is None for v in kept.values()):
        log("per-arm keep sets not cached; rerun src/eda/router_oracle.py first")
        return

    rows = []
    for stem in multi:
        for t in ARMS + ["MEAN8"]:
            rows.append(dict(stem=stem, arm=t,
                             pqA=per_image_pq(kept[t], gA, stem),
                             pqB=per_image_pq(kept[t], gB, stem)))
    sc = pd.DataFrame(rows).merge(feats, on="stem")
    sc["arm_id"] = sc["arm"].map({t: i for i, t in enumerate(ARMS + ["MEAN8"])}).astype(float)
    X = fcols + ["arm_id"]
    log(f"training rows {len(sc)}")

    rng = np.random.default_rng(0)
    order = rng.permutation(multi)
    folds_img = {s: i % 5 for i, s in enumerate(order)}
    sc["cv"] = sc["stem"].map(folds_img)
    sc["pred"] = np.nan
    for k in range(5):
        tr, te = sc["cv"] != k, sc["cv"] == k
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=20, l2_regularization=1.0, random_state=0)
        m.fit(sc.loc[tr, X].to_numpy(np.float64), sc.loc[tr, "pqA"].to_numpy())
        sc.loc[te, "pred"] = m.predict(sc.loc[te, X].to_numpy(np.float64))

    pick = sc.loc[sc.groupby("stem")["pred"].idxmax(), ["stem", "arm"]]
    chosen = dict(zip(pick["stem"], pick["arm"]))
    log("\nrouter picks: " + ", ".join(
        f"{t}x{sum(1 for v in chosen.values() if v == t)}" for t in ARMS + ["MEAN8"]
        if any(v == t for v in chosen.values())))

    def assemble(f):
        return pd.concat([kept[f(s)][kept[f(s)]["stem"] == s] for s in multi])

    gBm = gB[gB["stem"].isin(multi)]
    def sc_on_B(df):
        return score(gBm[["filament_id", "segmentation_rle"]],
                     df[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id",
                                                            "rle": "segmentation_rle"}))["pq"]
    mean_pq = sc_on_B(assemble(lambda s: "MEAN8"))
    learned = sc_on_B(assemble(lambda s: chosen[s]))
    log(f"\nscored on the held-out annotator:")
    log(f"  equal-weight mean of eight   {mean_pq:.4f}")
    log(f"  LEARNED router               {learned:.4f}  ({learned - mean_pq:+.4f})")
    log(f"  the transferable ceiling was +0.0079, so this captures "
        f"{100 * (learned - mean_pq) / 0.0079:.0f}% of it")
    json.dump(dict(mean8=float(mean_pq), learned=float(learned),
                   delta=float(learned - mean_pq),
                   fraction_of_ceiling=float((learned - mean_pq) / 0.0079),
                   picks={t: int(sum(1 for v in chosen.values() if v == t)) for t in ARMS + ["MEAN8"]}),
              open(OUT / "router_learned.json", "w"), indent=1, default=float)
    log(f"wrote {OUT / 'router_learned.json'}")


if __name__ == "__main__":
    main()
