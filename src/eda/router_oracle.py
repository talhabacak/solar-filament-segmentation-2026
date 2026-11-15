"""E153 -- would a learnable router beat plain averaging? The mixture-of-experts question, bounded.

Talha added arXiv 2606.18123v2 (MixTIME), a computational-pathology model whose
central idea is a mixture of experts with a LEARNABLE ROUTER that weights each
expert per input, rather than summing or averaging them. The domain is unrelated
to ours, but the idea lands squarely on our committee: our eight semantic members
are combined by a fixed equal-weight mean, and nothing in the pipeline lets the
weights depend on the image.

E140's law bounds plain averaging: the gain is A*(1 - 1/k) with A = +0.0092, so
at the deployed twelve members 92% is banked and at most +0.0008 remains. A router
is NOT averaging, so it is not bound by that law, and the question deserves a
measurement rather than an extrapolation. The one learned combiner this project
has tried, E135's per-pixel stacker, lost badly (-0.0186 to -0.0958), but it was
per-PIXEL and local; a per-IMAGE router is a different and much smaller function.

This measures the ceiling of the strongest simple router: top-1 gating, i.e. for
each image, use the single best member. That is the upper bound on any hard router
and a good proxy for a soft one, because with members correlating 0.93-0.96 the
soft optimum sits between the mean and the best member.

    production (clDice alone)      the deployed semantic half
    equal-weight mean of eight     the committee, what averaging achieves
    ORACLE top-1 per image         what a perfect router achieves
    the same, chosen on annotator A the part that transfers (E152's rule, now
                                    standing for every ceiling in this project)

If the oracle sits near the mean, routing adds nothing over averaging and the MoE
idea is closed for us. If it sits above but does not transfer, it is closed for the
same reason the ranking ceiling was.

Usage: python src/eda/router_oracle.py [--build]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/eda", "src/models"):
    sys.path.insert(0, str(ROOT / p))
from filament.pq import score  # noqa: E402

OUT = ROOT / "outputs" / "eda"
SEM = ROOT / "outputs" / "semantic"
ARMS = ["cldice", "1536wdec4832", "2048b2", "upernet1536", "b5_1536", "saaf1536", "gn8_1536", "rdn08"]
GATE = (0.40, 0.60)
FOLD = 0


def log(*a):
    print(*a, flush=True)


def build_fields():
    blob = np.load(SEM / f"fold{FOLD}_probs_blob1536.npz")
    for t in ARMS:
        f = SEM / f"fold{FOLD}_probs_sm_{t}.npz"
        if f.exists():
            continue
        z = np.load(SEM / f"fold{FOLD}_probs_{t}.npz")
        out = {s: np.clip(np.rint(0.5 * blob[s].astype(np.float32) + 0.5 * z[s].astype(np.float32)),
                          0, 255).astype(np.uint8) for s in blob.files}
        np.savez_compressed(f, **out)
        log(f"wrote {f.name}")


def build_pools():
    for t in ARMS:
        p = OUT / f"fused_pool_f{FOLD}_sm_{t}_t020_labelled.parquet"
        if p.exists():
            log(f"pool {t}: exists"); continue
        subprocess.run([sys.executable, "src/models/build_fused_pool.py", "--folds", str(FOLD),
                        "--sem-tag", f"_sm_{t}", "--sem-thr", "0.2",
                        "--out", f"fused_pool_f{FOLD}_sm_{t}_t020.parquet"],
                       cwd=ROOT, stdout=open(OUT / f"_pool_sm_{t}.log", "w"), stderr=subprocess.STDOUT)
        subprocess.run([sys.executable, "outputs/eda/_label_one_pool.py",
                        f"fused_pool_f{FOLD}_sm_{t}_t020"], cwd=ROOT)
        log(f"pool {t}: built")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    a_ = ap.parse_args()
    if a_.build:
        build_fields(); build_pools(); return

    import joblib
    from e115c_threshold_ci import make_cls, make_reg
    from extra_features import EXTRA, extra_features, structure_tensor
    from filament.rle import decode_rle
    import cv2

    feats = list(joblib.load(ROOT / "outputs/selector/selector_v5_shrunkblob5prodts.joblib")["features"]) + list(EXTRA)
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    prod = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    prod = prod.merge(pd.read_parquet(OUT / "extra_features_t020.parquet")[["cand_id"] + EXTRA],
                      on="cand_id", how="left")
    prod["fold"] = prod["stem"].map(fold_of).astype(int)
    trn = prod[prod["fold"] != FOLD]
    X, y, v = trn[feats].to_numpy(np.float64), trn["y"].to_numpy(), trn["iou_sum"].to_numpy(np.float64)
    inner = np.zeros(len(trn))
    for g in [k for k in range(5) if k != FOLD]:
        itr = (trn["fold"] != g).to_numpy(); msk = (trn["fold"] == g).to_numpy()
        c = make_cls().fit(X[itr], y[itr]); r = make_reg().fit(X[itr][y[itr] == 1], v[itr][y[itr] == 1])
        inner[msk] = c.predict_proba(X[msk])[:, 1] * np.clip(r.predict(X[msk]), 0, None)
    ref = np.sort(inner)
    cls, reg = make_cls().fit(X, y), make_reg().fit(X[y == 1], v[y == 1])
    g0 = gt[gt["stem"].map(fold_of) == FOLD]
    IMG = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"

    def kept_of(path):
        a = pd.read_parquet(path)
        if any(c not in a.columns for c in EXTRA):
            rows = []
            for stem, gg in a.groupby("stem"):
                img = cv2.imread(str(IMG / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
                T = structure_tensor(img)
                for cid, rle in zip(gg["cand_id"], gg["rle"]):
                    d = extra_features(decode_rle(rle).astype(bool), img, T); d["cand_id"] = cid; rows.append(d)
            a = a.merge(pd.DataFrame(rows), on="cand_id", how="left")
        Z = a[feats].to_numpy(np.float64)
        a["p"] = np.searchsorted(ref, cls.predict_proba(Z)[:, 1] * np.clip(reg.predict(Z), 0, None),
                                 side="right") / len(ref)
        return a[(a["conf"] >= GATE[0]) & (a["p"] >= GATE[1])][["cand_id", "rle", "stem"]]

    def sc(df, gtr):
        return score(gtr[["filament_id", "segmentation_rle"]],
                     df[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id",
                                                            "rle": "segmentation_rle"}))["pq"]

    keeps = {t: kept_of(OUT / f"fused_pool_f{FOLD}_sm_{t}_t020_labelled.parquet") for t in ARMS}
    keeps["MEAN8"] = kept_of(OUT / f"fused_pool_f{FOLD}_df_ref8_t020_labelled.parquet")
    # cached so a learned router can be fitted without refitting the selector
    for t, d in keeps.items():
        d.to_parquet(OUT / f"router_keep_{t}.parquet", index=False)
    log("single members and the equal-weight mean, fold 0:")
    for t, d in keeps.items():
        log(f"  {t:14s} kept {len(d):4d}  PQ {sc(d, g0):.4f}")

    # top-1 routing: per image, take the member whose own candidates score best there.
    # Annotator split so the choice can be made on one opinion and scored on another.
    ncop = gt.groupby("stem")["entry_id"].nunique()
    stems = sorted(set(g0["stem"]))
    multi = [s for s in stems if ncop.get(s, 1) >= 2]
    first = g0.groupby("stem")["entry_id"].transform("min")
    gA, gB = g0[g0["entry_id"] == first], g0[g0["entry_id"] != first]
    log(f"\n{len(multi)} of {len(stems)} fold-0 images carry more than one annotator")

    members = [t for t in ARMS]
    per_image = {}
    for s in multi:
        row = {}
        for t in members + ["MEAN8"]:
            d = keeps[t][keeps[t]["stem"] == s]
            row[t] = (sc(d, gA[gA["stem"] == s]), sc(d, gB[gB["stem"] == s]), d)
        per_image[s] = row

    def assemble(chooser):
        return pd.concat([per_image[s][chooser(s)][2] for s in multi])

    mean_pq = sc(assemble(lambda s: "MEAN8"), gB[gB["stem"].isin(multi)])
    prod_pq = sc(assemble(lambda s: "cldice"), gB[gB["stem"].isin(multi)])
    orc_pq = sc(assemble(lambda s: max(members + ["MEAN8"], key=lambda t: per_image[s][t][1])),
                gB[gB["stem"].isin(multi)])
    trf_pq = sc(assemble(lambda s: max(members + ["MEAN8"], key=lambda t: per_image[s][t][0])),
                gB[gB["stem"].isin(multi)])
    log("\nscored on the held-out annotator, multi-annotator images only:")
    log(f"  production, clDice alone                   {prod_pq:.4f}")
    log(f"  equal-weight mean of eight (the committee) {mean_pq:.4f}  ({mean_pq - prod_pq:+.4f})")
    log(f"  ORACLE top-1 router, chosen on the SAME    {orc_pq:.4f}  ({orc_pq - mean_pq:+.4f} over the mean)")
    log(f"  top-1 router chosen on the OTHER annotator {trf_pq:.4f}  ({trf_pq - mean_pq:+.4f} over the mean)")
    frac = (trf_pq - mean_pq) / (orc_pq - mean_pq) if orc_pq > mean_pq else float("nan")
    log(f"\n  routing beats averaging by {orc_pq - mean_pq:+.4f} with the answer in hand, "
        f"and {trf_pq - mean_pq:+.4f} without it ({100 * frac:.0f}% transfers)")
    json.dump(dict(production=float(prod_pq), mean8=float(mean_pq), oracle=float(orc_pq),
                   transfer=float(trf_pq), transferred_fraction=float(frac),
                   singles={t: float(sc(keeps[t], g0)) for t in ARMS}),
              open(OUT / "router_oracle.json", "w"), indent=1, default=float)
    log(f"\nwrote {OUT / 'router_oracle.json'}")


if __name__ == "__main__":
    main()
