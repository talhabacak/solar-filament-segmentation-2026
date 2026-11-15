"""E212 -- a regime-conditioned gate: the selector threshold set by the frame's hf.

Ledger 46-48: in busy (high-hf) frames the loss is all recall and half of it is observed at the gate;
the selector's score is on an absolute scale fitted to the pooled training set. The calibration table
of sub036's out-of-fold scores says the scale really moves: the realised value of a candidate at a
given score (v = iou_sum - PQ * n_copies / 2, report eq. 5; adding the candidate raises PQ iff v > 0)
is higher in busy frames in the band just below the gate (p 0.55-0.60: Q1 -0.009, Q2 +0.001,
Q3 +0.049, Q4 +0.076), so the break-even score sits lower there than the deployed 0.60.

The policy keeps sub036 exactly (conf >= 0.40, bagged score p) and changes one thing: the p threshold
is t_low in frames with hf at or below the split and t_high above it. hf is read from the image, so
the policy is legal at inference. The split is fixed in advance at the training median of hf (587),
the natural two-regime cut, not tuned.

Honesty protocol (the lessons of the oracle that did not transfer, ledger 48):
  * the thresholds are chosen by NESTED cross-validation -- on the four other folds, by the summed
    break-even value against the deployed keep set -- and applied to the held-out fold;
  * the resulting keep set is scored with the exact metric, per fold, and against sub036 with a
    paired image bootstrap;
  * the same is done for the single-threshold family (t_low = t_high), the control that says whether
    any gain needs the regime at all;
  * annotator transfer: on the images with two or more annotator copies, thresholds chosen on half
    A of the copies are scored against half B (the test the oracle failed at -0.365).

Usage: python src/eda/regime_gate.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/eda", "src/models"):
    sys.path.insert(0, str(ROOT / p))
from e115c_threshold_ci import per_image, pq  # noqa: E402

OUT = ROOT / "outputs" / "eda"
CONF, T0 = 0.40, 0.60
GRID_LOW = [0.55, 0.575, 0.60, 0.625, 0.65]
GRID_HIGH = [0.45, 0.50, 0.525, 0.55, 0.575, 0.60, 0.625]
PQ_REF = 0.4431
C4 = ["tp", "fp", "fn", "iou"]


def load():
    a = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet").merge(
        pd.read_parquet(OUT / "bag5_oof_scores.parquet"), on="cand_id")
    folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    a["fold"] = a["stem"].map(fold_of).astype(int)
    q = pd.read_parquet(OUT / "image_quality.parquet")
    q = q[q.split == "train"]
    split = float(q["hf"].median())
    a = a.merge(q[["stem", "hf"]], on="stem", how="left")
    assert a["hf"].notna().all()
    a["busy"] = a["hf"] > split
    a["v"] = a["iou_sum"] - PQ_REF * a["n_copies"] / 2
    return a, fold_of, split


def keep(a, t_low, t_high):
    t = np.where(a["busy"].to_numpy(), t_high, t_low)
    return (a["conf"].to_numpy() >= CONF) & (a["p"].to_numpy() >= t)


def value(a, mask, base):
    """Break-even value of switching from the deployed keep set to `mask`."""
    v = a["v"].to_numpy()
    return v[mask & ~base].sum() - v[base & ~mask].sum()


def nested(a, grid_low, grid_high):
    base = keep(a, T0, T0)
    chosen, kept = {}, np.zeros(len(a), bool)
    f = a["fold"].to_numpy()
    for k in range(5):
        tr = f != k
        best = max(((tl, th) for tl in grid_low for th in grid_high),
                   key=lambda t: value(a[tr], keep(a[tr], *t), base[tr]))
        chosen[k] = best
        kept[~tr] = keep(a[~tr], *best)
    return kept, chosen


def score(a, mask, gt, stems, ncopy):
    st = per_image(a[mask][["cand_id", "rle", "stem"]], gt, stems, ncopy)[0]
    return pd.DataFrame({"stem": stems, **{c: np.asarray(st[c], float) for c in C4}})


def main():
    a, fold_of, split = load()
    print(f"hf split (training median) {split:.0f}; busy frames {a.loc[a.busy, 'stem'].nunique()} "
          f"of {a.stem.nunique()} with candidates", flush=True)
    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    stems = np.array(sorted(fold_of))
    gt = gt[gt["stem"].isin(set(stems))]
    ncopy = gt.groupby("stem")["entry_id"].nunique().reindex(stems).fillna(1).to_numpy()
    fold_img = pd.Series(fold_of).reindex(stems).to_numpy()

    base = keep(a, T0, T0)
    arms = {"sub036": base}
    arms["regime (nested)"], ch_r = nested(a, GRID_LOW, GRID_HIGH)
    # control: the single-threshold family, one value for both regimes
    one = sorted(set(GRID_LOW) | set(GRID_HIGH))
    f = a["fold"].to_numpy()
    kept_s, ch_s = np.zeros(len(a), bool), {}
    for k in range(5):
        tr = f != k
        best = max(one, key=lambda t: value(a[tr], keep(a[tr], t, t), base[tr]))
        ch_s[k] = (best, best)
        kept_s[~tr] = keep(a[~tr], best, best)
    arms["single (nested)"] = kept_s
    print("chosen (t_low, t_high) per held-out fold, regime:", ch_r)
    print("chosen threshold per held-out fold, single:", {k: v[0] for k, v in ch_s.items()})

    parts = {}
    for name, m in arms.items():
        parts[name] = score(a, m, gt, stems, ncopy)
        P = parts[name][C4].to_numpy()
        per = [pq(*P[fold_img == k].sum(0)) for k in range(5)]
        print(f"  {name:16s} kept {int(m.sum()):5d}  PQ {pq(*P.sum(0)):.4f}  folds "
              + " ".join(f"{x:.4f}" for x in per), flush=True)

    res = {"split_hf": split, "chosen_regime": {str(k): v for k, v in ch_r.items()},
           "chosen_single": {str(k): v[0] for k, v in ch_s.items()}}
    A = parts["sub036"][C4].to_numpy()
    busy_img = pd.read_parquet(OUT / "image_quality.parquet").query("split == 'train'").set_index(
        "stem")["hf"].reindex(stems).to_numpy() > split
    rng = np.random.default_rng(0)
    for name in ("regime (nested)", "single (nested)"):
        Bp = parts[name][C4].to_numpy()
        d = pq(*Bp.sum(0)) - pq(*A.sum(0))
        dfold = [pq(*Bp[fold_img == k].sum(0)) - pq(*A[fold_img == k].sum(0)) for k in range(5)]
        boots = []
        for _ in range(4000):
            i = rng.integers(0, len(stems), len(stems))
            boots.append(pq(*Bp[i].sum(0)) - pq(*A[i].sum(0)))
        slices = {lab: pq(*Bp[m].sum(0)) - pq(*A[m].sum(0)) for lab, m in (("quiet", ~busy_img), ("busy", busy_img))}
        rec = {lab: Bp[m, 0].sum() / (Bp[m, 0].sum() + Bp[m, 2].sum()) - A[m, 0].sum() / (A[m, 0].sum() + A[m, 2].sum())
               for lab, m in (("quiet", ~busy_img), ("busy", busy_img))}
        res[name] = dict(delta=d, ci=[float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
                         folds=dfold, positive_folds=int(sum(x > 0 for x in dfold)), slices=slices, recall=rec)
        print(f"\n{name}: dPQ {d:+.4f}  95% CI [{np.quantile(boots, .025):+.4f}, {np.quantile(boots, .975):+.4f}]  "
              f"folds " + " ".join(f"{x:+.4f}" for x in dfold) + f"  ({sum(x > 0 for x in dfold)}/5)")
        print(f"  quiet frames dPQ {slices['quiet']:+.4f} (recall {rec['quiet']:+.4f})   "
              f"busy frames dPQ {slices['busy']:+.4f} (recall {rec['busy']:+.4f})")
    res["annotator_transfer"] = transfer_test(a, gt)
    json.dump(res, open(OUT / "regime_gate.json", "w"), indent=1, default=float)


def value_against(entries, G, ent, P):
    """iou_sum and copy count of every candidate against the given annotator entries."""
    sel = np.isin(ent, entries)
    g = G[sel]
    v = np.zeros(len(P))
    for i in range(len(P)):
        inter = (g & P[i]).sum((1, 2)).astype(float)
        iou = inter / (g.sum((1, 2)) + P[i].sum() - inter)
        v[i] = iou[iou > 0.5].sum()
    return v, np.full(len(P), len(set(entries)), float)


def transfer_test(a, gt):
    """Thresholds chosen against annotator half A, scored against half B (ledger 48's protocol).

    floor = deployed gate, transfer = (t_low, t_high) chosen on A, within = chosen on B; all three
    scored against B with the exact metric. transferred fraction = (transfer - floor) / (within - floor).
    """
    from filament import IMG_H, IMG_W
    from filament.pq import score as exact
    from filament.rle import rles_to_layers

    multi = gt.groupby("stem")["entry_id"].nunique()
    stems = sorted(set(multi[multi >= 2].index) & set(a["stem"]))
    print(f"\nannotator transfer on {len(stems)} images with two or more copies", flush=True)
    sub = a[a["stem"].isin(stems)].copy()
    vA = np.zeros(len(sub)); cA = np.zeros(len(sub)); vB = np.zeros(len(sub)); cB = np.zeros(len(sub))
    gtB = []
    pos = {cid: i for i, cid in enumerate(sub["cand_id"])}
    for k, stem in enumerate(stems):
        g = gt[gt["stem"] == stem].reset_index(drop=True)
        ent = g["entry_id"].to_numpy()
        uniq = sorted(set(ent))
        A, B = uniq[:len(uniq) // 2], uniq[len(uniq) // 2:]
        c = sub[sub["stem"] == stem]
        G = rles_to_layers(g["segmentation_rle"].tolist(), IMG_H, IMG_W).astype(bool)
        P = rles_to_layers(c["rle"].tolist(), IMG_H, IMG_W).astype(bool)
        ix = [pos[x] for x in c["cand_id"]]
        vA[ix], cA[ix] = value_against(A, G, ent, P)
        vB[ix], cB[ix] = value_against(B, G, ent, P)
        gtB.append(g[np.isin(ent, B)])
        if k % 100 == 0:
            print(f"  {k}/{len(stems)}", flush=True)
    base = keep(sub, T0, T0)

    def choose(v, c):
        sub["v"] = v - PQ_REF * c / 2
        return max(((tl, th) for tl in GRID_LOW for th in GRID_HIGH),
                   key=lambda t: value(sub, keep(sub, *t), base))

    tA, tB = choose(vA, cA), choose(vB, cB)
    gB = pd.concat(gtB)[["filament_id", "segmentation_rle"]]
    out, cache = {"thresholds_A": tA, "thresholds_B": tB}, {}
    for name, m in (("floor", base), ("transfer", keep(sub, *tA)), ("within", keep(sub, *tB))):
        s = sub[m]
        df = pd.DataFrame({"filament_id": [f"{st}_{i}" for i, st in enumerate(s["stem"])],
                           "segmentation_rle": s["rle"].to_numpy()})
        r = exact(gB, df, gt_cache=cache)
        out[name] = float(r["pq"])
        print(f"  {name:9s} kept {int(m.sum()):5d}  PQ against B {r['pq']:.4f}", flush=True)
    span = out["within"] - out["floor"]
    out["transferred_fraction"] = (out["transfer"] - out["floor"]) / span if abs(span) > 1e-9 else float("nan")
    print(f"  thresholds from A {tA}, from B {tB}; transferred fraction {out['transferred_fraction']:.3f}")
    return out


if __name__ == "__main__":
    main()
