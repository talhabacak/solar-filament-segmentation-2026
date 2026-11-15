"""E209b -- is the hf oracle ceiling REAL, or is it annotator-specific noise?

E209 says a perfect per-candidate decision recovers +0.0498 in the top hf quartile, rising
monotonically with hf. Before that funds anything it has to survive the test this project made a
standing rule after the contour-level ceiling turned out to be 85% selection-on-noise
(feedback-oracle-selection-bias) and after both headline oracles went negative across annotators
(project-annotator-ceiling).

The oracle decides with `v = iou_sum`, the value REALISED against the annotator entries of that
image. If the decision is a property of the image it must transfer to a different annotator's
copies of the same image; if it is a property of one person's pen it will not.

Restricted to images with at least two entries, split the entries in half (A, B) and compare:

  floor     the deployed gate, scored against B
  transfer  keep iff break-even computed on A, scored against B      <- the honest ceiling
  within    keep iff break-even computed on B, scored against B      <- the inflated ceiling

transferred fraction = (transfer - floor) / (within - floor). The contour-level version of this
number was 0.146.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/eda", "src/models"):
    sys.path.insert(0, str(ROOT / p))
from filament.rle import rles_to_layers  # noqa: E402
from filament import IMG_H, IMG_W  # noqa: E402
from filament.pq import score  # noqa: E402

OUT = ROOT / "outputs" / "eda"
GATE, P0 = (0.40, 0.60), 0.4433


def value_against(entries, G, ent, P, idx):
    """iou_sum and copy count of each candidate against the given annotator entries."""
    sel = np.isin(ent, entries)
    if not sel.any():
        return np.zeros(len(idx)), np.zeros(len(idx))
    g = G[sel]
    v = np.zeros(len(idx))
    for j, i in enumerate(idx):
        inter = (g & P[i]).sum((1, 2)).astype(float)
        iou = inter / (g.sum((1, 2)) + P[i].sum() - inter)
        v[j] = iou[iou > 0.5].sum()
    return v, np.full(len(idx), len(set(entries)), float)


def main():
    pool = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    pool = pool.merge(pd.read_parquet(OUT / "bag5_oof_scores.parquet"), on="cand_id")
    q = pd.read_parquet(OUT / "image_quality.parquet")
    pool = pool.merge(q[q.split == "train"][["stem", "hf"]], on="stem", how="left")
    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    multi = gt.groupby("stem")["entry_id"].nunique()
    stems = sorted(multi[multi >= 2].index)
    print(f"{len(stems)} cok-anotatorlu goruntu", flush=True)
    rows = {"floor": [], "transfer": [], "within": []}
    gtB = []
    for k, stem in enumerate(stems):
        g = gt[gt["stem"] == stem].reset_index(drop=True)
        ent = g["entry_id"].to_numpy()
        uniq = sorted(set(ent))
        A, B = uniq[:len(uniq) // 2] or uniq[:1], uniq[len(uniq) // 2:]
        c = pool[pool["stem"] == stem].reset_index(drop=True)
        if not len(c):
            continue
        G = rles_to_layers(g["segmentation_rle"].tolist(), IMG_H, IMG_W).astype(bool)
        P = rles_to_layers(c["rle"].tolist(), IMG_H, IMG_W).astype(bool)
        idx = np.arange(len(c))
        vA, cA = value_against(A, G, ent, P, idx)
        kept = ((c["conf"] >= GATE[0]) & (c["p"] >= GATE[1])).to_numpy()
        vB, cB = value_against(B, G, ent, P, idx)
        pol = {"floor": kept, "transfer": vA > 0.5 * cA * P0, "within": vB > 0.5 * cB * P0}
        for name, keep in pol.items():
            for i in np.flatnonzero(keep):
                rows[name].append((f"{stem}_{i}", c["rle"].iloc[i]))
        gtB.append(g[np.isin(ent, B)])
        if k % 120 == 0:
            print(f"  {k}/{len(stems)}", flush=True)
    gB = pd.concat(gtB)[["filament_id", "segmentation_rle"]]
    res, cache = {}, {}
    for name in ("floor", "transfer", "within"):
        df = pd.DataFrame(rows[name], columns=["filament_id", "segmentation_rle"])
        r = score(gB, df, gt_cache=cache)
        res[name] = dict(pq=float(r["pq"]), n=len(df))
        print(f"{name:9s} {len(df):5d} tahmin  PQ {r['pq']:.4f}")
    frac = (res["transfer"]["pq"] - res["floor"]["pq"]) / max(res["within"]["pq"] - res["floor"]["pq"], 1e-9)
    res["transferred_fraction"] = float(frac)
    print(f"\naktarilan pay = (transfer - floor) / (within - floor) = {frac:.3f}")
    print("  kontur-seviyesi surumunde bu sayi 0.146 cikmisti ve eksen kapanmisti")
    json.dump(res, open(OUT / "hf_oracle_transfer.json", "w"), indent=1)


if __name__ == "__main__":
    main()
