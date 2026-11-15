"""Is the test-time semantic field a different object from the OOF field? (ledger 19a)

Three local gains in one night (committee, committee4, threshold 0.2 + refit)
each ranked below sub025 on the board. Every OOF measurement uses ONE model per
fold (the fold's held-out clDice and blob models); the shipped field is the
FIVE-fold mean of each plus two heavy members. Averaging smooths: fewer
speckles, lower peak probabilities on faint objects, wider mid-probability
skirts -- exactly what a threshold change or a committee interacts with.

Per-image component statistics of a field P at threshold t (min area 256):
  n_comp, mean area, fraction of pixels with 0.2 <= P < 0.5 (skirt), mean P
  inside components, fraction of components with max P < 0.7 (faint), and the
  ratio n_comp(0.2)/n_comp(0.3).

Fields compared:
  OOF     blob5ens fold maps (0.5 blob + 0.5 clDice, single models), 703 stems
  TEST-1  0.5 blob_f0 + 0.5 clDice_f0 on the 180 test images (single models,
          the OOF construction applied to test)
  TEST-5  0.5 mean(blob f0-4) + 0.5 mean(clDice f0-4) (recipe_blob5_pure)
  TEST-H  recipe_blob5_50 as shipped (five-fold means + heavies)

If TEST-5/TEST-H differ from TEST-1 the way they differ from OOF, the mismatch
is the averaging, and OOF numbers for field-sensitive changes cannot transfer.

Usage: python src/eda/field_smoothness.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEM = ROOT / "outputs" / "semantic"
OUT = ROOT / "outputs" / "eda"


def log(*a):
    print(*a, flush=True)


def stats(P, t):
    m = (P >= t).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 256]
    areas = [int(st[i, cv2.CC_STAT_AREA]) for i in keep]
    faint = 0
    for i in keep:
        if P[lab == i].max() < 0.7:
            faint += 1
    return dict(n=len(keep), area=float(np.mean(areas)) if areas else 0.0,
                faint=faint / max(len(keep), 1),
                pin=float(P[np.isin(lab, keep)].mean()) if keep else 0.0)


def summarise(name, get, stems):
    rows = []
    for s in stems:
        P = get(s)
        a3, a2 = stats(P, 0.3), stats(P, 0.2)
        rows.append(dict(n3=a3["n"], area3=a3["area"], faint3=a3["faint"], pin3=a3["pin"], n2=a2["n"],
                         skirt=float(((P >= 0.2) & (P < 0.5)).mean() * 1e3), fg3=float((P >= 0.3).mean() * 1e3)))
    d = pd.DataFrame(rows)
    med = d.median()
    log(f"{name:7s} n={len(stems):3d}  comp@0.3 {med.n3:5.1f}  area {med.area3:7.0f}  faint(max<0.7) {d.faint3.mean():.3f}  "
        f"P_in {med.pin3:.3f}  comp@0.2/0.3 {d.n2.sum()/max(d.n3.sum(),1):.3f}  skirt‰ {med.skirt:.2f}  fg‰@0.3 {med.fg3:.2f}")
    return d.mean().to_dict()


def main():
    res = {}
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    Z = {k: np.load(SEM / f"fold{k}_probs_blob5ens.npz") for k in range(5)}
    stems = sorted(fold_of)
    res["OOF"] = summarise("OOF", lambda s: Z[fold_of[s]][s].astype(np.float32) / 255, stems)
    M = {t: np.load(SEM / f"test_probs_m_{t}.npz") for t in ["cldice_f0", "blob_f0"] + [f"cldice_f{k}" for k in range(1, 5)] + [f"blob_f{k}" for k in range(1, 5)]}
    tstems = sorted(M["cldice_f0"].files)
    res["TEST-1"] = summarise("TEST-1", lambda s: 0.5 * M["blob_f0"][s].astype(np.float32) / 255 + 0.5 * M["cldice_f0"][s].astype(np.float32) / 255, tstems)
    def five(s):
        c = np.mean([M[f"cldice_f{k}"][s].astype(np.float32) for k in range(5)], axis=0) / 255
        b = np.mean([M[f"blob_f{k}"][s].astype(np.float32) for k in range(5)], axis=0) / 255
        return 0.5 * b + 0.5 * c
    res["TEST-5"] = summarise("TEST-5", five, tstems)
    H = np.load(SEM / "test_probs_recipe_blob5_50.npz")
    res["TEST-H"] = summarise("TEST-H", lambda s: H[s].astype(np.float32) / 255, tstems)
    json.dump(res, open(OUT / "field_smoothness.json", "w"), indent=1, default=float)
    log(f"wrote {OUT / 'field_smoothness.json'}")


if __name__ == "__main__":
    main()
