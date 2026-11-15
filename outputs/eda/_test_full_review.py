"""All 180 test images with our predictions, plus the one check a visual pass cannot make by eye.

The public and private boards are two undisclosed halves of these same 180 images, so reviewing
the private part means reviewing all of them. Two outputs:

1. Longitude of kept masks, test against training out of fold. The annotation protocol ignores
   filaments beyond +-70 deg heliographic longitude (ledger 27n: 14 pool candidates beyond it, all
   negative), so a kept mask out there is a false positive by construction. If test keeps more of
   them than training, that is a test-side failure the local score cannot show.
   Longitude from the disc position with B0 = 0 (the solar tilt is at most 7.25 deg, so the
   boundary is approximate by a few degrees near the poles).

2. Contact sheets, 20 images each, every tile with sub036 masks (green), marginal-band candidates
   (yellow), and the approximate +-70 deg longitude boundary (grey), labelled with hf quartile,
   halo flag and kept / candidates. Ordered by year so epoch effects are visible.

Diagnosis only: nothing here feeds back into the submission.
"""
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))
from filament.rle import decode_rle  # noqa: E402
from submit_from_pool import scores  # noqa: E402

TEST = ROOT / "data/MAGFiLO_1.0_Kaggle_2026/test/test_images"
R_SUN, C = 904.1, 1023.5
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)


def lon_deg(cx, cy):
    """Heliographic longitude from the central meridian, B0 = 0, disc radius R_SUN."""
    x = (np.asarray(cx, float) - C) / R_SUN
    y = (np.asarray(cy, float) - C) / R_SUN
    y = np.clip(y, -0.999, 0.999)
    cosb = np.sqrt(1 - y ** 2)
    return np.degrees(np.arcsin(np.clip(x / np.maximum(cosb, 1e-6), -1, 1)))


q = pd.read_parquet(ROOT / "outputs/eda/image_quality.parquet")
edges = q[q.split == "train"]["hf"].quantile([.25, .5, .75]).to_numpy()
q["q"] = pd.cut(q["hf"], [-np.inf, *edges, np.inf], labels=["Q1", "Q2", "Q3", "Q4"])
halo = pd.read_parquet(ROOT / "outputs/eda/sky_halo.parquet")[["stem", "split", "bright_halo"]]
q = q.merge(halo, on=["stem", "split"], how="left")

b = joblib.load(ROOT / "outputs/selector/selector_t020pox_bag5.joblib")
te = pd.read_parquet(ROOT / "outputs/eda/test_pool_t020pox.parquet")
te["p"] = scores(b, te[b["features"]].to_numpy(np.float64))
tr = pd.read_parquet(ROOT / "outputs/eda/fused_pool_all_blob5_t020_labelled.parquet").merge(
    pd.read_parquet(ROOT / "outputs/eda/bag5_oof_scores.parquet"), on="cand_id")

print("LONGITUDE OF KEPT MASKS (|lon|, B0 = 0)")
for name, d in (("train OOF", tr), ("test", te)):
    k = d[(d.conf >= .40) & (d.p >= .60)].copy()
    k["lon"] = np.abs(lon_deg(k.cx, k.cy))
    bins = pd.cut(k["lon"], [0, 50, 60, 65, 70, 90], right=False)
    t = k.groupby(bins, observed=False).size()
    share = (t / len(k)).round(4)
    extra = ""
    if "y" in k.columns:
        extra = "  positive rate " + " ".join(f"{v:.2f}" for v in k.groupby(bins, observed=False)["y"].mean())
    print(f"  {name:10s} n={len(k):5d}  " + "  ".join(f"{str(i)}:{s:.3f}" for i, s in share.items()) + extra)
    if name == "test":
        far = k[k["lon"] >= 70]
        print(f"  test masks at |lon| >= 70: {len(far)} in {far.stem.nunique()} images: "
              + ", ".join(sorted(far.stem.unique()))[:300])

sub = pd.read_csv(ROOT / "outputs/submissions/sub036_t020pox_bag5.csv")
sub["stem"] = sub["filament_id"].str.rsplit("_", n=1).str[0]
per = te.assign(kept=(te.conf >= .40) & (te.p >= .60)).groupby("stem").agg(
    cand=("cand_id", "size"), kept=("kept", "sum"))
qt = q[q.split == "test"].set_index("stem")
stems = sorted(p.stem for p in TEST.glob("*.jpeg"))
stems.sort(key=lambda s: (s[:4], s))

T = 380
bnd = np.zeros((T, T), np.uint8)
yy, xx = np.mgrid[0:T, 0:T]
s2 = 2048 / T
lon_map = np.abs(lon_deg(xx * s2, yy * s2))
r_map = np.hypot(xx * s2 - C, yy * s2 - C) / R_SUN
bnd[(np.abs(lon_map - 70) < 0.8) & (r_map < 1.0)] = 1
for sheet in range(0, len(stems), 20):
    tiles = []
    for s in stems[sheet:sheet + 20]:
        img = cv2.imread(str(TEST / f"{s}.jpeg"), cv2.IMREAD_GRAYSCALE)
        body = img[img > 20]
        lo, hi = np.percentile(body, [1, 99.5]) if body.size else (0, 255)
        vis = np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
        vis = cv2.cvtColor(cv2.resize(vis, (T, T), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
        vis[bnd == 1] = (150, 150, 150)
        c = te[te.stem == s]
        for rle in c[(c.conf >= .40) & (c.p >= .40) & (c.p < .60)]["rle_shrunk"]:
            m = cv2.resize(decode_rle(rle).astype(np.uint8), (T, T), interpolation=cv2.INTER_NEAREST)
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(vis, cs, -1, (0, 215, 255), 1)
        for rle in sub[sub.stem == s]["segmentation_rle"]:
            m = cv2.resize(decode_rle(rle).astype(np.uint8), (T, T), interpolation=cv2.INTER_NEAREST)
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(vis, cs, -1, (60, 220, 60), 2)
        i = qt.loc[s]
        kc = per.loc[s] if s in per.index else None
        lab = (f"{s[:8]} {s[-2]} {i.q}{' H' if i.bright_halo else ''} "
               f"{int(kc.kept) if kc is not None else 0}/{int(kc.cand) if kc is not None else 0}")
        cv2.rectangle(vis, (0, 0), (T, 20), (0, 0, 0), -1)
        cv2.putText(vis, lab, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(vis)
    while len(tiles) < 20:
        tiles.append(np.zeros((T, T, 3), np.uint8))
    grid = np.vstack([np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(5)])
    f = OUT / f"sheet_{sheet // 20 + 1:02d}.png"
    cv2.imwrite(str(f), grid)
    print("wrote", f.name, stems[sheet][:8], "->", stems[min(sheet + 19, len(stems) - 1)][:8])
