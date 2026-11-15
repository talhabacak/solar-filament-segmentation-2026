"""E166 -- WA-05 prerequisite: does a second, near-in-time H-alpha frame carry information about the first frame's annotation?

The temporal-teacher idea (docs/30 WA-05) assumes seeing noise is independent between
frames minutes apart, so a neighbour frame's field is a second, partly independent
measurement of the same filaments. Before fetching anything, that can be measured on
the training set itself: 27% of images have a cross-fold neighbour within 24 h, and
some pairs are minutes apart from different GONG sites.

For each training pair (A, B) within 6 h, take the production OOF field on A (honest
for A) and the production OOF field on B, register B onto A (both are full-disk,
north-up, disk geometry measured in E155 -- a scale+shift is enough at this level; solar
rotation is NOT modelled: 0.55 deg/h, so up to ~3 deg over 6 h, ~50 px at disk centre at
2048 -- which is why the >=1 h pairs read as misregistered; the <1 h subset is the honest one), and ask:

    AUC on A's annotation of   field_A            the baseline
                               field_B (registered)   what the neighbour alone knows
                               mean(field_A, field_B) the temporal teacher, k=2

The uplift of the mean over field_A is the ceiling of a two-frame teacher before any
model is trained. Both directions of every pair are used. If the uplift is ~0 the
axis closes; if it is comparable to the magnetogram's +0.015 it earns a fetch.

Usage: python src/eda/temporal_pair_ceiling.py
"""
import sys, json, pathlib, numpy as np, pandas as pd, cv2
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from filament.rle import decode_rle
from sklearn.metrics import roc_auc_score
OUT = ROOT / "outputs" / "eda"; SEM = ROOT / "outputs" / "semantic"
MAX_H = 6.0

def main():
    folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
    folds["stem"] = folds["file_name"].str.replace(".jpeg", "", regex=False)
    folds["t"] = pd.to_datetime(folds["stem"].str[:14], format="%Y%m%d%H%M%S")
    fold_of = dict(zip(folds["stem"], folds["fold"]))
    geo = pd.read_parquet(OUT / "disk_geometry.parquet"); geo = geo[geo["r_iqr"] < 5].set_index("stem")
    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet"); gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    st = folds.sort_values("t").reset_index(drop=True)
    pairs = []
    for i in range(len(st)):
        for j in range(i + 1, len(st)):
            dh = (st.loc[j, "t"] - st.loc[i, "t"]).total_seconds() / 3600
            if dh > MAX_H: break
            pairs.append((st.loc[i, "stem"], st.loc[j, "stem"], dh))
    print(f"{len(pairs)} training pairs within {MAX_H} h (median gap {np.median([p[2] for p in pairs]):.2f} h)")
    fields = {k: np.load(SEM / f"fold{k}_probs_blob5ens.npz") for k in range(5)}
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:2048, 0:2048].astype(np.float32)
    rows = []
    for a, b, dh in pairs:
        for tgt, src in ((a, b), (b, a)):
            if tgt not in geo.index or src not in geo.index: continue
            ga, gb = geo.loc[tgt], geo.loc[src]
            # map target pixels to source pixels: same solar coords -> shift+scale
            sc = float(gb["radius"] / ga["radius"])
            sx = (float(gb["cx"]) + (xx - float(ga["cx"])) * sc).astype(np.float32); sy = (float(gb["cy"]) + (yy - float(ga["cy"])) * sc).astype(np.float32)
            fa = fields[fold_of[tgt]][tgt].astype(np.float32) / 255
            fb = cv2.remap(fields[fold_of[src]][src].astype(np.float32) / 255, sx, sy, cv2.INTER_LINEAR, borderValue=0)
            m = np.zeros((2048, 2048), bool)
            for r in gt.loc[gt["stem"] == tgt, "segmentation_rle"]: m |= decode_rle(r).astype(bool)
            rr = np.hypot(xx - ga["cx"], yy - ga["cy"]) / ga["radius"]
            fg = np.flatnonzero((m & (rr < 0.97)).ravel()); band = np.flatnonzero((~m & (rr < 0.97) & (cv2.dilate(m.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0)).ravel())
            far = np.flatnonzero((~m & (rr < 0.95)).ravel())
            sel = np.concatenate([rng.choice(fg, min(3000, len(fg)), replace=False), rng.choice(band, min(3000, len(band)), replace=False), rng.choice(far, 300, replace=False)])
            y = m.ravel()[sel].astype(int); A, B = fa.ravel()[sel], fb.ravel()[sel]
            if y.min() == y.max(): continue
            rows.append(dict(target=tgt, source=src, gap_h=dh, same_site=tgt[-2] == src[-2],
                             auc_A=roc_auc_score(y, A), auc_B=roc_auc_score(y, B), auc_mean=roc_auc_score(y, 0.5 * (A + B))))
    d = pd.DataFrame(rows); d.to_parquet(OUT / "temporal_pair_ceiling.parquet", index=False)
    print(f"{len(d)} directed pairs scored")
    for name, sub in (("all", d), ("same site", d[d["same_site"]]), ("different site", d[~d["same_site"]]), ("gap < 1 h", d[d["gap_h"] < 1])):
        if len(sub):
            print(f"  {name:15s} n {len(sub):3d}  AUC field_A {sub['auc_A'].mean():.4f}  neighbour alone {sub['auc_B'].mean():.4f}  mean of two {sub['auc_mean'].mean():.4f}  uplift {sub['auc_mean'].mean() - sub['auc_A'].mean():+.4f}")
    json.dump(dict(n=len(d), uplift=float(d["auc_mean"].mean() - d["auc_A"].mean()), auc_A=float(d["auc_A"].mean()), auc_B=float(d["auc_B"].mean())), open(OUT / "temporal_pair_ceiling.json", "w"), indent=1)
if __name__ == "__main__":
    main()
