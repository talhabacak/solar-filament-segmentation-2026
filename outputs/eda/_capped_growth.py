"""Cap how far a 0.10 mask may grow beyond its 0.20 core.

t0.10 recovers 151 shape copies and loses 65 already-matched ones: masks that were truncated get
longer (good) and masks that were already right overshoot the ground truth (bad). The loss has no
learned component -- it is unbounded growth -- so it takes one global parameter, not a per-
candidate decision on the closed contour-level axis:

    mask_k = mask_0.10  AND  dilate(mask_0.10 AND field>=0.20, k)

k = 0 is roughly the 0.20 mask, k = inf is the 0.10 mask. A candidate already agreeing with its
core cannot grow much; a truncated one can extend along its axis. The operation only REMOVES
pixels from the fused 0.10 mask, so pixel-disjointness -- a validator requirement, preserved by
fusion's de-overlap -- survives by construction.

Fold 0, selector-free reach first (pool_complementarity), exactly as the threshold sweep was read.

Usage: python outputs/eda/_capped_growth.py 2 4 8
"""
import sys, pathlib, time, numpy as np, pandas as pd, cv2
ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in ("src", "src/models"): sys.path.insert(0, str(ROOT / p))
from filament.rle import decode_rle, encode_mask

MIN_AREA = 64
ks = [int(x) for x in sys.argv[1:]] or [2, 4, 8]
pool = pd.read_parquet(ROOT / "outputs/selector/fused_pool_all_blob5_t010.parquet")
folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
f0 = set(folds.loc[folds["fold"] == 0, "file_name"].str.replace(".jpeg", "", regex=False))
pool = pool[pool["stem"].isin(f0)].reset_index(drop=True)
prob = np.load(ROOT / "outputs/semantic/fold0_probs_blob5ens.npz")
print(f"fold 0: {len(pool)} aday, {pool['stem'].nunique()} goruntu", flush=True)

for k in ks:
    out, kept_core, shrunk_n, t0 = [], 0, 0, time.time()
    K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)) if k > 0 else None
    for stem, g in pool.groupby("stem", sort=False):
        f = prob[stem].astype(np.float32) / 255.0 if stem in prob else None
        for rle in g["rle"]:
            m = decode_rle(rle).astype(bool)
            if f is None:
                out.append(encode_mask(m.astype(np.uint8))); continue
            core = m & (f >= 0.20)
            if core.sum() < MIN_AREA:          # no core: the candidate is all weak field.
                kept_core += 1                  # leave it alone rather than delete a candidate,
                out.append(encode_mask(m.astype(np.uint8))); continue   # so the pool stays comparable
            allow = core if K is None else (cv2.dilate(core.astype(np.uint8), K) > 0)
            q = m & allow
            if q.sum() < MIN_AREA:
                out.append(encode_mask(m.astype(np.uint8))); continue
            shrunk_n += int(q.sum() < m.sum())
            out.append(encode_mask(q.astype(np.uint8)))
    d = pool.copy(); d["rle"] = out
    name = f"fused_pool_f0_cap{k}"
    d.to_parquet(ROOT / "outputs/selector" / f"{name}.parquet", index=False)
    print(f"  k={k}: {len(d)} aday yazildi, {shrunk_n} kirpildi, {kept_core} cekirdeksiz "
          f"({time.time()-t0:.0f}s) -> {name}", flush=True)
