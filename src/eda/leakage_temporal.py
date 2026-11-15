"""Temporal / content leakage analysis between train and test (and across CV folds).

Stems: YYYYMMDDHHMMSS + site letter + 'h'  (e.g. 20110109104734Ch.jpeg)
Disk geometry (FIXED, confirmed): center (1023.5, 1023.5), R = 900.0 px at 2048.
At 512 (factor 4): center (255.5, 255.5), R = 225.0.

Outputs:
  outputs/eda/leakage_temporal_pairs.csv / .parquet   (per-pair content correlations, incremental)
  outputs/eda/leakage_temporal.md                     (report + verdict)
"""
import sys, time, json
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import cv2

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"
TRAIN_DIR = DATA / "train" / "train_images"
TEST_DIR = DATA / "test" / "test_images"
OUT = ROOT / "outputs" / "eda"
OUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)
H = 3600.0

def log(*a):
    print(*a, flush=True)

# ---------------------------------------------------------------- 1. parse stems
def parse_dir(d):
    rows = []
    for p in sorted(d.glob("*.jpeg")):
        stem = p.stem                       # e.g. 20110109104734Ch
        ts = pd.to_datetime(stem[:14], format="%Y%m%d%H%M%S")
        rows.append({"file_name": p.name, "stem": stem, "ts": ts, "site": stem[14]})
    df = pd.DataFrame(rows)
    df["t"] = df["ts"].astype("int64") // 10**9   # unix seconds
    return df

tr = parse_dir(TRAIN_DIR)
te = parse_dir(TEST_DIR)
log(f"parsed train={len(tr)} test={len(te)}")
assert len(tr) == 707 and len(te) == 180

def nearest_dt(query_t, ref_t):
    """abs seconds to nearest ref for each query, plus argmin index into ref (sorted internally)."""
    order = np.argsort(ref_t)
    rs = ref_t[order]
    idx = np.searchsorted(rs, query_t)
    best = np.full(len(query_t), np.inf)
    best_j = np.zeros(len(query_t), dtype=int)
    for off in (-1, 0):
        j = np.clip(idx + off, 0, len(rs) - 1)
        d = np.abs(query_t - rs[j])
        upd = d < best
        best[upd] = d[upd]
        best_j[upd] = j[upd]
    return best, order[best_j]

def dist_stats(dt_sec, label):
    dt_h = dt_sec / H
    pct = {p: float(np.percentile(dt_h, p)) for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)}
    counts = {th: int((dt_h < th).sum()) for th in (1, 6, 24, 72, 168)}
    s = {"label": label, "n": len(dt_h), "min_h": float(dt_h.min()), "max_h": float(dt_h.max()),
         "mean_h": float(dt_h.mean()), "pct_h": pct, "counts_under_h": counts}
    log(f"[{label}] n={s['n']} min={s['min_h']:.3f}h med={pct[50]:.1f}h "
        f"| <1h:{counts[1]} <6h:{counts[6]} <24h:{counts[24]} <72h:{counts[72]} <7d:{counts[168]}")
    return s

# --- test -> train, any site
dt_any, j_any = nearest_dt(te["t"].values, tr["t"].values)
te["dt_any_s"] = dt_any
te["nn_train"] = tr["file_name"].values[j_any]
te["nn_train_site"] = tr["site"].values[j_any]
stats_any = dist_stats(dt_any, "test->train any-site")

# --- test -> train, same site
dt_same = np.full(len(te), np.nan)
for site in te["site"].unique():
    q = te["site"].values == site
    ref = tr.loc[tr["site"] == site, "t"].values
    if len(ref):
        d, _ = nearest_dt(te.loc[q, "t"].values, ref)
        dt_same[q] = d
te["dt_same_s"] = dt_same
stats_same = dist_stats(dt_same[~np.isnan(dt_same)], "test->train same-site")
n_no_same = int(np.isnan(dt_same).sum())

closest20 = te.nsmallest(20, "dt_any_s")[["file_name", "site", "nn_train", "nn_train_site", "dt_any_s"]].copy()
closest20["dt_h"] = closest20["dt_any_s"] / H
log("closest 20 test->train pairs:")
log(closest20.to_string(index=False))

# ---------------------------------------------------------------- 2. CV folds
folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
folds = folds.merge(tr[["file_name", "t", "site"]].rename(columns={"site": "site_stem"}), on="file_name")
assert len(folds) == 707

cv_dt = np.full(len(folds), np.nan)
for f in sorted(folds["fold"].unique()):
    val = folds["fold"].values == f
    d, _ = nearest_dt(folds.loc[val, "t"].values, folds.loc[~val, "t"].values)
    cv_dt[val] = d
folds["cv_dt_s"] = cv_dt
stats_cv = dist_stats(cv_dt, "CV val->trainfold (folds_v1, pooled)")

hold = folds["temporal_holdout"].values.astype(bool)
d_hold, _ = nearest_dt(folds.loc[hold, "t"].values, folds.loc[~hold, "t"].values)
stats_hold = dist_stats(d_hold, "temporal_holdout val->train")

# ---------------------------------------------------------------- 3+4. content similarity
# all train-test pairs with |dt| < 48h
tt, rt = te["t"].values, tr["t"].values
diff = np.abs(tt[:, None] - rt[None, :])
ii, jj = np.where(diff < 48 * H)
pairs = pd.DataFrame({
    "test_file": te["file_name"].values[ii], "train_file": tr["file_name"].values[jj],
    "test_site": te["site"].values[ii], "train_site": tr["site"].values[jj],
    "dt_s": diff[ii, jj].astype(float)})
pairs["dt_h"] = pairs["dt_s"] / H
n_pairs_48h = len(pairs)
log(f"train-test pairs with dt<48h: {n_pairs_48h}")

# within 30 minutes (sec 4)
pairs30 = pairs[pairs["dt_s"] <= 1800].sort_values("dt_s").reset_index(drop=True)
log(f"train-test pairs within 30 min: {len(pairs30)}")
log(pairs30.to_string(index=False))

CAP = 500
if n_pairs_48h > CAP:
    keep = pairs["dt_h"] < 6          # keep every close pair
    rest = pairs.index[~keep].values
    take = RNG.choice(rest, size=CAP - int(keep.sum()), replace=False)
    sampled = pd.concat([pairs[keep], pairs.loc[take]]).sort_values(["test_file", "dt_s"])
else:
    sampled = pairs.sort_values(["test_file", "dt_s"])
sampled = sampled.reset_index(drop=True)
sampled["kind"] = "near"
log(f"near pairs for content check: {len(sampled)} (all dt<6h kept: {int((pairs['dt_h']<6).sum())})")

# far baseline: dt > 30d
fi, fj = np.where(diff > 30 * 24 * H)
sel = RNG.choice(len(fi), size=200, replace=False)
far = pd.DataFrame({
    "test_file": te["file_name"].values[fi[sel]], "train_file": tr["file_name"].values[fj[sel]],
    "test_site": te["site"].values[fi[sel]], "train_site": tr["site"].values[fj[sel]],
    "dt_s": diff[fi[sel], fj[sel]].astype(float)})
far["dt_h"] = far["dt_s"] / H
far["kind"] = "far"
far = far.sort_values("test_file").reset_index(drop=True)

work = pd.concat([sampled, far], ignore_index=True)

# --- image machinery (512, disk interior r<0.88 of R=225 centered 255.5) ---
S = 512
yy, xx = np.mgrid[0:S, 0:S]
MASK = (np.sqrt((xx - 255.5) ** 2 + (yy - 255.5) ** 2) / 225.0) < 0.88
HANN = cv2.createHanningWindow((S, S), cv2.CV_32F)
MASKF = MASK.astype(np.float32)

_cache = OrderedDict()
def load_norm(fname, is_test):
    key = fname
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    p = (TEST_DIR if is_test else TRAIN_DIR) / fname
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA).astype(np.float32)
    v = img[MASK]
    z = (img - v.mean()) / (v.std() + 1e-6)
    z[~MASK] = 0.0
    _cache[key] = z
    if len(_cache) > 48:
        _cache.popitem(last=False)
    return z

def corr_masked(a, b, m):
    av, bv = a[m], b[m]
    av = av - av.mean(); bv = bv - bv.mean()
    den = np.sqrt((av * av).sum() * (bv * bv).sum())
    return float((av * bv).sum() / den) if den > 0 else 0.0

def shift_img(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (S, S))

def pair_metrics(a, b):
    c0 = corr_masked(a, b, MASK)
    (dx, dy), _resp = cv2.phaseCorrelate(a, b, HANN)
    best = c0; best_shift = 0.0
    for sgn in (1, -1):
        bs = shift_img(b, sgn * dx, sgn * dy)
        ms = shift_img(MASKF, sgn * dx, sgn * dy) > 0.5
        m = MASK & ms
        if m.sum() > 1000:
            c = corr_masked(a, bs, m)
            if c > best:
                best = c; best_shift = float(np.hypot(dx, dy))
    return c0, best, float(np.hypot(dx, dy))

csv_path = OUT / "leakage_temporal_pairs.csv"
res = []
t0 = time.time()
for k, row in work.iterrows():
    a = load_norm(row["test_file"], True)
    b = load_norm(row["train_file"], False)
    c0, ca, mag = pair_metrics(a, b)
    res.append({**row.to_dict(), "corr_direct": c0, "corr_aligned": ca, "shift_px": mag})
    if (k + 1) % 50 == 0 or k + 1 == len(work):
        pd.DataFrame(res).to_csv(csv_path, index=False)
        log(f"  content {k+1}/{len(work)}  ({time.time()-t0:.0f}s)")
rdf = pd.DataFrame(res)
rdf.to_parquet(OUT / "leakage_temporal_pairs.parquet", index=False)

near = rdf[rdf["kind"] == "near"]
farr = rdf[rdf["kind"] == "far"]

def cstats(df, col):
    return dict(mean=float(df[col].mean()), p50=float(df[col].median()),
                p90=float(df[col].quantile(.90)), p99=float(df[col].quantile(.99)),
                max=float(df[col].max()))

log("near direct:", cstats(near, "corr_direct"))
log("near aligned:", cstats(near, "corr_aligned"))
log("far direct:", cstats(farr, "corr_direct"))
log("far aligned:", cstats(farr, "corr_aligned"))

buckets = [(0, 1), (1, 6), (6, 24), (24, 48)]
brows = []
for lo, hi in buckets:
    sub = near[(near["dt_h"] >= lo) & (near["dt_h"] < hi)]
    if len(sub):
        brows.append({"bucket": f"{lo}-{hi}h", "n": len(sub),
                      "direct_med": float(sub["corr_direct"].median()),
                      "aligned_med": float(sub["corr_aligned"].median()),
                      "aligned_max": float(sub["corr_aligned"].max()),
                      "n_aligned_gt_0.9": int((sub["corr_aligned"] > 0.9).sum())})
bdf = pd.DataFrame(brows)
log(bdf.to_string(index=False))

far_p99_al = float(farr["corr_aligned"].quantile(.99))
far_max_al = float(farr["corr_aligned"].max())
n_near_gt09 = int((near["corr_aligned"] > 0.9).sum())
n_near_gt_far99 = int((near["corr_aligned"] > far_p99_al).sum())
n_near_gt_farmax = int((near["corr_aligned"] > far_max_al).sum())
log(f"near pairs with aligned corr>0.9: {n_near_gt09} | > far p99 ({far_p99_al:.3f}): {n_near_gt_far99}"
    f" | > far max ({far_max_al:.3f}): {n_near_gt_farmax}")

top_near = near.nlargest(15, "corr_aligned")[
    ["test_file", "train_file", "dt_h", "test_site", "train_site", "corr_direct", "corr_aligned"]]
log(top_near.to_string(index=False))

# unique test images with a >0.9 partner
n_test_gt09 = near.loc[near["corr_aligned"] > 0.9, "test_file"].nunique()
n_test_gt08 = near.loc[near["corr_aligned"] > 0.8, "test_file"].nunique()

# ---------------------------------------------------------------- 5. report
def fmt_stats(s):
    p = s["pct_h"]; c = s["counts_under_h"]
    return (f"| {s['label']} | {s['n']} | {s['min_h']:.2f} | {p[5]:.1f} | {p[25]:.1f} | {p[50]:.1f} | "
            f"{p[75]:.1f} | {p[95]:.1f} | {c[1]} | {c[6]} | {c[24]} | {c[72]} | {c[168]} |")

md = []
md.append("# Temporal / content leakage: train vs test, and CV folds\n")
md.append(f"Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}. Train n={len(tr)}, test n={len(te)}. "
          "SHA-256 exact overlap = 0 (previously verified); 32x32 aHash degenerate on full-disk solar images.\n")
md.append("## 1. Nearest-neighbour Delta-t distributions\n")
md.append("| population | n | min h | p5 | p25 | p50 | p75 | p95 | <1h | <6h | <24h | <72h | <7d |")
md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for s in (stats_any, stats_same, stats_cv, stats_hold):
    md.append(fmt_stats(s))
md.append(f"\n(same-site: {n_no_same} test images had no same-site train image.)\n")
md.append("### 20 closest test->train pairs (any site)\n")
md.append(closest20[["file_name", "site", "nn_train", "nn_train_site", "dt_h"]].to_markdown(index=False))
md.append("\n## 2. CV folds vs the real train/test split\n")
fr_te = {th: stats_any["counts_under_h"][th] / stats_any["n"] for th in (6, 24, 72)}
fr_cv = {th: stats_cv["counts_under_h"][th] / stats_cv["n"] for th in (6, 24, 72)}
fr_ho = {th: stats_hold["counts_under_h"][th] / stats_hold["n"] for th in (6, 24, 72)}
md.append(f"Fraction of val images with a train neighbour under 6h/24h/72h: "
          f"test {fr_te[6]:.1%}/{fr_te[24]:.1%}/{fr_te[72]:.1%} ; "
          f"CV folds_v1 {fr_cv[6]:.1%}/{fr_cv[24]:.1%}/{fr_cv[72]:.1%} ; "
          f"temporal_holdout {fr_ho[6]:.1%}/{fr_ho[24]:.1%}/{fr_ho[72]:.1%}.\n")
md.append("\n## 3. Content similarity (512px, disk interior r<0.88, zero-mean/unit-std)\n")
md.append(f"Near pairs (dt<48h): {n_pairs_48h} total, {len(near)} scored (all {int((pairs['dt_h']<6).sum())} "
          f"with dt<6h kept, remainder sampled). Far baseline: {len(farr)} random pairs with dt>30d.\n")
md.append("| set | corr | mean | p50 | p90 | p99 | max |")
md.append("|---|---|---|---|---|---|---|")
for nm, df_, col in (("near", near, "corr_direct"), ("near", near, "corr_aligned"),
                     ("far", farr, "corr_direct"), ("far", farr, "corr_aligned")):
    s = cstats(df_, col)
    md.append(f"| {nm} | {col} | {s['mean']:.3f} | {s['p50']:.3f} | {s['p90']:.3f} | {s['p99']:.3f} | {s['max']:.3f} |")
md.append("\n### By Delta-t bucket (near pairs)\n")
md.append(bdf.to_markdown(index=False))
md.append(f"\nSeparability: far-baseline aligned-corr p99 = {far_p99_al:.3f}, max = {far_max_al:.3f}. "
          f"Near pairs above far-p99: {n_near_gt_far99}/{len(near)}; above far-max: {n_near_gt_farmax}.\n")
md.append(f"Near pairs with aligned corr > 0.9 (essentially same content): {n_near_gt09} "
          f"({n_test_gt09} unique test images; {n_test_gt08} unique test images have a partner > 0.8).\n")
md.append("### Highest-correlation near pairs\n")
md.append(top_near.to_markdown(index=False))
md.append("\n## 4. Cross-site near-simultaneous pairs (dt <= 30 min)\n")
if len(pairs30):
    p30 = pairs30.merge(rdf[["test_file", "train_file", "corr_direct", "corr_aligned"]],
                        on=["test_file", "train_file"], how="left")
    md.append(p30.to_markdown(index=False))
else:
    md.append("None. No train-test pair exists within 30 minutes.")
md.append("\n## 5. Verdict\n")
verdict_placeholder = "@@VERDICT@@"
md.append(verdict_placeholder)

report = "\n".join(md)
(OUT / "leakage_temporal.md").write_text(report, encoding="utf-8")
log("report skeleton written; verdict to be filled by driver")

summary = {
    "stats_any": stats_any, "stats_same": stats_same, "stats_cv": stats_cv, "stats_hold": stats_hold,
    "n_no_same_site": n_no_same, "n_pairs_48h": n_pairs_48h, "n_pairs_30min": len(pairs30),
    "near_scored": len(near), "far_scored": len(farr),
    "near_direct": cstats(near, "corr_direct"), "near_aligned": cstats(near, "corr_aligned"),
    "far_direct": cstats(farr, "corr_direct"), "far_aligned": cstats(farr, "corr_aligned"),
    "far_p99_aligned": far_p99_al, "far_max_aligned": far_max_al,
    "n_near_gt09": n_near_gt09, "n_near_gt_far99": n_near_gt_far99, "n_near_gt_farmax": n_near_gt_farmax,
    "n_test_gt09": n_test_gt09, "n_test_gt08": n_test_gt08,
    "buckets": brows, "fr_te": fr_te, "fr_cv": fr_cv, "fr_ho": fr_ho,
    "closest20": closest20[["file_name", "nn_train", "dt_h"]].to_dict("records"),
    "pairs30": pairs30.to_dict("records"),
    "top_near": top_near.to_dict("records"),
}
(OUT / "leakage_temporal_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
log("DONE")
