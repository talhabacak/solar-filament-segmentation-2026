"""Is the leaderboard test set drawn from the same distribution as the 707 train images?

Talha's argument for submitting the GONG-pretrained model: it saw 10,001 frames
over 5,017 days, so it may generalise better than local validation can show. That
argument stands or falls on one fact -- whether the test set actually sits
somewhere the train set does not.

Our own folds cannot answer it. folds_v1 is near-perfectly stratified by station
(24/25/25/25/22/21 in every fold), so train and validation are matched by
construction and the split is blind to exactly the axis in question. But the 180
test images ship with the competition, unlabelled, so the covariates can be
measured directly rather than argued about.

Measured per image, on-disk only (r <= 0.97, matching the pipeline's mask):

  date / station   the two things the filename encodes (YYYYMMDDhhmmss + site
                   letter). A test set drawn from held-out YEARS or SITES is a
                   real shift; one interleaved with train is not.
  solar cycle      filament counts follow the ~11-year cycle, so a date shift is
                   also an activity shift -- the covariate that matters most.
  sharpness        seeing, as Laplacian variance
  level, contrast  exposure and transparency

If the test set is interleaved, broader pretraining has little to buy and the
local measurement (GONG 0.6996 vs ImageNet 0.6981) is the honest estimate. If it
is held out in time or site, local validation systematically understates the
pretrained model and the leaderboard probe is worth a submission slot.

Usage: python src/eda/train_test_shift.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "outputs" / "eda"
DATA = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"
S = 768


def log(*a):
    print(*a, flush=True)


def disk_mask(S, frac=0.97):
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    c = (S - 1) / 2.0
    return ((xx - c) ** 2 + (yy - c) ** 2) <= (frac * c) ** 2


def describe(p, dk):
    im = cv2.resize(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), (S, S),
                    interpolation=cv2.INTER_AREA)
    v = im[dk].astype(np.float32)
    stem = p.name[:14]
    return dict(file=p.name,
                ts=pd.to_datetime(stem, format="%Y%m%d%H%M%S", errors="coerce"),
                station=p.name[14],
                sharp=float(cv2.Laplacian(im, cv2.CV_32F)[dk].var()),
                level=float(v.mean()),
                contrast=float((np.percentile(v, 84) - np.percentile(v, 16)) / 2))


def main():
    dk = disk_mask(S)
    rows = []
    for split, sub in (("train", "train/train_images"), ("test", "test/test_images")):
        for p in sorted((DATA / sub).rglob("*.jpeg")):
            r = describe(p, dk)
            r["split"] = split
            rows.append(r)
    df = pd.DataFrame(rows)
    df["year"] = df["ts"].dt.year
    log(f"{(df.split=='train').sum()} train, {(df.split=='test').sum()} test\n")

    # ---- 1. time ---------------------------------------------------------
    log("1. DATE -- is the test set held out in time?")
    for s in ("train", "test"):
        d = df[df.split == s]
        log(f"   {s:5s} {d['ts'].min().date()} .. {d['ts'].max().date()}  "
            f"({d['year'].nunique()} distinct years)")
    yt = set(df[df.split == "train"]["year"])
    ys = set(df[df.split == "test"]["year"])
    log(f"   years only in test : {sorted(ys - yt) or 'none'}")
    log(f"   years only in train: {sorted(yt - ys) or 'none'}")
    log(f"   shared years       : {len(yt & ys)}")
    tab = pd.crosstab(df["year"], df["split"])
    log("\n   per-year counts:")
    log("   " + tab.to_string().replace("\n", "\n   "))

    # do the exact days overlap?
    dtr = set(df[df.split == "train"]["ts"].dt.date)
    dte = set(df[df.split == "test"]["ts"].dt.date)
    log(f"\n   distinct days: train {len(dtr)}, test {len(dte)}, "
        f"SHARED {len(dtr & dte)}")
    log("   (shared days would mean near-duplicate frames across the split -- "
        "the same filaments, hours apart)")

    # ---- 2. station ------------------------------------------------------
    log("\n2. STATION -- is any site held out?")
    st = pd.crosstab(df["station"], df["split"])
    st["train_%"] = 100 * st["train"] / st["train"].sum()
    st["test_%"] = 100 * st["test"] / st["test"].sum()
    log("   " + st.to_string(float_format=lambda v: f"{v:.1f}").replace("\n", "\n   "))

    # ---- 3. covariates ---------------------------------------------------
    log("\n3. IMAGE COVARIATES -- does the test set look different?")
    log(f"   {'':10s} {'train mean':>11s} {'test mean':>10s} {'diff':>8s} "
        f"{'train sd':>9s} {'sd units':>9s}")
    for c in ("sharp", "level", "contrast"):
        a = df[df.split == "train"][c]
        b = df[df.split == "test"][c]
        d = b.mean() - a.mean()
        log(f"   {c:10s} {a.mean():11.2f} {b.mean():10.2f} {d:+8.2f} "
            f"{a.std():9.2f} {d/a.std():+9.2f}")
    log("   ('sd units' is the shift in train standard deviations -- under ~0.2 "
        "the two sets are the same population for that covariate)")

    OUT.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["ts"]).to_parquet(OUT / "train_test_shift.parquet", index=False)
    summary = dict(
        n_train=int((df.split == "train").sum()),
        n_test=int((df.split == "test").sum()),
        test_only_years=sorted(int(y) for y in (ys - yt)),
        shared_days=len(dtr & dte),
        station_pct={s: [float(st.loc[s, "train_%"]), float(st.loc[s, "test_%"])]
                     for s in st.index},
        covariate_sd_shift={c: float((df[df.split == "test"][c].mean() -
                                      df[df.split == "train"][c].mean()) /
                                     df[df.split == "train"][c].std())
                            for c in ("sharp", "level", "contrast")})
    with open(OUT / "train_test_shift.json", "w") as f:
        json.dump(summary, f, indent=1)
    log(f"\nsaved -> {OUT/'train_test_shift.json'}")


if __name__ == "__main__":
    main()
