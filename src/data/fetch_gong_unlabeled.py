"""Pilot download of unlabelled GONG H-alpha frames, with the safety checks first.

Why this is worth doing (measured, docs/14): the data-scaling curve has not
flattened — 25/50/100% of the training images give 0.6739 / 0.6866 / 0.6981, so
each doubling is worth about +0.012 pixel Dice. More frames is a live lever. The
initialisation axis, by contrast, is worth only +0.0169 end to end (ImageNet vs
random), which bounds in-domain SSL pretraining to a fraction of that.

Rules position (docs/01 §8): GONG frames are public, free and equally accessible,
so Sec. 2.6.a permits them. What is prohibited is external GROUND TRUTH — the
Host's ruling in discussion 729396 — so this fetches IMAGES ONLY and never
touches MAGFiLO annotations.

Two barriers against contaminating ourselves:
  1. every downloaded frame's stem and sha256 go into external_data_registry.csv
  2. any frame whose stem matches a competition TEST stem is flagged. It is not
     deleted — training on an unlabelled test frame is transductive learning and
     is allowed — but it must be visible and declarable in the report, and it
     must never be paired with an external label.

This is a PILOT: it fetches a small sample so the archive layout, file format and
image characteristics can be verified against the competition data before any
bulk download.

Usage: python src/data/fetch_gong_unlabeled.py --limit 40
"""

import argparse
import hashlib
import re
import ssl
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "data" / "external" / "gong_unlabeled"
REG = ROOT / "outputs" / "external" / "external_data_registry.csv"
TEST = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "test" / "test_images"
TRAIN = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
BASE = "https://nispdata.nso.edu/ftp/HA/haf"
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def log(*a):
    print(*a, flush=True)


def listing(url):
    with urllib.request.urlopen(url, timeout=60, context=CTX) as r:
        html = r.read().decode("utf-8", "replace")
    return re.findall(r'href="([^"?][^"]*)"', html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--month", default=None, help="e.g. 201506; default: probe")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    REG.parent.mkdir(parents=True, exist_ok=True)

    test_stems = {p.stem for p in TEST.glob("*.jpeg")}
    train_stems = {p.stem for p in TRAIN.glob("*.jpeg")}
    log(f"competition stems: {len(train_stems)} train, {len(test_stems)} test")

    months = [a.month] if a.month else None
    if months is None:
        top = listing(BASE + "/")
        months = [m.strip("/") for m in top if re.fullmatch(r"\d{6}/?", m)]
        log(f"archive months available: {len(months)} "
            f"({months[0] if months else '-'} .. {months[-1] if months else '-'})")
        months = months[len(months) // 2:len(months) // 2 + 1]
    log(f"probing month {months[0]}")

    files = [f for f in listing(f"{BASE}/{months[0]}/")
             if f.lower().endswith((".jpg", ".jpeg", ".fits.fz", ".fits"))]
    log(f"  {len(files)} entries; sample: {files[:3]}")
    if not files:
        subs = [s.strip("/") for s in listing(f"{BASE}/{months[0]}/")
                if s.endswith("/") and not s.startswith("/")]
        log(f"  no images at month level; subdirs: {subs[:6]}")
        if not subs:
            return
        files = [f"{subs[0]}/{f}" for f in listing(f"{BASE}/{months[0]}/{subs[0]}/")
                 if f.lower().endswith((".jpg", ".jpeg", ".fits.fz", ".fits"))]
        log(f"  {len(files)} entries under {subs[0]}; sample: {files[:3]}")
    if not files:
        return

    rows, t0 = [], time.time()
    for f in files[:a.limit]:
        name = f.rsplit("/", 1)[-1]
        dst = OUT / name
        url = f"{BASE}/{months[0]}/{f}"
        if not dst.exists():
            try:
                with urllib.request.urlopen(url, timeout=120, context=CTX) as r:
                    dst.write_bytes(r.read())
            except Exception as e:
                log(f"  ERR {name}: {type(e).__name__}")
                continue
        h = hashlib.sha256(dst.read_bytes()).hexdigest()
        stem = dst.stem
        img = cv2.imread(str(dst), cv2.IMREAD_GRAYSCALE)
        rows.append(dict(name=name, url=url, sha256=h, bytes=dst.stat().st_size,
                         h=None if img is None else img.shape[0],
                         w=None if img is None else img.shape[1],
                         mean=None if img is None else float(img.mean()),
                         matches_test_stem=stem in test_stems,
                         matches_train_stem=stem in train_stems))
    df = pd.DataFrame(rows)
    log(f"\nfetched {len(df)} files ({time.time()-t0:.0f}s)")
    if df.empty:
        return
    log(f"  sizes: {df['w'].dropna().unique()[:4]} x {df['h'].dropna().unique()[:4]}")
    log(f"  stems matching a TEST image : {int(df['matches_test_stem'].sum())}")
    log(f"  stems matching a TRAIN image: {int(df['matches_train_stem'].sum())}")
    log(f"  mean intensity {df['mean'].dropna().mean():.1f} "
        f"(competition frames for comparison below)")

    comp = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            for p in list(TRAIN.glob('*.jpeg'))[:8]]
    comp = [c for c in comp if c is not None]
    if comp:
        log(f"  competition: {comp[0].shape}, mean intensity "
            f"{np.mean([c.mean() for c in comp]):.1f}")

    if REG.exists():
        df = pd.concat([pd.read_csv(REG), df], ignore_index=True).drop_duplicates("sha256")
    df.to_csv(REG, index=False)
    log(f"registry -> {REG} ({len(df)} rows)")


if __name__ == "__main__":
    main()
