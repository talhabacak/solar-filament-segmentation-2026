"""Stratified, time-separated sample of unlabelled GONG H-alpha frames.

Why the sampling matters more than the count. GONG takes a frame every ~20 s at
each site, and our own leakage study measured 0.61-0.70 correlation between
20-second cross-site pairs — consecutive frames are very nearly the same
observation. Downloading 5000 CONSECUTIVE frames would be perhaps 200 distinct
observations, not the 7x more data the number suggests. So this samples with a
minimum time separation, spread over sites and years.

The `hag` branch is the competition's own product, verified rather than assumed:
the competition's stems exist in the archive, and comparing three of them pixel
by pixel gives correlation 0.9825 inside the disk, median offset 0, mean absolute
difference 1.9 grey levels, with every difference above 50 lying OUTSIDE the
disk where our pipeline already masks at r <= 0.97.

Rules: GONG frames are public, free and equally accessible, which Sec. 2.6.a
permits. Only external GROUND TRUTH is prohibited (Host, discussion 729396), and
this fetches images. Every frame's stem is checked against the competition's
train and test stems and recorded in external_data_registry.csv — a frame that
matches a test stem is kept and FLAGGED, since training on an unlabelled test
frame is transductive and allowed but must be declarable.

Politeness: the archive 404s under rapid repeated listing, so directory listings
are cached to disk and requests are paced.

Usage:
  python src/data/sample_gong_archive.py --target 5000 --min-gap-min 60
"""

import argparse
import hashlib
import json
import random
import re
import ssl
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "data" / "external" / "gong_unlabeled"
CACHE = ROOT / "outputs" / "external" / "listing_cache"
REG = ROOT / "outputs" / "external" / "external_data_registry.csv"
TEST = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "test" / "test_images"
TRAIN = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
BASE = "https://nispdata.nso.edu/ftp/HA/hag"
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
STEM_RE = re.compile(r"^(\d{14})([A-Z])h\.jpg$")


def log(*a):
    print(*a, flush=True)


def get(url, retries=3, pause=1.5):
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
            with urllib.request.urlopen(req, timeout=90, context=CTX) as r:
                return r.read()
        except Exception as e:
            if k == retries - 1:
                raise
            time.sleep(pause * (2 ** k))
    return None


def listing(path, pause):
    """Directory listing, cached to disk so a rerun costs no requests."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = CACHE / (hashlib.sha1(path.encode()).hexdigest() + ".json")
    if key.exists():
        return json.loads(key.read_text())
    body = get(f"{BASE}/{path}").decode("utf-8", "replace")
    items = re.findall(r'href="([^"?][^"]*)"', body)
    key.write_text(json.dumps(items))
    time.sleep(pause)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=5000)
    ap.add_argument("--min-gap-min", type=int, default=60,
                    help="minimum minutes between two kept frames from the SAME "
                         "site on the same day; consecutive GONG frames are ~20 s "
                         "apart and nearly identical")
    ap.add_argument("--pause", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent downloads. Measured single-threaded rate "
                         "was 27 frames/min, i.e. 3 h for 5000; a few "
                         "connections to a public research archive is ordinary "
                         "use, so 4 keeps it polite and finishes in about an hour.")
    ap.add_argument("--list-only", action="store_true",
                    help="build the sample plan without downloading")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    REG.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)

    test_stems = {p.stem for p in TEST.glob("*.jpeg")}
    train_stems = {p.stem for p in TRAIN.glob("*.jpeg")}

    # STAGE-1 SAMPLING IS ABOUT DIVERSITY, NOT DISTRIBUTION MATCH.
    #
    # An earlier version drew year quotas proportional to the competition's own
    # year counts and kept only its six sites. That is domain-adaptation logic,
    # and it is the wrong logic here: stage 2 fine-tunes on the competition data
    # and supplies the target distribution by itself, so constraining stage 1 to
    # the same marginals throws away free variety — 2010 and 2023-2026, and the
    # archive's other stations.
    #
    # What is kept is the DAY spread. Pixel correlation decorrelates within an
    # hour (measured on downloaded frames: -0.028 at 1-6 h, 0.0001 beyond 30
    # days, on the high-passed disk interior), but that measures pixels, not
    # content: filaments live for days to weeks, so frames from one day largely
    # show the same filaments. Seeing new filament populations means new days.
    from collections import Counter
    comp = [q.stem for q in TRAIN.glob("*.jpeg")] + [q.stem for q in TEST.glob("*.jpeg")]
    log(f"competition sites {sorted(set(s[14] for s in comp))}, "
        f"years {min(s[:4] for s in comp)}-{max(s[:4] for s in comp)}; "
        f"stage-1 sampling deliberately spans MORE than this")

    months = sorted(m.strip("/") for m in listing("", a.pause)
                    if re.fullmatch(r"\d{6}/?", m))
    log(f"{len(months)} archive months, {months[0]} .. {months[-1]}")

    # every day in the archive is a candidate; take the target as evenly over
    # them as possible, at most a couple of frames from any single day
    all_days = []
    for m in months:
        all_days += [(m, d.strip("/")) for d in listing(f"{m}/", a.pause)
                     if re.fullmatch(r"\d{8}/?", d)]
    log(f"{len(all_days)} days available across the span")
    rng.shuffle(all_days)
    per_day = max(1, -(-a.target // max(len(all_days), 1)))
    log(f"taking up to {per_day} frame(s) per day")

    plan, seen_days = [], 0
    for m, d in all_days:
        if len(plan) >= a.target:
            break
        files = [f for f in listing(f"{m}/{d}/", a.pause) if STEM_RE.match(f)]
        if not files:
            continue
        seen_days += 1
        by_site = defaultdict(list)
        for f in files:
            g = STEM_RE.match(f)
            by_site[g.group(2)].append((g.group(1), f))
        sites_here = sorted(by_site)
        rng.shuffle(sites_here)
        for si in range(min(per_day, len(sites_here))):
            site = sites_here[si]
            items = sorted(by_site[site])
            ts, f = items[rng.randrange(len(items))]
            plan.append((f"{m}/{d}/{f}", f, site, m[:4]))
        if seen_days % 400 == 0:
            log(f"  {seen_days} days visited, {len(plan)} frames planned")


    rng.shuffle(plan)
    plan = plan[:a.target]
    sites = defaultdict(int)
    years = defaultdict(int)
    for _, _, s, y in plan:
        sites[s] += 1
        years[y] += 1
    log(f"\nplan: {len(plan)} frames from {seen_days} days")
    log(f"  sites: {dict(sorted(sites.items()))}")
    log(f"  years: {dict(sorted(years.items()))}")
    if a.list_only:
        pd.DataFrame(plan, columns=["path", "name", "site", "year"]).to_csv(
            REG.parent / "gong_sample_plan.csv", index=False)
        log(f"plan -> {REG.parent/'gong_sample_plan.csv'} (no download)")
        return

    from concurrent.futures import ThreadPoolExecutor

    def fetch(item):
        path, name, site, year = item
        dst = OUT / name
        if not dst.exists():
            try:
                dst.write_bytes(get(f"{BASE}/{path}"))
                time.sleep(a.pause)
            except Exception as e:
                return None, f"{name}: {type(e).__name__}"
        stem = dst.stem
        return dict(name=name, url=f"{BASE}/{path}", site=site, year=year,
                    sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),
                    bytes=dst.stat().st_size,
                    matches_test_stem=stem in test_stems,
                    matches_train_stem=stem in train_stems), None

    rows, t0, errs = [], time.time(), 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, (row, err) in enumerate(ex.map(fetch, plan)):
            if err:
                errs += 1
                if errs <= 5:
                    log(f"  ERR {err}")
            else:
                rows.append(row)
            if (i + 1) % 500 == 0:
                log(f"  {i+1}/{len(plan)} ({time.time()-t0:.0f}s, {errs} errors)")

    df = pd.DataFrame(rows)
    log(f"\nfetched {len(df)} frames ({time.time()-t0:.0f}s)")
    log(f"  matching a TEST stem : {int(df['matches_test_stem'].sum())} (kept, flagged)")
    log(f"  matching a TRAIN stem: {int(df['matches_train_stem'].sum())}")
    if REG.exists():
        df = pd.concat([pd.read_csv(REG), df], ignore_index=True).drop_duplicates("sha256")
    df.to_csv(REG, index=False)
    log(f"registry -> {REG} ({len(df)} rows)")


if __name__ == "__main__":
    main()
