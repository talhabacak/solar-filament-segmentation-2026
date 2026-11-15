"""E154 -- can we add a second modality? A feasibility probe for co-temporal HMI magnetograms.

The real lesson of arXiv 2606.18123v2 (MixTIME) is not its router. It is that the
model's gain comes from having MODALITIES the baseline lacks -- image, text and
transcriptomics where others had image alone. Our pipeline has one modality,
grayscale H-alpha, and E152 has just shown that everything downstream of it is at
the inter-annotator ceiling. New information is the only thing left that the
ceiling argument does not close.

There is exactly one second modality available for this dataset, and it is the
strongest physical prior in solar physics for this problem: filaments form and sit
along POLARITY INVERSION LINES, the curves where the line-of-sight magnetic field
changes sign. A magnetogram shows those lines directly.

Three facts make it possible here:

  every file name carries a full UTC timestamp and an observatory letter
      (20110109104734Ch -> 2011-01-09 10:47:34, GONG site C), for the 707 training
      images AND the 180 test images;
  the range is 2011-01-09 to 2022-08-03, entirely inside the SDO/HMI era
      (HMI has produced a full-disk line-of-sight magnetogram every 720 s since
      2010), so a co-temporal magnetogram exists for every single image;
  magnetograms are an independent physical observation, not filament labels, so
      they fall under the external-data permission rather than the external-
      ground-truth prohibition.

This probe does not build anything. It answers the three questions that decide
whether the axis is worth the work:

  1. can we retrieve a magnetogram for one of our exact timestamps?
  2. does its header carry the geometry needed to register it to our 2048x2048
     H-alpha frame (disk centre, plate scale, solar radius, roll angle)?
  3. how big is the whole fetch?

Usage: python src/eda/magnetogram_probe.py [--stem 20110109104734Ch]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
IMG = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
OUT = ROOT / "outputs" / "eda"
JSOC = "http://jsoc.stanford.edu"
KEYS = "T_REC,CRPIX1,CRPIX2,CDELT1,CDELT2,RSUN_OBS,CROTA2,CRLT_OBS,DSUN_OBS,QUALITY"


def log(*a):
    print(*a, flush=True)


def stem_to_trec(stem):
    """20110109104734Ch -> 2011.01.09_10:47:34_TAI (TAI and UTC differ by ~35 s here,
    which is far inside the 720 s cadence)."""
    m = re.match(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", stem)
    y, mo, d, h, mi, s = m.groups()
    return f"{y}.{mo}.{d}_{h}:{mi}:{s}_TAI"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default=None)
    ap.add_argument("--n", type=int, default=3, help="how many stems to probe")
    a_ = ap.parse_args()
    stems = ([a_.stem] if a_.stem else
             [p.stem for p in sorted(IMG.glob("*.jpeg"))[:: max(len(list(IMG.glob('*.jpeg'))) // a_.n, 1)]][:a_.n])
    log(f"probing {len(stems)} stems: {', '.join(stems)}\n")

    results = []
    for stem in stems:
        trec = stem_to_trec(stem)
        # nearest record at or after the H-alpha timestamp; the [$] style range keeps
        # the query to a single record so the response stays small
        ds = f"hmi.M_720s[{trec}/1h@720s]"
        url = f"{JSOC}/cgi-bin/ajax/jsoc_info"
        try:
            r = requests.get(url, params={"op": "rs_list", "ds": ds, "key": KEYS,
                                          "seg": "magnetogram", "n": 1}, timeout=60)
            j = r.json()
        except Exception as e:
            log(f"{stem}: query failed {type(e).__name__} {e}")
            continue
        if j.get("status") != 0 or not j.get("count"):
            log(f"{stem}: no record for {trec} (status {j.get('status')}, count {j.get('count')})")
            continue
        keys = {k["name"]: k["values"][0] for k in j["keywords"]}
        seg = j["segments"][0]["values"][0]
        log(f"{stem}  ->  {keys.get('T_REC')}   quality {keys.get('QUALITY')}")
        log(f"  disk centre  CRPIX ({keys.get('CRPIX1')}, {keys.get('CRPIX2')})")
        log(f"  plate scale  CDELT {keys.get('CDELT1')} arcsec/px, solar radius "
            f"RSUN_OBS {keys.get('RSUN_OBS')} arcsec  ->  {float(keys['RSUN_OBS']) / float(keys['CDELT1']):.1f} px")
        log(f"  roll CROTA2 {keys.get('CROTA2')}, B0 CRLT_OBS {keys.get('CRLT_OBS')}")
        log(f"  segment      {seg}")
        results.append(dict(stem=stem, t_rec=keys.get("T_REC"), seg=seg,
                            crpix=(keys.get("CRPIX1"), keys.get("CRPIX2")),
                            cdelt=keys.get("CDELT1"), rsun=keys.get("RSUN_OBS"),
                            crota2=keys.get("CROTA2"), quality=keys.get("QUALITY")))

    if not results:
        log("\nno records retrieved; the axis cannot be built from this route")
        return

    # question 3: size. Fetch one file and measure it.
    seg = results[0]["seg"]
    log(f"\nfetching one magnetogram to measure the cost: {seg}")
    try:
        rr = requests.get(JSOC + seg, timeout=300, stream=True)
        raw = rr.content
        p = OUT / "_probe_magnetogram.fits"
        p.write_bytes(raw)
        log(f"  {len(raw) / 1e6:.1f} MB for one full-disk frame")
        from astropy.io import fits
        with fits.open(p, ignore_missing_simple=True) as hdul:
            d = [h for h in hdul if h.data is not None][0]
            arr = np.asarray(d.data, dtype=np.float32)
        finite = np.isfinite(arr)
        log(f"  array {arr.shape}, {100 * finite.mean():.0f}% finite, "
            f"range [{np.nanmin(arr):.0f}, {np.nanmax(arr):.0f}] Gauss")
        log(f"  polarity split: {100 * (arr[finite] > 20).mean():.1f}% above +20 G, "
            f"{100 * (arr[finite] < -20).mean():.1f}% below -20 G")
        total = len(raw) * 887 / 1e9
        log(f"\n  887 images (707 train + 180 test) at this size: {total:.1f} GB")
        log(f"  a 1024x1024 rebin would be {total / 16:.2f} GB, which is what the "
            f"pipeline actually needs at 2048 H-alpha resolution")
    except Exception as e:
        log(f"  fetch failed: {type(e).__name__} {e}")

    json.dump(results, open(OUT / "magnetogram_probe.json", "w"), indent=1, default=str)
    log(f"\nwrote {OUT / 'magnetogram_probe.json'}")


if __name__ == "__main__":
    main()
