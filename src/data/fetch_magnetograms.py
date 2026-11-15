"""E154b -- fetch a co-temporal HMI magnetogram for every image, train and test.

The one modality this dataset admits. Filaments form and sit along polarity
inversion lines, the curves where the line-of-sight magnetic field changes sign,
and a magnetogram shows those lines directly. That is information the H-alpha
frame does not contain, which matters because E152 showed everything downstream
of the H-alpha frame is already at the inter-annotator ceiling.

Permitted: competition rules Sec. 2.6 allow external data and models provided
they are available free to all participants. SDO/HMI is NASA open data. A
magnetogram is a physical observation, not a filament label, so it does not touch
the external-ground-truth prohibition.

Feasible: every file name carries a full UTC timestamp and the range is
2011-01-09 to 2022-08-03, entirely inside the HMI era (a full-disk line-of-sight
magnetogram every 720 s since 2010). A probe of three stems matched all three
within 12 minutes.

This step only downloads. It stores the frame rebinned 4x (4096 -> 1024, which is
still finer than our 2048 frame's 904-pixel solar radius needs) plus the header
keywords registration needs, and leaves registration to a separate step so the
geometry can be fixed without fetching 887 files again.

  outputs/magnetograms/mag_<stem>.npy    float16, 1024x1024, Gauss
  outputs/magnetograms/index.parquet     one row per stem with T_REC and geometry

Resumable: stems already on disk are skipped, so it can be interrupted freely.

Usage:
  python src/data/fetch_magnetograms.py                 # everything, 4 workers
  python src/data/fetch_magnetograms.py --limit 20      # a first batch
"""

import argparse
import io
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
TEST = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "test" / "test_images"
DEST = ROOT / "outputs" / "magnetograms"
JSOC = "http://jsoc.stanford.edu"
KEYS = "T_REC,CRPIX1,CRPIX2,CDELT1,CDELT2,RSUN_OBS,CROTA2,CRLT_OBS,CRLN_OBS,DSUN_OBS,QUALITY"
REBIN = 4
_lock = threading.Lock()


def log(*a):
    with _lock:
        print(*a, flush=True)


def stem_to_trec(stem):
    y, mo, d, h, mi, s = re.match(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", stem).groups()
    return f"{y}.{mo}.{d}_{h}:{mi}:{s}_TAI"


SERIES = {"hmi": dict(ds="hmi.M_720s[{t}/{w}@720s]", seg="magnetogram", prefix="mag"),
          "aia304": dict(ds="aia.lev1_euv_12s[{t}/{w}@12s][304]", seg="image", prefix="a304"),
          "aia193": dict(ds="aia.lev1_euv_12s[{t}/{w}@12s][193]", seg="image", prefix="a193")}
CUR = SERIES["hmi"]


def query(stem, session, window="2h"):
    """The records in a window around the H-alpha time, so a bad-quality frame can be
    stepped over rather than accepted."""
    ds = CUR["ds"].format(t=stem_to_trec(stem), w=window)
    r = session.get(f"{JSOC}/cgi-bin/ajax/jsoc_info",
                    params={"op": "rs_list", "ds": ds, "key": KEYS, "seg": CUR["seg"]},
                    timeout=(15, 60))
    j = r.json()
    if j.get("status") != 0 or not j.get("count"):
        return []
    keys = {k["name"]: k["values"] for k in j["keywords"]}
    segs = j["segments"][0]["values"]
    out = []
    for i in range(j["count"]):
        row = {k: v[i] for k, v in keys.items()}
        row["seg"] = segs[i]
        out.append(row)
    return out


def fetch_one(stem, session, retries=3):
    dest = DEST / f"{CUR['prefix']}_{stem}.npy"
    if dest.exists() and dest.stat().st_size > 1000:
        return None
    recs = []
    for w in (("2h", "12h") if CUR is SERIES["hmi"] else ("2m", "30m")):
        recs = [r for r in query(stem, session, w) if str(r.get("QUALITY", "1")) in ("0", "0x00000000")]
        if recs:
            break
    if not recs:
        log(f"{stem}: no good-quality record")
        return dict(stem=stem, status="no_record")
    # the record nearest in time to the H-alpha frame
    want = pd.Timestamp(stem_to_trec(stem).replace("_TAI", "").replace("_", " ").replace(".", "-", 2))
    def dt(r):
        # HMI T_REC is "2011.01.09_10:48:00_TAI"; AIA's is ISO "2011-01-09T10:47:02Z"
        t = r["T_REC"].replace("_TAI", "").replace("_", " ").replace("Z", "")
        if t[4] == ".":
            t = t[:4] + "-" + t[5:7] + "-" + t[8:]
        return abs((pd.Timestamp(t).tz_localize(None) - want).total_seconds())
    rec = min(recs, key=dt)
    for attempt in range(retries):
        try:
            rr = session.get(JSOC + rec["seg"], timeout=(10, 45), stream=True)
            rr.raise_for_status()
            buf = io.BytesIO()
            for chunk in rr.iter_content(1 << 20):
                buf.write(chunk)
            rr.close()
            from astropy.io import fits
            buf.seek(0)
            with fits.open(buf, ignore_missing_simple=True) as hdul:
                hdu = [h for h in hdul if h.data is not None][0]
                arr = np.asarray(hdu.data, dtype=np.float32)
            h, w = arr.shape
            arr = arr[: h // REBIN * REBIN, : w // REBIN * REBIN]
            with np.errstate(invalid="ignore"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    arr = np.nanmean(arr.reshape(h // REBIN, REBIN, w // REBIN, REBIN), axis=(1, 3))
            np.save(dest, arr.astype(np.float16))
            return dict(stem=stem, status="ok", t_rec=rec["T_REC"], dt_s=dt(rec),
                        crpix1=float(rec["CRPIX1"]), crpix2=float(rec["CRPIX2"]),
                        cdelt1=float(rec["CDELT1"]), rsun_obs=float(rec["RSUN_OBS"]),
                        crota2=float(rec["CROTA2"]), crlt_obs=float(rec["CRLT_OBS"]),
                        rebin=REBIN, shape=arr.shape[0])
        except Exception as e:
            if attempt == retries - 1:
                log(f"{stem}: fetch failed after {retries} tries, {type(e).__name__} {e}")
                return dict(stem=stem, status=f"error:{type(e).__name__}")
            time.sleep(5 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    # runs that were killed lost their index rows while their files stayed on disk;
    # this re-queries the header only (no download) for every such stem
    ap.add_argument("--reindex", action="store_true")
    ap.add_argument("--series", default="hmi", choices=list(SERIES))
    a_ = ap.parse_args()
    global CUR
    CUR = SERIES[a_.series]
    if a_.reindex:
        idx_path = DEST / "index.parquet"
        have = pd.read_parquet(idx_path) if idx_path.exists() else pd.DataFrame(columns=["stem"])
        ok_stems = set(have.loc[have.get("status", pd.Series(dtype=str)) == "ok", "stem"]) if len(have) else set()
        on_disk = [p.stem.replace("mag_", "") for p in DEST.glob("mag_*.npy")]
        missing = [s for s in on_disk if s not in ok_stems]
        log(f"{len(on_disk)} on disk, {len(missing)} without geometry in the index")
        session = requests.Session(); session.headers["Connection"] = "close"
        rows = []
        for i, stem in enumerate(missing):
            recs = []
            for w in ("2h", "12h"):
                recs = [r for r in query(stem, session, w)
                        if str(r.get("QUALITY", "1")) in ("0", "0x00000000")]
                if recs:
                    break
            if not recs:
                rows.append(dict(stem=stem, status="no_record")); continue
            want = pd.Timestamp(stem_to_trec(stem).replace("_TAI", "").replace("_", " ").replace(".", "-", 2))
            def dt(r):
                t = r["T_REC"].replace("_TAI", "").replace("_", " ")
                return abs((pd.Timestamp(t[:4] + "-" + t[5:7] + "-" + t[8:]) - want).total_seconds())
            rec = min(recs, key=dt)
            rows.append(dict(stem=stem, status="ok", t_rec=rec["T_REC"], dt_s=dt(rec),
                             crpix1=float(rec["CRPIX1"]), crpix2=float(rec["CRPIX2"]),
                             cdelt1=float(rec["CDELT1"]), rsun_obs=float(rec["RSUN_OBS"]),
                             crota2=float(rec["CROTA2"]), crlt_obs=float(rec["CRLT_OBS"]),
                             rebin=REBIN, shape=1024))
            if (i + 1) % 25 == 0:
                log(f"  reindexed {i + 1}/{len(missing)}")
        new = pd.DataFrame(rows)
        idx = pd.concat([have, new], ignore_index=True) if len(have) else new
        idx = idx.drop_duplicates(subset=["stem"], keep="last")
        idx.to_parquet(idx_path, index=False)
        log(f"index now {len(idx)} rows, {(idx['status'] == 'ok').sum()} ok")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    stems = [p.stem for p in sorted(TRAIN.glob("*.jpeg"))] + [p.stem for p in sorted(TEST.glob("*.jpeg"))]
    todo = [s for s in stems if not (DEST / f"{CUR['prefix']}_{s}.npy").exists()]
    have_n = len(stems) - len(todo)
    if a_.limit:
        todo = todo[: a_.limit]
    log(f"{len(stems)} images total, {have_n} already fetched, {len(todo)} to go this run")

    idx_path = DEST / ("index.parquet" if a_.series == "hmi" else f"index_{a_.series}.parquet")
    have = pd.read_parquet(idx_path) if idx_path.exists() else pd.DataFrame()
    rows, t0, done = [], time.time(), 0

    shared = requests.Session()
    shared.headers["Connection"] = "close"
    for scheme in ("http://", "https://"):
        shared.mount(scheme, requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1,
                                                           max_retries=0, pool_block=False))

    def work(s):
        nonlocal done
        t_one = time.time()
        r = fetch_one(s, shared)
        log(f"  {s} {'ok' if (r or {}).get('status') == 'ok' else (r or {}).get('status', 'skip')} "
            f"{time.time() - t_one:.1f}s")
        with _lock:
            done += 1
            if r:
                rows.append(r)
            if done % 50 == 0 and rows:
                cur = pd.DataFrame([x for x in rows if x])
                allr = pd.concat([have, cur], ignore_index=True) if len(have) else cur
                allr.drop_duplicates(subset=["stem"], keep="last").to_parquet(idx_path, index=False)
            if done % 20 == 0:
                el = time.time() - t0
                log(f"  {done}/{len(todo)} in {el:.0f}s ({el / max(done, 1):.1f}s each, "
                    f"~{(len(todo) - done) * el / max(done, 1) / 60:.0f} min left)")
        return r

    with ThreadPoolExecutor(max_workers=a_.workers) as ex:
        list(ex.map(work, todo))

    new = pd.DataFrame([r for r in rows if r])
    if len(new):
        idx = pd.concat([have, new], ignore_index=True) if len(have) else new
        idx = idx.drop_duplicates(subset=["stem"], keep="last")
        idx.to_parquet(idx_path, index=False)
        ok = (idx["status"] == "ok").sum()
        log(f"\nindex: {len(idx)} rows, {ok} ok")
        if "dt_s" in idx:
            d = idx.loc[idx["status"] == "ok", "dt_s"]
            log(f"  time offset from the H-alpha frame: median {d.median():.0f}s, max {d.max():.0f}s")
    n = len(list(DEST.glob("mag_*.npy")))
    size = sum(p.stat().st_size for p in DEST.glob("mag_*.npy")) / 1e9
    log(f"{n} magnetograms on disk, {size:.2f} GB")


if __name__ == "__main__":
    main()
