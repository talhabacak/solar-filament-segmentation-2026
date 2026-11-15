"""Should augmentation run live at native 2048, or on the 768 cache as it does now?

"On the fly" hides three separate questions and they have different answers.

  1. live vs precomputed     Already settled: __getitem__ augments every sample
                             every epoch, so nothing is frozen to disk. Freezing
                             would be strictly worse -- the model would see a
                             fixed finite set instead of a fresh draw.

  2. CPU workers vs GPU      A throughput question, answered by measurement, not
     (Kornia/DALI)           by preference.

  3. WHICH RESOLUTION        The real question. The cache is pre-downsampled to
                             768, so every augmentation is applied to already
                             -downsampled pixels. At inference the model sees
                             2048 frames downsampled to 768. For an augmentation
                             to model anything real, doing it at 768 has to be
                             equivalent to doing it at 2048 and then
                             downsampling. For some operations it is. For others
                             it is not, and then we are training against a
                             fiction.

The operations at risk are the ones with an intrinsic spatial scale:

  JPEG      blocks are 8x8 PIXELS. At 2048 a block spans 8/2048 of the frame; at
            768 it spans 8/768 -- 2.67x larger relative to the image. Worse, the
            2048->768 INTER_AREA downsample averages 2.67x2.67 pixels together,
            which largely destroys 8px blocking. So blocking applied at 768 is
            an artefact the model will never meet at inference.
  noise     same argument: pixel-scale noise at 2048 is attenuated by the area
            average, while noise injected at 768 survives intact.
  rotation  interpolating at 768 resamples structures that are already only a
            few pixels wide; rotating at 2048 first and then area-downsampling
            is anti-aliased. Filaments are thin, so this is where it could bite.

Measured here, per operation: how far does "do it at 768" sit from "do it at 2048
then downsample", against the natural spread between two images as the yardstick
for what counts as far.

Usage: python src/eda/aug_onthefly.py
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

OUT = ROOT / "outputs" / "eda"
S = 768
NAT = 2048
N = 12
KER = np.ones((15, 15), np.uint8)


def log(*a):
    print(*a, flush=True)


def down(x):
    return cv2.resize(x, (S, S), interpolation=cv2.INTER_AREA)


def jpeg(x, q):
    ok, enc = cv2.imencode(".jpg", x, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE) if ok else x


def rot(x, ang, interp):
    n = x.shape[0]
    c = (n - 1) / 2.0
    M = cv2.getRotationMatrix2D((c, c), ang, 1.0)
    return cv2.warpAffine(x, M, (n, n), flags=interp)


def filament_contrast(im, mk):
    """Filament minus its local ring -- the physical signal the model keys on."""
    m = mk > 0.5
    ring = (cv2.dilate(m.astype(np.uint8), KER, iterations=2) > 0) & ~m & (im > 8)
    if m.sum() < 200 or ring.sum() < 200:
        return np.nan
    return float(im[m].mean()) - float(im[ring].mean())


def main():
    idx = json.loads((ROOT / "data" / "semantic_cache" / f"index_{S}.json").read_text())
    imgs = np.load(ROOT / "data" / "semantic_cache" / f"images_{S}.npy", mmap_mode="r")
    msks = np.load(ROOT / "data" / "semantic_cache" / f"masks_{S}.npy", mmap_mode="r")
    fr = {f: i for i, f in enumerate(idx["files"])}
    er = {e: i for i, e in enumerate(idx["entry_ids"])}
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    ent = gt[["entry_id", "file_name"]].drop_duplicates().values
    raw_dir = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"
    paths = {p.name: p for p in raw_dir.rglob("*.jpeg")}

    rng = np.random.default_rng(0)
    picks = []
    for i in rng.permutation(len(ent)):
        eid, fn = ent[int(i)]
        if fn not in paths:
            continue
        m = np.unpackbits(msks[er[eid]].reshape(S, -1), axis=-1)[:, :S]
        if m.sum() > 800:
            picks.append((eid, fn, m.astype(np.float32)))
        if len(picks) >= N:
            break
    log(f"{len(picks)} image/entry pairs with a substantial mask\n")

    # ---- yardstick: how far apart are two DIFFERENT frames? -----------------
    base = []
    for eid, fn, mk in picks:
        base.append(down(cv2.imread(str(paths[fn]), cv2.IMREAD_GRAYSCALE)))
    between = [float(np.abs(base[i].astype(np.float32) -
                            base[j].astype(np.float32)).mean())
               for i in range(len(base)) for j in range(i + 1, len(base))]
    yard = float(np.mean(between))
    log(f"yardstick: mean |difference| between two different frames = {yard:.2f} DN")
    log("  an at-768 / at-2048 discrepancy is only meaningful next to this\n")

    rows = []

    # ---- 1. JPEG ------------------------------------------------------------
    log("1. JPEG quality jitter -- does 768 blocking survive to inference?")
    for q in (78, 85, 90):
        d_native, d_cache, blk_n, blk_c = [], [], [], []
        for k, (eid, fn, mk) in enumerate(picks):
            raw = cv2.imread(str(paths[fn]), cv2.IMREAD_GRAYSCALE)
            clean = down(raw)
            at_native = down(jpeg(raw, q))          # correct: encode at 2048
            at_cache = jpeg(clean, q)               # what we do now: encode at 768
            d_native.append(float(np.abs(at_native.astype(np.float32) -
                                         clean.astype(np.float32)).mean()))
            d_cache.append(float(np.abs(at_cache.astype(np.float32) -
                                        clean.astype(np.float32)).mean()))
            blk_n.append(blockiness(at_native))
            blk_c.append(blockiness(at_cache))
        log(f"   q={q}: change vs clean -- at 2048 then down {np.mean(d_native):.3f} DN, "
            f"at 768 {np.mean(d_cache):.3f} DN  ({np.mean(d_cache)/max(np.mean(d_native),1e-6):.1f}x)")
        log(f"          8px blockiness   -- at 2048 then down {np.mean(blk_n):.4f}, "
            f"at 768 {np.mean(blk_c):.4f}")
        rows.append(dict(op=f"jpeg_q{q}", native=np.mean(d_native),
                         cache=np.mean(d_cache),
                         block_native=np.mean(blk_n), block_cache=np.mean(blk_c)))

    # ---- 2. noise -----------------------------------------------------------
    log("\n2. Gaussian noise -- how much of a 2048 injection survives the downsample?")
    log("   (measured as the std of the injected residual ON DISK. A first pass used")
    log("    the whole-image std and got a NEGATIVE answer: global std is dominated")
    log("    by the disk-vs-black-corner step, and clipping lifts the 0-valued")
    log("    corners, which shrinks that step. The noise never entered the number.)")
    for sd in (2, 5, 7):
        surv, direct = [], []
        for eid, fn, mk in picks:
            raw = cv2.imread(str(paths[fn]), cv2.IMREAD_GRAYSCALE)
            clean = down(raw).astype(np.float32)
            dk = clean > 8
            g = np.random.default_rng(1)
            n2 = np.clip(raw.astype(np.float32) + g.normal(0, sd, raw.shape), 0, 255)
            surv.append(float((down(n2.astype(np.uint8)).astype(np.float32) - clean)[dk].std()))
            n7 = np.clip(clean + g.normal(0, sd, clean.shape), 0, 255)
            direct.append(float((n7 - clean)[dk].std()))
        log(f"   sd={sd}: injected at 2048 -> {np.mean(surv):.3f} DN survives to 768;  "
            f"injected at 768 -> {np.mean(direct):.3f} DN "
            f"({np.mean(direct)/max(np.mean(surv),1e-6):.1f}x too strong)")
        rows.append(dict(op=f"noise_sd{sd}", native=np.mean(surv),
                         cache=np.mean(direct)))

    # ---- 3. rotation --------------------------------------------------------
    log("\n3. Rotation -- does interpolating at 768 damage thin filaments?")
    for ang in (10.0, 20.0):
        c_nat, c_cac, c_clean = [], [], []
        for eid, fn, mk in picks:
            raw = cv2.imread(str(paths[fn]), cv2.IMREAD_GRAYSCALE)
            mk2 = cv2.resize(mk, (NAT, NAT), interpolation=cv2.INTER_NEAREST)
            a_nat = down(rot(raw, ang, cv2.INTER_LINEAR))
            m_nat = cv2.resize(rot(mk2, ang, cv2.INTER_NEAREST), (S, S),
                               interpolation=cv2.INTER_NEAREST)
            a_cac = rot(down(raw), ang, cv2.INTER_LINEAR)
            m_cac = rot(mk, ang, cv2.INTER_NEAREST)
            c_nat.append(filament_contrast(a_nat, m_nat))
            c_cac.append(filament_contrast(a_cac, m_cac))
            c_clean.append(filament_contrast(down(raw), mk))
        cn, cc, c0 = (float(np.nanmean(x)) for x in (c_nat, c_cac, c_clean))
        log(f"   {ang:4.0f} deg: filament contrast unrotated {c0:+.2f}, "
            f"rotated at 2048 then down {cn:+.2f}, rotated at 768 {cc:+.2f}")
        log(f"            loss vs unrotated: at 2048 {cn-c0:+.2f}, at 768 {cc-c0:+.2f}")
        rows.append(dict(op=f"rot{ang:.0f}", native=cn, cache=cc, clean=c0))

    # ---- 4. throughput ------------------------------------------------------
    log("\n4. Throughput -- can a live 2048 pipeline feed the GPU at all?")
    t0 = time.time()
    for eid, fn, mk in picks:
        raw = cv2.imread(str(paths[fn]), cv2.IMREAD_GRAYSCALE)
        _ = down(rot(raw, 12.0, cv2.INTER_LINEAR))
    live_ms = (time.time() - t0) / len(picks) * 1000
    t0 = time.time()
    for eid, fn, mk in picks:
        _ = rot(np.asarray(imgs[fr[fn]]), 12.0, cv2.INTER_LINEAR)
    cache_ms = (time.time() - t0) / len(picks) * 1000
    gpu_ms = 45_000 / 924
    log(f"   live  (decode 2048 + rotate + downsample): {live_ms:6.1f} ms/sample")
    log(f"   cache (rotate at 768)                    : {cache_ms:6.1f} ms/sample")
    log(f"   GPU step                                 : {gpu_ms:6.1f} ms/sample")
    log(f"   live on 3 workers = {live_ms/3:.1f} ms/sample wall clock -> "
        f"{'feeds the GPU' if live_ms/3 < gpu_ms else 'STARVES the GPU'}")
    log("   (masks are extra: the live path must also rasterise RLE at 2048)")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "aug_onthefly.json", "w") as f:
        json.dump(dict(yardstick=yard, rows=rows, live_ms=live_ms,
                       cache_ms=cache_ms, gpu_ms=gpu_ms), f, indent=1, default=float)
    log(f"\nsaved -> {OUT/'aug_onthefly.json'}")


def blockiness(x):
    """Ratio of gradient energy on the 8px JPEG grid to gradient energy off it.

    1.0 means no blocking; JPEG pushes it above 1 by putting discontinuities
    exactly on the block boundaries.
    """
    v = np.abs(np.diff(x.astype(np.float32), axis=1))
    on = v[:, 7::8].mean()
    off = np.delete(v, np.arange(7, v.shape[1], 8), axis=1).mean()
    return float(on / max(off, 1e-6))


if __name__ == "__main__":
    main()
