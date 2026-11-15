"""Parity cases built to BREAK the agreement, not to confirm it.

`scorer_parity.py` reports 0.00e+00 on all seven of its cases including a real
fold subset with 400 candidates, and an exact zero across hundreds of
floating-point additions is a fair thing to be suspicious of. Two explanations
are possible and they have very different consequences:

  benign    both implementations reduce to the same arithmetic on the same
            integers in the same order. IoU is intersection/union of integer
            pixel counts, IEEE-754 division is deterministic, TP/FP/FN are
            integer counts, and if the IoU list is summed in the same order the
            results are bit-identical rather than merely close.
  hidden    the cases never exercise a path where the two could differ.

This file assumes the second until the first is demonstrated. It targets the
places where the implementations are structurally different rather than where
they are obviously the same:

  * the official scorer imports `pycocotools.mask`, ours is pure numpy. The
    recorded pycocotools failure -- 32-bit overflow at exactly 1024 masks at
    2048x2048 -- would hit the OFFICIAL path and not ours, which is the one
    divergence that could silently corrupt a real submission rather than a test.
  * summation order: shuffling the input rows changes float addition order. If
    the delta stays exactly zero under shuffling, agreement is structural; if it
    moves to 1e-16, it was luck of ordering and the "exact" claim should be
    softened to "within float precision".
  * degenerate inputs the happy path never produces: an image with predictions
    and no ground truth, ground truth with no predictions, zero-area masks,
    duplicate identical predictions, and a prediction covering several ground
    truths at once.

Usage: python src/eda/scorer_parity_adversarial.py
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament.pq import score as our_score
from filament.rle import encode_mask

OFFICIAL = ROOT / "outputs" / "research" / "official_pq_scorer_extracted.py"


def log(*a):
    print(*a, flush=True)


def load_official():
    spec = importlib.util.spec_from_file_location("official", OFFICIAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return lambda g, p: mod.get_pq_score(mod.get_overlap_df(g, p))


def rect(y0, y1, x0, x1):
    m = np.zeros((2048, 2048), np.uint8)
    m[y0:y1, x0:x1] = 1
    return encode_mask(m)


def frame(rows, cols=("filament_id", "segmentation_rle")):
    return pd.DataFrame(rows, columns=cols)


def run(name, gt, pred, official, results):
    try:
        a = our_score(gt, pred)
        a = a["pq"] if isinstance(a, dict) else a
    except Exception as e:
        a = f"RAISED {type(e).__name__}: {str(e)[:60]}"
    try:
        b = official(gt, pred)
    except Exception as e:
        b = f"RAISED {type(e).__name__}: {str(e)[:60]}"
    if isinstance(a, float) and isinstance(b, float):
        d = abs(a - b)
        log(f"{name:52s} {a!r:>22} {b!r:>22}  delta {d:.3e}")
    else:
        log(f"{name:52s} ours={a}  official={b}")
        d = None
    results.append((name, a, b, d))


def main():
    official = load_official()
    res = []
    log(f"{'case':52s} {'ours':>22} {'official':>22}\n")

    # ---- degenerate shapes the production path never produces ---------------
    gt = frame([("A01-im1_1", rect(100, 200, 100, 200))])
    run("prediction on an image with no ground truth",
        gt, frame([("im2_1", rect(100, 200, 100, 200))]), official, res)

    run("ground truth with no prediction at all",
        gt, frame([], ), official, res)

    run("duplicate identical predictions (same mask twice)",
        gt, frame([("im1_1", rect(100, 200, 100, 200)),
                   ("im1_2", rect(100, 200, 100, 200))]), official, res)

    gt2 = frame([("A01-im1_1", rect(100, 200, 100, 200)),
                 ("A01-im1_2", rect(100, 200, 300, 400))])
    run("one prediction spanning two ground truths",
        gt2, frame([("im1_1", rect(100, 200, 100, 400))]), official, res)

    # a mask that is empty after decoding
    empty = encode_mask(np.zeros((2048, 2048), np.uint8))
    run("zero-area prediction mask",
        gt, frame([("im1_1", empty)]), official, res)
    run("zero-area ground-truth mask",
        frame([("A01-im1_1", empty)]), frame([("im1_1", rect(100, 200, 100, 200))]),
        official, res)

    # ---- summation order ---------------------------------------------------
    rng = np.random.default_rng(0)
    rows_g, rows_p = [], []
    for i in range(60):
        y, x = int(rng.integers(50, 1900)), int(rng.integers(50, 1900))
        rows_g.append((f"A01-im{i//12}_g{i}", rect(y, y + 60, x, x + 60)))
        # jitter so IoU lands on irrational-ish values rather than 1.0
        dy = int(rng.integers(0, 18))
        rows_p.append((f"im{i//12}_p{i}", rect(y + dy, y + 60 + dy, x, x + 60)))
    g_df, p_df = frame(rows_g), frame(rows_p)
    a0 = our_score(g_df, p_df)
    a0 = a0["pq"] if isinstance(a0, dict) else a0
    b0 = official(g_df, p_df)
    log("")
    run("60 jittered pairs, original row order", g_df, p_df, official, res)
    for seed in (1, 2, 3):
        r = np.random.default_rng(seed)
        gi = r.permutation(len(g_df))
        pi = r.permutation(len(p_df))
        run(f"  same data, rows shuffled (seed {seed})",
            g_df.iloc[gi].reset_index(drop=True),
            p_df.iloc[pi].reset_index(drop=True), official, res)
    log(f"  -> if our own score changes under shuffling, the metric is not")
    log(f"     order-invariant and every reported figure carries that noise.")
    a_shuf = [x[1] for x in res[-3:]]
    log(f"  our PQ across shuffles: {sorted(set(repr(v) for v in a_shuf))}")

    # ---- the structural divergence: pycocotools mask count -----------------
    log("")
    log("pycocotools 32-bit overflow probe (the official scorer imports")
    log("pycocotools.mask; ours does not -- this is the one path where a real")
    log("submission could diverge from what we measure locally):")
    for n in (300, 1000, 1024, 1100):
        rows = []
        side = int(np.ceil(np.sqrt(n)))
        step = 2040 // side
        k = 0
        for i in range(side):
            for j in range(side):
                if k >= n:
                    break
                y, x = i * step, j * step
                rows.append((f"imz_p{k}", rect(y, y + step - 2, x, x + step - 2)))
                k += 1
        gtz = frame([("A01-imz_g0", rect(0, step - 2, 0, step - 2))])
        t0 = time.time()
        run(f"  {n} predictions on one image", gtz, frame(rows), official, res)
        log(f"     ({time.time()-t0:.0f}s)")

    ok = [r for r in res if r[3] is not None]
    bad = [r for r in res if r[3] is None]
    log(f"\ncomparable cases: {len(ok)}, largest delta "
        f"{max((r[3] for r in ok), default=0):.3e}")
    if bad:
        log(f"cases where one side raised or the two are not comparable: {len(bad)}")
        for r in bad:
            log(f"  {r[0]}")
        log("These are the interesting ones: a divergence in BEHAVIOUR is worth")
        log("more than a divergence in the sixteenth decimal.")


if __name__ == "__main__":
    main()
