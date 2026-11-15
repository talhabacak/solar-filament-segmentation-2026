"""Why the board wants more shrink than the cross-fit does.

Two measurements disagree on the mask shrink and both are reproducible.

Locally, the true-fractional rule at 0.10 beats the production ring erosion:
0.4331 against 0.4286 on the base pool, and 0.4416 against 0.4368 on the blob5
pool. On the board, the ordering says the opposite -- sub018 (production) ranks
above sub015 (true-fractional 0.10), the two arms differing in nothing else.

The production rule removes about 19% of a mask's area; true-fractional 0.10
removes 10%. So the board is asking for MORE trimming than the cross-fit is.

A shrink pays exactly when a prediction over-covers its ground truth, because PQ
thresholds at IoU > 0.5 and trimming a systematically fat mask raises the ratio.
The obvious way for train and test to disagree about the optimal amount is for
the model to over-cover MORE on test than on the held-out training folds -- which
is what a harder distribution does: looser probability fields, fatter masks at the
same threshold.

That is directly checkable without any test labels. Compare the areas of the
instances the policy actually KEEPS, out-of-fold against test. If the test
instances are systematically larger, the board's preference has a mechanism and
the axis is worth pushing further than the production rule rather than merely
reverting to it. If they are the same size, the disagreement is not about
over-coverage and the local number should not be overridden on a story.

Usage: python src/eda/train_test_area_shift.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament.rle import decode_rle
from mask_shrink import shrink_frac, shrink_true_frac

OUT = ROOT / "outputs" / "eda"
SEL = ROOT / "outputs" / "selector"
SUB = ROOT / "outputs" / "submissions"


def log(*a):
    print(*a, flush=True)


def areas_from_csv(path):
    d = pd.read_csv(path)
    a = np.array([int(decode_rle(r).sum()) for r in d["segmentation_rle"]])
    stems = d["filament_id"].str.rsplit("_", n=1).str[0]
    return a, stems


def describe(name, a, per_img):
    q = np.percentile(a, [10, 25, 50, 75, 90])
    log(f"  {name:<22}{len(a):>6}{a.mean():>10.0f}{q[2]:>9.0f}"
        f"{q[0]:>8.0f}{q[1]:>8.0f}{q[3]:>8.0f}{q[4]:>8.0f}{per_img:>8.2f}")
    return q


def main():
    # test side: the instances actually submitted, from the two arms that
    # isolate the shrink rule
    log("kept instance areas (pixels at 2048), test vs out-of-fold\n")
    log(f"  {'arm':<22}{'n':>6}{'mean':>10}{'median':>9}"
        f"{'p10':>8}{'p25':>8}{'p75':>8}{'p90':>8}{'/img':>8}")

    rows = {}
    for tag, fn in (("sub015 test (true .10)", "sub015_recipe3_tf010_ts.csv"),
                    ("sub018 test (prod)", "sub018_prodshrink_ts.csv"),
                    ("sub019 test (blob5)", "sub019_blob5_ts.csv")):
        p = SUB / fn
        if not p.exists():
            continue
        a, stems = areas_from_csv(p)
        rows[tag] = a
        describe(tag, a, len(a) / stems.nunique())

    # train side: the same policy applied to the out-of-fold pool, so the
    # comparison is between instances that survived an equivalent decision
    import joblib

    for tag, pool, bundle_name, true_frac, frac in (
            ("sub015 OOF (true .10)", "fused_pool_all.parquet",
             "selector_v5_shrunk_tf010ts3.joblib", True, 0.10),
            ("sub019 OOF (blob5)", "fused_pool_all_blob5.parquet",
             "selector_v5_shrunkblob5ts.joblib", True, 0.10)):
        bp = SEL / bundle_name
        pp = SEL / pool
        if not (bp.exists() and pp.exists()):
            continue
        b = joblib.load(bp)
        cand = pd.read_parquet(pp)
        sh = shrink_true_frac if true_frac else shrink_frac
        # The selector was fitted on SHRUNK candidates, so area and log10_area
        # must be recomputed after shrinking before the model is asked anything.
        # Accumulate the areas only: holding the shrunk masks in a list is 4 MiB
        # each at 2048^2, which for ten thousand candidates is 39 GB and killed
        # this script once on a 31 GiB box.
        areas = np.fromiter(
            (int(sh(decode_rle(r).astype(bool), frac).sum()) for r in cand["rle"]),
            dtype=np.int64, count=len(cand))
        cand["area"] = areas
        cand["log10_area"] = np.log10(np.maximum(areas, 1))
        feats = b["features"]
        X = cand[feats].to_numpy(np.float64)
        p = b["model"].predict_proba(X)[:, 1]
        if b.get("iou_model") is not None:
            p = p * np.clip(b["iou_model"].predict(X), 0.0, None)
            ref = b.get("two_stage_cdf")
            if ref is not None:
                p = np.searchsorted(ref, p, side="right") / len(ref)
        pol = b["policy"]
        sel_mask = (cand["conf"] >= pol["conf"]).to_numpy() & (p >= pol["p"])
        keep = cand[sel_mask]
        a = areas[sel_mask]
        rows[tag] = a
        describe(tag, a, len(a) / keep["stem"].nunique())

    log("\nthe comparison that matters -- same recipe, test against out-of-fold:")
    for t, o in (("sub015 test (true .10)", "sub015 OOF (true .10)"),
                 ("sub019 test (blob5)", "sub019 OOF (blob5)")):
        if t in rows and o in rows:
            rt, ro = np.median(rows[t]), np.median(rows[o])
            log(f"  {o:<24} median {ro:>7.0f}   ->   {t:<24} median {rt:>7.0f}"
                f"   ratio {rt/ro:.3f}")

    json.dump({k: dict(n=len(v), mean=float(v.mean()), median=float(np.median(v)))
               for k, v in rows.items()},
              open(OUT / "train_test_area_shift.json", "w"), indent=1)
    log(f"\nwrote {OUT/'train_test_area_shift.json'}")
    log("\nA ratio above 1 means the submitted instances are fatter than the ones")
    log("the shrink was tuned on, which is the condition under which the board")
    log("would want a larger trim than the cross-fit chose. A ratio near 1 leaves")
    log("the disagreement unexplained, and the local number should then stand.")


if __name__ == "__main__":
    main()
