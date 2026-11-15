"""Does the augmentation preserve image/mask alignment, and what does it cost?

Two failure modes would make the study meaningless and neither shows up as an
exception:

  silent de-alignment   if a geometric op reaches the image but not the mask (or
                        applies a different interpolation), training still runs
                        and simply learns nothing. The test: filaments are DARK
                        absorption features, so the mean intensity under the
                        mask must stay below its surroundings. Break the
                        alignment and that contrast collapses towards zero.

                        The reference must be a LOCAL ring, not the disk mean.
                        Measured: against the whole disk, filaments come out
                        +9.2 BRIGHTER, because limb darkening drags the disk
                        average down while filaments sit at mid-latitudes near
                        disk centre. Against a 2x15px dilated ring they are
                        -15.7 darker in 100% of instances (sd 4.2) — the correct
                        sign, and a 3.8-sigma probe instead of a 1.3-sigma one.
  a CPU bottleneck      JPEG re-encode and motion blur at 768^2 run per sample on
                        3 workers. If a sample costs more than the GPU step, the
                        heavy arm is slower for a reason that has nothing to do
                        with augmentation.

Also reports how much the mask survives: rotation pushes pixels off the frame,
and if heavy augmentation quietly deleted a chunk of the target that would be a
confound rather than a regulariser.

Usage: python src/eda/aug_sanity.py
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from augment import apply_aug

OUT = ROOT / "outputs" / "eda"
S = 768
N = 120
KER = np.ones((15, 15), np.uint8)


def log(*a):
    print(*a, flush=True)


def main():
    import finetune_aug                      # import check, the build_fused_pool lesson
    log(f"import ok: augment, finetune_aug ({finetune_aug.__name__})")

    import pandas as pd
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    entries = list(gt[["entry_id", "file_name"]].drop_duplicates()
                   .itertuples(index=False, name=None))
    ds = finetune_aug.AugCached(entries, S, train=False, aug="none")
    ds._open()

    rng0 = np.random.default_rng(0)
    picks = rng0.choice(len(entries), N, replace=False)
    res = {}
    for level in ("none", "light", "medium", "heavy"):
        rng = np.random.default_rng(1)
        contr, keep, t0 = [], [], time.time()
        for i in picks:
            eid, fn = entries[int(i)]
            img = np.asarray(ds.imgs[ds.file_row[fn]])
            mask = ds._unpack(ds.msks, ds.entry_row[eid]).astype(np.float32)
            a0 = float(mask.sum())
            if a0 < 50:
                continue
            im, mk, _ = apply_aug(img, mask, None, S, level, rng)
            m = mk > 0.5
            if m.sum() < 200:
                continue
            ring = (cv2.dilate(m.astype(np.uint8), KER, iterations=2) > 0) \
                & ~m & (im > 8)               # local surroundings, on-disk only
            if ring.sum() < 200:
                continue
            contr.append(float(im[m].mean()) - float(im[ring].mean()))
            keep.append(float(m.sum()) / a0)
        dt = (time.time() - t0) / max(len(contr), 1) * 1000
        res[level] = dict(contrast=float(np.mean(contr)),
                          contrast_sd=float(np.std(contr)),
                          mask_kept=float(np.mean(keep)),
                          ms_per_sample=round(dt, 2))
        log(f"{level:7s} filament-minus-ring intensity {np.mean(contr):+7.2f} "
            f"(sd {np.std(contr):5.2f})   mask kept {np.mean(keep):.3f}   "
            f"{dt:6.2f} ms/sample   n={len(contr)}")

    log("\ninterpretation")
    base = res["none"]["contrast"]
    for level in ("light", "medium", "heavy"):
        r = res[level]
        drift = r["contrast"] - base
        # Blur genuinely reduces contrast — that is the point of it — so the
        # test is for a COLLAPSE towards zero, which is what de-alignment does.
        verdict = "aligned" if abs(r["contrast"]) > 0.5 * abs(base) else "DE-ALIGNED"
        log(f"  {level:7s} contrast {r['contrast']:+6.2f} "
            f"({drift:+.2f} vs unaugmented {base:+.2f}) -> {verdict}")
    gpu_ms = 45_000 / 924                 # reference run: 45 s for 924 samples
    log(f"\n  the 768 GPU step costs ~{gpu_ms:.0f} ms/sample (reference run: "
        f"45 s / 924 train entries at batch 4). On 3 workers the heavy arm "
        f"costs {res['heavy']['ms_per_sample']/3:.1f} ms/sample of wall clock "
        f"-> {'not a bottleneck' if res['heavy']['ms_per_sample']/3 < gpu_ms else 'BOTTLENECK'}")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "aug_sanity.json", "w") as f:
        json.dump(res, f, indent=1)
    log(f"\nsaved -> {OUT/'aug_sanity.json'}")


if __name__ == "__main__":
    main()
