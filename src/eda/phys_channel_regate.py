"""Re-ask the physical-channels negative against the gate that actually matters.

The rejected result. `--channels phys` feeds [raw, local contrast, radial
position] instead of a replicated greyscale, and was rejected at val pixel-Dice
0.6971 against the baseline's 0.7019 (ledger, section 3).

Why that gate may have been wrong. Foreground is 0.42% of the frame, so
pixel-Dice is dominated by the large, high-contrast filaments both models find
easily. The population the contrast channel is supposed to help is a faint
MINORITY: fold-0's never-proposed ground-truth rows have median contrast 0.089
against 0.114 for the rows we match, while their area (1352 vs 1490) and spine
length (119 vs 123) are indistinguishable. They are ordinary filaments that are
simply faint, and 75.5% of them sit on entries the annotators agreed about --
13.6% at full 3/3 consensus. A model could recover a hundred of them and move
pixel-Dice by less than the run-to-run noise.

This is the same class of error as `eval_tiled.py` demanding that a single lane
beat the whole pipeline's oracle, and as the mAP trap recorded in the ledger
(crack-init led on val mAP50 and lost on PQ). So the negative is re-asked, not
overturned: the question is whether the contrast channel REACHES ground truth
the baseline never proposes, which is the only property that earns a place here.

Both probability maps are already cached at 2048 from the matched pair of runs
(sem_b3_f0_1536_cldice and sem_b3_f0_1536_cldice_phys), so this costs no GPU and
changes nothing about how either was trained.

Usage: python src/eda/phys_channel_regate.py
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

from filament.rle import decode_rle

SEM = ROOT / "outputs" / "semantic"
OUT = ROOT / "outputs" / "eda"
ARMS = {"baseline": "fold0_probs_cldice.npz", "phys": "fold0_probs_phys.npz"}
THR, MIN_AREA, CLOSE_K = 0.5, 200, 5


def log(*a):
    print(*a, flush=True)


def boxed(m):
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    return (y0, y1, x0, x1), m[y0:y1, x0:x1], int(m.sum())


def iou(a, b):
    (ay0, ay1, ax0, ax1), asub, asum = a
    (by0, by1, bx0, bx1), bsub, bsum = b
    y0, y1 = max(ay0, by0), min(ay1, by1)
    x0, x1 = max(ax0, bx0), min(ax1, bx1)
    if y0 >= y1 or x0 >= x1:
        return 0.0
    i = int(np.logical_and(asub[y0 - ay0:y1 - ay0, x0 - ax0:x1 - ax0],
                           bsub[y0 - by0:y1 - by0, x0 - bx0:x1 - bx0]).sum())
    u = asum + bsum - i
    return i / u if u else 0.0


def instances(prob_u8, thr=THR, min_area=MIN_AREA, close_k=CLOSE_K):
    b = (prob_u8 > int(round(thr * 255))).astype(np.uint8)
    if close_k:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    return [(lab == i) for i in range(1, n)
            if stats[i, cv2.CC_STAT_AREA] >= min_area]


def main():
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    f0 = set(folds.loc[folds["fold"] == 0, "file_name"]
             .str.replace(".jpeg", "", regex=False))
    d = pd.read_parquet(OUT / "error_profile_fn_gt.parquet")
    d["matched"] = d["matched"].astype(bool)
    d["in_raw_pool"] = d["in_raw_pool"].astype(bool)
    never = d[d["stem"].isin(f0) & ~d["matched"] & ~d["in_raw_pool"]]
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt = gt.set_index("filament_id")["segmentation_rle"]
    log(f"fold-0 never-proposed GT rows: {len(never)} "
        f"(median contrast {never['contrast'].median():.3f} against "
        f"{d[d['matched']]['contrast'].median():.3f} for matched rows)")

    res = {}
    for arm, fn in ARMS.items():
        z = np.load(SEM / fn)
        t0, rows = time.time(), []
        for stem, g in never.groupby("stem"):
            if stem not in z:
                for r in g.itertuples():
                    rows.append(dict(filament_id=r.filament_id, best=0.0))
                continue
            pm = [boxed(m) for m in instances(np.asarray(z[stem]))]
            pm = [x for x in pm if x is not None]
            for r in g.itertuples():
                gm = boxed(decode_rle(gt.loc[r.filament_id]).astype(bool))
                rows.append(dict(filament_id=r.filament_id,
                                 best=max((iou(gm, x) for x in pm), default=0.0)
                                 if gm else 0.0))
        rf = pd.DataFrame(rows).set_index("filament_id")["best"]
        res[arm] = rf
        log(f"\n{arm} ({fn}, thr {THR}, min_area {MIN_AREA}, close {CLOSE_K}) "
            f"[{time.time()-t0:.0f}s]")
        for t in (0.3, 0.4, 0.5, 0.6):
            n = int((rf > t).sum())
            log(f"  reaches IoU > {t}: {n:4d} ({100*n/len(rf):.1f}%)")

    j = pd.DataFrame(res).dropna()
    B, P = j["baseline"] > 0.5, j["phys"] > 0.5
    log(f"\nhead to head on {len(j)} never-proposed rows:")
    log(f"  baseline reaches      : {int(B.sum())}")
    log(f"  phys reaches          : {int(P.sum())}")
    log(f"  both                  : {int((B & P).sum())}")
    log(f"  phys only             : {int((P & ~B).sum())}")
    log(f"  baseline only         : {int((B & ~P).sum())}")
    log(f"  union                 : {int((B | P).sum())}")
    log(f"\n  mean best IoU: baseline {j['baseline'].mean():.4f}  "
        f"phys {j['phys'].mean():.4f}")

    faint = j.join(never.set_index("filament_id")["contrast"]).dropna()
    faint["q"] = pd.qcut(faint["contrast"], 4,
                         labels=["C1 faintest", "C2", "C3", "C4"])
    log("\n  reach by contrast quartile of the never-proposed rows "
        "(this is where a contrast channel has to show up):")
    tab = faint.groupby("q", observed=True).apply(
        lambda g: pd.Series({"n": len(g),
                             "baseline": (g["baseline"] > 0.5).sum(),
                             "phys": (g["phys"] > 0.5).sum()}))
    log("   " + tab.to_string().replace("\n", "\n   "))

    json.dump({k: dict(reach_050=int((v > 0.5).sum()), n=int(len(v)),
                       mean_best=float(v.mean())) for k, v in res.items()},
              open(OUT / "phys_channel_regate.json", "w"), indent=1)
    log(f"\nwrote {OUT/'phys_channel_regate.json'}")


if __name__ == "__main__":
    main()
