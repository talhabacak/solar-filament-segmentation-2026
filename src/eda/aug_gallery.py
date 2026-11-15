"""Look at what each augmentation level actually does to a filament.

Numbers caught the miscalibrations; eyes catch the absurdities numbers do not
score — a blur that erases a thin filament while the mask still claims it, a
contrast scale that pushes the disk to clipping, a JPEG quality that puts 8x8
blocking right where the boundary is. The alignment probe would pass all three.

Two panels per row: the whole disk (does the frame still look like a solar
observation?) and a 256 px crop centred on one filament with its mask outlined
(is the target still supported by the pixels?).

Usage: python src/eda/aug_gallery.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from augment import apply_aug

OUT = ROOT / "outputs" / "gallery"
S = 768
CROP = 256
LEVELS = ("none", "light", "medium", "heavy")
N_ROWS = 6


def log(*a):
    print(*a, flush=True)


def main():
    idx = json.loads((ROOT / "data" / "semantic_cache" / f"index_{S}.json").read_text())
    imgs = np.load(ROOT / "data" / "semantic_cache" / f"images_{S}.npy", mmap_mode="r")
    msks = np.load(ROOT / "data" / "semantic_cache" / f"masks_{S}.npy", mmap_mode="r")
    fr = {f: i for i, f in enumerate(idx["files"])}
    er = {e: i for i, e in enumerate(idx["entry_ids"])}
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    ent = gt[["entry_id", "file_name"]].drop_duplicates().values

    # Pick rows spanning the stations, so the gallery shows the augmentation
    # acting on the natural contrast range (31.0 at T to 45.2 at M) rather than
    # six near-identical frames.
    cal = pd.read_parquet(ROOT / "outputs" / "eda" / "aug_calibration.parquet")
    by_st = {s: g.sort_values("contrast")["file"].iloc[len(g) // 2]
             for s, g in cal.groupby("station")}
    picks = []
    for st, fn in sorted(by_st.items()):
        for eid, f2 in ent:
            if f2 == fn:
                m = np.unpackbits(msks[er[eid]].reshape(S, -1), axis=-1)[:, :S]
                if m.sum() > 400:
                    picks.append((st, eid, fn))
                break
        if len(picks) >= N_ROWS:
            break

    tiles = []
    for st, eid, fn in picks:
        img0 = np.asarray(imgs[fr[fn]])
        msk0 = np.unpackbits(msks[er[eid]].reshape(S, -1), axis=-1)[:, :S].astype(np.float32)
        # Centre on the LARGEST component, not the whole mask's centroid: an
        # entry holds every filament in the frame, and their joint centroid lands
        # on empty disk between them.
        n, lab, stats, cent = cv2.connectedComponentsWithStats(
            (msk0 > 0.5).astype(np.uint8), 8)
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        cx, cy = int(cent[big][0]), int(cent[big][1])
        y0 = np.clip(cy - CROP // 2, 0, S - CROP)
        x0 = np.clip(cx - CROP // 2, 0, S - CROP)
        row = []
        for lv in LEVELS:
            rng = np.random.default_rng(7)          # same draw across levels
            im, mk, _ = apply_aug(img0.copy(), msk0.copy(), None, S, lv, rng)
            full = cv2.resize(im, (CROP, CROP), interpolation=cv2.INTER_AREA)
            full = cv2.cvtColor(full, cv2.COLOR_GRAY2BGR)
            crop = cv2.cvtColor(im[y0:y0 + CROP, x0:x0 + CROP], cv2.COLOR_GRAY2BGR)
            cm = (mk[y0:y0 + CROP, x0:x0 + CROP] > 0.5).astype(np.uint8)
            cnt, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(crop, cnt, -1, (0, 200, 255), 1)
            cv2.putText(full, f"{st} {lv}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 255, 0), 1, cv2.LINE_AA)
            row += [full, crop]
        tiles.append(np.hstack(row))
    sheet = np.vstack(tiles)

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "aug_levels.png"
    cv2.imwrite(str(p), sheet)
    log(f"{len(picks)} rows x {len(LEVELS)} levels (disk + 256px crop each)")
    log(f"columns: " + "  ".join(f"{lv}(full,crop)" for lv in LEVELS))
    log(f"saved -> {p}  ({sheet.shape[1]}x{sheet.shape[0]})")


if __name__ == "__main__":
    main()
