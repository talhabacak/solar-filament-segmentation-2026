"""Teacher pass: soft pseudo-labels for the unlabelled GONG frames.

Stage 1 of the two-stage plan (docs/14). The point is domain representation, not
label accuracy — the annotation convention is relearned in stage 2 from the real
MAGFiLO labels — so the target is the teacher's SOFT probability rather than a
threshold. Soft targets carry the teacher's uncertainty, which is exactly the
information a hard threshold throws away, and they degrade gracefully where the
teacher is unsure.

Teacher: the five held-out semantic models (sem_b3_f*_1536_cldice), averaged,
each with flip TTA. That is the strongest probability map we can produce, and
averaging five folds damps any single model's idiosyncrasy.

Writes a training-resolution cache in the same shape as the competition cache so
the pretraining stage reads it the same way:
    img_{S}.npy   uint8 (n, S, S)   the frame
    tgt_{S}.npy   uint8 (n, S, S)   teacher probability, 0-255
    meta_{S}.parquet

Usage: python src/models/pseudo_label_gong.py --imgsz 768
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament import IMG_H, IMG_W
from semantic_candidates import load_model

GONG = ROOT / "data" / "external" / "gong_unlabeled"
OUT = ROOT / "data" / "gong_pseudo"
REG = ROOT / "outputs" / "external" / "external_data_registry.csv"


def log(*a):
    print(*a, flush=True)


@torch.no_grad()
def teacher_prob(models, img, imgsz):
    """Mean probability over the fold models, flip TTA each, at imgsz."""
    small = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
    raw = small.astype(np.float32) / 255.0
    x = torch.from_numpy(np.stack([raw, raw, raw]))[None].cuda()
    acc = None
    for m in models:
        with torch.amp.autocast("cuda"):
            p = torch.sigmoid(m(x))
            p = p + torch.flip(torch.sigmoid(m(torch.flip(x, dims=[3]))), dims=[3])
            p = p / 2
        acc = p if acc is None else acc + p
    return (acc / len(models))[0, 0].float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--manifest", default=None,
                    help="text file listing the frames to label, one name per "
                         "line. Lets a labelling pass run on a SNAPSHOT while "
                         "more frames are still downloading, instead of racing "
                         "a growing directory.")
    ap.add_argument("--tag", default="",
                    help="suffix for the cache files, so several passes can "
                         "coexist and be read together at pretraining time")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    S = a.imgsz

    if a.manifest:
        names = [l.strip() for l in Path(a.manifest).read_text().splitlines() if l.strip()]
        frames = [GONG / n for n in names if (GONG / n).exists()]
        log(f"manifest {Path(a.manifest).name}: {len(frames)}/{len(names)} present")
    else:
        frames = sorted(GONG.glob("*.jpg"))
    if a.limit:
        frames = frames[:a.limit]
    log(f"{len(frames)} GONG frames")
    if not frames:
        return

    # keep the registry's provenance flags alongside the cache
    reg = pd.read_csv(REG) if REG.exists() else pd.DataFrame()
    flags = {}
    if len(reg):
        flags = {r["name"]: bool(r.get("matches_test_stem", False))
                 for _, r in reg.iterrows()}

    models = []
    for fold in range(5):
        w = ROOT / "outputs" / "runs_semantic" / f"sem_b3_f{fold}_1536_cldice" / "best.pt"
        if not w.exists():
            log(f"  missing {w.name} for fold {fold}")
            continue
        m, imgsz_ck, _ = load_model(str(w))
        models.append(m)
        log(f"  teacher fold {fold} loaded (trained at {imgsz_ck})")
    if not models:
        raise SystemExit("no teacher models found")

    n = len(frames)
    imgs = np.lib.format.open_memmap(OUT / f"img_{S}{a.tag}.npy", mode="w+",
                                     dtype=np.uint8, shape=(n, S, S))
    tgts = np.lib.format.open_memmap(OUT / f"tgt_{S}{a.tag}.npy", mode="w+",
                                     dtype=np.uint8, shape=(n, S, S))
    meta, k, t0 = [], 0, time.time()
    for f in frames:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape != (IMG_H, IMG_W):
            continue
        p = teacher_prob(models, img, S)
        imgs[k] = cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA)
        tgts[k] = np.clip(p * 255.0, 0, 255).astype(np.uint8)
        meta.append(dict(idx=k, name=f.name, stem=f.stem, site=f.stem[14],
                         year=f.stem[:4], mean_prob=float(p.mean()),
                         frac_over_50=float((p >= 0.5).mean()),
                         is_test_frame=flags.get(f.name, False)))
        k += 1
        if k % 250 == 0:
            log(f"  {k}/{n} ({time.time()-t0:.0f}s)")
    imgs.flush()
    tgts.flush()
    md = pd.DataFrame(meta)
    md.to_parquet(OUT / f"meta_{S}{a.tag}.parquet", index=False)
    with open(OUT / f"index_{S}{a.tag}.json", "w") as fh:
        json.dump(dict(n=k, crop=S, teacher="sem_b3_f0-4_1536_cldice",
                       soft_target=True), fh, indent=1)
    log(f"\nwrote {k} pseudo-labelled frames at {S}px ({time.time()-t0:.0f}s)")
    log(f"  teacher activity: mean prob {md['mean_prob'].mean():.4f}, "
        f"frac>0.5 {md['frac_over_50'].mean():.5f}")
    log(f"  frames that are also competition TEST frames: "
        f"{int(md['is_test_frame'].sum())} (transductive, declarable)")
    log(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
