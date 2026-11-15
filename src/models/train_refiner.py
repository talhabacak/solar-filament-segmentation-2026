"""Refine a candidate mask against the consensus, at native resolution.

Target of the exercise, measured: under oracle selection the pool is PQ 0.5078 at
SQ 0.6666, and of the 0.12 SQ gap to the majority-consensus mask (0.7903), the
global width term is exhausted (+0.0000), per-instance width is worth +0.0305,
and +0.0906 is shape. This model goes after the shape term.

Why crops. The semantic lane predicts at 1536 and the result is upsampled to
2048, so a filament ~11 px wide has its boundary decided by interpolation. Here
each candidate gets a 320x320 window at FULL resolution (93% of candidates fit
inside one; median bbox side is 73 px).

Input is 3 channels — image, the current mask, and a clipped signed distance
field of that mask, so the network is told where the current boundary is rather
than having to infer it from a binary step.

Training data is two sources, because 4161 matched candidates alone is too few:
  gt    every GT instance, with its input mask SYNTHESISED by perturbing the
        target (offset, affine jitter, boundary noise, fragment dropout)
  cand  real pipeline candidates with the majority mask of the GT they matched,
        i.e. the true error distribution

Fold 0 is held out on both sources, and the reported number is the IoU of the
refined mask against the same target the scorer would use, next to the IoU the
unrefined candidate already had — so the gain is the SQ gain, not a proxy.

Usage:
  python src/models/train_refiner.py --epochs 20 --batch 16
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
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

CROPS = ROOT / "data" / "refine_crops"
OUT = ROOT / "outputs" / "runs_refine"
CROP = 320
SDF_CLIP = 20.0


def log(*a):
    print(*a, flush=True)


def sdf(mask):
    """Signed distance to the mask boundary, negative inside, clipped."""
    m = mask.astype(np.uint8)
    if m.max() == 0:
        return np.full((CROP, CROP), SDF_CLIP, np.float32)
    din = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    dout = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
    return np.clip(dout - din, -SDF_CLIP, SDF_CLIP).astype(np.float32)


def perturb(mask, rng):
    """Turn a GT mask into a plausible imperfect prediction.

    Ranges follow the measured error: the per-instance oracle offset runs from
    about -2 to +1 px, and the residual is shape, so this applies an offset AND
    a small affine plus boundary noise rather than offset alone.
    """
    m = mask.astype(np.uint8)
    if m.max() == 0:
        return m
    k = float(rng.normal(-0.5, 1.8))
    d = sdf(m)
    out = (d <= k).astype(np.uint8)

    if rng.random() < 0.7:                       # small rigid/scale error
        ang = rng.normal(0, 4.0)
        sc = float(np.exp(rng.normal(0, 0.05)))
        tx, ty = rng.normal(0, 2.0, 2)
        M = cv2.getRotationMatrix2D((CROP / 2, CROP / 2), ang, sc)
        M[:, 2] += (tx, ty)
        out = cv2.warpAffine(out, M, (CROP, CROP), flags=cv2.INTER_NEAREST)

    if rng.random() < 0.5:                       # ragged boundary
        noise = rng.normal(0, 1.0, (CROP, CROP)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), 4.0) * 12.0
        out = ((sdf(out) + noise) <= 0).astype(np.uint8)

    if rng.random() < 0.15 and out.sum() > 200:  # a piece goes missing
        ys, xs = np.nonzero(out)
        i = rng.integers(len(ys))
        cv2.circle(out, (int(xs[i]), int(ys[i])), int(rng.integers(10, 30)), 0, -1)

    return out if out.sum() >= 20 else m


class RefineSet(Dataset):
    """Serves (image, input mask, sdf) -> target from both crop sources.

    Memmaps are opened per worker, never pickled: shipping one through a Windows
    spawn materialises the whole array into the pipe.
    """

    def __init__(self, rows, train):
        self.rows = rows                 # (source, idx) pairs
        self.train = train
        self.gt_img = self.gt_msk = None
        self.c_img = self.c_in = self.c_tgt = None

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        s = self.__dict__.copy()
        for k in ("gt_img", "gt_msk", "c_img", "c_in", "c_tgt"):
            s[k] = None
        return s

    def _open(self):
        if self.gt_img is None:
            self.gt_img = np.load(CROPS / "gt_img.npy", mmap_mode="r")
            self.gt_msk = np.load(CROPS / "gt_msk.npy", mmap_mode="r")
            self.c_img = np.load(CROPS / "cand_img.npy", mmap_mode="r")
            self.c_in = np.load(CROPS / "cand_in.npy", mmap_mode="r")
            self.c_tgt = np.load(CROPS / "cand_tgt.npy", mmap_mode="r")

    def __getitem__(self, i):
        self._open()
        src, idx = self.rows[i]
        rng = np.random.default_rng((idx * 2654435761 + int(self.train)) % (2 ** 32))
        if src == 0:
            img = np.asarray(self.gt_img[idx])
            tgt = np.asarray(self.gt_msk[idx])
            inm = perturb(tgt, rng) if self.train else perturb(tgt, np.random.default_rng(idx))
        else:
            img = np.asarray(self.c_img[idx])
            inm = np.asarray(self.c_in[idx])
            tgt = np.asarray(self.c_tgt[idx])

        if self.train:                                    # flips only: the crop
            if rng.random() < 0.5:                        # is already centred
                img, inm, tgt = img[:, ::-1], inm[:, ::-1], tgt[:, ::-1]
            if rng.random() < 0.5:
                img, inm, tgt = img[::-1], inm[::-1], tgt[::-1]
            img, inm, tgt = (np.ascontiguousarray(a) for a in (img, inm, tgt))

        x = np.stack([img.astype(np.float32) / 255.0,
                      inm.astype(np.float32),
                      sdf(inm) / SDF_CLIP])
        return (torch.from_numpy(x),
                torch.from_numpy(tgt.astype(np.float32))[None],
                torch.from_numpy(inm.astype(np.float32))[None])


def dice_bce(logits, target):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + 1.0
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1.0
    return bce + (1 - num / den).mean()


def iou_batch(a, b):
    inter = (a * b).sum(dim=(1, 2, 3))
    union = ((a + b) > 0).float().sum(dim=(1, 2, 3))
    return (inter / union.clamp(min=1)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--encoder", default="timm-efficientnet-b0")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--name", default="refine_b0_320")
    a = ap.parse_args()
    dest = OUT / a.name
    dest.mkdir(parents=True, exist_ok=True)

    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"], folds["fold"]))
    gm = pd.read_parquet(CROPS / "gt_meta.parquet")
    cm = pd.read_parquet(CROPS / "cand_meta.parquet")
    gm["fold"] = gm["file_name"].map(fold_of)
    cm = cm[cm["n_gt"] > 0]                      # only matched candidates carry a target

    tr_rows = ([(0, int(i)) for i in gm.loc[gm["fold"] != a.fold, "idx"]] +
               [(1, int(i)) for i in cm.loc[cm["fold"] != a.fold, "idx"]])
    va_rows = [(1, int(i)) for i in cm.loc[cm["fold"] == a.fold, "idx"]]
    log(f"train {len(tr_rows)} crops ({int((gm['fold']!=a.fold).sum())} gt + "
        f"{int((cm['fold']!=a.fold).sum())} cand), val {len(va_rows)} real candidates")

    dl_tr = DataLoader(RefineSet(tr_rows, True), batch_size=a.batch, shuffle=True,
                       num_workers=4, pin_memory=True, drop_last=True,
                       persistent_workers=True)
    dl_va = DataLoader(RefineSet(va_rows, False), batch_size=a.batch, shuffle=False,
                       num_workers=2, pin_memory=True)

    import segmentation_models_pytorch as smp
    torch.backends.cudnn.benchmark = True
    model = smp.Unet(encoder_name=a.encoder, encoder_weights="imagenet",
                     in_channels=3, classes=1, decoder_attention_type="scse").cuda()
    log(f"  {a.encoder}: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr,
                                                total_steps=a.epochs * len(dl_tr),
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")
    logf = open(dest / "train.log", "a", buffering=1)

    def rec(m):
        log(m)
        logf.write(m + "\n")

    rec(f"{a.name}: {a.encoder} crop {CROP} batch {a.batch} lr {a.lr} epochs {a.epochs}")
    best, hist = -1.0, []
    for ep in range(a.epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for x, y, _ in dl_tr:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.amp.autocast("cuda"):
                loss = dice_bce(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            tot += loss.item()
            nb += 1
        model.eval()
        before, after = [], []
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for x, y, inm in dl_va:
                x, y, inm = x.cuda(), y.cuda(), inm.cuda()
                p = (torch.sigmoid(model(x)) > 0.5).float()
                before.append(iou_batch(inm, y))
                after.append(iou_batch(p, y))
        b, f = float(np.concatenate(before).mean()), float(np.concatenate(after).mean())
        hist.append(dict(epoch=ep, loss=tot / max(nb, 1), iou_before=b, iou_after=f))
        rec(f"epoch {ep}: loss {tot/max(nb,1):.4f}  val IoU {b:.4f} -> {f:.4f} "
            f"({f-b:+.4f})  ({time.time()-t0:.0f}s)")
        with open(dest / "history.json", "w") as fh:
            json.dump(hist, fh, indent=1)
        if f > best:
            best = f
            torch.save({"model": model.state_dict(), "encoder": a.encoder,
                        "crop": CROP, "fold": a.fold, "iou_after": f,
                        "iou_before": b}, dest / "best.pt")
    rec(f"done. best val IoU {best:.4f} (unrefined {hist[-1]['iou_before']:.4f}) "
        f"-> {dest/'best.pt'}")
    logf.close()


if __name__ == "__main__":
    main()
