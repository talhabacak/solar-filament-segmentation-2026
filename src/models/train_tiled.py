"""Native-resolution tiled semantic lane -- the docs/17 pivot.

Why this exists. Our oracle is local 0.5078, about LB 0.46: that is what we would
score picking PERFECTLY from the candidate pool we generate. Four teams are now
above it, so the pool itself is the ceiling, and docs/15 already located the
missing 0.19 PQ in mask quality. Every mask we produce has passed through a
2048 -> 768/1536 downsample. This lane does not downsample.

What is deliberately NOT carried over. clDice, the all-copies target and the 5%
mask shrink were all calibrated at 768 and are adopted there; none of them is
assumed here. At 768 a filament is a few pixels wide and continuity is a real
problem, which is why clDice paid; at native resolution it may not be. Each gets
re-measured on this lane rather than inherited.

Sampling. The union of annotators covers 0.302% of a frame, so a uniformly placed
1056px window is empty about as often as not. FG_PROB of the crops are anchored
on a real filament pixel; the rest are uniform, so the model still sees the
negative distribution it has to reject.

Augmentation is applied as ONE affine that reads straight from the full 2048
frame into the tile. Cropping first and warping after would need a 1.33x larger
source window and would reflect or pad at its edge -- inventing chromosphere the
model would learn to segment. Composing the two means every output pixel comes
from a real source pixel or from outside the frame, which is black sky and
correct.

Scale and translation ARE used here, unlike the whole-image lane where they were
excluded because the disk sits at a fixed centre. A tile is already an arbitrary
window into the frame, so there is no global geometry left for them to break.

Usage:
  python src/models/train_tiled.py --fold 0 --epochs 24 --batch 2
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

import tiling
from tiling import FULL, CROP, OFFSET, TILE

CACHE = ROOT / "data" / "tile_cache"
OUT = ROOT / "outputs" / "runs_tiled"


def log(*a):
    print(*a, flush=True)


class TiledEntries(Dataset):
    """One sample = one (annotator entry, window) pair.

    Memmaps open per worker and never enter the pickle: shipping one through a
    Windows spawn materialises the whole 3 GB array into the pipe.
    """

    def __init__(self, rows, fg, train=True, crops_per=4, fg_prob=0.7,
                 rotate=25.0, scale=(0.9, 1.1), translate=0.05):
        self.rows = list(rows)          # (entry_idx, img_idx, entry_id)
        self.fg = fg
        self.train = train
        self.crops_per = crops_per if train else 1
        self.fg_prob = fg_prob
        self.rotate, self.scale, self.translate = rotate, scale, translate
        self.imgs = self.msks = None
        self._rng = None

    def __len__(self):
        return len(self.rows) * self.crops_per

    def __getstate__(self):
        s = self.__dict__.copy()
        s["imgs"] = s["msks"] = s["_rng"] = None
        return s

    def _open(self):
        if self.imgs is None:
            self.imgs = np.load(CACHE / f"images_{FULL}.npy", mmap_mode="r")
            self.msks = np.load(CACHE / f"masks_{FULL}.npy", mmap_mode="r")
        if self._rng is None:
            self._rng = np.random.default_rng(np.random.randint(2 ** 31))

    def _centre(self, eid, rng):
        """Where to put the window's centre, in full-frame coordinates."""
        lo, hi = OFFSET + TILE // 2, OFFSET + CROP - TILE // 2
        pts = self.fg.get(eid)
        if self.train and pts is not None and pts.shape[1] > 0 \
                and rng.random() < self.fg_prob:
            k = rng.integers(pts.shape[1])
            cy, cx = float(pts[0, k]), float(pts[1, k])
            # jitter so the filament is not always dead centre, which would let
            # the model learn "the answer is in the middle"
            j = TILE * 0.35
            cy += rng.uniform(-j, j)
            cx += rng.uniform(-j, j)
        else:
            cy, cx = rng.uniform(lo, hi), rng.uniform(lo, hi)
        return float(np.clip(cy, lo, hi)), float(np.clip(cx, lo, hi))

    def __getitem__(self, i):
        self._open()
        rng = self._rng
        eidx, iidx, eid = self.rows[i // self.crops_per]
        img = self.imgs[iidx]
        msk = np.unpackbits(np.asarray(self.msks[eidx]), axis=-1)[:, :FULL]

        if self.train:
            cy, cx = self._centre(eid, rng)
            ang = rng.uniform(-self.rotate, self.rotate)
            sc = rng.uniform(*self.scale)
            cy += rng.uniform(-self.translate, self.translate) * TILE
            cx += rng.uniform(-self.translate, self.translate) * TILE
            # One affine: rotate+scale about (cx, cy) in the SOURCE, then shift
            # that point to the tile centre. Border pixels fall outside the frame
            # and become 0, which is the black sky beyond the disk.
            M = cv2.getRotationMatrix2D((cx, cy), ang, sc)
            M[0, 2] += TILE / 2.0 - cx
            M[1, 2] += TILE / 2.0 - cy
            tile = cv2.warpAffine(img, M, (TILE, TILE), flags=cv2.INTER_LINEAR,
                                  borderValue=0)
            tmsk = cv2.warpAffine(msk, M, (TILE, TILE), flags=cv2.INTER_NEAREST,
                                  borderValue=0)
            if rng.random() < 0.5:
                tile, tmsk = tile[:, ::-1], tmsk[:, ::-1]
            k = int(rng.integers(4))            # D4: the disk has no up
            if k:
                tile, tmsk = np.rot90(tile, k), np.rot90(tmsk, k)
            g = rng.uniform(0.85, 1.2)
            tile = np.clip(((tile / 255.0) ** g) * 255.0, 0, 255).astype(np.uint8)
        else:
            r, c = OFFSET, OFFSET
            tile = np.asarray(img[r:r + TILE, c:c + TILE])
            tmsk = msk[r:r + TILE, c:c + TILE]

        x = np.ascontiguousarray(tile).astype(np.float32) / 255.0
        y = np.ascontiguousarray(tmsk).astype(np.float32)
        return (torch.from_numpy(np.stack([x, x, x])),
                torch.from_numpy(y).unsqueeze(0))


def dice_bce(logits, target, bce_w=0.45, dice_w=0.55, pos_weight=None, eps=1e-6):
    """Weighted BCE + soft Dice over the whole tile.

    Kept deliberately plain. The metric-shaped losses we adopted at 768 (clDice)
    are re-measured on this lane as an ablation rather than assumed, because the
    reason clDice paid -- thin structures losing continuity under downsampling --
    may not exist at native resolution.
    """
    pw = None if pos_weight is None else torch.tensor(pos_weight, device=logits.device)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pw)
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return bce_w * bce + dice_w * (1 - num / den).mean()


def build(encoder, grad_ckpt=False):
    import segmentation_models_pytorch as smp
    m = smp.Unet(encoder_name=encoder, encoder_weights="imagenet", in_channels=3,
                 classes=1, decoder_attention_type="scse")
    if grad_ckpt and hasattr(m.encoder, "set_grad_checkpointing"):
        m.encoder.set_grad_checkpointing(True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--crops-per", type=int, default=4)
    ap.add_argument("--fg-prob", type=float, default=0.7)
    ap.add_argument("--encoder", default="timm-efficientnet-b3")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--wd", type=float, default=2e-2)
    ap.add_argument("--warmup-epochs", type=float, default=2.0)
    ap.add_argument("--pos-weight", type=float, default=None)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--val-every", type=int, default=1)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    name = a.name or f"tiled_{a.encoder.split('-')[-1]}_f{a.fold}_{TILE}"
    dest = OUT / name
    dest.mkdir(parents=True, exist_ok=True)

    idx = json.loads((CACHE / f"index_{FULL}.json").read_text())
    fgz = np.load(CACHE / f"fg_{FULL}.npz")
    fg = {k: fgz[k] for k in fgz.files}
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"], folds["fold"]))
    rows_tr, rows_va = [], []
    for j, (eid, ifile) in enumerate(zip(idx["entry_ids"], idx["entry_file"])):
        fn = idx["files"][ifile]
        (rows_va if fold_of[fn] == a.fold else rows_tr).append((j, ifile, eid))
    log(f"{name}: train {len(rows_tr)} entries, val {len(rows_va)} (fold {a.fold})")
    log(f"  geometry crop {CROP} tile {TILE} overlap {tiling.OVERLAP} "
        f"offset {OFFSET} -> {len(tiling.tile_origins())} tiles/image")

    ds_tr = TiledEntries(rows_tr, fg, True, a.crops_per, a.fg_prob)
    dl_tr = DataLoader(ds_tr, batch_size=a.batch, shuffle=True, num_workers=3,
                       pin_memory=True, drop_last=True, persistent_workers=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = build(a.encoder, a.grad_ckpt).cuda()
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * (len(dl_tr) // a.accum)
    warm = int(a.warmup_epochs * (len(dl_tr) // a.accum))

    def lr_at(s):
        if s < warm:
            return a.lr * (s + 1) / max(warm, 1)
        t = (s - warm) / max(steps - warm, 1)
        return a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + np.cos(np.pi * t))

    scaler = torch.amp.GradScaler("cuda")
    logf = open(dest / "train.log", "a", buffering=1)

    def rec(m):
        log(m)
        logf.write(m + "\n")

    rec(f"{name}: {a.encoder} {n_par:.1f}M tile={TILE} batch={a.batch}x{a.accum} "
        f"lr={a.lr} wd={a.wd} epochs={a.epochs} crops_per={a.crops_per} "
        f"fg_prob={a.fg_prob} pos_weight={a.pos_weight} "
        f"steps/epoch={len(dl_tr)//a.accum}")

    # validation images (unique files in the fold), scored as pixel-Dice against
    # the annotator union -- the cheap signal. PQ is the real metric and is run
    # by eval_tiled.py on the best checkpoint, since it needs the full mosaic.
    imgs = np.load(CACHE / f"images_{FULL}.npy", mmap_mode="r")
    msks = np.load(CACHE / f"masks_{FULL}.npy", mmap_mode="r")
    va_files = sorted({ifile for _, ifile, _ in rows_va})
    va_union = {}
    for ifile in va_files:
        u = None
        for j, jf, _ in rows_va:
            if jf != ifile:
                continue
            m = np.unpackbits(np.asarray(msks[j]), axis=-1)[:, :FULL] > 0
            u = m if u is None else (u | m)
        va_union[ifile] = u

    best, hist = -1.0, []
    gstep = 0
    for ep in range(a.epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(dl_tr):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.amp.autocast("cuda"):
                loss = dice_bce(model(x), y, pos_weight=a.pos_weight)
            scaler.scale(loss / a.accum).backward()
            if (i + 1) % a.accum == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(gstep)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                gstep += 1
            tot += loss.item()
            nb += 1
        tr_loss = tot / max(nb, 1)

        dice = float("nan")
        if (ep + 1) % a.val_every == 0 or ep == a.epochs - 1:
            model.eval()
            inter = union = 0
            for ifile in va_files:
                p = tiling.mosaic_predict(model, np.asarray(imgs[ifile]), tta=1)
                b, t = p > 0.5, va_union[ifile]
                inter += int((b & t).sum())
                union += int(b.sum() + t.sum())
            dice = 2 * inter / max(union, 1)
        hist.append(dict(epoch=ep, train_loss=tr_loss, val_dice=dice,
                         lr=opt.param_groups[0]["lr"],
                         secs=round(time.time() - t0, 1)))
        rec(f"epoch {ep}: loss {tr_loss:.4f}  val pixel-Dice {dice:.4f}  "
            f"({time.time()-t0:.0f}s)  peakVRAM "
            f"{torch.cuda.max_memory_allocated()/2**30:.1f}G")
        json.dump(hist, open(dest / "history.json", "w"), indent=1)
        if dice > best:
            best = dice
            torch.save({"model": model.state_dict(), "encoder": a.encoder,
                        "arch": "unet", "in_ch": 3, "tile": TILE, "fold": a.fold,
                        "val_dice": dice, "lane": "tiled", "channels": "rep"},
                       dest / "best.pt")
    rec(f"done. best val pixel-Dice {best:.4f} -> {dest/'best.pt'}")
    rec("next: python src/models/eval_tiled.py --weights "
        f"{dest/'best.pt'} --fold {a.fold}   (the real metric, with TTA)")
    logf.close()


if __name__ == "__main__":
    main()
