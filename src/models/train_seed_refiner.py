"""Segment a filament from a BOX, in a scale-normalised crop.

Our earlier crop refiner was handed the pipeline's mask and asked to correct it;
it learned to copy the input and topped out at +0.0038 before overfitting. This
follows the community notebook in docs/ (yolo11s detect -> crop -> U-Net, local
PQ ~0.409) instead, whose seed is a FILLED BOX. The network is given
localisation and nothing else, so producing the mask is the whole task and there
is no input mask to imitate.

The second difference is scale. A fixed native-resolution window shows a 70 px
filament and a 400 px one at wildly different apparent sizes; here every crop is
1.5x the box and resized to 512, so filaments fill the frame comparably.

Channels are image, CLAHE(2.0, 8x8) and the seed, the notebook's exact
combination — and CLAHE is where the histogram-equalisation idea finally earns
its place, as a channel beside the raw image rather than a replacement for it.

Training seeds are jittered boxes (shift, scale, pad) so the model never sees a
perfectly centred box, which is what it gets from a detector at inference.
Validation reports both seed types: the GT box (the notebook's number) and the
DETECTOR box, which is the one that matters for the pipeline.

Fold 0 is held out.

Usage: python src/models/train_seed_refiner.py --epochs 16 --batch 8
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

CROPS = ROOT / "data" / "seed_crops"
OUT = ROOT / "outputs" / "runs_seed"
PAD_RATIO = 0.40


def log(*a):
    print(*a, flush=True)


def seed_from_box(bx, by, bw, bh, S, pad=PAD_RATIO):
    """Filled box grown by `pad`, as the notebook's seed representation."""
    m = np.zeros((S, S), np.float32)
    px, py = bw * pad / 2, bh * pad / 2
    x0 = int(max(0, round(bx - px)))
    y0 = int(max(0, round(by - py)))
    x1 = int(min(S, round(bx + bw + px)))
    y1 = int(min(S, round(by + bh + py)))
    if x1 > x0 and y1 > y0:
        m[y0:y1, x0:x1] = 1.0
    return m


def jitter_box(bx, by, bw, bh, S, rng):
    """A detector's box is never the GT box; train on that distribution."""
    sc = float(np.exp(rng.normal(0, 0.08)))
    nw, nh = bw * sc, bh * sc
    cx, cy = bx + bw / 2 + rng.normal(0, 0.05 * bw), by + bh / 2 + rng.normal(0, 0.05 * bh)
    return cx - nw / 2, cy - nh / 2, nw, nh


class SeedSet(Dataset):
    """(image, CLAHE, seed) -> GT mask, seeds regenerated per epoch.

    Two sources. `gt` crops are centred on GT boxes and the seed is jittered to
    imitate a detector. `detbox` crops are centred on REAL out-of-fold detector
    boxes with the GT they matched as the target, so no jitter is needed or
    wanted — train and apply then draw from the same distribution, which the
    jittered version did not (measured: the detector's log-scale error has p90
    0.255 against the trained sigma 0.08's ~0.13).

    Memmaps open per worker and never enter the pickle: shipping one through a
    Windows spawn materialises the whole 2 GB array into the pipe.
    """

    def __init__(self, meta, S, train, root=CROPS, jitter=True):
        self.rows = meta[["idx", "bx", "by", "bw", "bh"]].to_numpy(np.float64)
        self.S = S
        self.train = train
        self.root = root
        self.jitter = jitter
        self.img = self.tgt = None
        self._clahe = None

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        s = self.__dict__.copy()
        s["img"] = s["tgt"] = s["_clahe"] = None
        return s

    def _open(self):
        if self.img is None:
            self.img = np.load(self.root / f"img_{self.S}.npy", mmap_mode="r")
            self.tgt = np.load(self.root / f"tgt_{self.S}.npy", mmap_mode="r")
            self._clahe = cv2.createCLAHE(2.0, (8, 8))

    def __getitem__(self, i):
        self._open()
        idx, bx, by, bw, bh = self.rows[i]
        j = int(idx)
        img = np.asarray(self.img[j])
        tgt = np.asarray(self.tgt[j])
        rng = np.random.default_rng((j * 2654435761 + int(self.train) * 977) % (2 ** 32))
        if self.train and self.jitter:
            bx, by, bw, bh = jitter_box(bx, by, bw, bh, self.S, rng)
            pad = float(rng.uniform(0.2, 0.6))
        else:
            pad = PAD_RATIO
        seed = seed_from_box(bx, by, bw, bh, self.S, pad)
        cl = self._clahe.apply(img)
        if self.train:
            if rng.random() < 0.5:
                img, cl, tgt, seed = (a[:, ::-1] for a in (img, cl, tgt, seed))
            if rng.random() < 0.5:
                img, cl, tgt, seed = (a[::-1] for a in (img, cl, tgt, seed))
            img, cl, tgt, seed = (np.ascontiguousarray(a) for a in (img, cl, tgt, seed))
        x = np.stack([img.astype(np.float32) / 255.0,
                      cl.astype(np.float32) / 255.0,
                      seed.astype(np.float32)])
        return torch.from_numpy(x), torch.from_numpy(tgt.astype(np.float32))[None]


def dice_bce(logits, target):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + 1.0
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1.0
    return bce + (1 - num / den).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--encoder", default="timm-efficientnet-b3")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--name", default="seed_b3_512")
    ap.add_argument("--context", type=float, default=1.5,
                    help="crop context the data was built with; selects the crop "
                         "directory and is stored in the checkpoint so inference "
                         "cannot silently use a different one")
    ap.add_argument("--source", default="gt", choices=["gt", "detbox"],
                    help="gt = GT boxes with synthetic jitter; detbox = real "
                         "out-of-fold detector boxes (no jitter needed)")
    a = ap.parse_args()
    if a.source == "gt":
        root = CROPS if a.context == 1.5 else Path(str(CROPS) + f"_c{a.context:g}")
    else:
        root = ROOT / "data" / "detbox_crops"
    jitter = a.source == "gt"
    dest = OUT / a.name
    dest.mkdir(parents=True, exist_ok=True)

    meta = pd.read_parquet(root / f"meta_{a.crop}.parquet")
    if "fold" not in meta.columns:
        folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
        meta["fold"] = meta["file_name"].map(dict(zip(folds["file_name"], folds["fold"])))
    tr, va = meta[meta["fold"] != a.fold], meta[meta["fold"] == a.fold]
    log(f"source {a.source}: train {len(tr)} crops, val {len(va)} "
        f"(fold {a.fold} held out)")

    dl_tr = DataLoader(SeedSet(tr, a.crop, True, root, jitter), batch_size=a.batch,
                       shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
                       persistent_workers=True)
    dl_va = DataLoader(SeedSet(va, a.crop, False, root, jitter), batch_size=a.batch,
                       shuffle=False, num_workers=2, pin_memory=True)

    import segmentation_models_pytorch as smp
    torch.backends.cudnn.benchmark = True
    model = smp.Unet(encoder_name=a.encoder, encoder_weights="imagenet",
                     in_channels=3, classes=1,
                     decoder_attention_type="scse").cuda()
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

    rec(f"{a.name}: {a.encoder} crop {a.crop} batch {a.batch} lr {a.lr} "
        f"epochs {a.epochs} fold {a.fold}")
    best, hist = -1.0, []
    for ep in range(a.epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for x, y in dl_tr:
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
        inter = union = 0.0
        ious, has_tgt, pred_empty = [], [], []
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for x, y in dl_va:
                x, y = x.cuda(), y.cuda()
                p = (torch.sigmoid(model(x)) > 0.5).float()
                inter += (p * y).sum().item()
                union += p.sum().item() + y.sum().item()
                i = (p * y).sum(dim=(1, 2, 3))
                u = ((p + y) > 0).float().sum(dim=(1, 2, 3)).clamp(min=1)
                ious.append((i / u).cpu().numpy())
                has_tgt.append((y.sum(dim=(1, 2, 3)) > 0).cpu().numpy())
                pred_empty.append((p.sum(dim=(1, 2, 3)) < 64).cpu().numpy())
        # With the detbox source a third of the crops have an EMPTY target
        # (detector candidates that match no GT). Those score IoU 0 whatever the
        # model does, so a plain mean is not comparable across sources — report
        # the mean over crops that HAVE a target, and separately how often an
        # empty one is correctly left empty.
        arr = np.concatenate(ious)
        has = np.concatenate(has_tgt)
        dice = 2 * inter / max(union, 1)
        miou = float(arr[has].mean()) if has.any() else 0.0
        over = float((arr[has] > 0.5).mean()) if has.any() else 0.0
        empty_ok = float(np.concatenate(pred_empty)[~has].mean()) if (~has).any() else float("nan")
        hist.append(dict(epoch=ep, loss=tot / max(nb, 1), dice=dice,
                         mean_iou=miou, frac_over_50=over, empty_ok=empty_ok))
        rec(f"epoch {ep}: loss {tot/max(nb,1):.4f}  val Dice {dice:.4f}  "
            f"mean IoU {miou:.4f}  frac IoU>0.5 {over:.4f}  "
            f"empty-left-empty {empty_ok:.4f}  ({time.time()-t0:.0f}s)  "
            f"peakVRAM {torch.cuda.max_memory_allocated()/2**30:.1f}G")
        with open(dest / "history.json", "w") as fh:
            json.dump(hist, fh, indent=1)
        if miou > best:
            best = miou
            torch.save({"model": model.state_dict(), "encoder": a.encoder,
                        "crop": a.crop, "fold": a.fold, "mean_iou": miou,
                        "pad_ratio": PAD_RATIO, "context": a.context},
                       dest / "best.pt")
    rec(f"done. best val mean IoU {best:.4f} "
        f"(fused-pool SQ reference 0.6666) -> {dest/'best.pt'}")
    logf.close()


if __name__ == "__main__":
    main()
