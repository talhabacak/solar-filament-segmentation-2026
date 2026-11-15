"""Stage 1: pretrain the encoder on pseudo-labelled GONG frames.

Two-stage plan (docs/14), Talha's design: learn the DOMAIN here — chromospheric
texture, disk geometry, seeing noise, what a filament looks like in context —
and learn the annotation CONVENTION in stage 2 from the real labels. Because the
convention is relearned last, the pseudo-labels' accuracy is secondary, which is
what makes training on our own teacher's output safe rather than
self-reinforcing.

Distillation, not hard labels: the target is the teacher's probability map and
the loss is soft BCE plus a soft Dice. A threshold would discard exactly the
information that makes a noisy teacher usable — where it is unsure.

The measurement that justifies this: the data-scaling curve has not flattened
(25/50/100% of training images give 0.6739/0.6866/0.6981, about +0.012 per
doubling), while the whole ImageNet-vs-random axis is worth +0.0169. More data
is the live lever.

Also honours Talha's point about bands. ImageNet's stem is 3-channel and smp
folds it into 1 when in_channels=1, which measured worse (0.6973 vs 0.7019);
a domain pretraining has no such handicap, so --in-ch 1 is a first-class option
here rather than an afterthought.

Usage:
  python src/models/pretrain_gong.py --imgsz 768 --epochs 12 --in-ch 1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from train_semantic_arch import build_model, make_schedule, EarlyStop
from train_semantic import cldice_loss

CROPS = ROOT / "data" / "gong_pseudo"
OUT = ROOT / "outputs" / "runs_pretrain"


def log(*a):
    print(*a, flush=True)


class GongPseudo(Dataset):
    """Frame -> teacher probability, across one or more labelling batches.

    The frames arrive in batches (the archive is downloaded in passes), each
    written to its own tagged cache, so a sample is addressed by (tag, row) and
    the batches are concatenated without copying 3 GB of memmap around.

    Memmaps open per worker and never enter the pickle: shipping one through a
    Windows spawn materialises the whole array into the pipe.
    """

    def __init__(self, rows, S, in_ch, train=True):
        self.rows = list(rows)              # (tag, local_idx)
        self.S = S
        self.in_ch = in_ch
        self.train = train
        self.img = self.tgt = None

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        s = self.__dict__.copy()
        s["img"] = s["tgt"] = None
        return s

    def _open(self):
        if self.img is None:
            tags = sorted({t for t, _ in self.rows})
            self.img = {t: np.load(CROPS / f"img_{self.S}{t}.npy", mmap_mode="r")
                        for t in tags}
            self.tgt = {t: np.load(CROPS / f"tgt_{self.S}{t}.npy", mmap_mode="r")
                        for t in tags}

    def __getitem__(self, i):
        self._open()
        tag, j = self.rows[i]
        j = int(j)
        img = np.asarray(self.img[tag][j]).astype(np.float32) / 255.0
        tgt = np.asarray(self.tgt[tag][j]).astype(np.float32) / 255.0
        if self.train:
            rng = np.random.default_rng(j)
            if rng.random() < 0.5:
                img, tgt = img[:, ::-1], tgt[:, ::-1]
            if rng.random() < 0.5:
                img, tgt = img[::-1], tgt[::-1]
            img, tgt = np.ascontiguousarray(img), np.ascontiguousarray(tgt)
        x = img[None] if self.in_ch == 1 else np.stack([img, img, img])
        return torch.from_numpy(x), torch.from_numpy(tgt)[None]


def soft_loss(logits, target, cldice_w=0.0):
    """BCE against a soft target plus a soft Dice, both distillation-friendly.

    clDice is added here for the same reason it is the fine-tune's one adopted
    loss-side ablation (pixel-Dice 0.6919 -> 0.7019, instance PQ 0.4011 ->
    0.4066): it rewards centre-line continuity, which is what keeps thin
    filaments alive as connected components. Talha's instruction was that both
    stages train to the same standard, and the pretraining was the stage missing
    it. The soft-skeleton uses min/max pooling, so it is defined on a
    probability target and does not need a threshold.
    """
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + 1.0
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1.0
    loss = bce + (1 - num / den).mean()
    if cldice_w > 0:
        loss = loss + cldice_w * cldice_loss(logits, target)
    return loss



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=0,
                    help="early stopping patience in epochs; 0 disables")
    ap.add_argument("--cosine", action="store_true",
                    help="cosine decay from --lr to --min-lr instead of OneCycle")
    ap.add_argument("--encoder", default="timm-efficientnet-b3")
    ap.add_argument("--in-ch", type=int, default=3, choices=[1, 3])
    ap.add_argument("--scratch", action="store_true",
                    help="start from random rather than ImageNet")
    ap.add_argument("--cldice", type=float, default=0.3,
                    help="clDice weight — the fine-tune's adopted loss ablation, "
                         "applied here too so both stages match")
    ap.add_argument("--cldice-warmup", type=int, default=5)
    ap.add_argument("--swin-satlas", default=None, choices=["s1-si", "s1-mi"],
                    help="pretrain the Satlas Sentinel-1 Swin-V2-B instead of "
                         "the smp encoder (2-band input, no weight surgery)")
    ap.add_argument("--tags", default="_a",
                    help="comma-separated cache tags to train on together, e.g. "
                         "'_a,_b' once the second download batch is labelled")
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    name = a.name or f"pre_gong_{a.encoder.split('-')[-1]}_{a.imgsz}_{a.in_ch}ch"
    dest = OUT / name
    dest.mkdir(parents=True, exist_ok=True)

    rows, parts = [], []
    for tag in a.tags.split(","):
        md = pd.read_parquet(CROPS / f"meta_{a.imgsz}{tag}.parquet")
        rows += [(tag, int(i)) for i in md["idx"]]
        parts.append((tag, len(md), int(md.get("is_test_frame", pd.Series([0])).sum())))
        log(f"  cache {tag}: {len(md)} frames "
            f"({int(md.get('is_test_frame', pd.Series([0])).sum())} are competition "
            f"test frames — transductive, declarable)")
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(rows))
    n_val = max(16, int(len(rows) * a.val_frac))
    va_rows = [rows[i] for i in perm[:n_val]]
    tr_rows = [rows[i] for i in perm[n_val:]]
    log(f"{name}: {len(tr_rows)} pretrain frames, {len(va_rows)} held out "
        f"(distillation val, not a real metric)")

    dl_tr = DataLoader(GongPseudo(tr_rows, a.imgsz, a.in_ch, True), batch_size=a.batch,
                       shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
                       persistent_workers=True)
    dl_va = DataLoader(GongPseudo(va_rows, a.imgsz, a.in_ch, False), batch_size=a.batch,
                       shuffle=False, num_workers=2, pin_memory=True)

    torch.backends.cudnn.benchmark = True
    model = build_model("unet", a.encoder, a.in_ch, a.imgsz,
                        weights=None if a.scratch else "imagenet",
                        swin_satlas=a.swin_satlas).cuda()
    log(f"  {a.swin_satlas or a.encoder} in_ch={a.in_ch} "
        f"{a.swin_satlas or ('random' if a.scratch else 'imagenet')} init, "
        f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    if a.cosine:
        sched = make_schedule(opt, a, len(dl_tr))
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr,
                                                    total_steps=a.epochs * len(dl_tr),
                                                    pct_start=0.1)
    stopper = EarlyStop(a.patience, mode="min")     # distillation val: lower is better
    scaler = torch.amp.GradScaler("cuda")
    logf = open(dest / "train.log", "a", buffering=1)

    def rec(m):
        log(m)
        logf.write(m + "\n")

    rec(f"{name}: {a.encoder} in_ch={a.in_ch} imgsz={a.imgsz} batch={a.batch} "
        f"lr={a.lr} wd={a.wd} epochs={a.epochs} frames={len(tr_rows)} "
        f"tags={a.tags} cldice={a.cldice} cosine={a.cosine} patience={a.patience}")
    hist, best = [], 1e9
    for ep in range(a.epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for x, y in dl_tr:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.amp.autocast("cuda"):
                loss = soft_loss(model(x), y,
                                 a.cldice if ep >= a.cldice_warmup else 0.0)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            tot += loss.item()
            nb += 1
        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for x, y in dl_va:
                vl += soft_loss(model(x.cuda()), y.cuda(), 0.0).item()
                vn += 1
        v = vl / max(vn, 1)
        hist.append(dict(epoch=ep, train=tot / max(nb, 1), val=v))
        rec(f"epoch {ep}: train {tot/max(nb,1):.4f}  val {v:.4f}  "
            f"({time.time()-t0:.0f}s)  peakVRAM "
            f"{torch.cuda.max_memory_allocated()/2**30:.1f}G")
        with open(dest / "history.json", "w") as fh:
            json.dump(hist, fh, indent=1)
        stop_now = stopper.step(v)
        if v < best:
            best = v
            torch.save({"model": model.state_dict(), "encoder": a.encoder,
                        "in_ch": a.in_ch, "imgsz": a.imgsz, "val": v,
                        "stage": "gong_pretrain"}, dest / "best.pt")
        if stop_now:
            rec(f"early stop: no improvement for {a.patience} epochs "
                f"(best {stopper.best:.4f})")
            break
    rec(f"done. best distillation val {best:.4f} -> {dest/'best.pt'}")
    rec("next: fine-tune on the 707 real images with heavy augmentation "
        "(train_semantic_arch.py --init this checkpoint)")
    logf.close()


if __name__ == "__main__":
    main()
