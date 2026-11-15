"""Transductive self-training: heavy-60 recipe on real GT + soft test pseudo-labels.

The largest legal lever not yet tried. The test IMAGES are competition data; the
soft targets are our own ensemble's outputs (recipe3 mean probability maps), so
no label information enters that we do not already have. Declared in the final
writeup as transductive self-training.

Why it can work when every local axis is saturated: all our models were fitted
to the 707 train images only. Training on the test images with soft targets
adapts BatchNorm statistics and features to the exact frames being scored, and
consolidates the ensemble's knowledge into one student (classic distillation
gain). The catch, stated honestly: the benefit is INVISIBLE to local validation
by construction -- fold val can only show whether the pseudo mixing HURT the
train-distribution fit. Adoption therefore needs an LB probe, and the change is
sized to clear the board's 0.005 resolution or it is not submitted.

Mechanics: GT samples take the heavy-60 path unchanged (heavy augmentation,
dice_bce + clDice); pseudo samples take light augmentation (flips only, like the
GONG distillation) and a soft BCE+Dice loss at --pseudo-w weight. One pseudo
batch is interleaved every ceil(n_gt/n_pseudo) GT batches so the 180 test frames
are seen about once per epoch without dominating.

Usage:
  python src/models/train_mixed.py --fold 0            # sanity: val must hold ~0.70
  python src/models/train_mixed.py --all-data          # submission member
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

from finetune_aug import AugCached
from train_semantic import dice_bce_loss, cldice_loss, log
from train_semantic_arch import build_model

PSEUDO = ROOT / "data" / "gong_pseudo"
OUT = ROOT / "outputs" / "runs_aug"
S = 768


class TestPseudo(Dataset):
    """The 180 test frames with the ensemble's soft probability targets."""

    def __init__(self):
        self.img = self.tgt = None
        self.n = len(pd.read_parquet(PSEUDO / f"meta_{S}_test.parquet"))

    def __len__(self):
        return self.n

    def __getstate__(self):
        s = self.__dict__.copy()
        s["img"] = s["tgt"] = None
        return s

    def _open(self):
        if self.img is None:
            self.img = np.load(PSEUDO / f"img_{S}_test.npy", mmap_mode="r")
            self.tgt = np.load(PSEUDO / f"tgt_{S}_test.npy", mmap_mode="r")

    def __getitem__(self, i):
        self._open()
        img = np.asarray(self.img[i]).astype(np.float32) / 255.0
        tgt = np.asarray(self.tgt[i]).astype(np.float32) / 255.0
        rng = np.random.default_rng()
        if rng.random() < 0.5:
            img, tgt = img[:, ::-1], tgt[:, ::-1]
        if rng.random() < 0.5:
            img, tgt = img[::-1], tgt[::-1]
        img, tgt = np.ascontiguousarray(img), np.ascontiguousarray(tgt)
        return (torch.from_numpy(np.stack([img, img, img])),
                torch.from_numpy(tgt)[None])


def soft_loss(logits, target):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + 1.0
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1.0
    return bce + (1 - num / den).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--all-data", action="store_true",
                    help="train on all 707 images (no honest val; fixed epochs)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--pct-start", type=float, default=0.1)
    ap.add_argument("--cldice", type=float, default=0.3)
    ap.add_argument("--cldice-warmup", type=int, default=5)
    ap.add_argument("--pseudo-w", type=float, default=0.5)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    name = a.name or ("mixed_all707" if a.all_data else f"mixed_f{a.fold}_768")
    dest = OUT / name
    dest.mkdir(parents=True, exist_ok=True)

    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    entries = gt[["entry_id", "file_name"]].drop_duplicates()
    entries["fold"] = entries["file_name"].map(dict(zip(folds["file_name"],
                                                        folds["fold"])))
    if a.all_data:
        tr = list(entries[["entry_id", "file_name"]].itertuples(index=False, name=None))
        va = []
    else:
        tr = list(entries.loc[entries["fold"] != a.fold, ["entry_id", "file_name"]]
                  .itertuples(index=False, name=None))
        va = list(entries.loc[entries["fold"] == a.fold, ["entry_id", "file_name"]]
                  .itertuples(index=False, name=None))

    fgz = None      # AugCached needs no fg anchors; it augments full frames
    ds_tr = AugCached(tr, S, train=True, aug="heavy")
    dl_tr = DataLoader(ds_tr, batch_size=a.batch, shuffle=True, num_workers=3,
                       pin_memory=True, drop_last=True, persistent_workers=True)
    ds_ps = TestPseudo()
    dl_ps = DataLoader(ds_ps, batch_size=a.batch, shuffle=True, num_workers=1,
                       pin_memory=True, drop_last=True, persistent_workers=True)
    dl_va = (DataLoader(AugCached(va, S, train=False, aug="none"),
                        batch_size=a.batch, shuffle=False, num_workers=2,
                        pin_memory=True) if va else None)
    every = max(1, len(dl_tr) // max(len(dl_ps), 1))

    torch.backends.cudnn.benchmark = True
    model = build_model("unet", "timm-efficientnet-b3", 3, S).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * len(dl_tr), pct_start=a.pct_start)
    scaler = torch.amp.GradScaler("cuda")
    logf = open(dest / "train.log", "a", buffering=1)

    def rec(m):
        log(m)
        logf.write(m + "\n")

    rec(f"{name}: b3 heavy-60 + test-pseudo(w={a.pseudo_w}, every {every} steps) "
        f"train={len(tr)} pseudo={len(ds_ps)} val={len(va)} epochs={a.epochs}")

    best, hist = -1.0, []
    for ep in range(a.epochs):
        model.train()
        it_ps = iter(dl_ps)
        t0, tot, nb = time.time(), 0.0, 0
        for i, (x, y, w) in enumerate(dl_tr):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss = dice_bce_loss(logits, y)
                if a.cldice > 0 and ep >= a.cldice_warmup:
                    loss = loss + a.cldice * cldice_loss(logits, y)
            if (i + 1) % every == 0:
                try:
                    xp, yp = next(it_ps)
                except StopIteration:
                    it_ps = iter(dl_ps)
                    xp, yp = next(it_ps)
                with torch.amp.autocast("cuda"):
                    loss = loss + a.pseudo_w * soft_loss(
                        model(xp.cuda(non_blocking=True)),
                        yp.cuda(non_blocking=True))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            tot += loss.item()
            nb += 1

        dice = float("nan")
        if dl_va is not None:
            model.eval()
            inter = union = 0
            with torch.no_grad(), torch.amp.autocast("cuda"):
                for x, y, _ in dl_va:
                    p = torch.sigmoid(model(x.cuda())) > 0.5
                    yv = y.cuda() > 0.5
                    inter += (p & yv).sum().item()
                    union += p.sum().item() + yv.sum().item()
            dice = 2 * inter / max(union, 1)
        hist.append(dict(epoch=ep, train_loss=tot / max(nb, 1), val_dice=dice,
                         secs=round(time.time() - t0, 1)))
        rec(f"epoch {ep}: loss {tot/max(nb,1):.4f}  val pixel-Dice {dice:.4f}  "
            f"({time.time()-t0:.0f}s)")
        json.dump(hist, open(dest / "history.json", "w"), indent=1)
        save = (dl_va is None) or (dice > best)
        if dl_va is not None and dice > best:
            best = dice
        if save:
            torch.save({"model": model.state_dict(),
                        "encoder": "timm-efficientnet-b3", "arch": "unet",
                        "in_ch": 3, "imgsz": S, "val_dice": dice,
                        "channels": "rep", "lane": "mixed"}, dest / "best.pt")
    rec(f"done. {'best val ' + format(best, '.4f') if dl_va is not None else 'all-data run'} "
        f"-> {dest/'best.pt'}")
    logf.close()


if __name__ == "__main__":
    main()
