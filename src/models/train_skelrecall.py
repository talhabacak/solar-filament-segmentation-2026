"""Skeleton Recall Loss on the main mask output, with the HUMAN spine as the skeleton.

Why this exists, and why it is not the experiment that just failed. Our spine
attempt attached a second output channel and hoped the shared encoder would
benefit; it lost 0.0022 and shaped fewer of the unreachable instances than the
baseline. The literature does it differently: Skeleton Recall Loss (Kirchhoff
et al., ECCV 2024) puts the skeleton term on the MAIN prediction, rewarding the
mask itself for covering a precomputed tubed skeleton --

    L = L_generic + w * L_SkelRecall,
    L_SkelRecall = - sum(Y_tube * P) / sum(Y_tube)

-- so the gradient shapes the output we actually score instead of a sibling
head. On five benchmarks it beats clDice (e.g. ToothFairy clDice 92.05 vs 89.67)
at 2% VRAM overhead against clDice's 52%, and it avoids the perforated skeletons
that differentiable soft-skeleton approximations produce.

Two substitutions of our own, each measured rather than assumed:
  * the skeleton is the ANNOTATORS' spine, not an algorithmic skeletonisation of
    the mask. The organizers confirmed this metadata is permitted supervision
    (discussion 736543), and we measured the centreline to be the more stable
    annotation (cross-annotator ribbon IoU 0.708 vs mask IoU 0.629).
  * the tube is dilated at NATIVE resolution before downsampling, because a
    radius-2 kernel applied after downsampling is as wide as the filament
    itself: containment in the label is 94.9% built our way against 68.9% built
    naively.

Arms: --cldice controls the existing centreline loss so the term can be
compared against clDice (replacement) and stacked with it (addition).

Usage:
  python src/models/train_skelrecall.py --fold 0 --skel-w 1.0 --cldice 0.0
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from finetune_aug import AugCached
from train_semantic import dice_bce_loss, cldice_loss, log
from train_semantic_arch import build_model

CACHE = ROOT / "data" / "semantic_cache"
OUT = ROOT / "outputs" / "runs_aug"
S = 768


class AugCachedTube(AugCached):
    """AugCached that also serves the entry's tubed spine, warped identically.

    The tube rides the augmentation through apply_aug's third plane, the same
    slot the weight map uses, so it cannot drift out of register with the image
    and mask -- the failure that would leave the loss rewarding the wrong pixels
    while training proceeded normally.
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.tubes = None

    def __getstate__(self):
        s = super().__getstate__()
        s["tubes"] = None
        return s

    def _open(self):
        super()._open()
        if self.tubes is None:
            self.tubes = np.load(CACHE / f"tubes_{self.size}.npy", mmap_mode="r")

    def __getitem__(self, i):
        self._open()
        if self._rng is None:
            self._rng = np.random.default_rng(np.random.randint(2 ** 31))
        entry_id, fn = self.entries[i]
        img = np.asarray(self.imgs[self.file_row[fn]])
        mask = self._unpack(self.msks, self.entry_row[entry_id]).astype(np.float32)
        tube = np.asarray(self.tubes[self.entry_row[entry_id]]).astype(np.float32)
        if self.train:
            from augment import apply_aug
            img, mask, tube = apply_aug(img, mask, tube, self.size, self.aug,
                                        self._rng, wmap_border=0.0)
        raw = np.ascontiguousarray(img).astype(np.float32) / 255.0
        return (torch.from_numpy(np.stack([raw, raw, raw])).float(),
                torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0),
                torch.from_numpy(np.ascontiguousarray(tube)).float().unsqueeze(0))


def skeleton_recall(logits, tube, eps=1e-6):
    """Soft recall of the tubed skeleton by the predicted foreground.

    Entries whose tube is empty contribute nothing rather than a division by
    zero; averaging over the non-empty ones keeps the term's scale independent
    of how many images in a batch happen to carry spines.
    """
    p = torch.sigmoid(logits)
    num = (tube * p).sum(dim=(1, 2, 3))
    den = tube.sum(dim=(1, 2, 3))
    has = den > eps
    if not bool(has.any()):
        return logits.sum() * 0.0
    return 1.0 - (num[has] / den[has]).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--pct-start", type=float, default=0.1)
    ap.add_argument("--cldice", type=float, default=0.3,
                    help="0 replaces clDice with the skeleton term; >0 stacks")
    ap.add_argument("--cldice-warmup", type=int, default=5)
    ap.add_argument("--skel-w", type=float, default=1.0,
                    help="the paper sweeps 0.1 and 1.0")
    ap.add_argument("--encoder", default="timm-efficientnet-b3")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    name = a.name or f"skelrec_f{a.fold}_w{a.skel_w}_cld{a.cldice}"
    dest = OUT / name
    dest.mkdir(parents=True, exist_ok=True)

    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    entries = gt[["entry_id", "file_name"]].drop_duplicates()
    entries["fold"] = entries["file_name"].map(dict(zip(folds["file_name"],
                                                        folds["fold"])))
    tr = list(entries.loc[entries["fold"] != a.fold, ["entry_id", "file_name"]]
              .itertuples(index=False, name=None))
    va = list(entries.loc[entries["fold"] == a.fold, ["entry_id", "file_name"]]
              .itertuples(index=False, name=None))

    dl_tr = DataLoader(AugCachedTube(tr, S, train=True, aug="heavy"),
                       batch_size=a.batch, shuffle=True, num_workers=3,
                       pin_memory=True, drop_last=True, persistent_workers=True)
    dl_va = DataLoader(AugCachedTube(va, S, train=False, aug="none"),
                       batch_size=a.batch, shuffle=False, num_workers=2,
                       pin_memory=True)

    torch.backends.cudnn.benchmark = True
    model = build_model("unet", a.encoder, 3, S, weights="imagenet").cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * len(dl_tr), pct_start=a.pct_start)
    scaler = torch.amp.GradScaler("cuda")
    logf = open(dest / "train.log", "a", buffering=1)

    def rec(m):
        log(m)
        logf.write(m + "\n")

    rec(f"{name}: heavy-60 + SkelRecall(w={a.skel_w}, human spine) "
        f"clDice={a.cldice} train={len(tr)} val={len(va)} fold={a.fold} "
        f"epochs={a.epochs}  [baseline heavy-60 + clDice = 0.7028]")

    best, hist = -1.0, []
    for ep in range(a.epochs):
        model.train()
        t0, tot, tsk, nb = time.time(), 0.0, 0.0, 0
        for x, y, tube in dl_tr:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            tube = tube.cuda(non_blocking=True)
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss = dice_bce_loss(logits, y)
                if a.cldice > 0 and ep >= a.cldice_warmup:
                    loss = loss + a.cldice * cldice_loss(logits, y)
                if a.skel_w > 0:
                    sk = skeleton_recall(logits, tube)
                    loss = loss + a.skel_w * sk
                    tsk += float(sk.item())
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            tot += float(loss.item())
            nb += 1

        model.eval()
        inter = union = 0
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for x, y, _ in dl_va:
                p = torch.sigmoid(model(x.cuda())) > 0.5
                yv = y.cuda() > 0.5
                inter += int((p & yv).sum().item())
                union += int(p.sum().item() + yv.sum().item())
        dice = 2 * inter / max(union, 1)
        hist.append(dict(epoch=ep, loss=tot / max(nb, 1),
                         skel=tsk / max(nb, 1), val_dice=dice))
        rec(f"epoch {ep}: loss {tot/max(nb,1):.4f}  skel {tsk/max(nb,1):.4f}  "
            f"val pixel-Dice {dice:.4f}  ({time.time()-t0:.0f}s)")
        json.dump(hist, open(dest / "history.json", "w"), indent=1)
        if dice > best:
            best = dice
            torch.save({"model": model.state_dict(), "encoder": a.encoder,
                        "arch": "unet", "in_ch": 3, "imgsz": S, "fold": a.fold,
                        "val_dice": dice, "channels": "rep",
                        "lane": "skelrecall"}, dest / "best.pt")
    rec(f"done. best val pixel-Dice {best:.4f} (heavy-60 + clDice = 0.7028)")
    logf.close()


if __name__ == "__main__":
    main()
