"""E191b -- the two-lane refiner: learn the annotator's mask from both lanes' evidence.

Talha, 2026-09-14. Trained on crops built by `build_fusion_crops.py`, whose five channels are
the image, the U-Net probability field, the detector's mask union, the current candidate mask
and its signed distance field. The target is the ground-truth copy the candidate matches best,
or an empty mask when it matches nothing.

What makes this different from the refiner this project already measured (+0.0043 then
overfitting, ledger 7): that one saw only the fused mask, after the confidence-weighted vote had
discarded the disagreement between the two lanes. Everything measured since says the
disagreement is the signal -- the boundary band is 24.5% of ground-truth copies, its candidates
are systematically too small, and the fusion vote is what makes them small.

Honest protocol, because a refiner is exactly the kind of model that can cheat:
  * trained on the folds OTHER than the one it is applied to (the crops carry their fold);
  * the ground truth is used only as the training target, never at inference;
  * reported first as the IoU it achieves against the same target the scorer uses, against the
    IoU the unrefined candidate already had -- so the number is the realised shape gain, split
    by where the candidate started (matched, boundary band, or hallucination).

Usage: python src/models/train_fusion_refiner.py --fold 0 --epochs 12
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
CROPS = ROOT / "data" / "fusion_crops"
OUT = ROOT / "outputs" / "runs_refiner"


def log(*a):
    print(*a, flush=True)


class Crops(Dataset):
    def __init__(self, idx, size, train):
        self.idx, self.size, self.train = idx, size, train
        self.X = self.Y = None

    def __len__(self):
        return len(self.idx)

    def __getstate__(self):
        s = self.__dict__.copy()
        s["X"] = s["Y"] = None
        return s

    def __getitem__(self, i):
        if self.X is None:
            self.X = np.load(CROPS / f"x_{self.size}.npy", mmap_mode="r")
            self.Y = np.load(CROPS / f"y_{self.size}.npy", mmap_mode="r")
        j = self.idx[i]
        x = np.asarray(self.X[j]).astype(np.float32) / 255.0
        y = (np.asarray(self.Y[j]) > 127).astype(np.float32)
        if self.train:
            if np.random.rand() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
            if np.random.rand() < 0.5:
                x, y = x[:, ::-1, :], y[::-1, :]
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(np.ascontiguousarray(y))[None])


def soft_dice(logits, y, eps=1.0):
    p = torch.sigmoid(logits)
    num = 2 * (p * y).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3)) + eps
    return (1 - num / den).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--crop", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--encoder", default="timm-efficientnet-b0")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--residual", action="store_true",
                    help="E191d: predict a CORRECTION to the candidate instead of regenerating "
                         "the mask. Two runs without this were negative everywhere (band IoU "
                         "0.300 -> 0.243 with a single-copy target, 0.295 -> 0.233 with the "
                         "consensus target), and the reason is structural: a b0 U-Net trained on "
                         "7944 crops cannot re-derive from scratch what a twelve-member ensemble "
                         "plus a detector already produced at full resolution. With the input "
                         "mask carried into the logits and a zero-initialised head, the refiner "
                         "STARTS as the identity and can only learn where to differ")
    a = ap.parse_args()
    import segmentation_models_pytorch as smp
    md = pd.read_parquet(CROPS / f"meta_{a.crop}.parquet")
    tr = np.flatnonzero((md["fold"] != a.fold).to_numpy())
    te = np.flatnonzero((md["fold"] == a.fold).to_numpy())
    log(f"{len(md)} crops: {len(tr)} train (folds != {a.fold}), {len(te)} held out")
    dl_tr = DataLoader(Crops(tr, a.crop, True), batch_size=a.batch, shuffle=True,
                       num_workers=a.workers, pin_memory=True, drop_last=True,
                       persistent_workers=a.workers > 0)
    dl_te = DataLoader(Crops(te, a.crop, False), batch_size=a.batch, shuffle=False,
                       num_workers=min(a.workers, 2), pin_memory=True)
    torch.backends.cudnn.benchmark = True
    model = smp.Unet(encoder_name=a.encoder, encoder_weights="imagenet", in_channels=5,
                     classes=1, decoder_attention_type="scse").cuda()
    if a.residual:
        import torch.nn as nn
        head = model.segmentation_head[0]
        nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
        base = model.forward

        def fwd(x, _b=base):
            # 4.0 puts the identity at sigmoid(+/-4) = 0.982 / 0.018, i.e. the candidate mask
            # reproduced exactly at threshold 0.5 before a single gradient step
            return 4.0 * (2.0 * x[:, 3:4] - 1.0) + _b(x)
        model.forward = fwd
        log("residual mode: the refiner starts as the identity on the candidate mask")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(dl_tr),
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"fusion_refiner_f{a.fold}{'_res' if a.residual else ''}"
    dest.mkdir(exist_ok=True)
    for ep in range(a.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in dl_tr:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(x)
                loss = F.binary_cross_entropy_with_logits(out.float(), y) + soft_dice(out.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item(); n += 1
        log(f"epoch {ep}: loss {tot / max(n, 1):.4f} ({time.time() - t0:.0f}s)")
        torch.save({"model": model.state_dict(), "encoder": a.encoder, "crop": a.crop,
                    "fold": a.fold, "epoch": ep}, dest / "last.pt")

    # held-out reading: IoU before and after, split by where the candidate started
    model.eval()
    before, after = [], []
    with torch.no_grad():
        k = 0
        for x, y in dl_te:
            x = x.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                p = torch.sigmoid(model(x)).float().cpu().numpy()[:, 0]
            cur = x[:, 3].cpu().numpy() > 0.5
            tgt = y.numpy()[:, 0] > 0.5
            new = p > 0.5
            for i in range(len(p)):
                u0 = (cur[i] | tgt[i]).sum()
                u1 = (new[i] | tgt[i]).sum()
                before.append((cur[i] & tgt[i]).sum() / max(u0, 1))
                after.append((new[i] & tgt[i]).sum() / max(u1, 1))
                k += 1
    d = md.iloc[te].copy()
    d["iou_before"], d["iou_after"] = before[: len(d)], after[: len(d)]
    d.to_parquet(dest / "heldout.parquet", index=False)
    band = (d["best_iou"] > 0.05) & (d["best_iou"] <= 0.5)
    matched = d["best_iou"] > 0.5
    halluc = d["best_iou"] <= 0.05
    log(f"\nheld-out fold {a.fold}: {len(d)} candidates")
    for name, sel in (("matched (IoU > 0.5)", matched), ("boundary band", band), ("hallucination", halluc)):
        if sel.sum():
            log(f"  {name:22s} n {int(sel.sum()):5d}  IoU {d.loc[sel, 'iou_before'].mean():.4f} -> "
                f"{d.loc[sel, 'iou_after'].mean():.4f}  ({d.loc[sel, 'iou_after'].mean() - d.loc[sel, 'iou_before'].mean():+.4f})")
    crossed = int(((d["iou_before"] <= 0.5) & (d["iou_after"] > 0.5)).sum())
    lost = int(((d["iou_before"] > 0.5) & (d["iou_after"] <= 0.5)).sum())
    log(f"  candidates crossing IoU 0.5 upward: {crossed}; falling below: {lost}; net {crossed - lost:+d}")
    json.dump(dict(n=len(d), crossed=crossed, lost=lost,
                   band_before=float(d.loc[band, "iou_before"].mean()) if band.sum() else 0.0,
                   band_after=float(d.loc[band, "iou_after"].mean()) if band.sum() else 0.0),
              open(dest / "heldout.json", "w"), indent=1)
    log("REFINER_DONE")


if __name__ == "__main__":
    main()
