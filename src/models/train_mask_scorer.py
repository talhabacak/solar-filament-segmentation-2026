"""Look at the crop to judge a candidate, instead of describing it with numbers.

There is still real selection headroom: oracle on the fold-0 pool is 0.5321
against 0.4383 realised, and 0.5078 against 0.4286 pooled. selector v6 predicts
the metric-derived value (iou_sum / n_copies) from 32 tabular features and gets
Spearman 0.68 — good enough to match v5, not good enough to close 0.08.

The tabular features describe a mask (area, eccentricity, skeleton length, the
semantic probability under and around it) but never look at it. A CNN over the
same 320x320 native-resolution crop the refiner used can, and judging a mask is
a much easier task than correcting one — the refiner failed at correction
(+0.0038 at best, then overfitting) because the boundary is an annotation
convention the image does not determine, but whether a mask is a plausible
filament drawing is visible.

Predicts the same value the rule uses, so the closed-form keep threshold still
applies. Output is meant to join the tabular features as one more column, not to
replace them — this reports both the CNN alone and the CNN stacked into v6.

Usage: python src/models/train_mask_scorer.py --epochs 15 --batch 24
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
OUT = ROOT / "outputs" / "runs_scorer"
CROP = 320
SDF_CLIP = 20.0


def log(*a):
    print(*a, flush=True)


def sdf(mask):
    m = mask.astype(np.uint8)
    if m.max() == 0:
        return np.full((CROP, CROP), SDF_CLIP, np.float32)
    din = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    dout = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
    return np.clip(dout - din, -SDF_CLIP, SDF_CLIP).astype(np.float32)


class ScoreSet(Dataset):
    """(image, mask, sdf) crop -> the candidate's value.

    Memmaps open per worker and are stripped from the pickle: shipping one
    through a Windows spawn materialises the whole array into the pipe.
    """

    def __init__(self, idx, y, train):
        self.idx = np.asarray(idx)
        self.y = np.asarray(y, np.float32)
        self.train = train
        self.img = self.inm = None

    def __len__(self):
        return len(self.idx)

    def __getstate__(self):
        s = self.__dict__.copy()
        s["img"] = s["inm"] = None
        return s

    def _open(self):
        if self.img is None:
            self.img = np.load(CROPS / "cand_img.npy", mmap_mode="r")
            self.inm = np.load(CROPS / "cand_in.npy", mmap_mode="r")

    def __getitem__(self, i):
        self._open()
        j = int(self.idx[i])
        img = np.asarray(self.img[j])
        m = np.asarray(self.inm[j])
        if self.train:
            rng = np.random.default_rng((j * 2654435761 + 7) % (2 ** 32))
            if rng.random() < 0.5:
                img, m = img[:, ::-1], m[:, ::-1]
            if rng.random() < 0.5:
                img, m = img[::-1], m[::-1]
            img, m = np.ascontiguousarray(img), np.ascontiguousarray(m)
        x = np.stack([img.astype(np.float32) / 255.0, m.astype(np.float32),
                      sdf(m) / SDF_CLIP])
        return torch.from_numpy(x), torch.tensor(self.y[i])


class Scorer(torch.nn.Module):
    def __init__(self, encoder="efficientnet_b0"):
        super().__init__()
        import timm
        self.enc = timm.create_model(encoder, pretrained=True, in_chans=3,
                                     num_classes=0, global_pool="avg")
        d = self.enc.num_features
        self.head = torch.nn.Sequential(
            torch.nn.Linear(d, 128), torch.nn.SiLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 1))

    def forward(self, x):
        return self.head(self.enc(x)).squeeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--encoder", default="efficientnet_b0")
    ap.add_argument("--name", default="scorer_b0_320")
    a = ap.parse_args()
    dest = OUT / a.name
    dest.mkdir(parents=True, exist_ok=True)

    meta = pd.read_parquet(CROPS / "cand_meta.parquet")
    pool = pd.read_parquet(ROOT / "outputs" / "selector" / "fused_pool_all.parquet")
    pool["value"] = pool["iou_sum"] / pool["n_copies"].clip(lower=1)
    meta = meta.merge(pool[["cand_id", "value", "conf", "fold"]].rename(
        columns={"fold": "pool_fold"}), on="cand_id", how="left")
    meta = meta.dropna(subset=["value"])
    log(f"{len(meta)} crops with labels; positives (value>0) "
        f"{int((meta['value']>0).sum())}")

    tr = meta[meta["fold"] != a.fold]
    va = meta[meta["fold"] == a.fold]
    log(f"train {len(tr)}, val {len(va)}")

    dl_tr = DataLoader(ScoreSet(tr["idx"], tr["value"], True), batch_size=a.batch,
                       shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
                       persistent_workers=True)
    dl_va = DataLoader(ScoreSet(va["idx"], va["value"], False), batch_size=a.batch,
                       shuffle=False, num_workers=2, pin_memory=True)

    torch.backends.cudnn.benchmark = True
    model = Scorer(a.encoder).cuda()
    log(f"  {a.encoder}: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr,
                                                total_steps=a.epochs * len(dl_tr),
                                                pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda")
    logf = open(dest / "train.log", "a", buffering=1)

    def rec(m):
        log(m)
        logf.write(m + "\n")

    from scipy.stats import spearmanr
    rec(f"{a.name}: {a.encoder} crop {CROP} batch {a.batch} lr {a.lr} "
        f"epochs {a.epochs} fold {a.fold}")
    best, hist = -1.0, []
    for ep in range(a.epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for x, y in dl_tr:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.amp.autocast("cuda"):
                loss = torch.nn.functional.mse_loss(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            tot += loss.item()
            nb += 1
        model.eval()
        preds = []
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for x, _ in dl_va:
                preds.append(model(x.cuda()).float().cpu().numpy())
        p = np.concatenate(preds)
        rho = float(spearmanr(p, va["value"].to_numpy()).statistic)
        hist.append(dict(epoch=ep, loss=tot / max(nb, 1), spearman=rho))
        rec(f"epoch {ep}: mse {tot/max(nb,1):.5f}  val Spearman {rho:.4f}  "
            f"({time.time()-t0:.0f}s)")
        with open(dest / "history.json", "w") as fh:
            json.dump(hist, fh, indent=1)
        if rho > best:
            best = rho
            torch.save({"model": model.state_dict(), "encoder": a.encoder,
                        "crop": CROP, "fold": a.fold, "spearman": rho},
                       dest / "best.pt")
            np.save(dest / "val_pred.npy", p)
            va[["cand_id", "value"]].to_parquet(dest / "val_meta.parquet", index=False)
    rec(f"done. best val Spearman {best:.4f} (tabular v6 reference 0.6822) "
        f"-> {dest/'best.pt'}")
    logf.close()


if __name__ == "__main__":
    main()
