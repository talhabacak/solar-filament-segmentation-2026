"""WA-06 pilot (docs/30) -- masked-image pretraining of the segmentation U-Net on H-alpha.

Every encoder this project fine-tuned started from ImageNet, and nine encoders lost to
b3 at matched resolution: with 707 labelled frames the constraint is data, not capacity.
The audit's remaining representation question is whether a pretext task ON THE H-ALPHA
FRAMES THEMSELVES (no labels) gives the same network a better starting point than
ImageNet does. This is the smallest honest version of that question:

  * the exact production architecture (smp U-Net, timm-efficientnet-b3, scSE decoder),
    ImageNet-initialised as usual, trained to reconstruct block-masked H-alpha frames
    at the production size (1536), L1 on the masked pixels inside the disk only;
  * block masking (64 px blocks, 40% of the on-disk blocks) so a filament segment is
    hidden whole and has to be continued from its surroundings -- the audit's warning is
    that a pixel-level mask degenerates into filling flat quiet-sun background;
  * the frames of the fold that will later be evaluated are EXCLUDED (--exclude-fold),
    so the fine-tune on that fold is judged on frames the encoder never saw at all,
    exactly like a test frame;
  * the fine-tune is the unchanged recipe with --init pointing at mim.pt; judged at
    matched member count against the deployment-faithful reference, like everything.

Outputs (outputs/runs_semantic/<name>/): last.pt every epoch (resumable), mim.pt at the
end (model weights + a "mim" tag), history.json with the train and held-out losses.

Usage:
  python src/models/pretrain_mim.py --exclude-fold 0 --epochs 40 --name mim_x0
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
CACHE = ROOT / "data" / "semantic_cache"
RUNS = ROOT / "outputs" / "runs_semantic"


def log(*a):
    print(*a, flush=True)


class Frames(Dataset):
    """Raw frames from the semantic cache, same photometric/geometric augmentation as
    CachedSemantic (flip, +/-10 degree rotation about the centre, gamma 0.8..1.25)."""

    def __init__(self, size, files, train):
        idx = json.loads((CACHE / f"index_{size}.json").read_text())
        want = set(files)
        self.rows = [i for i, f in enumerate(idx["files"]) if f in want]
        self.size, self.train, self.imgs = size, train, None

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        s = self.__dict__.copy()
        s["imgs"] = None          # never pickle a memmap into a spawn worker
        return s

    def __getitem__(self, i):
        if self.imgs is None:
            self.imgs = np.load(CACHE / f"images_{self.size}.npy", mmap_mode="r")
        img = np.ascontiguousarray(self.imgs[self.rows[i]])
        if self.train:
            if np.random.rand() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
            ang = np.random.uniform(-10, 10)
            if abs(ang) > 0.5:
                c = (self.size - 1) / 2.0
                M = cv2.getRotationMatrix2D((c, c), ang, 1.0)
                img = cv2.warpAffine(img, M, (self.size, self.size), flags=cv2.INTER_LINEAR)
            g = np.random.uniform(0.8, 1.25)
            img = np.clip(((img / 255.0) ** g) * 255.0, 0, 255).astype(np.uint8)
        return torch.from_numpy(img.astype(np.float32) / 255.0)[None]


def disk_blocks(size, block):
    """Which blocks of the block grid lie (mostly) on the solar disk. The disk is fixed
    at the frame centre with radius 900 px at 2048 (E155), and every augmentation used
    here is a flip or a rotation about the centre, so this never changes."""
    n = size // block
    c = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size]
    disk = (np.hypot(xx - c, yy - c) / (900.0 * size / 2048.0)) < 1.0
    cov = disk.reshape(n, block, n, block).mean(axis=(1, 3))
    return torch.from_numpy(disk), torch.from_numpy(cov > 0.5)


def make_mask(bs, size, block, on_disk, ratio, gen):
    """(bs, 1, size, size) bool: True where the input is hidden and the loss is paid."""
    n = size // block
    m = torch.zeros(bs, n, n, dtype=torch.bool)
    idx = torch.nonzero(on_disk).tolist()
    k = int(round(ratio * len(idx)))
    for b in range(bs):
        pick = torch.randperm(len(idx), generator=gen)[:k]
        for j in pick.tolist():
            m[b, idx[j][0], idx[j][1]] = True
    return m.repeat_interleave(block, 1).repeat_interleave(block, 2)[:, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--imgsz", type=int, default=1536)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--ratio", type=float, default=0.4)
    ap.add_argument("--exclude-fold", type=int, default=0,
                    help="frames of this fold are held out entirely (and report the held-out loss)")
    ap.add_argument("--encoder", default="timm-efficientnet-b3")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: this many frames per split")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    import segmentation_models_pytorch as smp
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    files = sorted(folds["file_name"].unique())
    held = set(folds.loc[folds["fold"] == a.exclude_fold, "file_name"])
    tr_files = [f for f in files if f not in held]
    va_files = sorted(held)
    if a.limit:
        tr_files, va_files = tr_files[: a.limit], va_files[: a.limit]
    ds_tr, ds_va = Frames(a.imgsz, tr_files, True), Frames(a.imgsz, va_files, False)
    log(f"MIM on {len(ds_tr)} frames, fold {a.exclude_fold} held out ({len(ds_va)} frames); "
        f"block {a.block}, ratio {a.ratio}, {a.epochs} epochs at {a.imgsz}")
    dl_tr = DataLoader(ds_tr, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                       pin_memory=True, drop_last=True, persistent_workers=a.workers > 0)
    dl_va = DataLoader(ds_va, batch_size=a.batch, shuffle=False, num_workers=min(a.workers, 2),
                       pin_memory=True)
    disk, on_disk = disk_blocks(a.imgsz, a.block)
    disk = disk.cuda()[None, None]

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # the production architecture, defaults everywhere, so the weights drop into
    # train_semantic --init without any renaming
    model = smp.Unet(encoder_name=a.encoder, encoder_weights="imagenet", in_channels=3,
                     classes=1, decoder_attention_type="scse").cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr,
                                                total_steps=a.epochs * len(dl_tr), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")
    dest = RUNS / a.name
    dest.mkdir(parents=True, exist_ok=True)
    start_ep, hist = 0, []
    if a.resume and (dest / "last.pt").exists():
        ck = torch.load(dest / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_ep, hist = ck["epoch"] + 1, ck["hist"]
        log(f"resumed at epoch {start_ep}")
    gen = torch.Generator().manual_seed(0)

    def save_atomic(obj, path):
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(obj, tmp)
        tmp.replace(path)

    def step(raw, train):
        bs = raw.shape[0]
        m = make_mask(bs, a.imgsz, a.block, on_disk, a.ratio, gen).cuda(non_blocking=True)
        x = raw.cuda(non_blocking=True)
        xin = torch.where(m, torch.zeros_like(x), x).repeat(1, 3, 1, 1)
        with torch.autocast("cuda", dtype=torch.float16):
            out = torch.sigmoid(model(xin))
        pay = (m & disk).float()
        return ((out.float() - x).abs() * pay).sum() / pay.sum().clamp(min=1.0)

    for ep in range(start_ep, a.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for i, raw in enumerate(dl_tr):
            loss = step(raw, True)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item(); n += 1
            if (i + 1) % 100 == 0:
                log(f"  ep{ep} {i+1}/{len(dl_tr)} masked L1 {tot/n:.4f} ({time.time()-t0:.0f}s)")
        model.eval()
        vt, vn = 0.0, 0
        with torch.no_grad():
            for raw in dl_va:
                vt += step(raw, False).item(); vn += 1
        rec = dict(epoch=ep, train_l1=tot / max(n, 1), heldout_l1=vt / max(vn, 1), sec=time.time() - t0)
        hist.append(rec)
        log(f"epoch {ep}: masked L1 train {rec['train_l1']:.4f}  held-out {rec['heldout_l1']:.4f}  ({rec['sec']:.0f}s)")
        save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                     "scaler": scaler.state_dict(), "epoch": ep, "hist": hist, "encoder": a.encoder,
                     "imgsz": a.imgsz, "mim": True}, dest / "last.pt")
        (dest / "history.json").write_text(json.dumps(hist, indent=1))
    save_atomic({"model": model.state_dict(), "encoder": a.encoder, "imgsz": a.imgsz, "mim": True,
                 "exclude_fold": a.exclude_fold, "block": a.block, "ratio": a.ratio, "epochs": a.epochs},
                dest / "mim.pt")
    log(f"wrote {dest / 'mim.pt'}")


if __name__ == "__main__":
    main()
