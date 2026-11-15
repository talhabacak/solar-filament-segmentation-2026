"""Fine-tune augmentation study: does a 707-image dataset want more augmentation?

Why this is worth measuring rather than assuming. The fine-tune stage sees 707
physical images / 1154 annotator entries — 924 of them after the fold-0 split —
and the current recipe augments them with three operations only: horizontal
flip, +-10 degrees, gamma 0.8-1.25. For a 13M-parameter network that is thin,
and two independent measurements say overfitting is the live constraint:

  * the data-scaling curve is still climbing (25/50/100% of images ->
    0.6739/0.6866/0.6981, about +0.012 per doubling), so the model is
    data-limited, and augmentation is synthetic data
  * the 100-epoch schedule DEGRADED under the current light recipe. That is the
    classic signature of a model that runs out of new information and starts
    memorising, not of a schedule that is simply too long.

That second point is the interesting one, and it is why this script can also run
the long schedule: light-aug + long = overfit is the expected result, but
heavy-aug + long is a different experiment that has never been run. If heavy
augmentation is doing its job, it should show up exactly there.

Design. Arms are cumulative (light < medium < heavy, see augment.py) so a
difference is attributable to the group that was added. Everything else is held
fixed: same fold, same encoder, same init, same schedule, same effective batch.
A `light` arm is run as the internal control rather than comparing against the
recorded 0.6981, because that reference used batch 4 and this study runs batch 2
with accum 2 to leave VRAM for the concurrent pretraining — same optimisation
step, different BatchNorm sample count.

What is reported. Final val pixel-Dice is the headline, but the mechanism check
is the train-loss/val-Dice gap over epochs: augmentation is supposed to work by
closing that gap, and an arm that improves val without closing the gap improved
for some other reason.

Usage:
  python src/models/finetune_aug.py --aug heavy --epochs 30 --name aug_heavy
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

from augment import apply_aug
from train_semantic import CachedSemantic, dice_bce_loss, cldice_loss, log
from train_semantic_arch import build_model, make_schedule, EarlyStop

OUT = ROOT / "outputs" / "runs_aug"


class AugCached(CachedSemantic):
    """CachedSemantic with the augmentation level swapped out.

    The parent's __getitem__ hard-codes its three operations, so it is replaced
    wholesale rather than wrapped — wrapping would apply the light recipe first
    and then the heavy one on top, which is not the comparison being run.

    RNG: one Generator per worker, created on first use so it is seeded from the
    per-worker global numpy state PyTorch sets, and never re-created — with
    persistent_workers the stream then keeps advancing across epochs instead of
    replaying the same augmentations every epoch.
    """

    def __init__(self, *args, aug="light", **kw):
        super().__init__(*args, **kw)
        self.aug = aug
        self._rng = None

    def __getstate__(self):
        s = super().__getstate__()
        s["_rng"] = None
        return s

    def __getitem__(self, i):
        self._open()
        if self._rng is None:
            self._rng = np.random.default_rng(np.random.randint(2 ** 31))
        entry_id, fn = self.entries[i]
        img = np.asarray(self.imgs[self.file_row[fn]])
        mask = self._unpack(self.msks, self.entry_row[entry_id]).astype(np.float32)
        wmap = None
        if self.u_weight != 1.0:
            lr = self._unpack(self.msks_lr, self.entry_row[entry_id]).astype(np.float32)
            wmap = np.where((mask > 0) & (lr == 0), np.float32(self.u_weight),
                            np.float32(1.0))
        if self.train:
            img, mask, wmap = apply_aug(img, mask, wmap, self.size, self.aug,
                                        self._rng)
        raw = np.ascontiguousarray(img).astype(np.float32) / 255.0
        x = torch.from_numpy(np.stack([raw, raw, raw])).float()
        y = torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0)
        w = torch.from_numpy(np.ascontiguousarray(wmap)).float().unsqueeze(0) \
            if wmap is not None else torch.ones_like(y)
        return x, (y > 0.5).float(), w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", default="light",
                    choices=["none", "light", "medium", "heavy"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=2,
                    help="effective batch = batch * accum; 2*2 matches the "
                         "reference run's step while halving VRAM")
    ap.add_argument("--encoder", default="timm-efficientnet-b3")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=float, default=0.0)
    ap.add_argument("--pct-start", type=float, default=0.1)
    ap.add_argument("--cosine", action="store_true")
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--cldice", type=float, default=0.3)
    ap.add_argument("--cldice-warmup", type=int, default=5)
    ap.add_argument("--u-weight", type=float, default=1.0)
    ap.add_argument("--init", default=None,
                    help="checkpoint to start from (e.g. a GONG pretraining)")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="encoder gradient checkpointing -- needed at 1536, "
                         "where batch 2 would otherwise want ~27 GB (1024 at "
                         "batch 2 already peaks at 12.3 GB and 1536 is 2.25x "
                         "the pixels)")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    name = a.name or f"aug_{a.aug}_f{a.fold}_{a.imgsz}_{a.epochs}ep"
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

    ds_tr = AugCached(tr, a.imgsz, train=True, u_weight=a.u_weight, aug=a.aug)
    ds_va = AugCached(va, a.imgsz, train=False, aug="none")
    dl_tr = DataLoader(ds_tr, batch_size=a.batch, shuffle=True, num_workers=3,
                       pin_memory=True, drop_last=True, persistent_workers=True)
    dl_va = DataLoader(ds_va, batch_size=a.batch, shuffle=False, num_workers=2,
                       pin_memory=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = build_model("unet", a.encoder, 3, a.imgsz, a.grad_ckpt,
                        weights=None if a.init else "imagenet")
    init_note = ""
    if a.init:
        ck = torch.load(ROOT / a.init, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        init_note = (f"init from {a.init} (stage={ck.get('stage','?')}, "
                     f"missing={len(missing)}, unexpected={len(unexpected)})")
        if len(missing) > 20 or len(unexpected) > 20:
            raise SystemExit("init checkpoint does not match this architecture")
    model = model.cuda()

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps_per_ep = len(dl_tr) // a.accum
    if a.cosine:
        sched = make_schedule(opt, a, steps_per_ep)
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=a.lr, total_steps=a.epochs * steps_per_ep,
            pct_start=a.pct_start)
    stopper = EarlyStop(a.patience, mode="max")
    scaler = torch.amp.GradScaler("cuda")
    logf = open(dest / "train.log", "a", buffering=1)

    def rec(m):
        log(m)
        logf.write(m + "\n")

    rec(f"{name}: aug={a.aug} {a.encoder} imgsz={a.imgsz} batch={a.batch}x{a.accum} "
        f"lr={a.lr} wd={a.wd} epochs={a.epochs} cosine={a.cosine} "
        f"cldice={a.cldice} train={len(tr)} val={len(va)} fold={a.fold}")
    if init_note:
        rec("  " + init_note)

    best, hist = -1.0, []
    total_steps = a.epochs * steps_per_ep
    for ep in range(a.epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, (x, y, w) in enumerate(dl_tr):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            wt = w.cuda(non_blocking=True) if a.u_weight != 1.0 else None
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss = dice_bce_loss(logits, y, weight=wt)
                if a.cldice > 0 and ep >= a.cldice_warmup:
                    loss = loss + a.cldice * cldice_loss(logits, y)
            scaler.scale(loss / a.accum).backward()
            if (i + 1) % a.accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if a.cosine or sched.last_epoch < total_steps - 1:
                    sched.step()
            tot += loss.item()
            nb += 1
        # Train-set Dice on the UNAUGMENTED view is what makes the gap readable:
        # the training loss above is computed on augmented inputs, so comparing
        # it across arms would confound "harder inputs" with "less overfitting".
        model.eval()
        inter = union = 0.0
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for x, y, _ in dl_va:
                p = torch.sigmoid(model(x.cuda())) > 0.5
                yv = y.cuda() > 0.5
                inter += (p & yv).sum().item()
                union += p.sum().item() + yv.sum().item()
        dice = 2 * inter / max(union, 1)
        hist.append(dict(epoch=ep, train_loss=tot / max(nb, 1), val_dice=dice,
                         lr=opt.param_groups[0]["lr"],
                         secs=round(time.time() - t0, 1)))
        rec(f"epoch {ep}: loss {tot/max(nb,1):.4f}  val pixel-Dice {dice:.4f}  "
            f"({time.time()-t0:.0f}s)  peakVRAM "
            f"{torch.cuda.max_memory_allocated()/2**30:.1f}G")
        with open(dest / "history.json", "w") as f:
            json.dump(hist, f, indent=1)
        stop_now = stopper.step(dice)
        if dice > best:
            best = dice
            torch.save({"model": model.state_dict(), "encoder": a.encoder,
                        "arch": "unet", "in_ch": 3, "imgsz": a.imgsz,
                        "fold": a.fold, "val_dice": dice, "aug": a.aug,
                        "channels": "rep"}, dest / "best.pt")
        if stop_now:
            rec(f"early stop: no improvement for {a.patience} epochs "
                f"(best {stopper.best:.4f})")
            break
    rec(f"done. aug={a.aug} best val pixel-Dice {best:.4f} -> {dest/'best.pt'}")
    logf.close()


if __name__ == "__main__":
    main()
