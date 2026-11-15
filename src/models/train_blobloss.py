"""Instance-balanced (blob) loss, aimed at the one deficit still open.

The measurement that earned this. 30% of ground-truth rows are unreachable by
any pool candidate, and the reason turned out not to be what the project assumed.
The model is NOT blind to them: the median maximum predicted probability inside a
never-proposed filament is 1.000, exactly as for a matched one. What differs is
COVERAGE -- at p >= 0.7 the model claims 47.3% of a never-proposed filament's
area against 79.6% of a matched one's. We detect these filaments and then draw
half of them, which is how a well-detected instance lands below IoU 0.5.

That deficit is unequally distributed across instances in the precise way an
image-level loss produces, and the gap is measurable:

    image-level pixel-Dice, which is what the loss optimises   0.7019
    mean PER-INSTANCE Dice at the production threshold         0.6472
      -- matched instances                                     0.7725
      -- never-proposed instances                              0.4478  (44.5% below 0.5)

A global Dice is a pixel-weighted average, so a bright filament the model already
covers contributes more gradient than a faint one it half-misses. Blob loss
(Kofler et al., IPMI 2023) fixes exactly this: compute Dice per instance and
average over INSTANCES, so every filament pulls equally regardless of how many
pixels or how much contrast it brought.

Why this is not the Skeleton Recall failure repeated. That term was RECALL-ONLY:
it rewarded covering the skeleton and never penalised spilling, so in our 0.42%
foreground regime it inflated predictions to 1.09x the ground-truth foreground
and cost 0.018. A per-instance DICE is two-sided -- over-prediction on an
instance lowers that instance's own Dice -- and the global term is kept alongside
it, exactly as the paper specifies, so precision has something holding it.

    L = dice_bce + 0.3 * clDice + blob_w * L_blob

Everything else is heavy-60 exactly (same augmentation, schedule, optimiser,
fold), so the only variable is the new term. Baseline to beat on fold 0: 0.7028.

Implementation note. The naive form builds an (N, H, W) one-hot and masks each
instance in turn. The same quantity falls out of one scatter-add: for instance i
with predicted mass t_i inside it and area a_i, masking out the OTHER instances'
foreground leaves

    Dice_i = 2 t_i / (S_p - S_fg + t_i + a_i)

where S_p is the predicted mass over the whole frame and S_fg the predicted mass
over all labelled pixels. `index_add_` is used rather than `bincount` because
bincount does not carry gradients to its weights.

Usage:
  python src/models/train_blobloss.py --fold 0 --blob-w 1.0
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


class AugCachedLabels(AugCached):
    """AugCached that also serves the entry's instance label map, warped alike.

    The label plane rides apply_aug's third slot -- the one the weight map uses
    -- so it cannot fall out of register with the image and mask. `wmap_border`
    must be 0.0 here: the slot defaults to 1.0 because a weight map's neutral
    value is 1, and carrying a label plane at that default would flood every
    rotated corner with instance 1. That exact mistake corrupted two earlier
    spine experiments before it was caught.
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.labels = None

    def __getstate__(self):
        s = super().__getstate__()
        s["labels"] = None
        return s

    def _open(self):
        super()._open()
        if self.labels is None:
            self.labels = np.load(CACHE / f"labels_{self.size}.npy", mmap_mode="r")

    def __getitem__(self, i):
        self._open()
        if self._rng is None:
            self._rng = np.random.default_rng(np.random.randint(2 ** 31))
        entry_id, fn = self.entries[i]
        img = np.asarray(self.imgs[self.file_row[fn]])
        mask = self._unpack(self.msks, self.entry_row[entry_id]).astype(np.float32)
        lab = np.asarray(self.labels[self.entry_row[entry_id]]).astype(np.float32)
        if self.train:
            from augment import apply_aug
            img, mask, lab = apply_aug(img, mask, lab, self.size, self.aug,
                                       self._rng, wmap_border=0.0)
        raw = np.ascontiguousarray(img).astype(np.float32) / 255.0
        return (torch.from_numpy(np.stack([raw, raw, raw])).float(),
                torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0),
                torch.from_numpy(np.ascontiguousarray(lab)).float().unsqueeze(0))


def blob_loss(logits, labels, eps=1.0):
    """Mean over instances of (1 - Dice) computed per instance.

    Each instance is scored against the frame with the OTHER instances' pixels
    removed, so a neighbouring filament does not count as this one's false
    positive. Averaging over instances rather than pixels is the whole point:
    a 400-pixel faint filament and a 4000-pixel bright one pull equally.
    """
    p = torch.sigmoid(logits.float())
    total, n = logits.sum() * 0.0, 0
    for b in range(p.shape[0]):
        lab = labels[b, 0].round().long()
        nmax = int(lab.max().item())
        if nmax <= 0:
            continue
        flat = lab.reshape(-1)
        pb = p[b, 0].reshape(-1)
        t = torch.zeros(nmax + 1, device=p.device, dtype=p.dtype)
        t = t.index_add(0, flat, pb)
        a = torch.bincount(flat, minlength=nmax + 1).to(p.dtype)
        s_p, s_fg = pb.sum(), t[1:].sum()
        ti, ai = t[1:], a[1:]
        keep = ai > 0
        if not bool(keep.any()):
            continue
        den = s_p - s_fg + ti + ai
        dice = (2.0 * ti + eps) / (den + eps)
        total = total + (1.0 - dice[keep]).mean()
        n += 1
    return total / n if n else total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--pct-start", type=float, default=0.1)
    ap.add_argument("--cldice", type=float, default=0.3)
    ap.add_argument("--cldice-warmup", type=int, default=5)
    ap.add_argument("--blob-w", type=float, default=1.0)
    ap.add_argument("--blob-warmup", type=int, default=0,
                    help="epochs before the instance term switches on")
    ap.add_argument("--encoder", default="timm-efficientnet-b3")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    name = a.name or f"blob_f{a.fold}_w{a.blob_w}"
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

    dl_tr = DataLoader(AugCachedLabels(tr, S, train=True, aug="heavy"),
                       batch_size=a.batch, shuffle=True, num_workers=3,
                       pin_memory=True, drop_last=True, persistent_workers=True)
    dl_va = DataLoader(AugCachedLabels(va, S, train=False, aug="none"),
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

    rec(f"{name}: heavy-60 + blob loss (w={a.blob_w}, warmup {a.blob_warmup}) "
        f"clDice={a.cldice} {a.encoder} train={len(tr)} val={len(va)} "
        f"fold={a.fold} epochs={a.epochs}  [baseline heavy-60 + clDice = 0.7028]")

    best, hist = -1.0, []
    for ep in range(a.epochs):
        model.train()
        t0, tot, tbl, nb = time.time(), 0.0, 0.0, 0
        for x, y, lab in dl_tr:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            lab = lab.cuda(non_blocking=True)
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss = dice_bce_loss(logits, y)
                if a.cldice > 0 and ep >= a.cldice_warmup:
                    loss = loss + a.cldice * cldice_loss(logits, y)
            if a.blob_w > 0 and ep >= a.blob_warmup:
                # outside autocast: the per-instance sums run over 590k pixels
                # and fp16 accumulation loses the faint instances this term
                # exists to protect
                bl = blob_loss(logits.float(), lab)
                loss = loss + a.blob_w * bl
                tbl += float(bl.item())
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
                         blob=tbl / max(nb, 1), val_dice=dice))
        rec(f"epoch {ep}: loss {tot/max(nb,1):.4f}  blob {tbl/max(nb,1):.4f}  "
            f"val pixel-Dice {dice:.4f}  ({time.time()-t0:.0f}s)")
        json.dump(hist, open(dest / "history.json", "w"), indent=1)
        if dice > best:
            best = dice
            torch.save({"model": model.state_dict(), "encoder": a.encoder,
                        "arch": "unet", "in_ch": 3, "imgsz": S, "fold": a.fold,
                        "val_dice": dice, "channels": "rep",
                        "lane": "blobloss"}, dest / "best.pt")
    rec(f"done. best val pixel-Dice {best:.4f} (heavy-60 + clDice = 0.7028)")
    rec("NOTE: pixel-Dice is the WRONG gate for this term -- it is an image-level "
        "measure and the term is instance-level. The real gate is per-instance "
        "Dice and reach on the never-proposed rows; run src/eda/extraction_"
        "schemes.py style measurement on this checkpoint before believing either "
        "a win or a loss.")
    logf.close()


if __name__ == "__main__":
    main()
