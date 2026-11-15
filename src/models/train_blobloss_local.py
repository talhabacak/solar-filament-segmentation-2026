"""Instance-balanced loss with a LOCAL background, fixing the measured failure.

Why a second arm. The first blob-loss run lost on the gate built for it:
per-instance Dice -0.0149, net rows reached -48, and decisively COVERAGE went
DOWN rather than up (matched 0.743 -> 0.632, unreachable 0.456 -> 0.390). The
mechanism was diagnosed rather than guessed. In the paper's formulation the
per-instance denominator is

    Dice_i = 2 t_i / (B + t_i + a_i)

where B is the predicted probability mass over EVERY unlabelled pixel of the
frame, common to all instances. With about seven instances per entry, one
background false positive is charged seven times, so the term pushes precision
instead of coverage -- the exact mirror of the Skeleton Recall failure, which was
recall-only and inflated the foreground instead.

The fix follows from the diagnosis: give each instance its OWN background. Every
background pixel is assigned to the nearest instance within a radius, so

    Dice_i = 2 t_i / (b_i + t_i + a_i)

with b_i the predicted mass in instance i's neighbourhood only. A false positive
now costs once, next to the instance it sits beside, and the denominator scales
with the instance rather than with the frame -- which is what makes small and
faint instances pull as hard as large bright ones.

The balance is measured rather than hoped for, which is the check the first arm
skipped. Writing d_i = b_i + t_i + a_i, the pull toward covering an instance is
2(b_i + a_i)/d_i^2 and the push against spilling beside it is 2 t_i/d_i^2. On
real data a converged model leaves almost no probability in the background, so
b_i is about 0.20 a_i at ANY radius from 10 to 80 px, giving

    local    cover 0.50/a_i   vs   spill 0.41/a_i, charged once   -> coverage wins
    global   cover 0.38/a_i   vs   spill 0.13/a_i x 7 instances    -> precision wins

The second line is the arm that failed, and its 7x is exactly the frame-wide
background being charged once per instance. The radius barely moves b_i, so 20 px
is chosen simply for locality.

The ownership map is derived from the AUGMENTED label plane at load time rather
than warped through the augmentation, because a nearest-neighbour field is not a
quantity that survives resampling: rotating an ownership map and rotating the
labels then recomputing ownership give different answers, and only the second is
correct.

What this targets, measured. The instances the pipeline cannot reach are not
truncated at their ends -- their coverage deficit is uniform along the filament
(ends 0.393 against middle 0.412, where MATCHED instances show a larger end
deficit of -0.113) and equally low across the flanks (0.374 against 0.652). It is
a per-object confidence deficit spread evenly over the object, which is the
failure an instance-balanced loss exists to correct.

Baseline to beat on fold 0: 0.7028 pixel-Dice, but that is NOT the gate. Judge
this on src/eda/gate_instance_quality.py: per-instance Dice, coverage, and net
rows reached.

Usage:
  python src/models/train_blobloss_local.py --fold 0 --blob-w 1.0 --radius 40
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from finetune_aug import AugCached
from train_semantic import dice_bce_loss, cldice_loss, log
from train_semantic_arch import build_model

CACHE = ROOT / "data" / "semantic_cache"
OUT = ROOT / "outputs" / "runs_aug"


def ownership(lab, radius):
    """Assign each background pixel to the nearest instance within `radius`.

    One distance transform does it: with the instances as the zero set,
    DIST_LABEL_PIXEL hands every pixel the label of its nearest instance pixel,
    and reading the label map at instance positions builds the lookup back to
    instance ids. Pixels further than `radius` from any instance belong to
    nobody and enter no instance's denominator.
    """
    inst = lab > 0
    if not inst.any():
        return np.zeros_like(lab, np.int32)
    src = (~inst).astype(np.uint8)
    dist, lbl = cv2.distanceTransformWithLabels(
        src, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros(int(lbl.max()) + 1, np.int32)
    lut[lbl[inst]] = lab[inst].astype(np.int32)
    own = lut[lbl]
    own[dist > radius] = 0
    own[inst] = lab[inst]
    return own


class AugCachedLabelsLocal(AugCached):
    """AugCached plus the instance label plane and its ownership field."""

    def __init__(self, *args, radius=40, **kw):
        super().__init__(*args, **kw)
        self.labels = None
        self.radius = radius

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
            # wmap_border=0.0: the third plane defaults to 1.0 for weight maps,
            # and carrying labels at that default floods every rotated corner
            # with instance 1
            img, mask, lab = apply_aug(img, mask, lab, self.size, self.aug,
                                       self._rng, wmap_border=0.0)
        labi = np.ascontiguousarray(lab).round().astype(np.int32)
        own = ownership(labi, self.radius)
        raw = np.ascontiguousarray(img).astype(np.float32) / 255.0
        return (torch.from_numpy(np.stack([raw, raw, raw])).float(),
                torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0),
                torch.from_numpy(labi).long(),
                torch.from_numpy(own).long())


def blob_loss_local(logits, lab, own, eps=1.0):
    """Mean over instances of (1 - Dice), each scored in its own neighbourhood."""
    p = torch.sigmoid(logits.float())
    total, n = logits.sum() * 0.0, 0
    for b in range(p.shape[0]):
        lb, ob, pb = lab[b].reshape(-1), own[b].reshape(-1), p[b, 0].reshape(-1)
        nmax = int(lb.max().item())
        if nmax <= 0:
            continue
        t = torch.zeros(nmax + 1, device=p.device, dtype=p.dtype).index_add(0, lb, pb)
        o = torch.zeros(nmax + 1, device=p.device, dtype=p.dtype).index_add(0, ob, pb)
        a = torch.bincount(lb, minlength=nmax + 1).to(p.dtype)
        ti, ai = t[1:], a[1:]
        bi = o[1:] - ti                      # neighbourhood mass outside instance i
        keep = ai > 0
        if not bool(keep.any()):
            continue
        dice = (2.0 * ti + eps) / (ti + bi.clamp(min=0) + ai + eps)
        total = total + (1.0 - dice[keep]).mean()
        n += 1
    return total / n if n else total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--size", type=int, default=768,
                    help="training resolution; the label cache must exist "
                         "at this size (src/data/build_label_cache.py)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--pct-start", type=float, default=0.1)
    ap.add_argument("--cldice", type=float, default=0.3)
    ap.add_argument("--cldice-warmup", type=int, default=5)
    ap.add_argument("--blob-w", type=float, default=1.0)
    ap.add_argument("--radius", type=int, default=40,
                    help="px at 768; 40 is about nine filament widths")
    ap.add_argument("--encoder", default="timm-efficientnet-b3")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    S = a.size
    name = a.name or f"bloblocal_f{a.fold}_s{S}_w{a.blob_w}_r{a.radius}"
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

    dl_tr = DataLoader(AugCachedLabelsLocal(tr, S, train=True, aug="heavy",
                                            radius=a.radius),
                       batch_size=a.batch, shuffle=True, num_workers=3,
                       pin_memory=True, drop_last=True, persistent_workers=True)
    dl_va = DataLoader(AugCachedLabelsLocal(va, S, train=False, aug="none",
                                            radius=a.radius),
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

    rec(f"{name}: heavy-60 + LOCAL blob loss (w={a.blob_w}, radius={a.radius}px) "
        f"clDice={a.cldice} train={len(tr)} val={len(va)} fold={a.fold} "
        f"epochs={a.epochs}  [global-background arm lost: coverage 0.743->0.632]")

    best, hist = -1.0, []
    for ep in range(a.epochs):
        model.train()
        t0, tot, tbl, nb = time.time(), 0.0, 0.0, 0
        for x, y, lab, own in dl_tr:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            lab, own = lab.cuda(non_blocking=True), own.cuda(non_blocking=True)
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss = dice_bce_loss(logits, y)
                if a.cldice > 0 and ep >= a.cldice_warmup:
                    loss = loss + a.cldice * cldice_loss(logits, y)
            if a.blob_w > 0:
                bl = blob_loss_local(logits.float(), lab, own)
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
            for x, y, _, _ in dl_va:
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
                        "lane": "blobloss_local"}, dest / "best.pt")
    rec(f"done. best val pixel-Dice {best:.4f}")
    rec("GATE: run src/eda/gate_instance_quality.py on this checkpoint. "
        "pixel-Dice is an image-level measure and this term is instance-level; "
        "the arm is judged on per-instance Dice, coverage and net rows reached.")
    logf.close()


if __name__ == "__main__":
    main()
