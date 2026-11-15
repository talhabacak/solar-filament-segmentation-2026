"""Spine-auxiliary training: the winning recipe plus a centreline head.

Mechanism and its measured precondition. Annotators agree on the CENTRELINE
substantially better than on the boundary (ribbon IoU 0.708 vs mask IoU 0.629,
measured in one unit over 2,144 cross-annotator pairs), and the population we
miss is precisely the one where boundary convention is unstable -- faint, thin
instances whose recall falls from 0.74 at full consensus to 0.17 at 1-of-3,
with the annotators' own pairwise IoU the strongest predictor of whether we
match them (Cohen's d 0.85). A second head supervised by the more stable signal
should push the shared encoder toward width-invariant, continuity-aware
features, which is what has to improve for those instances to form components
that clear IoU 0.5.

This is NOT the rejected spine-ribbon experiment. That one used the spine as
the OUTPUT representation and failed because width is unrecoverable from a
centreline (IoU 0.6572 vs 0.6666 even given oracle width). Here the mask head
and its loss are untouched; the spine is auxiliary supervision only, and at
inference the spine head is simply ignored.

Everything else is heavy-60 exactly: same augmentation level, schedule, loss,
optimiser and fold, so the only variable is the auxiliary head. Baseline to
beat on fold 0: 0.7028 pixel-Dice.

Usage: python src/models/train_spine_aux.py --fold 0 --spine-w 0.3
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
from train_semantic_arch import build_model, EarlyStop

CACHE = ROOT / "data" / "semantic_cache"
OUT = ROOT / "outputs" / "runs_aug"
S = 768


class AugCachedSpine(AugCached):
    """AugCached plus the entry's spine target, warped by the SAME geometry.

    The spine plane rides through the augmentation as the weight map does, so
    it cannot drift out of alignment with the image and mask -- the failure
    mode that would silently teach nothing.
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.spines = None

    def __getstate__(self):
        s = super().__getstate__()
        s["spines"] = None
        return s

    def _open(self):
        super()._open()
        if self.spines is None:
            self.spines = np.load(CACHE / f"spines_{self.size}.npy", mmap_mode="r")

    def __getitem__(self, i):
        self._open()
        if self._rng is None:
            self._rng = np.random.default_rng(np.random.randint(2 ** 31))
        entry_id, fn = self.entries[i]
        img = np.asarray(self.imgs[self.file_row[fn]])
        mask = self._unpack(self.msks, self.entry_row[entry_id]).astype(np.float32)
        spine = np.asarray(self.spines[self.entry_row[entry_id]]).astype(np.float32) / 255.0
        if self.train:
            # apply_aug carries a third plane through as the weight map; nearest
            # interpolation keeps the ridge crisp, which is what we want here
            from augment import apply_aug
            img, mask, spine = apply_aug(img, mask, spine, self.size, self.aug,
                                         self._rng, wmap_border=0.0)
        raw = np.ascontiguousarray(img).astype(np.float32) / 255.0
        x = torch.from_numpy(np.stack([raw, raw, raw])).float()
        y = torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0)
        sp = torch.from_numpy(np.ascontiguousarray(spine)).float().unsqueeze(0)
        return x, (y > 0.5).float(), sp


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
    ap.add_argument("--spine-w", type=float, default=0.3,
                    help="auxiliary loss weight; 0 reproduces heavy-60 exactly")
    ap.add_argument("--encoder", default="timm-efficientnet-b3")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    name = a.name or f"spineaux_f{a.fold}_{S}_w{a.spine_w}"
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

    dl_tr = DataLoader(AugCachedSpine(tr, S, train=True, aug="heavy"),
                       batch_size=a.batch, shuffle=True, num_workers=3,
                       pin_memory=True, drop_last=True, persistent_workers=True)
    dl_va = DataLoader(AugCachedSpine(va, S, train=False, aug="none"),
                       batch_size=a.batch, shuffle=False, num_workers=2,
                       pin_memory=True)

    torch.backends.cudnn.benchmark = True
    # two output channels: [mask logit, spine logit]; channel 0 is scored, and
    # at inference the second is discarded, so the deployed head is unchanged
    model = build_model("unet", a.encoder, 3, S, weights="imagenet")
    import torch.nn as nn
    head = model.segmentation_head[0]
    model.segmentation_head[0] = nn.Conv2d(head.in_channels, 2, head.kernel_size,
                                           padding=head.padding)
    model = model.cuda()

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * len(dl_tr), pct_start=a.pct_start)
    scaler = torch.amp.GradScaler("cuda")
    logf = open(dest / "train.log", "a", buffering=1)

    def rec(m):
        log(m)
        logf.write(m + "\n")

    rec(f"{name}: heavy-60 + spine aux (w={a.spine_w}) {a.encoder} "
        f"train={len(tr)} val={len(va)} fold={a.fold} epochs={a.epochs}")

    best, hist = -1.0, []
    for ep in range(a.epochs):
        model.train()
        t0, tot, taux, nb = time.time(), 0.0, 0.0, 0
        for x, y, sp in dl_tr:
            x, y, sp = (x.cuda(non_blocking=True), y.cuda(non_blocking=True),
                        sp.cuda(non_blocking=True))
            with torch.amp.autocast("cuda"):
                out = model(x)
                mlog, slog = out[:, :1], out[:, 1:]
                loss = dice_bce_loss(mlog, y)
                if a.cldice > 0 and ep >= a.cldice_warmup:
                    loss = loss + a.cldice * cldice_loss(mlog, y)
                if a.spine_w > 0:
                    aux = torch.nn.functional.binary_cross_entropy_with_logits(slog, sp)
                    loss = loss + a.spine_w * aux
                    taux += float(aux.item())
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
                p = torch.sigmoid(model(x.cuda())[:, :1]) > 0.5
                yv = y.cuda() > 0.5
                inter += int((p & yv).sum().item())
                union += int(p.sum().item() + yv.sum().item())
        dice = 2 * inter / max(union, 1)
        hist.append(dict(epoch=ep, loss=tot / max(nb, 1),
                         aux=taux / max(nb, 1), val_dice=dice))
        rec(f"epoch {ep}: loss {tot/max(nb,1):.4f}  aux {taux/max(nb,1):.4f}  "
            f"val pixel-Dice {dice:.4f}  ({time.time()-t0:.0f}s)")
        json.dump(hist, open(dest / "history.json", "w"), indent=1)
        if dice > best:
            best = dice
            torch.save({"model": model.state_dict(), "encoder": a.encoder,
                        "arch": "unet", "in_ch": 3, "imgsz": S, "fold": a.fold,
                        "val_dice": dice, "channels": "rep",
                        "lane": "spine_aux", "out_channels": 2}, dest / "best.pt")
    rec(f"done. best val pixel-Dice {best:.4f} (heavy-60 baseline 0.7028)")
    logf.close()


if __name__ == "__main__":
    main()
