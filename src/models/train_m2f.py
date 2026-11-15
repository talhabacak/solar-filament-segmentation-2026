"""The last untried structural family: query-based instance segmentation.

Why this exists, after two days of closing every other axis by measurement. Nine
capture attempts say the candidate-level selection signal is saturated for the
YOLO-proposal and connected-components families; the recall budget says the gap
to the honest-band ceiling is missed detections. Mask2Former detects by LEARNED
QUERIES with global attention -- a mechanism unrelated to anchor/proposal boxes
or probability thresholds -- and the one pipeline principle that never failed
here is that a lane with a genuinely DIFFERENT error profile adds (two-lane
fusion, the only unabsorbed gain: LB 0.36 -> 0.37). Whether the query mechanism
has a different miss profile on thin curvilinear structures is not measurable
without fine-tuning it, so this is the minimal viable version of that
measurement, not a commitment.

Design choices, each tied to something already measured:
  * per-ENTRY samples (all annotator copies as separate training samples) --
    the all-copies target measured +0.0051 on the YOLO lane
  * 1024 input from the cached pyramid; batch 2 fits a 16 GB card with Swin-T
  * light geometric augmentation only (flips + rot90) for the first run; the
    heavy recipe is an ablation later IF the lane earns one
  * COCO-instance pretrained start (facebook/mask2former-swin-tiny), class head
    reinitialised for nc=1

HF's Mask2FormerForUniversalSegmentation computes its own Hungarian-matched
loss when given mask_labels/class_labels, so no custom loss code exists here to
get wrong.

Usage: python src/models/train_m2f.py --fold 0 --epochs 25
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

from filament.rle import decode_rle

OUT = ROOT / "outputs" / "runs_m2f"
CACHE = ROOT / "data" / "semantic_cache"
S = 1024
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


def log(*a):
    print(*a, flush=True)


class EntryInstances(Dataset):
    """One sample per annotator entry: image + that entry's instance masks."""

    def __init__(self, entries, gt_by_entry, train=True):
        self.entries = entries              # (entry_id, file_name)
        self.gt = gt_by_entry               # entry_id -> list of RLE strings
        self.train = train
        idx = json.loads((CACHE / f"index_{S}.json").read_text())
        self.file_row = {f: i for i, f in enumerate(idx["files"])}
        self.imgs = None

    def __len__(self):
        return len(self.entries)

    def __getstate__(self):
        s = self.__dict__.copy()
        s["imgs"] = None
        return s

    def _open(self):
        if self.imgs is None:
            self.imgs = np.load(CACHE / f"images_{S}.npy", mmap_mode="r")

    def __getitem__(self, i):
        self._open()
        eid, fn = self.entries[i]
        img = np.asarray(self.imgs[self.file_row[fn]])
        masks = []
        for r in self.gt[eid]:
            m = decode_rle(r)
            m = cv2.resize(m, (S, S), interpolation=cv2.INTER_NEAREST)
            if m.sum() >= 16:
                masks.append(m)
        if self.train:
            rng = np.random.default_rng()
            if rng.random() < 0.5:
                img = img[:, ::-1]
                masks = [m[:, ::-1] for m in masks]
            k = int(rng.integers(4))
            if k:
                img = np.rot90(img, k)
                masks = [np.rot90(m, k) for m in masks]
        x = (np.stack([img] * 3, -1).astype(np.float32) - MEAN) / STD
        x = np.ascontiguousarray(x.transpose(2, 0, 1))
        mk = (np.stack([np.ascontiguousarray(m) for m in masks]).astype(np.float32)
              if masks else np.zeros((0, S, S), np.float32))
        return (torch.from_numpy(x), torch.from_numpy(mk),
                torch.zeros(len(masks), dtype=torch.long))


def collate(batch):
    xs = torch.stack([b[0] for b in batch])
    return xs, [b[1] for b in batch], [b[2] for b in batch]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--backbone-lr-mult", type=float, default=0.1)
    ap.add_argument("--model", default="facebook/mask2former-swin-tiny-coco-instance")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    name = a.name or f"m2f_swint_f{a.fold}_{S}"
    dest = OUT / name
    dest.mkdir(parents=True, exist_ok=True)

    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"], folds["fold"]))
    gt_by_entry = gt.groupby("entry_id")["segmentation_rle"].apply(list).to_dict()
    entries = gt[["entry_id", "file_name"]].drop_duplicates()
    tr = [(e, f) for e, f in entries.itertuples(index=False, name=None)
          if fold_of[f] != a.fold]
    log(f"{name}: {len(tr)} training entries (fold {a.fold} held out)")

    dl = DataLoader(EntryInstances(tr, gt_by_entry, True), batch_size=a.batch,
                    shuffle=True, num_workers=3, pin_memory=True, drop_last=True,
                    collate_fn=collate, persistent_workers=True)

    from transformers import Mask2FormerForUniversalSegmentation
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        a.model, num_labels=1, ignore_mismatched_sizes=True).cuda()
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    back = [p for n, p in model.named_parameters() if "pixel_level_module.encoder" in n]
    rest = [p for n, p in model.named_parameters() if "pixel_level_module.encoder" not in n]
    opt = torch.optim.AdamW(
        [{"params": back, "lr": a.lr * a.backbone_lr_mult}, {"params": rest, "lr": a.lr}],
        weight_decay=0.05)
    steps = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + np.cos(np.pi * s / max(steps, 1))))

    logf = open(dest / "train.log", "a", buffering=1)

    def rec(m):
        log(m)
        logf.write(m + "\n")

    rec(f"{name}: {a.model} {n_par:.1f}M, batch {a.batch}, lr {a.lr} "
        f"(backbone x{a.backbone_lr_mult}), epochs {a.epochs}, "
        f"steps/epoch {len(dl)}")

    hist = []
    for ep in range(a.epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for xs, mks, cls in dl:
            out = model(pixel_values=xs.cuda(non_blocking=True),
                        mask_labels=[m.cuda(non_blocking=True) for m in mks],
                        class_labels=[c.cuda(non_blocking=True) for c in cls])
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            tot += float(loss.item())
            nb += 1
        rec(f"epoch {ep}: loss {tot/max(nb,1):.4f}  ({time.time()-t0:.0f}s)  "
            f"peakVRAM {torch.cuda.max_memory_allocated()/2**30:.1f}G")
        hist.append(dict(epoch=ep, loss=tot / max(nb, 1)))
        json.dump(hist, open(dest / "history.json", "w"), indent=1)
        torch.save({"model": model.state_dict(), "hf_name": a.model,
                    "imgsz": S, "fold": a.fold, "lane": "m2f"},
                   dest / "last.pt")
    rec(f"done -> {dest/'last.pt'}")
    logf.close()


if __name__ == "__main__":
    main()
