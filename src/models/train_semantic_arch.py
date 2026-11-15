"""Architecture / input-band experiments for the semantic lane.

Kept separate from train_semantic.py so the shared dataset and losses can be
edited-free while a run is in flight (Windows DataLoader workers re-import the
module, and editing it mid-run kills them).

Two questions this answers, both single-variable against the current reference
(smp.Unet + timm-efficientnet-b3, 3 replicated channels, 1536, clDice 0.3 ->
val pixel-Dice 0.7019, standalone instance PQ 0.4066):

  --in-ch 1     Does feeding the grey image once beat replicating it three
                times? smp folds the pretrained stem weights across channels, so
                this is the clean test of whether the replicated bands cost or
                buy anything.
  --arch upernet --encoder tu-swinv2_cr_tiny_ns_224
                Hierarchical shifted-window attention + UPerNet's pyramid
                fusion, i.e. a genuinely different inductive bias from a
                convolutional U-Net. Different error profile is what the
                ensemble rewards (measured: same-mechanism levers get absorbed).

Usage:
  python src/models/train_semantic_arch.py --arch unet --in-ch 1 --name sem_b3_f0_1536_1ch
  python src/models/train_semantic_arch.py --arch upernet --encoder tu-swinv2_cr_tiny_ns_224 \
      --imgsz 1024 --batch 2 --name sem_swin_f0_1024
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

from train_semantic import CachedSemantic, dice_bce_loss, cldice_loss, log

OUT = ROOT / "outputs" / "runs_semantic"


class NBandCached(CachedSemantic):
    """Same cache, but the sample carries `n` copies of the grey channel.

    n=2 exists for the Sentinel-1 weights: their stem is natively (64, 2, 7, 7),
    so feeding the grey frame duplicated loads all 320 tensors untouched and
    leaves the checkpoint's BatchNorm statistics operating at their native scale
    — no weight surgery, and therefore no scale bug to introduce. Feeding
    (x, x) through [w_VV, w_VH] equals (w_VV + w_VH) * x, i.e. it is exactly the
    1-channel sum adaptation, but obtained without touching the weights.
    """

    n = 1

    def __getitem__(self, i):
        x, y, w = super().__getitem__(i)
        return x[:self.n], y, w


class OneBandCached(NBandCached):
    n = 1


class TwoBandCached(NBandCached):
    n = 2


def build_model(arch, encoder, in_ch, imgsz, grad_ckpt=False, weights="imagenet",
                swin_satlas=None):
    """weights=None trains from scratch.

    That is the cheap test of whether the initialisation axis is live at all: if
    ImageNet barely beats random on 707 images x 30 epochs, then a BETTER
    pretraining (in-domain GONG MAE, EO foundation models) has little room
    either, and the days it would cost are not justified.
    """
    if swin_satlas:
        # TorchGeo's Satlas Sentinel-1 Swin-V2 is torchvision-based and cannot
        # go through smp's timm encoders; the wrapper exposes it as one. Only
        # Swin_V2_B has Sentinel-1 weights — the tiny variant is Sentinel-2 only.
        from swin_satlas_encoder import build_swin_satlas_unet
        return build_swin_satlas_unet(in_ch=in_ch, weights=swin_satlas)
    import segmentation_models_pytorch as smp
    kw = {}
    if encoder.startswith("tu-swin"):
        kw["img_size"] = imgsz          # swin ties relative-position bias to size
    if arch == "unet":
        m = smp.Unet(encoder_name=encoder, encoder_weights=weights,
                     in_channels=in_ch, classes=1,
                     decoder_attention_type="scse", **kw)
    elif arch == "upernet":
        m = smp.UPerNet(encoder_name=encoder, encoder_weights=weights,
                        in_channels=in_ch, classes=1, **kw)
    else:
        raise ValueError(arch)
    if grad_ckpt:
        # Measured (vram_probe): SwinV2+UPerNet at 768/batch 2 needs 14.8 GB
        # without this and 6.3 GB with it. On Windows the driver silently spills
        # past the 16 GB card into host RAM instead of raising OOM, so the
        # un-checkpointed run is not merely tight — it is 1513 ms/step against
        # 324 ms/step here. Checkpointing is the FASTER setting, not a trade.
        inner = getattr(m.encoder, "model", m.encoder)
        if not hasattr(inner, "set_grad_checkpointing"):
            raise SystemExit(f"{encoder} does not support gradient checkpointing")
        inner.set_grad_checkpointing(True)
    return m


def load_torchgeo_weights(model, tag, in_ch):
    """Load a TorchGeo Sentinel-1 encoder into an smp model.

    Verified: 317 of the 320 tensors of resnet50 SENTINEL1_ALL_MOCO transfer
    exactly into smp's tu-resnet50 encoder; the only mismatch is conv1, because
    Sentinel-1 is 2-channel (VV, VH) and our frames are 1. That stem is adapted
    by averaging the two source channels and tiling to in_ch, which preserves
    the filter's response scale rather than dropping one polarisation.

    Fair comparison note: the encoder is a resnet50, not our b3, so a
    Sentinel-1 run only isolates the WEIGHTS when it is compared against
    resnet50 + ImageNet — otherwise architecture and pretraining are confounded.
    """
    import torchgeo.models as tgm
    enum, name = {
        "s1-moco": (tgm.ResNet50_Weights.SENTINEL1_ALL_MOCO, "resnet50"),
        "s1-decur": (tgm.ResNet50_Weights.SENTINEL1_ALL_DECUR, "resnet50"),
        "s1-softcon": (tgm.ResNet50_Weights.SENTINEL1_GRD_SOFTCON, "resnet50"),
    }[tag]
    src = getattr(tgm, name)(weights=enum).state_dict()
    inner = getattr(model.encoder, "model", model.encoder)
    tgt = inner.state_dict()
    ok = {k: v for k, v in src.items() if k in tgt and v.shape == tgt[k].shape}
    stem = "conv1.weight"
    if stem in src and stem in tgt and src[stem].shape != tgt[stem].shape:
        # Scale matters here, not just shape: conv1 is followed by a BatchNorm
        # whose running statistics come from the same checkpoint, so a stem that
        # halves the activation scale feeds it inputs it was never normalised
        # for. Measured against the native 2-channel response on a grey frame:
        # averaging the polarisations and dividing by the target channel count
        # gives exactly 0.500 of it, while timm's adapt_input_conv rule gives
        # 0.991 for 3 channels and 1.000 for 1. Using timm's rule.
        w, n_in = src[stem], tgt[stem].shape[1]
        c_src = w.shape[1]
        if n_in == 1:
            ok[stem] = w.sum(dim=1, keepdim=True)
        else:
            rep = -(-n_in // c_src)
            ok[stem] = w.repeat(1, rep, 1, 1)[:, :n_in] * (c_src / float(n_in))
    missing, unexpected = inner.load_state_dict(ok, strict=False)
    log(f"  torchgeo {tag}: loaded {len(ok)}/{len(src)} tensors "
        f"(missing {len(missing)}, unexpected {len(unexpected)})")
    return model


def make_schedule(opt, a, steps_per_epoch):
    """Cosine decay from --lr to --min-lr, optionally with a short warmup.

    Replaces OneCycle for the two-stage recipe: OneCycle RAMPS UP to max_lr,
    which measurably disturbed a pretrained encoder (the first GONG fine-tune
    sagged from 0.6719 at epoch 0 to 0.6743 at epoch 7 while the LR peaked).
    Cosine starts at the requested LR and only decays, so a pretrained
    representation is never pushed harder than asked.
    """
    total = a.epochs * steps_per_epoch
    warm = int(a.warmup_epochs * steps_per_epoch)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(total - warm, 1), eta_min=a.min_lr)
    if warm <= 0:
        return cos
    warmup = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.05, total_iters=warm)
    return torch.optim.lr_scheduler.SequentialLR(
        opt, [warmup, cos], milestones=[warm])


class EarlyStop:
    """Stop when the monitored score has not improved for `patience` epochs."""

    def __init__(self, patience, mode="max"):
        self.patience, self.mode = patience, mode
        self.best, self.bad = None, 0

    def step(self, value):
        better = (self.best is None or
                  (value > self.best if self.mode == "max" else value < self.best))
        if better:
            self.best, self.bad = value, 0
        else:
            self.bad += 1
        return self.bad >= self.patience > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=1536)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--arch", default="unet", choices=["unet", "upernet"])
    ap.add_argument("--encoder", default="timm-efficientnet-b3")
    ap.add_argument("--in-ch", type=int, default=3, choices=[1, 2, 3])
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=0,
                    help="early stopping patience in epochs; 0 disables")
    ap.add_argument("--cosine", action="store_true",
                    help="cosine decay from --lr to --min-lr instead of OneCycle")
    ap.add_argument("--pct-start", type=float, default=0.1,
                    help="OneCycle warmup fraction. The default was chosen for "
                         "training from scratch; a pretrained encoder gets its "
                         "features disturbed by the ramp to max_lr, which showed "
                         "up as a dip at epoch 7 of the first GONG fine-tune.")
    ap.add_argument("--cldice", type=float, default=0.3)
    ap.add_argument("--cldice-warmup", type=int, default=5)
    ap.add_argument("--u-weight", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="fraction of TRAINING IMAGES to keep (all annotator "
                         "entries of a kept image are kept, so the fold "
                         "structure stays honest). Used to measure the data "
                         "scaling curve: if val is still climbing at 100%%, more "
                         "data — including pseudo-labelled GONG frames — is a "
                         "live lever; if it has flattened, it is not.")
    ap.add_argument("--init", default=None,
                    help="checkpoint to start from — stage 2 of the two-stage "
                         "plan loads the GONG-pretrained encoder here and then "
                         "fine-tunes on the real labels, which is what keeps the "
                         "annotation convention coming from truth")
    ap.add_argument("--swin-satlas", default=None,
                    choices=["s1-si", "s1-mi", "imagenet", "none"],
                    help="Satlas Swin-V2-B encoder. 's1-si' is Single-Image "
                         "Sentinel-1, which matches our single-frame task; MI is "
                         "the multi-image variant. 'none' builds the same "
                         "architecture untrained, as the matched control.")
    ap.add_argument("--tg-weights", default=None,
                    choices=["s1-moco", "s1-decur", "s1-softcon"],
                    help="TorchGeo Sentinel-1 SAR encoder weights instead of "
                         "ImageNet. Needs a resnet50 encoder (use "
                         "--encoder tu-resnet50) and must be compared against "
                         "tu-resnet50 + ImageNet to isolate the weights.")
    ap.add_argument("--scratch", action="store_true",
                    help="random init instead of ImageNet (measures how much the "
                         "initialisation axis is worth before investing in a "
                         "better pretraining)")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="encoder gradient checkpointing (see build_model)")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    if a.arch == "upernet" and a.batch < 2:
        # UPerNet's pyramid-pooling branch reduces to 1x1, and BatchNorm cannot
        # compute statistics from a single sample. Trade VRAM with --accum, not
        # with batch 1.
        raise SystemExit("upernet needs --batch >= 2 (use --accum to trade VRAM)")
    name = a.name or f"sem_{a.arch}_{a.encoder.replace('/', '_')}_f{a.fold}_{a.imgsz}_{a.in_ch}ch"
    dest = OUT / name
    dest.mkdir(parents=True, exist_ok=True)

    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"], folds["fold"]))
    entries = gt[["entry_id", "file_name"]].drop_duplicates()
    entries["fold"] = entries["file_name"].map(fold_of)
    tr_e = entries.loc[entries["fold"] != a.fold]
    if a.train_frac < 1.0:
        imgs = np.sort(tr_e["file_name"].unique())
        rng = np.random.default_rng(0)
        keep = set(rng.choice(imgs, max(1, int(round(len(imgs) * a.train_frac))),
                              replace=False))
        tr_e = tr_e[tr_e["file_name"].isin(keep)]
        log(f"  train-frac {a.train_frac}: {len(keep)}/{len(imgs)} images kept")
    tr = list(tr_e[["entry_id", "file_name"]].itertuples(index=False, name=None))
    va = list(entries.loc[entries["fold"] == a.fold,
                          ["entry_id", "file_name"]].itertuples(index=False, name=None))
    log(f"{name}: train {len(tr)}, val {len(va)} (fold {a.fold})")

    DS = {1: OneBandCached, 2: TwoBandCached}.get(a.in_ch, CachedSemantic)
    ds_tr = DS(tr, a.imgsz, train=True, u_weight=a.u_weight)
    ds_va = DS(va, a.imgsz, train=False)
    dl_tr = DataLoader(ds_tr, batch_size=a.batch, shuffle=True, num_workers=3,
                       pin_memory=True, drop_last=True, persistent_workers=True)
    dl_va = DataLoader(ds_va, batch_size=a.batch, shuffle=False, num_workers=2,
                       pin_memory=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = build_model(
        a.arch, a.encoder, a.in_ch, a.imgsz, a.grad_ckpt,
        None if (a.scratch or a.init or a.tg_weights) else "imagenet",
        swin_satlas=None if a.swin_satlas is None else a.swin_satlas)
    if a.tg_weights:
        model = load_torchgeo_weights(model, a.tg_weights, a.in_ch)
    init_note = ""
    if a.init:
        ck = torch.load(ROOT / a.init, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        # Recorded into train.log below, not just stdout: the first GONG run was
        # launched with stdout to /dev/null and the load could then only be
        # confirmed indirectly, from an unusually high epoch-0 score.
        init_note = (f"init from {a.init} (stage={ck.get('stage','?')}, "
                     f"in_ch={ck.get('in_ch')}, missing={len(missing)}, "
                     f"unexpected={len(unexpected)})")
        log("  " + init_note)
        if len(missing) > 20 or len(unexpected) > 20:
            raise SystemExit("init checkpoint does not match this architecture")
    model = model.cuda()
    # NOTE: channels_last measured 4.7x SLOWER for the EfficientNet U-Net
    # (1339 vs 284 ms/step) — depthwise-separable convs hit poor cuDNN kernels.
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    log(f"  {a.arch} / {a.encoder} / in_ch={a.in_ch}: {n_par:.1f}M params")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * (len(dl_tr) // a.accum)
    if a.cosine:
        sched = make_schedule(opt, a, len(dl_tr) // a.accum)
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr,
                                                    total_steps=steps,
                                                    pct_start=a.pct_start)
    stopper = EarlyStop(a.patience, mode="max")
    scaler = torch.amp.GradScaler("cuda")

    # Progress has to be readable from the run directory while the run is in
    # flight — history.json written only at the end left no way to see where a
    # job was, and the live stdout goes to a temp task file.
    logf = open(dest / "train.log", "a", buffering=1)

    def rec(msg):
        log(msg)
        logf.write(msg + "\n")

    # wd used to be a hardcoded "wd=1e-4" here while --wd was already settable,
    # so every run launched with a different decay logged the wrong one.
    rec(f"{name}: {a.arch}/{a.encoder} in_ch={a.in_ch} imgsz={a.imgsz} "
        f"batch={a.batch}x{a.accum} lr={a.lr} wd={a.wd} "
        f"{'cosine' if a.cosine else f'onecycle pct_start={a.pct_start}'} "
        f"min_lr={a.min_lr} patience={a.patience} cldice={a.cldice} "
        f"epochs={a.epochs}")
    if init_note:
        rec("  " + init_note)

    best, hist = -1.0, []
    for ep in range(a.epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(dl_tr):
            x, y = batch[0], batch[1]
            w = batch[2] if a.u_weight != 1.0 else None
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            if w is not None:
                w = w.cuda(non_blocking=True)
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss = dice_bce_loss(logits, y, weight=w)
                if a.cldice > 0 and ep >= a.cldice_warmup:
                    loss = loss + a.cldice * cldice_loss(logits, y)
            scaler.scale(loss / a.accum).backward()
            if (i + 1) % a.accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if a.cosine or sched.last_epoch < steps - 1:
                    sched.step()
            tot += loss.item()
            nb += 1
            if (i + 1) % 150 == 0:
                log(f"  ep{ep} {i+1}/{len(dl_tr)} loss {tot/nb:.4f} ({time.time()-t0:.0f}s)")
        model.eval()
        inter = union = 0.0
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for batch in dl_va:
                x = batch[0].cuda()
                p = torch.sigmoid(model(x)) > 0.5
                yv = batch[1].cuda() > 0.5
                inter += (p & yv).sum().item()
                union += p.sum().item() + yv.sum().item()
        dice = 2 * inter / max(union, 1)
        hist.append(dict(epoch=ep, train_loss=tot / max(nb, 1), val_dice=dice,
                         secs=round(time.time() - t0, 1)))
        rec(f"epoch {ep}: loss {tot/max(nb,1):.4f}  val pixel-Dice {dice:.4f}  "
            f"({time.time()-t0:.0f}s)  peakVRAM "
            f"{torch.cuda.max_memory_allocated()/2**30:.1f}G")
        with open(dest / "history.json", "w") as f:
            json.dump(hist, f, indent=1)
        if stopper.step(dice):
            rec(f"early stop: no improvement for {a.patience} epochs "
                f"(best {stopper.best:.4f})")
            if dice <= best:
                break
        if dice > best:
            best = dice
            torch.save({"model": model.state_dict(), "encoder": a.encoder,
                        "arch": a.arch, "in_ch": a.in_ch, "imgsz": a.imgsz,
                        "fold": a.fold, "val_dice": dice,
                        "channels": "one" if a.in_ch == 1 else "rep"},
                       dest / "best.pt")
        if stopper.bad >= a.patience > 0:
            break
    rec(f"done. best val pixel-Dice {best:.4f} -> {dest/'best.pt'}")
    logf.close()


if __name__ == "__main__":
    main()
