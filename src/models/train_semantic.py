"""Semantic segmentation lane — a COMPLEMENTARY proposal source for the detector.

Why this exists (measured, not assumed): 2528 of 3431 pooled-OOF false negatives
(73.7%) are never proposed by the YOLO detector at any confidence. Every
detector-side knob is saturated (resolution ladder, capacity, mask_ratio, epochs,
NMS). A detector assigns positives by score x IoU, so faint/small filaments that
start with a low score can never become positives — they are simply not learned.
A semantic model has dense per-pixel supervision and no assignment step, so it
sees faint structure. Its weakness (instance separation) is not our problem:
merge errors are only 4.9% of FN.

Targets follow the all-copies finding (E021, +0.0051): every annotator entry is a
separate sample, so the model learns the annotator distribution rather than a
collapsed consensus.

Usage:
  python src/models/train_semantic.py --fold 0 --imgsz 1536 --epochs 30
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament.rle import rles_to_layers

DATA = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
OUT = ROOT / "outputs" / "runs_semantic"
DISK_C, DISK_R = 1023.5, 900.0


def log(*a):
    print(*a, flush=True)


DISK_C, DISK_R = 1023.5, 900.0


def radial_channel(size):
    """r/R_sun per pixel. The disk geometry is FIXED for every image (measured:
    centre 1023.5, R=900), so this is a constant field — it adds no per-image
    information, but a conv stack is translation-equivariant and cannot derive
    absolute position on its own. Near-limb behaviour is where our hallucination
    rate explodes (pure-FP share 0.23 at r<0.3 vs 0.60 at r>=0.9), so telling the
    model where the limb is has a measured target."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    s = 2048.0 / size
    return np.hypot(xx * s - DISK_C, yy * s - DISK_C) / DISK_R


def denoise(img8, d=9, sc=40.0, ss=9.0):
    """Edge-preserving removal of the JPEG's lossy-compression noise.

    Measured against the original GONG FITS on 317 filaments over 14 images: the
    competition JPEG separates a filament from its local background a factor 1.13
    worse than the FITS, and 8-bit quantisation explains almost none of that (the
    step is 0.1% of the local variance, worth 1.0005) -- what is missing is
    compression noise. 80% of the deficit survives the resize to 1536, so the
    network really is fed it.

    A bilateral filter applied at training resolution recovers the whole gap:
    d' 1.598 raw, 1.763 filtered, 1.756 for the FITS itself, and on the
    unreachable rows 1.415 / 1.547 / 1.511. Downloading 661 FITS files and
    retraining five folds would buy less than this call does.

    Bilateral rather than Gaussian because the signal here IS an edge: filaments
    are absorption features a few pixels wide, and a plain low-pass that reaches
    the noise also reaches the filament.
    """
    return cv2.bilateralFilter(img8, d, sc, ss)


def radial_denoise(img8, r0, ramp=0.05):
    """Bilateral-filtered pixels outside r0 (of R_sun), raw inside, blended
    over `ramp` in r so the transition is smooth. Test-time use of exactly
    this on the production model gave +11 reached rows (ledger, radial
    denoise profile); this makes it a TRAINING input."""
    den = denoise(img8)
    r = radial_channel(img8.shape[0])
    w = np.clip((r - r0) / ramp, 0.0, 1.0).astype(np.float32)
    out = img8.astype(np.float32) * (1.0 - w) + den.astype(np.float32) * w
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def local_contrast(img8, ksize=31):
    """High-pass / local-contrast channel: (I - blur(I)) normalised. Filaments are
    absorption features, so their signal is a local darkening; the raw channel
    carries it entangled with the chromospheric background."""
    blur = cv2.GaussianBlur(img8, (ksize, ksize), 0).astype(np.float32)
    hp = img8.astype(np.float32) - blur
    return np.clip(hp / 40.0 + 0.5, 0.0, 1.0)


def local_contrast_norm(img8, ksize=31, k=4.0):
    """The same high-pass, but with the gain set PER FRAME instead of at a fixed 40 grey levels.

    E205/E207 measured the regime the pipeline is blind to: across high-frequency quartiles SQ and
    precision are flat while recall falls 0.623 -> 0.540, and the filaments of 2021-2022 are ~32%
    shallower than those of the cycle-24 maximum. `local_contrast` removes the chromospheric
    background but keeps the ABSOLUTE depth, so a genuinely shallower frame still yields a
    shallower channel: measured on the twelve lowest- and twelve highest-contrast frames, the
    disk-interior pixel sd ratio is 2.42x on raw and still 1.67x through local_contrast.

    Dividing by the frame's own robust scale (the MAD of the high-pass inside r < 0.85, where the
    ground truth lives) removes what is left. The scale is measured on the disk only, so the sky
    outside the limb cannot set it, and k spreads a typical frame across the unit interval.

    This deliberately discards absolute depth from the MODEL's input. The selector keeps its
    absolute evidence features, so the division of labour is: the model finds structure at
    whatever amplitude the epoch provides, the gate still prices it on the pooled scale.
    """
    blur = cv2.GaussianBlur(img8, (ksize, ksize), 0).astype(np.float32)
    hp = img8.astype(np.float32) - blur
    h, w = hp.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    c = (h - 1) / 2.0
    disk = ((xx - c) ** 2 + (yy - c) ** 2) < (0.85 * 900.0 * h / 2048.0) ** 2
    mad = float(np.median(np.abs(hp[disk]))) if disk.any() else 0.0
    scale = max(mad * 1.4826, 1e-3)
    return np.clip(hp / (k * scale) + 0.5, 0.0, 1.0)


class FilamentSemantic(Dataset):
    """Training samples for the semantic lane.

    target='copies' : one sample per annotator entry, binary union mask of that
        entry. Under BCE the model converges in expectation to the per-pixel
        annotator agreement fraction.
    target='soft'   : one sample per PHYSICAL image, target = that agreement
        fraction itself (k/N per pixel). Same Bayes-optimal solution with lower
        gradient variance, and 1154 -> 707 samples.

    Why a soft target rather than agreement-weighting: the metric scores our mask
    against every copy, so adding a pixel drawn by k of N annotators changes mean
    IoU by [k - (N-k)*IoU]/|union| — i.e. it pays whenever k/N > IoU/(1+IoU),
    which at our measured IoU 0.67 is 0.40. The optimal boundary policy is
    therefore INCLUSIVE of disputed pixels, while up-weighting agreed pixels
    would push the model the other way.

    channels:
      'rep'  = [raw, raw, raw]              (ImageNet-style, 2 of 3 wasted)
      'phys' = [raw, local_contrast, r_norm] (same stem, informative channels)
    """

    def __init__(self, entries, gt, imgsz, train=True, channels="rep", target="copies"):
        self.target = target
        self.by_entry = {e: g["segmentation_rle"].tolist()
                         for e, g in gt.groupby("entry_id")}
        if target == "soft":
            by_file = {}
            for (fn,), g in gt.groupby(["file_name"]):
                by_file[fn] = sorted(g["entry_id"].unique())
            files = sorted({fn for _, fn in entries})
            self.entries = [(by_file[fn], fn) for fn in files]
        else:
            self.entries = entries      # list of (entry_id, file_name)
        self.imgsz = imgsz
        self.train = train
        self.channels = channels
        self._radial = radial_channel(imgsz)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        entry_id, fn = self.entries[i]
        img = cv2.imread(str(DATA / fn), cv2.IMREAD_GRAYSCALE)
        if self.target == "soft":
            acc = np.zeros((2048, 2048), np.float32)
            for e in entry_id:                      # entry_id is a list here
                layers = rles_to_layers(self.by_entry[e])
                if len(layers):
                    acc += (layers.sum(axis=0) > 0).astype(np.float32)
            mask = acc / max(len(entry_id), 1)      # per-pixel agreement fraction
        else:
            layers = rles_to_layers(self.by_entry[entry_id])
            mask = (layers.sum(axis=0) > 0).astype(np.float32) if len(layers) else \
                np.zeros((2048, 2048), np.float32)

        if self.train:                                   # physics-safe augmentation
            if np.random.rand() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            ang = np.random.uniform(-10, 10)
            if abs(ang) > 0.5:
                M = cv2.getRotationMatrix2D((1023.5, 1023.5), ang, 1.0)
                img = cv2.warpAffine(img, M, (2048, 2048), flags=cv2.INTER_LINEAR)
                mask = cv2.warpAffine(mask, M, (2048, 2048), flags=cv2.INTER_NEAREST)
            g = np.random.uniform(0.8, 1.25)             # gamma / exposure jitter
            img = np.clip(((img / 255.0) ** g) * 255.0, 0, 255).astype(np.uint8)

        img = cv2.resize(img, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)
        raw = np.ascontiguousarray(img).astype(np.float32) / 255.0
        if self.channels == "phys":
            stack = np.stack([raw, local_contrast(img), self._radial])
        elif self.channels == "physn":
            # the same stack with the high-pass gain set per frame (E208). Measured on the twelve
            # lowest- and twelve highest-contrast frames, the disk-interior pixel sd ratio is
            # 2.42x through raw, 1.67x through local_contrast and 1.02x through this one, i.e.
            # this is the channel that actually removes the regime the pipeline is blind to.
            stack = np.stack([raw, local_contrast_norm(img), self._radial])
        elif self.channels == "mag":
            raise NotImplementedError("channels='mag' needs the cached dataset (--cache)")
        else:
            stack = np.stack([raw, raw, raw])
        x = torch.from_numpy(np.ascontiguousarray(stack)).float()
        y = torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0)
        # 'copies' targets are binary but INTER_AREA made them continuous — re-binarise.
        # 'soft' targets ARE the agreement fraction and must stay continuous.
        return x, y if self.target == "soft" else (y > 0.5).float()


class CachedSemantic(Dataset):
    """Same samples as FilamentSemantic, served from the pre-rendered cache.

    The live dataset paid ~100 ms of CPU per sample (2048^2 JPEG decode, seven
    RLE rasterisations at 2048^2, warp at 2048^2). Here images and masks are
    already at the training size, so augmentation runs at S and the per-sample
    cost is a memmap read plus one warp.

    weight map: pixels covered ONLY by Unidentifiable instances get `u_weight`
    (1.0 disables the mechanism).
    """

    def __init__(self, entries, size, train=True, channels="rep", u_weight=1.0,
                 denoise_input=False, denoise_radial=0.0, faint_weight=0.0, limb_weight=0.0):
        import json as _json
        cdir = ROOT / "data" / "semantic_cache"
        idx = _json.loads((cdir / f"index_{size}.json").read_text())
        # Store PATHS, not memmaps: pickling a np.memmap for a Windows spawn
        # worker materialises the whole array into the pipe (1.7 GB -> OSError 22
        # / truncated pickle, and a full copy in every worker's RSS). Each worker
        # opens its own memmap lazily instead.
        self._cdir, self._size = cdir, size
        self.imgs = self.msks = self.msks_lr = None
        # E162: the magnetogram channels, same row order as the images. Opened lazily
        # like the others; absent on disk -> every frame is treated as missing.
        self.mags = None
        self._mag_path = cdir / f"mag_{size}.npy"
        hp = cdir / f"mag_{size}_has.npy"
        self.aux_mag = False          # set by main() with --aux-mag: return the maps as a target
        self.mag_has = np.load(hp) if ((channels == "mag" or hp.exists()) and hp.exists()) else None
        self.mag_drop = 0.15          # training-time channel dropout, see __getitem__
        self.file_row = {f: i for i, f in enumerate(idx["files"])}
        self.entry_row = {e: i for i, e in enumerate(idx["entry_ids"])}
        self.entries = entries
        self.size = size
        self.train = train
        self.channels = channels
        self.u_weight = u_weight
        # physics-derived members (ledger 18, saf_bilim): faint instances are a real
        # population (Doppler/seeing), not noise; limb instances show their height
        # and are a distinct morphology. Both enter as per-pixel BCE weights.
        self.faint_weight, self.limb_weight = float(faint_weight), float(limb_weight)
        self.want_comps = False      # set by main() when --inst-iou is used
        self.limb_aug, self.limb_aug_max = 0.0, 45.0   # set by main() with --limb-aug
        self.elastic, self.elastic_amp = 0.0, 3.0      # set by main() with --elastic
        self.depth_jitter = 0.0                        # set by main() with --depth-jitter
        self._dj_rng = None                            # its own stream: see --seed
        c = (size - 1) / 2.0
        yy, xx = np.mgrid[0:size, 0:size]
        self._rr = (np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / (900.0 * size / 2048.0)).astype(np.float32)
        self.denoise_input = denoise_input
        self.denoise_radial = float(denoise_radial)
        self._radial = radial_channel(size)

    def __len__(self):
        return len(self.entries)

    def __getstate__(self):
        """Never ship the memmaps through a pickle — see __init__."""
        s = self.__dict__.copy()
        s["imgs"] = s["msks"] = s["msks_lr"] = None
        return s

    def _open(self):
        if self.imgs is None:
            S, d = self._size, self._cdir
            self.imgs = np.load(d / f"images_{S}.npy", mmap_mode="r")
            self.msks = np.load(d / f"masks_{S}.npy", mmap_mode="r")
            self.msks_lr = np.load(d / f"masks_lr_{S}.npy", mmap_mode="r")
        if (self.channels == "mag" or self.aux_mag) and self.mags is None and self._mag_path.exists():
            self.mags = np.load(self._mag_path, mmap_mode="r")

    def _unpack(self, arr, row):
        S = self.size
        return np.unpackbits(arr[row].reshape(S, -1), axis=-1)[:, :S]

    def _physics_weights(self, img, mask, wmap):
        """Multiply the BCE weight of GT pixels by (1 + faint_weight * faintness)
        per connected instance -- faintness = clip((0.15 - contrast) / 0.15, 0, 1),
        contrast = (ring - fg) / ring on the raw image (0.15 is the TP median of
        ledger 18b) -- and by (1 + limb_weight) beyond r 0.8. Computed before
        augmentation; flips and rotations about the centre preserve both."""
        m8 = (mask > 0).astype(np.uint8)
        if self.faint_weight > 0 and m8.any():
            n, lab, st, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
            imf = img.astype(np.float32)
            ring_all = cv2.dilate(m8, np.ones((15, 15), np.uint8)).astype(bool) & ~m8.astype(bool)
            for k in range(1, n):
                if st[k, cv2.CC_STAT_AREA] < 32:
                    continue
                comp = lab == k
                x0, y0, w, h = st[k, cv2.CC_STAT_LEFT], st[k, cv2.CC_STAT_TOP], st[k, cv2.CC_STAT_WIDTH], st[k, cv2.CC_STAT_HEIGHT]
                ya, yb, xa, xb = max(y0 - 8, 0), min(y0 + h + 8, self.size), max(x0 - 8, 0), min(x0 + w + 8, self.size)
                local_ring = ring_all[ya:yb, xa:xb] & cv2.dilate(comp[ya:yb, xa:xb].astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
                if local_ring.sum() < 10:
                    continue
                fg, bg = float(imf[comp].mean()), float(imf[ya:yb, xa:xb][local_ring].mean())
                contrast = (bg - fg) / max(bg, 1.0)
                faint = float(np.clip((0.15 - contrast) / 0.15, 0.0, 1.0))
                if faint > 0:
                    wmap[comp] *= np.float32(1.0 + self.faint_weight * faint)
        if self.limb_weight > 0:
            sel = (m8 > 0) & (self._rr > 0.8)
            if sel.any():
                wmap[sel] *= np.float32(1.0 + self.limb_weight)
        return wmap

    def __getitem__(self, i):
        self._open()
        entry_id, fn = self.entries[i]
        img = np.asarray(self.imgs[self.file_row[fn]])
        mag = None
        if self.channels == "mag" or self.aux_mag:
            row = self.file_row[fn]
            present = self.mags is not None and self.mag_has is not None and self.mag_has[row] == 1
            # channel dropout: a frame without a magnetogram at test time (SDO eclipse
            # seasons, gaps, quality flags) must be in-distribution, so training shows
            # the neutral channels at a fixed rate and whenever the frame has none
            drop = self.train and np.random.rand() < self.mag_drop and not self.aux_mag
            self._mag_present = bool(present)
            if present and not drop:
                mag = np.asarray(self.mags[row]).copy()          # (2, S, S) uint8
            else:
                mag = np.stack([np.full((self.size, self.size), 128, np.uint8),
                                np.full((self.size, self.size), 255, np.uint8)])
        if self.denoise_input:
            # before augmentation, so the warp and the gamma jitter act on the
            # cleaned image exactly as they do on the raw one
            img = denoise(np.ascontiguousarray(img))
        elif self.denoise_radial > 0:
            # same placement; flip and rotation about the disc centre keep r
            img = radial_denoise(np.ascontiguousarray(img), self.denoise_radial)
        mask = self._unpack(self.msks, self.entry_row[entry_id]).astype(np.float32)
        wmap = None
        if self.u_weight != 1.0:
            lr = self._unpack(self.msks_lr, self.entry_row[entry_id]).astype(np.float32)
            wmap = np.where((mask > 0) & (lr == 0), np.float32(self.u_weight),
                            np.float32(1.0))
        if self.faint_weight > 0 or self.limb_weight > 0:
            if wmap is None:
                wmap = np.ones_like(mask, dtype=np.float32)
            wmap = self._physics_weights(img, mask, wmap)

        if self.train and getattr(self, "depth_jitter", 0.0) > 0:
            # E204: the test set weights 2021 at 1.95x and 2022 at 2.28x, and our PQ on
            # 2019-2022 is -0.058 against the rest (P 0.992). Those years are the solar minimum
            # and the rise of cycle 25: the same filaments rendered ~32% shallower (ring contrast
            # 0.078 against 0.114). The recipe's only photometric augmentation is a gamma jitter,
            # which is a global tone curve -- it moves the background with the filament and so
            # leaves the DEPTH of a local darkening almost untouched. This compresses every local
            # darkening toward the large-scale chromospheric background by a random factor, which
            # is what a weaker-contrast epoch looks like.
            #
            # It deliberately does NOT touch the masks. Scaling only labelled pixels would teach
            # "faint smudge => filament" and that is the mechanism that sank --faint-weight
            # (ledger 8p: agreement rises monotonically with model confidence, so up-weighting
            # low-confidence foreground buys one TP and pays the rest as FP). Here labelled and
            # unlabelled structure are compressed alike, so the decision boundary is unchanged
            # and only the amplitude the model must work at moves.
            if self._dj_rng is None:
                # a private stream, so drawing alpha does not shift the flips, rotations, gamma
                # and sample order that the control run sees -- the pair then differs in the
                # jitter alone and the seed floor (ledger 8o: +-42 rows on fold 0) cancels
                self._dj_rng = np.random.default_rng(
                    1000 + (torch.utils.data.get_worker_info().id if torch.utils.data.get_worker_info() else 0))
            alpha = float(self._dj_rng.uniform(1.0 - self.depth_jitter, 1.0))
            if alpha < 0.999:
                sigma = max(3.0, 25.0 * self.size / 2048.0)     # > filament width, < supergranule
                base = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma)
                img = np.clip(base + alpha * (img.astype(np.float32) - base), 0, 255).astype(np.uint8)

        if self.train and getattr(self, "elastic", 0.0) > 0 and np.random.rand() < self.elastic:
            # E177: smooth shape jitter inside the disk, tapered to zero before the limb
            from elastic_aug import elastic as _el
            img, mask, wmap = _el(np.ascontiguousarray(img), mask, wmap, self.size,
                                  np.random.default_rng(), amp_px_2048=self.elastic_amp)
        if self.train and self.limb_aug > 0 and mag is None and np.random.rand() < self.limb_aug:
            # docs/v2 A5: rotate the Sun so central filaments are seen foreshortened near the
            # limb, label and weights follow, far-side pixels get no gradient (limb_aug.py)
            from limb_aug import sphere_rotate
            delta = np.random.uniform(15.0, self.limb_aug_max) * (1.0 if np.random.rand() < 0.5 else -1.0)
            img, mask, wmap = sphere_rotate(np.ascontiguousarray(img), mask, wmap, delta, self.size)
        if self.train:
            if np.random.rand() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
                if wmap is not None:
                    wmap = wmap[:, ::-1]
                if mag is not None:
                    mag = mag[:, :, ::-1]
            ang = np.random.uniform(-10, 10)
            if abs(ang) > 0.5:
                c = (self.size - 1) / 2.0
                M = cv2.getRotationMatrix2D((c, c), ang, 1.0)
                img = cv2.warpAffine(img, M, (self.size, self.size), flags=cv2.INTER_LINEAR)
                mask = cv2.warpAffine(mask, M, (self.size, self.size), flags=cv2.INTER_NEAREST)
                if wmap is not None:
                    wmap = cv2.warpAffine(wmap, M, (self.size, self.size),
                                          flags=cv2.INTER_NEAREST, borderValue=1.0)
                if mag is not None:
                    mag = np.stack([cv2.warpAffine(mag[0], M, (self.size, self.size),
                                                   flags=cv2.INTER_LINEAR, borderValue=128),
                                    cv2.warpAffine(mag[1], M, (self.size, self.size),
                                                   flags=cv2.INTER_LINEAR, borderValue=255)])
            g = np.random.uniform(0.8, 1.25)
            img = np.clip(((img / 255.0) ** g) * 255.0, 0, 255).astype(np.uint8)

        raw = np.ascontiguousarray(img).astype(np.float32) / 255.0
        if self.channels == "phys":
            stack = np.stack([raw, local_contrast(img), self._radial])
        elif self.channels == "physn":
            # the same stack with the high-pass gain set per frame (E208). Measured on the twelve
            # lowest- and twelve highest-contrast frames, the disk-interior pixel sd ratio is
            # 2.42x through raw, 1.67x through local_contrast and 1.02x through this one, i.e.
            # this is the channel that actually removes the regime the pipeline is blind to.
            stack = np.stack([raw, local_contrast_norm(img), self._radial])
        elif self.channels == "mag":
            # signed field centred on zero (polarity is the information) and the PIL
            # distance; the raw image keeps its slot so the first-layer ImageNet filters
            # still see something photometric. Talha caught the first version of this
            # (2026-09-10): mag was loaded and never used, the tensor was [raw, raw, raw].
            assert mag is not None, "channels='mag' but no magnetic channels were built"
            stack = np.stack([raw, (mag[0].astype(np.float32) - 128.0) / 127.0,
                              mag[1].astype(np.float32) / 255.0])
        else:
            stack = np.stack([raw, raw, raw])
        x = torch.from_numpy(np.ascontiguousarray(stack)).float()
        y = torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0)
        w = torch.from_numpy(np.ascontiguousarray(wmap)).float().unsqueeze(0) \
            if wmap is not None else torch.ones_like(y)
        if self.want_comps:
            # instance ids of the (augmented) target, for the per-instance IoU loss
            _, lab = cv2.connectedComponents((mask > 0.5).astype(np.uint8), connectivity=8)
            c = torch.from_numpy(np.ascontiguousarray(lab.astype(np.int32))).long().unsqueeze(0)
            return x, (y > 0.5).float(), w, c
        if self.aux_mag:
            # the magnetic maps as a regression target in the network's output scale,
            # plus a flag so frames without a magnetogram contribute nothing to the loss
            t = np.stack([(mag[0].astype(np.float32) - 128.0) / 127.0, mag[1].astype(np.float32) / 255.0])
            return x, (y > 0.5).float(), w, torch.from_numpy(np.ascontiguousarray(t)).float(),                 torch.tensor(1.0 if self._mag_present else 0.0)
        return x, (y > 0.5).float(), w


def instance_iou_loss(logits, target, comps, eps=1.0):
    """Mean soft-IoU over the GROUND-TRUTH INSTANCES of the batch.

    The metric matches an instance only when its IoU exceeds 0.5 and then pays
    that IoU, and the honest error anatomy (ledger 22b) says 24.5% of disc rows
    sit in the pool with 0.05 < IoU <= 0.5 -- a population the objective never
    addresses, because Dice is computed over the whole image, where one large
    filament dominates hundreds of marginal ones.

    This computes, per connected ground-truth component c,

        softIoU_c = (sum_{i in dilate(c)} p_i y_i) / (sum p_i + sum y_i - sum p_i y_i)

    over a small dilation of the component (so the denominator sees the
    prediction's spill), and averages over components. It is the quantity the
    metric thresholds, expressed differentiably; it does NOT up-weight
    low-confidence foreground the way focal or Tversky do, which is the failure
    mode of ledger 6a-bis -- a component that the model already segments well
    contributes a gradient of nearly zero, and one it segments at IoU 0.4
    contributes a large one, regardless of how confident it is.

    comps: integer map (B,1,H,W), 0 = background, k = instance k of that sample.
    """
    p = torch.sigmoid(logits)
    losses = []
    for b in range(p.shape[0]):
        cb, pb, yb = comps[b, 0], p[b, 0], target[b, 0]
        ids = torch.unique(cb)
        ids = ids[ids > 0]
        if len(ids) == 0:
            continue
        for k in ids:
            m = cb == k
            if m.sum() < 32:
                continue
            # 9x9 dilation of the component, so the prediction's overflow counts
            md = F.max_pool2d(m.float().unsqueeze(0).unsqueeze(0), 9, 1, 4)[0, 0] > 0
            pi, yi = pb[md], yb[md]
            inter = (pi * yi).sum()
            union = pi.sum() + yi.sum() - inter
            losses.append(1.0 - (inter + eps) / (union + eps))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def dice_bce_loss(logits, target, weight=None, eps=1.0):
    """BCE (optionally per-pixel weighted) + soft Dice.

    Only the BCE term is weighted: Dice is a set-overlap ratio, and scaling parts
    of it would distort the overlap rather than express 'this pixel matters less'.
    """
    if weight is None:
        bce = F.binary_cross_entropy_with_logits(logits, target)
    else:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        bce = (bce * weight).sum() / weight.sum().clamp(min=1.0)
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return bce + (1 - num / den).mean()


def soft_erode(x):
    return -F.max_pool2d(-x, 3, 1, 1)


def soft_dilate(x):
    return F.max_pool2d(x, 3, 1, 1)


def soft_skeleton(x, iters=8):
    """Differentiable skeleton (Shit et al., clDice) via iterated morphology."""
    x1 = soft_dilate(soft_erode(x))
    skel = F.relu(x - x1)
    for _ in range(iters):
        x = soft_erode(x)
        x1 = soft_dilate(soft_erode(x))
        skel = skel + F.relu(x - x1) - skel * F.relu(x - x1)
    return skel


def cldice_loss(logits, target, iters=5, eps=1.0, scale=2):
    """1 - clDice: centerline overlap, the topology-aware complement to Dice.

    Targets thin-structure connectivity, where our matched IoU is worst (W1
    thinnest quartile 0.614 vs W4 0.698) and PQ responds linearly
    (dPQ = 0.6181 x mean IoU gain).

    Computed at 1/scale resolution and in the autocast dtype: the soft-skeleton
    is a topology regulariser, not a boundary-precision term, and full-resolution
    fp32 morphology cost 6x the epoch time for no measured extra benefit.
    """
    p = torch.sigmoid(logits)
    if scale > 1:
        p = F.avg_pool2d(p, scale)
        target = F.avg_pool2d(target, scale)
    sp, st = soft_skeleton(p, iters), soft_skeleton(target, iters)
    tprec = (sp * target).sum(dim=(1, 2, 3)) + eps
    tprec = tprec / (sp.sum(dim=(1, 2, 3)) + eps)
    tsens = (st * p).sum(dim=(1, 2, 3)) + eps
    tsens = tsens / (st.sum(dim=(1, 2, 3)) + eps)
    return (1 - 2 * tprec * tsens / (tprec + tsens)).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0,
                    help="E204: the trainer had no seeding at all, so any two runs differed by "
                         "the seed floor as well as by the arm. Seeding makes a control/arm pair "
                         "differ in one thing.")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--folds-file", default="folds_v1.parquet",
                    help="E204: a split other than the production one. folds_year.parquet puts "
                         "2019-2022 alone in fold 0, which turns --fold 0 into a direct "
                         "domain-generalisation read: train on the cycle-24 years, validate on "
                         "the minimum and the rise of cycle 25, where the deployed pipeline is "
                         "-0.058 and which the test set weights about twice.")
    # E144: the nested reference. "--fold 1 --exclude 0" trains on folds 2,3,4 only, so the
    # model has never seen fold 0; four such models average on fold 0 honestly and give the
    # same fold-diverse structure the submitted twelve-member field has.
    ap.add_argument("--exclude", default="", help="extra folds dropped from training entirely")
    ap.add_argument("--imgsz", type=int, default=1536)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--encoder", default="timm-efficientnet-b3")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--name", default=None)
    ap.add_argument("--cldice", type=float, default=0.0, help="clDice loss weight")
    ap.add_argument("--cldice-warmup", type=int, default=5,
                    help="epochs before clDice is switched on (needs a sane mask first)")
    # E163 (privileged information): the student sees H-alpha only, but a second head
    # must regress the two magnetic channels from it. The head is dropped at inference,
    # so the final model reads nothing but the provided frames -- which is what the
    # organiser's answer in 729396 requires ("inference takes place solely on the
    # H-Alpha images"). Knowledge of the field enters through the encoder's features.
    ap.add_argument("--aux-mag", type=float, default=0.0,
                    help="weight of the auxiliary magnetic-regression head (0 disables)")
    ap.add_argument("--channels", default="rep", choices=["rep", "phys", "physn", "mag"],
                    help="rep=[raw,raw,raw]; phys=[raw,local_contrast,r_norm]")
    ap.add_argument("--target", default="copies", choices=["copies", "soft"],
                    help="copies=one sample per annotator entry; soft=agreement fraction")
    ap.add_argument("--cache", action="store_true",
                    help="serve pre-rendered images/masks (much less CPU and RAM)")
    ap.add_argument("--decoder-norm", default="batchnorm",
                    choices=["batchnorm", "instancenorm", "groupnorm", "layernorm", "identity"],
                    help="decoder normalisation. BatchNorm at batch 2 is marginal "
                         "(batch 1 cost -53 net rows); instancenorm is batch-independent "
                         "and the nnU-Net default, at no VRAM cost")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="encoder gradient checkpointing; the BN-safe way to fit "
                         "a run in VRAM (prefer this over --batch 1 --accum 2)")
    ap.add_argument("--accum", type=int, default=1,
                    help="gradient accumulation steps; effective batch = "
                         "batch x accum. Use with --batch 1 when a run does "
                         "not fit in VRAM")
    ap.add_argument("--decoder-channels", default=None,
                    help="comma-separated decoder stage widths, default "
                         "256,128,64,32,16; the last value is the "
                         "full-resolution stage and is the narrowest layer "
                         "in the network")
    ap.add_argument("--workers", type=int, default=None,
                    help="dataloader workers; default 3 cached / 4 live")
    ap.add_argument("--denoise", action="store_true",
                    help="bilateral-filter the input; see denoise() for the "
                         "FITS measurement that motivates it")
    ap.add_argument("--inst-iou", type=float, default=0.0,
                    help="weight of the per-instance soft-IoU term: the quantity the metric "
                         "thresholds, averaged over ground-truth instances not over the image")
    ap.add_argument("--faint-weight", type=float, default=0.0,
                    help="physics member: BCE weight of a GT instance grows with its faintness, "
                         "1 + W * clip((0.15 - contrast)/0.15, 0, 1); 0 = off")
    ap.add_argument("--limb-weight", type=float, default=0.0,
                    help="physics member: BCE weight of GT pixels beyond r 0.8 is multiplied by (1 + L); 0 = off")
    ap.add_argument("--u-weight", type=float, default=1.0,
                    help="BCE weight for pixels covered only by Unidentifiable instances")
    ap.add_argument("--denoise-radial", type=float, default=0.0,
                    help="bilateral-denoise only OUTSIDE this r/R_sun (raw inside, 0.05 "
                         "ramp); 0 = off. Test-time use outside 0.8 gave +11 rows on "
                         "the production model -- the only positive limb-side signal")
    ap.add_argument("--decoder-att", default="scse", choices=["scse", "coordatt"],
                    help="decoder attention: production scSE, or Coordinate Attention (coord_att.py)")
    ap.add_argument("--decoder-act", default="relu", choices=["relu", "silu", "gelu"],
                    help="decoder activation; the encoder is already SiLU, smp's decoder "
                         "blocks are ReLU. A second-order change, run as an A100 filler")
    ap.add_argument("--saaf", action="store_true",
                    help="SAAF-lite: learnable per-channel scale + spatial gate on the "
                         "up-sampled map BEFORE it meets the skip (docs/24 D1b; ledger "
                         "§8j found scSE's spatial gate ~0.01 at the finest stage)")
    ap.add_argument("--arch", default="unet", choices=["unet", "unetpp", "upernet"],
                    help="E187: UnetPlusPlus was never tried here. Its nested dense skips act "
                         "exactly at the stages E171 found carry the load (384 and 96 px), and "
                         "it costs 0.5M parameters over Unet")
    ap.add_argument("--blurpool", action="store_true",
                    help="E181: anti-aliased downsampling -- every stride-2 encoder conv becomes "
                         "conv(stride 1) + BlurPool(stride 2), so the subsampling is low-passed. "
                         "The encoder has no pooling to fix; this fixes the strided convs instead")
    ap.add_argument("--decoder-interp", default="nearest", choices=["nearest", "bilinear", "bicubic"],
                    help="E180: smp upsamples the decoder with NEAREST by default, so four "
                         "doublings from 96 px quantise every boundary to a coarse grid before "
                         "the convolutions can refine it. Free to change; the encoder has no "
                         "pooling at all (five strided convs), so this is the pipeline's one "
                         "remaining lossy resampling inside the network")
    ap.add_argument("--depth-jitter", type=float, default=0.0,
                    help="E204: compress every local darkening toward the large-scale background "
                         "by alpha ~ U(1-A, 1). A=0.35 spans the measured late-year regime "
                         "(ring contrast 0.078 against 0.114 = 0.68x). 0 = off")
    ap.add_argument("--elastic", type=float, default=0.0,
                    help="E177: probability of elastic shape jitter inside the disk")
    ap.add_argument("--elastic-amp", type=float, default=3.0,
                    help="displacement sd in pixels at 2048; 3 moves a mask to IoU 0.73 of itself, inside the annotators' own 0.629 median agreement (elastic_aug.py)")
    ap.add_argument("--ctx-stage", type=int, default=-1,
                    help="E174: attach a dilated-context module to this encoder skip (4 = the "
                         "96 px stage that E171 found carries the faint population); -1 = off")
    ap.add_argument("--limb-aug", type=float, default=0.0,
                    help="docs/v2 A5: probability of the sphere-rotation centre-to-limb augmentation")
    ap.add_argument("--limb-aug-max-deg", type=float, default=45.0)
    ap.add_argument("--init", default=None,
                    help="WA-06: start from these weights (pretrain_mim.py's mim.pt) instead of "
                         "ImageNet; the segmentation head is left at its fresh init")
    ap.add_argument("--hf-film", action="store_true",
                    help="E213: condition every encoder skip on the frame's disc hf (FiLM, src/models/"
                         "hf_film.py); starts as the identity, so the pair differs only in the conditioning")
    ap.add_argument("--balance-era", type=float, default=0.0,
                    help="E214: sample training entries with weight n_era^-POWER over the solar-cycle eras "
                         "<=2015 / 2016-2018 / >=2019 (0 = uniform shuffle, 0.5 = square-root balance, "
                         "1 = every era equally often)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <run>/last.pt if it exists (Colab kills idle runtimes; "
                         "a kill then costs one epoch, not the run)")
    a = ap.parse_args()
    import random as _random
    _random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    name = a.name or f"sem_{a.encoder}_f{a.fold}_{a.imgsz}"
    dest = OUT / name
    dest.mkdir(parents=True, exist_ok=True)

    import segmentation_models_pytorch as smp

    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    folds = pd.read_parquet(ROOT / "data" / "splits" / a.folds_file)
    fold_of = dict(zip(folds["file_name"], folds["fold"]))
    entries = gt[["entry_id", "file_name"]].drop_duplicates()
    entries["fold"] = entries["file_name"].map(fold_of)
    drop = {int(x) for x in a.exclude.split(",") if x.strip() != ""}
    keep = (entries["fold"] != a.fold) & ~entries["fold"].isin(drop)
    tr = list(entries.loc[keep, ["entry_id", "file_name"]].itertuples(index=False, name=None))
    va = list(entries.loc[entries["fold"] == a.fold, ["entry_id", "file_name"]].itertuples(index=False, name=None))
    log(f"train entries {len(tr)}, val entries {len(va)} (fold {a.fold}"
        + (f", excluding fold(s) {sorted(drop)})" if drop else ")"))

    if a.cache:
        ds_tr = CachedSemantic(tr, a.imgsz, train=True, channels=a.channels,
                               u_weight=a.u_weight, faint_weight=a.faint_weight, limb_weight=a.limb_weight, denoise_input=a.denoise,
                               denoise_radial=a.denoise_radial)
        ds_va = CachedSemantic(va, a.imgsz, train=False, channels=a.channels,
                               denoise_input=a.denoise, denoise_radial=a.denoise_radial)
    else:
        ds_tr = FilamentSemantic(tr, gt, a.imgsz, train=True, channels=a.channels,
                                 target=a.target)
        # validation always uses per-entry binary masks so the metric stays
        # comparable across target strategies
        ds_va = FilamentSemantic(va, gt, a.imgsz, train=False, channels=a.channels,
                                 target="copies")
    # cached samples are cheap; fewer workers = less RAM. Overridable because this
    # box shares its 31 GiB with other work and has already lost a run to an
    # allocation failure at 99% -- when memory is tight, buy it back here.
    nw = a.workers if a.workers is not None else (3 if a.cache else 4)
    if a.inst_iou > 0:
        ds_tr.want_comps = True
        log(f"per-instance soft-IoU term on, weight {a.inst_iou}")
    sampler = None
    if a.balance_era > 0:
        # E214 (Talha, 2026-09-25: "eğitim datası dengeli olmalı"). The training set is balanced
        # by station and by hf, but not by solar-cycle phase: 2019-2022 are 8.9 % of the images
        # and 3.9 % of the filaments, score -0.058 PQ against the rest (ledger 43), and the test
        # set weights them about 2x. This draws entries with weight n_era^-POWER, so the rare era
        # is seen more often without changing a single pixel of any sample.
        era = np.array([0 if int(fn[:4]) <= 2015 else 1 if int(fn[:4]) <= 2018 else 2 for _, fn in tr])
        counts = np.bincount(era, minlength=3).astype(float)
        wts = counts[era] ** -a.balance_era
        g = torch.Generator().manual_seed(a.seed)
        sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(wts, dtype=torch.double),
                                                         num_samples=len(tr), replacement=True, generator=g)
        share = [float(wts[era == k].sum() / wts.sum()) for k in range(3)]
        log(f"era-balanced sampling, power {a.balance_era}: entries per era {counts.astype(int).tolist()}, "
            f"share drawn {[round(s, 3) for s in share]} (uniform would be "
            f"{[round(c / counts.sum(), 3) for c in counts]})")
    dl_tr = DataLoader(ds_tr, batch_size=a.batch, shuffle=sampler is None, sampler=sampler,
                       num_workers=nw, pin_memory=True, drop_last=True, persistent_workers=True,
                       prefetch_factor=2)
    dl_va = DataLoader(ds_va, batch_size=a.batch, shuffle=False, num_workers=2,
                       pin_memory=True)

    # Fixed input shape lets cuDNN pick its algorithms once; TF32 is free accuracy
    if a.aux_mag > 0:
        assert a.cache, "--aux-mag needs the cached dataset"
        ds_tr.aux_mag = ds_va.aux_mag = True
    if a.elastic > 0:
        assert a.cache, "--elastic needs the cached dataset"
        ds_tr.elastic, ds_tr.elastic_amp = a.elastic, a.elastic_amp
        ds_tr.depth_jitter = a.depth_jitter
        log(f"elastic shape jitter p={a.elastic}, sd {a.elastic_amp} px at 2048, tapered before the limb")
    if a.limb_aug > 0:
        assert a.cache, "--limb-aug needs the cached dataset"
        ds_tr.limb_aug, ds_tr.limb_aug_max = a.limb_aug, a.limb_aug_max_deg
        log(f"sphere-rotation limb augmentation p={a.limb_aug}, |delta| in 15..{a.limb_aug_max_deg} deg")
    # we do not need. NOTE: channels_last was measured 4.7x SLOWER here
    # (1339 vs 284 ms/step) — EfficientNet is depthwise-separable and cuDNN's
    # channels_last depthwise kernels are poor for it. Do not re-add it.
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Decoder width is the one capacity axis this project has never swept. Nine
    # ENCODERS have been tried and every larger one lost at matched resolution
    # (b3 13.2M 0.6981, effnetv2-b3 0.6930, swin-base 0.6921, resnet50 0.6913,
    # convnext-tiny 0.6889 at 1536 against b3's 0.7019) -- with 707 images at
    # 0.42% foreground the constraint is data, not encoder capacity.
    #
    # The decoder is a different argument. It is 2.5M of the model's 13.2M, and
    # smp's default (256,128,64,32,16) makes the FULL-RESOLUTION stage the
    # narrowest layer in the network at 16 channels. That stage is where a
    # two-pixel-wide filament is either resolved or lost, and the population this
    # project keeps failing on is exactly thin and faint. Widening the last two
    # stages to (48,32) costs 0.1M parameters, so it adds capacity where the data
    # constraint bites least: per-pixel refinement rather than semantic
    # abstraction.
    dec = tuple(int(c) for c in a.decoder_channels.split(",")) \
        if a.decoder_channels else (256, 128, 64, 32, 16)
    # smp accepts a dict {"type": ..., **kwargs} for the decoder norm. The
    # string "instancenorm" builds InstanceNorm2d with affine=False, whose
    # weight is None, and initialize_decoder then fails on constant_(). The
    # affine form also keeps a per-channel scale/shift, which BatchNorm had.
    # smp 0.5.0 has no groupnorm option: build BatchNorm and swap afterwards
    gn_groups = 8 if a.decoder_norm == "groupnorm" else 0
    dnorm = ({"type": "instancenorm", "affine": True} if a.decoder_norm == "instancenorm"
             else "batchnorm" if gn_groups else a.decoder_norm)
    n_out = 1 + (2 if a.aux_mag > 0 else 0)
    if a.arch == "unetpp":
        model = smp.UnetPlusPlus(encoder_name=a.encoder, encoder_weights="imagenet",
                                 in_channels=3, classes=n_out, decoder_attention_type="scse",
                                 decoder_channels=dec, decoder_use_norm=dnorm).cuda()
    elif a.arch == "upernet":
        model = smp.UPerNet(encoder_name=a.encoder, encoder_weights="imagenet",
                            in_channels=3, classes=n_out).cuda()
    else:
        model = smp.Unet(encoder_name=a.encoder, encoder_weights="imagenet",
                         in_channels=3, classes=n_out, decoder_attention_type="scse",
                         decoder_channels=dec, decoder_use_norm=dnorm,
                         decoder_interpolation=a.decoder_interp).cuda()
    if gn_groups:
        from decoder_norm import swap_decoder_bn_to_gn
        log(f"decoder GroupNorm({gn_groups}): {swap_decoder_bn_to_gn(model, gn_groups)} BatchNorm layers swapped")
        model = model.cuda()
    if a.decoder_att == "coordatt":
        from coord_att import swap_decoder_scse_to_coordatt
        log(f"decoder attention coordatt: {swap_decoder_scse_to_coordatt(model)} scSE modules swapped")
        model = model.cuda()   # the new modules were built on the CPU
    if a.decoder_act != "relu":
        from decoder_norm import swap_decoder_act
        log(f"decoder activation {a.decoder_act}: {swap_decoder_act(model, a.decoder_act)} ReLU layers swapped")
        model = model.cuda()
    if a.saaf:
        # must precede the checkpoint wrapper below, which wraps blk.forward
        from saaf import apply_saaf
        log(f"SAAF-lite: {apply_saaf(model)} pre-concatenation gates attached")
        model = model.cuda()
    if a.grad_ckpt:
        # The way to buy VRAM WITHOUT touching BatchNorm. Dropping to batch 1
        # with --accum keeps the gradient honest but leaves BN computing its
        # statistics on a single sample, and that alone cost the baseline
        # decoder 0.0135 val pixel-Dice at 1536 (0.7019 -> 0.6884) -- enough to
        # swamp any architecture effect being measured. Encoder checkpointing
        # recomputes activations in the backward pass at batch 2 instead.
        # The arch harness measured this on this box as the FASTER setting,
        # not a trade: the Windows driver spills past the card into host RAM
        # rather than raising OOM, and a spilled step is 4-5x slower than a
        # recomputed one.
        inner = getattr(model.encoder, "model", model.encoder)
        if not hasattr(inner, "set_grad_checkpointing"):
            raise SystemExit(f"{a.encoder} does not support gradient checkpointing")
        inner.set_grad_checkpointing(True)
        # The encoder alone is not enough when the DECODER is what grew: with a
        # widened full-resolution stage the extra activations live in the
        # decoder blocks, and encoder checkpointing left (48,32) spilling at
        # 14.6 GB and 25 s/step. Wrapping each DecoderBlock too brings
        # (40,24) to 12.1 GB and 535 ms/step at batch 2 on this card.
        from torch.utils.checkpoint import checkpoint as _ckpt
        for blk in model.decoder.blocks:
            blk.forward = (lambda f: (lambda *args, **kw:
                           _ckpt(f, *args, use_reentrant=False, **kw)))(blk.forward)
        log("encoder + decoder gradient checkpointing ON")
    if a.blurpool:
        from blurpool import apply_blurpool
        log(f"anti-aliased downsampling: {apply_blurpool(model)} strided convs low-passed")
        model = model.cuda()
    if a.ctx_stage >= 0:
        from stage_ctx import attach_stage_context
        n = attach_stage_context(model, a.ctx_stage)
        log(f"dilated context on encoder skip {a.ctx_stage}: {n / 1e6:.2f}M parameters, zero-initialised")
        model = model.cuda()
    if a.init:
        # WA-06: the pretext model is the same architecture at its defaults; everything but
        # the head transfers by name. Missing/unexpected keys are logged, not hidden, so an
        # architecture mismatch (a widened decoder, GroupNorm) shows up in the log at once.
        sd = torch.load(a.init, map_location="cpu", weights_only=False)["model"]
        sd = {k: v for k, v in sd.items() if not k.startswith("segmentation_head.")}
        res = model.load_state_dict(sd, strict=False)
        log(f"init from {a.init}: {len(sd)} tensors loaded, {len(res.missing_keys)} missing "
            f"({', '.join(res.missing_keys[:3])}{'...' if len(res.missing_keys) > 3 else ''}), "
            f"{len(res.unexpected_keys)} unexpected")
    if a.hf_film:
        assert a.cache, "--hf-film reads its normalisation from the cached training frames"
        from hf_film import attach, fit_norm
        film = attach(model, a.imgsz).cuda()
        ds_tr._open()
        rows = sorted({ds_tr.file_row[fn] for _, fn in tr})
        mu, sd, n, med = fit_norm(film, ds_tr.imgs, rows)
        log(f"hf-FiLM on {len(film.channels)} encoder skips ({sum(film.channels)} channels): "
            f"log hf mu {mu:.3f} sd {sd:.3f} over {n} training frames (median hf {med:.0f} at {a.imgsz})")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    # Gradient accumulation trades wall-clock for VRAM at a fixed effective
    # batch. The wide-decoder arm at 1536 needs it: its full-resolution
    # activations pushed a 16.3 GB card to 15.2 GB and the driver spilled to
    # system memory -- 30 s per step, 100x normal. With accum=2 and batch=1 the
    # effective batch is unchanged and the activation footprint halves.
    # OneCycleLR counts OPTIMISER steps, so its horizon is divided accordingly.
    accum = max(int(a.accum), 1)
    n_opt_steps = (len(dl_tr) + accum - 1) // accum
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * n_opt_steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")

    best = -1.0
    hist = []
    start_ep = 0
    # Colab dropped an A100 runtime 70 min into D1 (browser idle, not GPU idle)
    # and best.pt died with the VM. last.pt carries the full optimiser state so
    # a fresh VM continues the same schedule: OneCycleLR's step counter is in
    # its state_dict, so the LR resumes where it stopped, and the clDice
    # warmup keys on the absolute epoch below. Only the data order differs
    # (loader RNG is not restored), which is noise, not a schedule change.
    last_pt = dest / "last.pt"
    if a.resume and last_pt.exists():
        ck = torch.load(last_pt, map_location="cuda", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        best, hist, start_ep = ck["best"], ck["hist"], ck["epoch"] + 1
        log(f"resumed from last.pt at epoch {start_ep} (best so far {best:.4f})")
    elif a.resume:
        log("--resume given but no last.pt in the run directory: starting fresh")
    if start_ep >= a.epochs:
        log(f"nothing to do: last.pt already at epoch {start_ep - 1} of {a.epochs}")

    def save_atomic(obj, path):
        # write-then-rename: a kill mid-write, or a concurrent copy over the
        # tunnel, never sees a half-written checkpoint
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(obj, tmp)
        tmp.replace(path)

    opt.zero_grad(set_to_none=True)
    for ep in range(start_ep, a.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for i, batch in enumerate(dl_tr):
            x, y = batch[0], batch[1]
            w = batch[2] if len(batch) > 2 and (a.u_weight != 1.0 or a.faint_weight > 0 or a.limb_weight > 0) else None
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            if w is not None:
                w = w.cuda(non_blocking=True)
            with torch.amp.autocast("cuda"):
                out = model(x)
                logits = out[:, :1]
                loss = dice_bce_loss(logits, y, weight=w)
                if a.aux_mag > 0:
                    tgt = batch[3].cuda(non_blocking=True)
                    has = batch[4].cuda(non_blocking=True).view(-1, 1, 1, 1)
                    aux = ((out[:, 1:3].float() - tgt) ** 2 * has).mean() / has.mean().clamp_min(1e-3)
                    loss = loss + a.aux_mag * aux
                if a.inst_iou > 0 and len(batch) > 3:
                    loss = loss + a.inst_iou * instance_iou_loss(logits, y, batch[3].cuda(non_blocking=True))
                if a.cldice > 0 and ep >= a.cldice_warmup:
                    loss = loss + a.cldice * cldice_loss(logits, y)
            # the loss is averaged over the accumulation window so the gradient
            # magnitude matches a single step at the effective batch
            scaler.scale(loss / accum).backward()
            last = (i + 1) == len(dl_tr)
            if (i + 1) % accum == 0 or last:
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
            tot += loss.item()
            if (i + 1) % 100 == 0:
                log(f"  ep{ep} {i+1}/{len(dl_tr)} loss {tot/(i+1):.4f} "
                    f"({time.time()-t0:.0f}s)")
        # validation: pixel dice at 0.5
        model.eval()
        inter = union = 0.0
        # the same Dice split by solar-cycle era (>= 2019 against earlier), so a split that
        # validates on both can read a gain on the late years and a cost on the rest separately
        late_va = np.array([int(fn[:4]) >= 2019 for _, fn in va])
        sl = {"late": [0.0, 0.0], "early": [0.0, 0.0]}
        k0 = 0
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for batch in dl_va:
                x, y = batch[0], batch[1]
                p = torch.sigmoid(model(x.cuda())[:, :1]) > 0.5
                y = y.cuda() > 0.5
                inter += (p & y).sum().item()
                union += p.sum().item() + y.sum().item()
                for j in range(len(x)):
                    s = sl["late" if late_va[k0 + j] else "early"]
                    s[0] += (p[j] & y[j]).sum().item()
                    s[1] += p[j].sum().item() + y[j].sum().item()
                k0 += len(x)
        dice = 2 * inter / max(union, 1)
        d_late = 2 * sl["late"][0] / sl["late"][1] if sl["late"][1] else float("nan")
        d_early = 2 * sl["early"][0] / sl["early"][1] if sl["early"][1] else float("nan")
        hist.append(dict(epoch=ep, train_loss=tot / len(dl_tr), val_dice=dice,
                         val_dice_late=d_late, val_dice_early=d_early))
        log(f"epoch {ep}: loss {tot/len(dl_tr):.4f}  val pixel-Dice {dice:.4f}  "
            f"(late {d_late:.4f}, early {d_early:.4f})  ({time.time()-t0:.0f}s)")
        if dice > best:
            best = dice
            # decoder_channels must travel with the weights: load_model rebuilds
            # the architecture from this dict, and a checkpoint that does not say
            # how wide its decoder was cannot be loaded at all
            save_atomic({"model": model.state_dict(), "encoder": a.encoder,
                         "imgsz": a.imgsz, "fold": a.fold, "val_dice": dice,
                         "channels": a.channels, "classes": 1 + (2 if a.aux_mag > 0 else 0),
                         "ctx_stage": a.ctx_stage, "decoder_interp": a.decoder_interp, "blurpool": bool(a.blurpool), "arch": a.arch,
                         "decoder_channels": list(dec),
                         "decoder_use_norm": dnorm, "saaf": bool(a.saaf), "decoder_gn": gn_groups, "denoise_radial": a.denoise_radial, "decoder_act": a.decoder_act, "decoder_att": a.decoder_att, "faint_weight": a.faint_weight, "limb_weight": a.limb_weight, "inst_iou": a.inst_iou, "hf_film": bool(a.hf_film), "balance_era": a.balance_era}, dest / "best.pt")
        # every epoch: resumable state + the history so far, both atomic, so
        # the local watcher can copy them at any moment
        save_atomic({"model": model.state_dict(), "opt": opt.state_dict(),
                     "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                     "epoch": ep, "best": best, "hist": hist,
                     "encoder": a.encoder, "imgsz": a.imgsz, "fold": a.fold,
                     "channels": a.channels, "classes": 1 + (2 if a.aux_mag > 0 else 0), "decoder_channels": list(dec),
                     "decoder_use_norm": dnorm, "saaf": bool(a.saaf), "decoder_gn": gn_groups, "denoise_radial": a.denoise_radial, "decoder_act": a.decoder_act, "decoder_att": a.decoder_att, "faint_weight": a.faint_weight, "limb_weight": a.limb_weight, "inst_iou": a.inst_iou, "hf_film": bool(a.hf_film), "balance_era": a.balance_era}, last_pt)
        tmp = dest / "history.json.tmp"
        with open(tmp, "w") as f:
            json.dump(hist, f, indent=1)
        tmp.replace(dest / "history.json")
    log(f"done. best val pixel-Dice {best:.4f} -> {dest/'best.pt'}")


if __name__ == "__main__":
    main()
