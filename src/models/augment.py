"""Augmentation levels for the fine-tune stage, and why each one is here.

The recipe this replaces is light for a 707-image dataset: horizontal flip, +-10
degrees of rotation, gamma 0.8-1.25. Nothing else. With heavy overfitting risk on
that little data, more is worth measuring.

Two measurements shape everything below, and both overturned a first guess.

CALIBRATION (src/eda/aug_calibration.py). Ranges are set from the six GONG
stations' own spread, because that is variation the test set genuinely contains.
The stations barely differ in disk level (108.9-113.0 DN) -- the exposure loop
normalises it -- but differ by 46% in contrast (31.0 at T to 45.2 at M), because
transparency and seeing hit an absorption feature's contrast directly. So my
hand-picked +-12 DN brightness was 3.3x too WIDE while +-0.10 contrast was 2.4x
too NARROW: opposite errors, which is what accidental ranges look like.

RESOLUTION (src/eda/aug_onthefly.py). The cache is pre-downsampled from 2048, so
every augmentation lands on already-downsampled pixels while inference sees
2048 -> S. For an operation with an intrinsic PIXEL scale, doing it at S is not
the same as doing it at 2048 and downsampling:

  JPEG    DROPPED, measured to be a fiction. The competition file carries real
          q~90 blocking at 2048 (blockiness 1.1394), but the area-downsample
          erases it: 1.0025 at 1536, 1.0201 at 768. The model never meets JPEG
          blocking at any training resolution. Encoding at 768 instead produced
          1.1773 -- MORE blocking than the native file, at a 2.67x too coarse
          scale, changing the image 9.1x more than encoding at 2048 would have.
          docs/02 already said "avoid re-encoding augmentation"; that was right
          and I overrode it with a plausible argument about the GONG archive's
          smaller files. The archive difference is real but lives at 2048 and is
          destroyed before it reaches the model.
  noise   SCALED. Noise injected at 2048 arrives at 768 attenuated 2.5-3.0x, the
          expected sqrt(N) for a 2.67x2.67 area average. Injecting at S therefore
          has to be divided by 2048/S to mean sensor noise rather than an
          arbitrary number.
  blur    SCALED, same argument: sigma is in pixels.
  rotation KEPT AS IS, measured harmless. Rotating at 768 costs 0.71 DN of the
          17.0 DN filament contrast; rotating at 2048 first costs 1.14 DN (two
          resamplings instead of one, so the cache path is marginally SHARPER).
          Both are small against the 11.3 DN that separates two different frames.

Level design. `medium` matches nature: every range is the stations' measured
spread. `heavy` deliberately EXCEEDS it by 1.5x. That contrast is the experiment
worth running -- augmentation does not have to stay inside the test distribution
to regularise, and whether over-augmenting beats matching is exactly what a
707-image dataset makes worth asking.

What stays out, because this dataset's geometry breaks under it:

  translation / large scale   The disk sits at a FIXED centre (1023.5, 1023.5),
                              R about 900, and the pipeline masks at r <= 0.97.
  rotation beyond ~25 deg     Filaments follow the butterfly distribution in
                              latitude.
  colour ops                  Single-channel greyscale.
"""

import cv2
import numpy as np

LEVELS = ("light", "medium", "heavy")

# Measured natural spread across the six stations, expressed at 768 (see
# aug_calibration.py). `heavy` multiplies the magnitudes by HEAVY_GAIN.
NAT_CONTRAST = 0.22          # relative, natural +-0.243
NAT_BRIGHT = 5.0             # DN, natural +-3.6
NAT_SIGMA = (0.2, 0.7)       # px at 768; sigma 0.5 spans the sharpest->blurriest decile
NAT_NOISE = (2.0, 7.0)       # DN at NATIVE 2048, divided by the downsample below
HEAVY_GAIN = 1.5
NATIVE = 2048.0


def apply_aug(img, mask, wmap, size, level, rng, wmap_border=1.0):
    """img uint8 HxW, mask float HxW, wmap float HxW or None.

    `size` is the training resolution; the pixel-scale operations are expressed
    relative to the native 2048 frame so that 768 and 1536 mean the same
    physical thing.

    `wmap_border` is what rotation puts outside the source frame in the third
    plane. It defaults to 1.0 because that plane was introduced for a per-pixel
    WEIGHT map, whose neutral value is 1. Any other payload needs 0.0, and the
    difference is not cosmetic: a rotated 768 frame has a large border region,
    so a spine or skeleton target carried at the default flooded 6.4% of the
    frame with foreground and collapsed its containment inside the mask from
    1.00 to 0.075. Two experiments were run against that corrupted target
    before the check that caught it -- measuring CONTAINMENT rather than mere
    concentration.
    """
    if level == "none":
        return img, mask, wmap

    scale = size / NATIVE            # 0.375 at 768, 0.75 at 1536
    gain = HEAVY_GAIN if level == "heavy" else 1.0

    # --- geometry ---------------------------------------------------------
    if rng.random() < 0.5:
        img, mask = img[:, ::-1], mask[:, ::-1]
        if wmap is not None:
            wmap = wmap[:, ::-1]
    if level != "light" and rng.random() < 0.5:
        img, mask = img[::-1], mask[::-1]
        if wmap is not None:
            wmap = wmap[::-1]

    max_ang = 10.0 if level == "light" else 20.0
    ang = rng.uniform(-max_ang, max_ang)
    if abs(ang) > 0.5:
        c = (size - 1) / 2.0
        M = cv2.getRotationMatrix2D((c, c), ang, 1.0)
        img = cv2.warpAffine(np.ascontiguousarray(img), M, (size, size),
                             flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(np.ascontiguousarray(mask), M, (size, size),
                              flags=cv2.INTER_NEAREST)
        if wmap is not None:
            wmap = cv2.warpAffine(np.ascontiguousarray(wmap), M, (size, size),
                                  flags=cv2.INTER_NEAREST,
                                  borderValue=float(wmap_border))
    img = np.ascontiguousarray(img)

    # --- photometric ------------------------------------------------------
    g = rng.uniform(0.8, 1.25)
    img = np.clip(((img / 255.0) ** g) * 255.0, 0, 255).astype(np.uint8)

    if level != "light":
        if rng.random() < 0.5:                      # exposure / transparency
            a = 1.0 + rng.uniform(-NAT_CONTRAST, NAT_CONTRAST) * gain
            b = rng.uniform(-NAT_BRIGHT, NAT_BRIGHT) * gain
            img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        if rng.random() < 0.35:                     # atmospheric seeing
            sig = rng.uniform(*NAT_SIGMA) * gain * (size / 768.0)
            img = cv2.GaussianBlur(img, (0, 0), float(sig))
        if rng.random() < 0.30:                     # sensor / sky noise
            sd = rng.uniform(*NAT_NOISE) * gain * scale
            img = np.clip(img.astype(np.float32) + rng.normal(0, sd, img.shape),
                          0, 255).astype(np.uint8)

    if level == "heavy":
        if rng.random() < 0.35:                     # telescope tracking jitter
            k = int(rng.choice([3, 5, 7])) * max(1, int(round(size / 768.0)))
            k += 1 - (k % 2)                        # keep it odd
            ker = np.zeros((k, k), np.float32)
            ker[k // 2, :] = 1.0 / k
            M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5),
                                        rng.uniform(0, 180), 1.0)
            ker = cv2.warpAffine(ker, M, (k, k))
            s = ker.sum()
            if s > 1e-6:
                img = cv2.filter2D(img, -1, ker / s)

    return img, np.ascontiguousarray(mask), \
        (np.ascontiguousarray(wmap) if wmap is not None else None)
