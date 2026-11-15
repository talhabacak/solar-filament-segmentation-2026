"""E177 -- elastic deformation, the one augmentation family this project never tried.

The augmentation audit (src/models/augment.py) calibrated brightness, contrast, noise and blur
to the six GONG stations' measured spread and deliberately excluded translation, large scale
and colour operations because the disk sits at a fixed centre. Elastic deformation was never
considered either way. It is the standard augmentation for thin curvilinear structures
(vessels, roads, cracks) and it acts on exactly what our loss is made of: E170/E173 say the
addressable misses are a SHAPE problem (46% lateral, 38% width of the boundary band's symmetric
difference), and a model that has only ever seen one shape per filament has no reason to be
robust to the annotators' shape variation.

Physically careful version. A free displacement field would move the limb, and the disk
geometry is fixed information the model uses (the radial channel experiment, the fixed centre).
So the field is tapered to zero outside r 0.85 and at the frame border: the deformation acts on
the chromosphere, never on the disk edge. Amplitude is set in PIXELS AT 2048 and scaled to the
training size, following the same rule the noise/blur augmentations follow, and it is CALIBRATED
against the annotators rather than guessed: two annotators drawing the same filament agree at
median IoU 0.629, and the deformation moves a mask to IoU 0.846 / 0.806 / 0.729 / 0.666 / 0.565
of itself at 1.5 / 2 / 3 / 4 / 6 px. The default is **3 px** -- the largest amplitude whose
deformation stays clearly inside the label's own spread, so the model is taught the annotators'
shape variation and not a shape no annotator would draw. (A first version defaulted to 6 px,
which deforms MORE than two annotators differ; that is recorded because the same mistake -- a
hand-picked range wider than the measured one -- is what the augmentation calibration found in
2026-08.)
"""
import cv2
import numpy as np

DISK_R_2048 = 900.0


def elastic(img, mask, wmap, size, rng, amp_px_2048=3.0, sigma_px_2048=96.0, r_taper=0.85):
    """Smooth random displacement inside the disk. img uint8, mask/wmap float32 or None."""
    scale = size / 2048.0
    amp = amp_px_2048 * scale
    sigma = max(sigma_px_2048 * scale, 3.0)
    # a low-pass random field is a smooth displacement; normalise so `amp` is the sd in px
    dx = rng.standard_normal((size, size)).astype(np.float32)
    dy = rng.standard_normal((size, size)).astype(np.float32)
    k = int(sigma * 4) | 1
    dx = cv2.GaussianBlur(dx, (k, k), sigma)
    dy = cv2.GaussianBlur(dy, (k, k), sigma)
    sd = float(np.sqrt(dx.var() + dy.var()) / np.sqrt(2)) + 1e-8
    dx *= amp / sd
    dy *= amp / sd
    c = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    r = np.hypot(xx - c, yy - c) / (DISK_R_2048 * scale)
    taper = np.clip((r_taper - r) / 0.10, 0.0, 1.0).astype(np.float32)   # 1 inside, 0 by r 0.95
    dx *= taper
    dy *= taper
    mx = (xx + dx).astype(np.float32)
    my = (yy + dy).astype(np.float32)
    img_o = cv2.remap(img, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask_o = cv2.remap(mask, mx, my, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                       borderValue=0.0)
    wmap_o = None if wmap is None else cv2.remap(wmap, mx, my, interpolation=cv2.INTER_NEAREST,
                                                 borderMode=cv2.BORDER_CONSTANT, borderValue=1.0)
    return img_o, mask_o, wmap_o
