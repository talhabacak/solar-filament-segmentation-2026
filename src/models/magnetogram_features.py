"""E158 -- polarity inversion lines from the registered magnetogram, and per-candidate features from them.

Filaments sit on polarity inversion lines (PILs), the curves where the line-of-sight
field changes sign. A registered magnetogram (E154c, verified at 0 degrees) gives
that geometry directly, and this module turns it into two things:

  per image    a PIL map, a distance-to-PIL map, the PIL's local tangent as a
               director (cos 2theta, sin 2theta -- orientation, not direction, so
               10 and 190 degrees agree), the smoothed |B| and the field gradient

  per mask     features a selector or a diagnostic can use: how far the structure
               sits from the nearest PIL, what fraction of it lies within a few
               pixels of one, how strong the field is on either side, how well the
               structure's own axis aligns with the PIL tangent

The PIL is found on a smoothed field, because the raw 720-second magnetogram is
noisy at the few-Gauss level and every noise zero-crossing would otherwise count.
Smoothing sigma 4 px at the half-scale frame is about 9 arcsec, small against a
filament's length and large against the noise. A crossing only counts where the
field on both sides exceeds a floor, so quiet-Sun salt-and-pepper polarity does
not draw lines everywhere.

Everything is in the half-scale 1024x1024 frame that register_magnetograms writes;
masks from the 2048 pipeline are downsampled by two before sampling.

Missing magnetogram -> every feature is NaN. The selector (HistGradientBoosting)
handles NaN natively, so an image without a usable HMI record simply falls back to
the current model on that image. That is a design requirement, not a convenience:
SDO has eclipse seasons and gaps, and a test frame may land in one.
"""

from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
MAG = ROOT / "outputs" / "magnetograms"
# E159b swept sigma x floor against ground truth with a random-patch control. The
# enrichment of filament pixels near a PIL over random patches was 2.7x at sigma 4 /
# 2 G and 5.4x at sigma 8 / 1 G: filaments follow the LARGE-scale inversion line, so
# the field has to be smoothed to ~13 Mm before its zero-crossings mean anything.
SIGMA = 8.0          # px at half scale
# HMI 720 s noise is ~10 G per pixel; the 4x rebin brings it to ~2.5 G and the sigma-4
# smoothing to ~0.25 G. The first floor, 15 G, was sixty times that and erased the
# weak-field inversion lines that quiescent filaments -- most of this dataset -- sit
# on: every PIL feature came out zero (E159, first pass). 2 G is about eight sigma.
B_FLOOR = 1.0        # Gauss; both sides of a crossing must exceed this
PIL_NEAR = 6         # px at half scale (~12 px full), "on the PIL" band

PIL_FEATS = ["pil_dist_mean", "pil_dist_min", "pil_frac_near", "pil_align",
             "b_abs_mean", "b_grad_mean", "b_side_pos", "b_side_neg", "pil_strength"]


def load_registered(stem):
    p = MAG / f"reg_{stem}.npy"
    if not p.exists():
        return None
    return np.load(p).astype(np.float32)


def pil_maps(B):
    """Per-image maps from a registered half-scale B_LOS array (NaN off-disk)."""
    on = np.isfinite(B)
    Bs = ndimage.gaussian_filter(np.where(on, B, 0.0), SIGMA)
    wgt = ndimage.gaussian_filter(on.astype(np.float32), SIGMA)
    Bs = np.where(wgt > 0.2, Bs / np.maximum(wgt, 1e-3), 0.0)
    gy, gx = np.gradient(Bs)
    grad = np.hypot(gx, gy)
    # a crossing: the smoothed field changes sign between a pixel and a neighbour,
    # and both sides carry real field
    pos, neg = Bs > B_FLOOR, Bs < -B_FLOOR
    dp = ndimage.binary_dilation(pos, iterations=1)
    dn = ndimage.binary_dilation(neg, iterations=1)
    pil = dp & dn & on
    dist = ndimage.distance_transform_edt(~pil).astype(np.float32)
    # the tangent of an iso-line is perpendicular to the gradient; encode as a director
    theta = np.arctan2(gy, gx) + np.pi / 2
    cos2, sin2 = np.cos(2 * theta).astype(np.float32), np.sin(2 * theta).astype(np.float32)
    return dict(Bs=Bs.astype(np.float32), grad=grad.astype(np.float32), pil=pil,
                dist=dist, cos2=cos2, sin2=sin2, on=on)


def mask_axis(m):
    """Orientation of a mask's principal axis as a director (cos 2t, sin 2t)."""
    ys, xs = np.nonzero(m)
    if len(ys) < 3:
        return 0.0, 0.0
    x, y = xs - xs.mean(), ys - ys.mean()
    cxx, cyy, cxy = (x * x).mean(), (y * y).mean(), (x * y).mean()
    t = 0.5 * np.arctan2(2 * cxy, cxx - cyy)
    return float(np.cos(2 * t)), float(np.sin(2 * t))


def mask_features(mask_full, maps):
    """Features for one 2048-frame mask against the half-scale maps."""
    if maps is None:
        return {k: np.nan for k in PIL_FEATS}
    m = cv2.resize(mask_full.astype(np.uint8), (1024, 1024), interpolation=cv2.INTER_AREA) > 0.25
    m &= maps["on"]
    if m.sum() == 0:
        return {k: np.nan for k in PIL_FEATS}
    d = maps["dist"][m]
    c2, s2 = mask_axis(m)
    near = m & (maps["dist"] <= PIL_NEAR)
    # alignment between the mask's axis and the PIL tangent, measured where the mask
    # is actually near a PIL (elsewhere the tangent is not meaningful)
    if near.sum() >= 5:
        align = float(np.mean(c2 * maps["cos2"][near] + s2 * maps["sin2"][near]))
        strength = float(np.mean(maps["grad"][near]))
    else:
        align, strength = np.nan, 0.0
    # the field on either side: a ring around the mask, split by sign
    ring = ndimage.binary_dilation(m, iterations=8) & ~m & maps["on"]
    Br = maps["Bs"][ring] if ring.any() else np.array([0.0])
    return dict(pil_dist_mean=float(d.mean()), pil_dist_min=float(d.min()),
                pil_frac_near=float((d <= PIL_NEAR).mean()), pil_align=align,
                b_abs_mean=float(np.abs(maps["Bs"][m]).mean()),
                b_grad_mean=float(maps["grad"][m].mean()),
                b_side_pos=float(Br[Br > 0].mean()) if (Br > 0).any() else 0.0,
                b_side_neg=float(Br[Br < 0].mean()) if (Br < 0).any() else 0.0,
                pil_strength=strength)
