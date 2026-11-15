"""Centre-to-limb augmentation by rotating the solar sphere (docs/v2 suggestion 01 section 5).

The idea in the suggestion is right: upscaling a two-pixel limb filament creates nothing, but
showing the network a well-observed central filament AS IT WOULD LOOK near the limb teaches
what foreshortening does to real structure. The suggestion implements it as a local affine
compression; here it is done exactly instead. A frame is an orthographic view of a sphere
(B0 ignored, |B0| <= 7.25 deg), so rotating the Sun by DELTA degrees of longitude is a fixed
resampling: output pixel (x', y') -> heliographic (lat, lon') -> source (lat, lon' - DELTA)
-> source pixel. Filaments near the central meridian move toward the limb with the correct
anisotropic compression, the label moves with them (nearest neighbour), and pixels whose
source would be on the far side get NO gradient (weight 0) rather than a hallucinated fill.

Limb darkening: a surface element that moves from mu_src to mu_dst must also get darker
by the frame's own centre-to-limb profile, or the network learns "limb-shaped but
centre-bright" objects. The profile Q(r) is the per-image median intensity in radial bins
(quiet Sun dominates every bin), applied as I' = I * Q(r_dst) / Q(r_src).

Off-disc pixels are left untouched (they are not solar surface). Rotation of +/-15..50
degrees keeps at least the central 130 degrees of longitude populated.
"""
import numpy as np
import cv2

DISK_R_2048 = 900.0


def _radial_profile(img, rr, nbins=64):
    """Median intensity per radial bin inside the disc; monotone-smoothed; index by r."""
    inside = rr < 1.0
    b = np.clip((rr[inside] * nbins).astype(np.int32), 0, nbins - 1)
    v = img[inside].astype(np.float32)
    prof = np.array([np.median(v[b == k]) if (b == k).any() else np.nan for k in range(nbins)], np.float32)
    ok = np.isfinite(prof)
    prof[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), prof[ok]) if ok.any() else 1.0
    prof = np.maximum(prof, 1.0)
    return prof


def sphere_rotate(img, mask, wmap, delta_deg, size, margin=12):
    """Rotate the visible hemisphere by delta_deg of longitude.

    img  : (S, S) uint8       H-alpha frame at the training size
    mask : (S, S) float32     label (any float map; resampled nearest)
    wmap : (S, S) float32 or None   per-pixel loss weight (None -> ones)
    Returns (img, mask, wmap) with wmap = 0 where the output has no source.
    """
    S = size
    c = (S - 1) / 2.0
    R = DISK_R_2048 * S / 2048.0
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    X = (xx - c) / R
    Y = (c - yy) / R
    rr = np.hypot(X, Y)
    on = rr < 1.0
    lat = np.arcsin(np.clip(Y, -1.0, 1.0))
    coslat = np.maximum(np.cos(lat), 1e-6)
    lon = np.arcsin(np.clip(X / coslat, -1.0, 1.0))
    lon_src = lon - np.deg2rad(delta_deg)
    Xs = coslat * np.sin(lon_src)
    Ys = Y
    # a target pixel is valid only if its source lies inside r < 0.95: the outermost 5% of
    # the source disc is the bright limb rim (a line-of-sight effect at mu -> 0), and
    # stretching it across the terminator paints a white band into the frame (seen in the
    # first check image). Those targets get the original pixels and zero weight instead.
    valid = on & (np.abs(lon_src) < np.pi / 2 - 1e-3) & (np.hypot(Xs, Ys) < 0.95)
    map_x = (c + R * Xs).astype(np.float32)
    map_y = (c - R * Ys).astype(np.float32)
    # source pixels outside the disc cannot happen for valid targets; guard anyway
    map_x[~valid] = xx[~valid]
    map_y[~valid] = yy[~valid]

    img_f = img.astype(np.float32)
    out_img = cv2.remap(img_f, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    # limb darkening: brightness follows the destination radius, not the source one
    prof = _radial_profile(img, rr)
    nb = len(prof)
    r_src = np.hypot(Xs, Ys)
    q_dst = prof[np.clip((rr * nb).astype(np.int32), 0, nb - 1)]
    q_src = prof[np.clip((r_src * nb).astype(np.int32), 0, nb - 1)]
    out_img = np.where(valid, out_img * (q_dst / q_src), img_f)
    out_img = np.clip(out_img, 0, 255).astype(np.uint8)

    out_mask = cv2.remap(mask.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    out_mask = np.where(valid, out_mask, 0.0).astype(np.float32)
    w = np.ones((S, S), np.float32) if wmap is None else cv2.remap(wmap.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=1.0)
    # no gradient where the output has no source, plus a margin around that seam and outside the disc
    dead = on & ~valid
    if dead.any() and margin > 0:
        dead = cv2.dilate(dead.astype(np.uint8), np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)) > 0
    w = np.where(dead, 0.0, w).astype(np.float32)
    return out_img, out_mask, w
