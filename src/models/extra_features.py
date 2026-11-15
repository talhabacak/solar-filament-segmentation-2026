"""The photometry + orientation feature family (E124/E125), as ONE function.

Why a module rather than two analysis scripts: the ledger's most expensive
lesson is that a feature must be computed by the same code at training and at
deployment (the uint8/float drift flipped 5.9% of keep decisions). The selector
bundle `selector_v5_photoorient.joblib` is fitted on features produced here and
`submit_v2.py` produces the test-time features with the same call.

Ten features per candidate mask, on the SHRUNK mask (the selector's own object):

  photometry, from the raw image
    img_contrast  (ring median - foreground mean) / ring median
    dark_core     foreground 5th percentile / ring median
    fg_p05        foreground 5th percentile (absolute)
    ring_med      ring median (absolute)
  orientation, from the structure tensor of the raw image
    coh_in        mean coherence (l1-l2)/(l1+l2) inside the mask
    coh_ring      mean coherence in a 12 px ring
    tex_align     |angle between the mask's principal axis and the coherence-
                  weighted dominant texture orientation inside| (degrees)
    ring_align    the same against the ring's texture
    grad_norm     mean gradient magnitude on the mask boundary
    aniso_ratio   coh_in / (coh_ring + 0.05)

The ring is a 31x31 elliptical dilation minus the mask for photometry and a
25x25 one minus the 3x3-dilated mask for orientation, matching the two
experiments this family was measured with.
"""

import cv2
import numpy as np

PHOTO = ["img_contrast", "dark_core", "fg_p05", "ring_med"]
ORI = ["coh_in", "coh_ring", "tex_align", "ring_align", "grad_norm", "aniso_ratio"]
EXTRA = PHOTO + ORI

RING31 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
RING25 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
K3 = np.ones((3, 3), np.uint8)


def structure_tensor(gray):
    """Coherence, dominant texture orientation (degrees, mod 180) and gradient
    magnitude of a float32 image. Gradient sigma 1.5, integration sigma 4."""
    g = cv2.GaussianBlur(gray, (0, 0), 1.5)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 4.0)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 4.0)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 4.0)
    tr, det = jxx + jyy, jxx * jyy - jxy * jxy
    disc = np.sqrt(np.maximum(tr * tr / 4 - det, 0))
    coh = (2 * disc) / (tr + 1e-6)
    ori = (np.degrees(0.5 * np.arctan2(2 * jxy, jxx - jyy)) + 90.0) % 180.0
    return coh, ori, np.sqrt(gx * gx + gy * gy)


def principal_angle(m, ys=None, xs=None):
    if ys is None:
        ys, xs = np.nonzero(m)
    if len(xs) < 5:
        return 0.0
    c = np.cov(np.vstack([xs, ys]).astype(np.float64))
    v = np.linalg.eigh(c)[1]
    return float(np.degrees(np.arctan2(v[1, -1], v[0, -1])) % 180.0)


def angle_diff(a, b):
    d = abs(a - b) % 180.0
    return float(min(d, 180.0 - d))


def circ_mean_deg(angles_deg, weights=None):
    a = np.radians(angles_deg * 2.0)
    w = np.ones_like(a) if weights is None else weights
    return float((np.degrees(np.arctan2((w * np.sin(a)).sum(), (w * np.cos(a)).sum())) / 2.0) % 180.0)


def extra_features(mask, img, tensors=None):
    """mask: bool HxW; img: float32 HxW raw image; tensors: (coh, ori, gm) from
    structure_tensor(img), computed once per image by the caller."""
    coh, ori, gm = tensors if tensors is not None else structure_tensor(img)
    from maskops import bbox, nonzero_boxed, pad_box
    box = bbox(mask)
    if box is None:
        return {k: np.nan for k in EXTRA}
    ys, xs = nonzero_boxed(mask, box)
    H, W = mask.shape
    # --- photometry: a 31x31 ring, originally dilated over the whole frame. The ellipse has
    # radius 15, so a window padded by 15 holds every ring pixel, and cv2.dilate does not let
    # pixels beyond an image edge contribute, which is also what happens at a window edge that
    # coincides with it; the values read out of the window are those of the full frame, in the
    # same C order (shape_photometry_features.py is the reference).
    py0, py1, px0, px1 = pad_box(box, 15, mask.shape)
    mw = mask[py0:py1, px0:px1]
    iw = img[py0:py1, px0:px1]
    fg = iw[mw]
    ring = cv2.dilate(mw.astype(np.uint8), RING31).astype(bool) & ~mw
    rg = iw[ring]
    rm = float(np.median(rg)) if rg.size else float(fg.mean())
    out = dict(img_contrast=(rm - float(fg.mean())) / max(rm, 1.0),
               dark_core=float(np.percentile(fg, 5)) / max(rm, 1.0),
               fg_p05=float(np.percentile(fg, 5)), ring_med=rm)
    # --- orientation (bbox crop with 14 px pad, 25x25 ring: matches orientation_selector_features.py)
    y0, y1 = max(ys.min() - 14, 0), min(ys.max() + 15, H)
    x0, x1 = max(xs.min() - 14, 0), min(xs.max() + 15, W)
    mc = mask[y0:y1, x0:x1]
    ring2 = cv2.dilate(mc.astype(np.uint8), RING25).astype(bool) & ~mc
    cc, oc, gc = coh[y0:y1, x0:x1], ori[y0:y1, x0:x1], gm[y0:y1, x0:x1]
    bnd = mc & ~cv2.erode(mc.astype(np.uint8), K3).astype(bool)
    axis = principal_angle(mask, ys, xs)
    out["coh_in"] = float(cc[mc].mean())
    out["coh_ring"] = float(cc[ring2].mean()) if ring2.any() else 0.0
    out["tex_align"] = angle_diff(axis, circ_mean_deg(oc[mc], cc[mc]))
    out["ring_align"] = angle_diff(axis, circ_mean_deg(oc[ring2], cc[ring2])) if ring2.any() else 90.0
    out["grad_norm"] = float(gc[bnd].mean()) if bnd.any() else 0.0
    out["aniso_ratio"] = out["coh_in"] / (out["coh_ring"] + 0.05)
    return out
