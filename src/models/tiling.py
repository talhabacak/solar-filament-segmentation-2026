"""Tiling geometry and blended mosaic reassembly for the native-resolution lane.

The geometry is not copied from anywhere -- it comes out of src/eda/tile_geometry.py,
which measured our own frames:

  disk radius     median 991.7 px (914-1033 across frames)
  GT reach        903.1 px at most; p99 876.4
  foreground      0.302% of a frame, over the union of annotators

and then chose the CHEAPEST layout satisfying the constraints, which is not the
one with the most overlap -- maximising overlap picks 3x3 tiles of 1408 at step
231, nine forward passes for 84% redundancy that buys nothing. The overlap only
has to let the blend ramp cross-fade and let both tiles see a straddling filament
with context; filaments are 10-30 px wide, so 194 px is already generous.

  crop 1918 (offset 65)   2x2 tiles of 1056   step 862   overlap 194   margin 56

Every tile lies wholly inside the crop, so no pixel a tile sees is padded or
reflected -- reflecting at a tile edge would invent chromosphere that is not
there, and the model would learn to segment it.

The margin matters because the reach was measured on TRAIN; a test filament
sitting slightly further out still falls inside 959 px of half-crop.
"""

import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
_G = json.loads((ROOT / "outputs" / "eda" / "tile_geometry.json").read_text())

FULL = int(_G["full"])          # 2048
CROP = int(_G["crop"])          # 1918
OFFSET = int(_G["offset"])      # 65
TILE = int(_G["tile"])          # 1056
TILES = int(_G["tiles"])        # 2
STEP = int(_G["step"])          # 862
OVERLAP = int(_G["overlap"])    # 194
BLEND_FLOOR = 0.05


def tile_origins():
    """Top-left corners of every tile, in FULL-frame coordinates.

    The first and last tile are flush against the crop's edges, so the tiles
    cover the crop exactly and none of them reaches outside it.
    """
    starts = [OFFSET + i * STEP for i in range(TILES)]
    assert starts[-1] + TILE == OFFSET + CROP, "tiles must cover the crop exactly"
    return [(r, c) for r in starts for c in starts]


def blend_window():
    """TILE x TILE weights: flat 1 through the middle, tapering to BLEND_FLOOR at the rim.

    A pixel at a tile's edge was predicted with context on one side only. The same
    pixel sits well inside its neighbour, with context all round. Weighting each
    prediction by its distance from its OWN tile's edge lets the tile that saw it
    properly dominate, and the two cross-fade instead of meeting at a cut.

    Why a cut would be expensive here specifically: connected components are what
    turn the probability map into instances, so a seam is a place where one
    filament can be labelled as two -- the one-to-many fragmentation the metric
    charges for. Blending removes the seam rather than hiding it.

    The ramp is a raised cosine over exactly OVERLAP px, the width two neighbours
    share, so one fades out as the other fades in. It stops at BLEND_FLOOR rather
    than 0 because the outermost ring of the crop is covered by a single tile:
    there the floor is the only weight present and cancels in the division. A true
    zero would make that 0/0.
    """
    d = np.minimum(np.arange(TILE), np.arange(TILE)[::-1]).astype(np.float32)
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.clip(d / OVERLAP, 0.0, 1.0)))
    line = BLEND_FLOOR + (1.0 - BLEND_FLOOR) * ramp
    return np.outer(line, line).astype(np.float32)


_BLEND = None


def blend_gpu(device="cuda"):
    global _BLEND
    if _BLEND is None or _BLEND.device.type != torch.device(device).type:
        _BLEND = torch.from_numpy(blend_window()).to(device)
    return _BLEND


@torch.no_grad()
def mosaic_predict(model, img, tta=8, device="cuda", batch=4, amp_dtype=torch.float16):
    """Reassemble a FULL x FULL probability map from the overlapping tiles.

    img: uint8 (FULL, FULL). Returns float32 (FULL, FULL) in [0, 1].

    Pixels outside the crop are left at 0: no tile touches them, both accumulators
    stay 0 and the clamped division returns 0. They are black sky beyond the disk
    and, as measured, hold no ground truth.

    tta averages over the dihedral group in IMAGE space -- each transform is undone
    on the output before averaging, never on the input alone. 8 = 4 rotations x
    {identity, hflip}. The reason to want it here is the same as the reason to
    blend: the noise it cancels is the noise that breaks a faint filament into two
    components.
    """
    win = blend_gpu(device)
    acc = torch.zeros((FULL, FULL), device=device, dtype=torch.float32)
    wsum = torch.zeros((FULL, FULL), device=device, dtype=torch.float32)
    origins = tile_origins()

    def forward(x):
        if tta <= 1:
            return torch.sigmoid(model(x))
        outs = []
        for flip in range(2 if tta >= 8 else 1):
            xf = torch.flip(x, dims=[-1]) if flip else x
            for k in range(4 if tta >= 4 else 1):
                y = torch.sigmoid(model(torch.rot90(xf, k, dims=(-2, -1)).contiguous()))
                y = torch.rot90(y, -k, dims=(-2, -1))
                outs.append(torch.flip(y, dims=[-1]) if flip else y)
        return torch.stack(outs).mean(0)

    for k in range(0, len(origins), batch):
        chunk = origins[k:k + batch]
        tiles = np.stack([img[r:r + TILE, c:c + TILE] for r, c in chunk])
        x = torch.from_numpy(tiles).to(device).float().div_(255.0).unsqueeze(1)
        x = x.repeat(1, 3, 1, 1)
        with torch.autocast(device, dtype=amp_dtype):
            p = forward(x).float()
        for (r, c), pi in zip(chunk, p):
            acc[r:r + TILE, c:c + TILE] += pi[0] * win
            wsum[r:r + TILE, c:c + TILE] += win
    return (acc / wsum.clamp(min=1e-6)).cpu().numpy()
