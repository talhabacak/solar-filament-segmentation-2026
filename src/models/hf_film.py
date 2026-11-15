"""E213 -- tell the segmentation model the frame's regime: FiLM on the disc hf.

Ledger 46: across hf (disc Laplacian variance) quartiles recall falls 0.623 -> 0.540 while SQ and
precision stay flat. The model-side attempts so far REMOVED the regime from the input (phys/physn,
ledger 49: amplitude 2.42x -> 1.02x, Dice -0.0026/-0.0056) or simulated it (depth jitter, ledger 45,
-0.0081). This does the opposite: it keeps the image as it is and gives the network the regime as
a conditioning signal, so the same local evidence can be read differently in a busy frame.

  z   = clip((log hf(x) - mu) / sd, -4, 4)       hf computed from the input itself, on the GPU,
                                                  exactly as outputs/eda/_image_quality.py does
                                                  (cv2 Laplacian ksize 3, variance inside r < 0.85)
  f_l <- f_l * (1 + gamma_l(z)) + beta_l(z)       every encoder skip l = 1..5, per channel

gamma/beta come from a small MLP whose last layer starts at zero, so at step 0 the network is
bit-identical to the control. The MLP is built inside torch.random.fork_rng, so building it does
not advance the global RNG and the control and FiLM runs see the same initial weights, flips,
rotations, gamma draws and sample order: the pair differs only in the conditioning.

No extra input at inference: hf is read from the image the model is given. mu and sd are
measured on the training frames at start-up and saved with the weights (buffers).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LAPLACE3 = torch.tensor([[2.0, 0.0, 2.0], [0.0, -8.0, 0.0], [2.0, 0.0, 2.0]])   # cv2 ksize=3


def disc_mask(size, r_frac=0.85, radius_2048=900.0):
    c = (size - 1) / 2.0
    yy, xx = torch.meshgrid(torch.arange(size, dtype=torch.float32),
                            torch.arange(size, dtype=torch.float32), indexing="ij")
    return (((xx - c) ** 2 + (yy - c) ** 2).sqrt() < r_frac * radius_2048 * size / 2048.0).float()


def disc_hf(x, mask):
    """Laplacian variance inside the disc, per image; x is (B, C, S, S) in [0, 1], channel 0 raw.
    Always in float32: the variance reaches ~1e6 and would overflow half precision under autocast."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        img = x[:, :1].float() * 255.0
        lap = F.conv2d(img, LAPLACE3.to(img.device).view(1, 1, 3, 3), padding=1)[:, 0]
        m = mask.to(img.device).float()
        n = m.sum()
        mean = (lap * m).sum((1, 2)) / n
        return ((lap - mean.view(-1, 1, 1)) ** 2 * m).sum((1, 2)) / n


class HfFiLM(nn.Module):
    def __init__(self, channels, size, hidden=32):
        super().__init__()
        self.channels = list(channels)
        with torch.random.fork_rng(devices=[]):
            self.mlp = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, 2 * sum(self.channels)))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.register_buffer("mask", disc_mask(size), persistent=False)
        self.register_buffer("mu", torch.tensor(0.0))
        self.register_buffer("sd", torch.tensor(1.0))

    def z(self, x):
        h = disc_hf(x, self.mask)
        return ((torch.log(h + 1.0) - self.mu) / self.sd).clamp(-4.0, 4.0)

    def forward(self, x, feats):
        gb = self.mlp(self.z(x).view(-1, 1))
        out, k = [feats[0]], 0
        for f, c in zip(feats[1:], self.channels):
            g, b = gb[:, k:k + c], gb[:, k + sum(self.channels): k + sum(self.channels) + c]
            out.append(f * (1.0 + g.to(f.dtype).view(-1, c, 1, 1)) + b.to(f.dtype).view(-1, c, 1, 1))
            k += c
        return out


def attach(model, size):
    """Add `model.hf_film` and route forward through it. State-dict keys of the base model are
    unchanged; the new ones live under `hf_film.`."""
    film = HfFiLM(model.encoder.out_channels[1:], size)
    model.hf_film = film

    def forward(x):
        feats = model.encoder(x)
        feats = model.hf_film(x, feats)
        return model.segmentation_head(model.decoder(feats))

    model.forward = forward
    return film


@torch.no_grad()
def fit_norm(film, images_uint8, rows, device="cuda", chunk=8):
    """mu, sd of log hf over the training frames `rows` of a (N, S, S) uint8 array (a memmap is
    read chunk by chunk, never whole)."""
    import numpy as np
    vals = []
    for i in range(0, len(rows), chunk):
        a = np.stack([np.asarray(images_uint8[r]) for r in rows[i:i + chunk]])
        x = torch.from_numpy(a).to(device).float().unsqueeze(1) / 255.0
        vals.append(torch.log(disc_hf(x, film.mask) + 1.0).cpu())
    v = torch.cat(vals)
    film.mu.fill_(float(v.mean()))
    film.sd.fill_(max(float(v.std()), 1e-3))
    return float(v.mean()), float(v.std()), len(v), float(math.exp(v.median()) - 1.0)
