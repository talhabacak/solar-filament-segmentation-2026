"""Coordinate Attention (Hou, Zhou, Feng, CVPR 2021) as a drop-in for smp's scSE.

Talha's question (2026-09-06): "did we try CoordAttention in the U-Net?" We had
not. scSE (channel + spatial squeeze-excitation) is the production decoder
attention; SAAF (a learned spatial gate on the skips) measured -16 rows as a
replacement, inside the single-seed noise floor. Coordinate attention factorises
the spatial pooling into two 1-D pools (along H and along W), so the gate keeps
POSITION information along each axis -- for elongated objects with preferred
orientations (filaments, and the polar-crown E-W bias at high latitude) this is
the attention whose inductive bias matches the object, unlike scSE's global
average.

  x -> pool_h (N,C,H,1), pool_w (N,C,1,W) -> concat along the pooled axis
    -> 1x1 conv (C -> C/r) -> BN -> hard-swish -> split -> 1x1 conv each (C/r -> C)
    -> sigmoid a_h (N,C,H,1), a_w (N,C,1,W) -> x * a_h * a_w

swap_decoder_scse_to_coordatt(model) replaces every SCSEModule inside
model.decoder (attention1 and attention2 of each UnetDecoderBlock) with a
CoordAtt of the same channel count; the trainer records "decoder_att":
"coordatt" and semantic_candidates.load_model performs the same swap before
load_state_dict. Measurement protocol per ledger 8o: two seeds x two folds --
a single fold-0 run cannot see a decoder change.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordAtt(nn.Module):
    def __init__(self, channels, reduction=32):
        super().__init__()
        mid = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mid, 1)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv_h = nn.Conv2d(mid, channels, 1)
        self.conv_w = nn.Conv2d(mid, channels, 1)

    def forward(self, x):
        n, c, h, w = x.shape
        x_h = F.adaptive_avg_pool2d(x, (h, 1))                     # N,C,H,1
        x_w = F.adaptive_avg_pool2d(x, (1, w)).permute(0, 1, 3, 2)  # N,C,W,1
        y = torch.cat([x_h, x_w], dim=2)
        y = F.hardswish(self.bn1(self.conv1(y)))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        # keep the gates in the feature map's dtype: under autocast the BN/sigmoid
        # path comes out float32 and the broadcast multiply would promote the whole
        # full-resolution feature map (measured 2.6x slower epochs before this cast)
        a_h = torch.sigmoid(self.conv_h(y_h)).to(x.dtype)
        a_w = torch.sigmoid(self.conv_w(y_w.permute(0, 1, 3, 2))).to(x.dtype)
        return x * (a_h * a_w)


def swap_decoder_scse_to_coordatt(model, reduction=32):
    """Replace every scSE module under model.decoder by CoordAtt(C). Returns the
    number swapped. Channel count is read from the scSE's own cSE branch."""
    n = 0
    for _, mod in list(model.decoder.named_modules()):
        for child_name, child in list(mod.named_children()):
            if type(child).__name__ == "SCSEModule":
                c = child.cSE[1].in_channels
                setattr(mod, child_name, CoordAtt(c, reduction))
                n += 1
    return n
