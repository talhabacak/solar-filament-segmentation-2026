"""SAAF-lite: re-weight the up-sampled decoder map BEFORE it is concatenated
with the skip (docs/24 D1b, _review_dfas_yolo.md §3.2).

The production U-Net's scSE attention runs AFTER the concatenation, and the
probe in ledger §8j found that at the finest attention stage (the 1/2-
resolution block) its spatial gate is ~0.01 everywhere -- effectively off --
while the channel gate does not suppress the up branch (up/skip 1.29 and 0.95
on the two finest blocks). So nothing in the production decoder re-weights
the coarse, up-sampled map spatially before it meets the thin structure in
the skip. This adds exactly that, and nothing else:

    up = interpolate(x, x2)
    g  = sigmoid(conv1x1(cat[up, skip]))        # one spatial gate, from both
    up = up * gamma[c] * g                       # gamma per channel, init 1
    x  = cat[up, skip]  ->  attention1 -> conv1 -> conv2 -> attention2 (as before)

gamma starts at 1 and the gate conv's bias at +3 (sigmoid 0.95), so at
initialisation the block is within 5% of the unmodified one and training
starts from the production behaviour. ~0.02M parameters on (256,128,64,32,16).
Applied to every block that has a skip; the last block has none and is left
alone. Loaded back by semantic_candidates.load_model through the checkpoint
key "saaf".
"""

import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAAFGate(nn.Module):
    def __init__(self, up_ch, skip_ch):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(up_ch))
        self.gate = nn.Conv2d(up_ch + skip_ch, 1, kernel_size=1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 3.0)

    def forward(self, up, skip):
        g = torch.sigmoid(self.gate(torch.cat([up, skip], dim=1)))
        return up * self.gamma[None, :, None, None] * g


def apply_saaf(model):
    """Attach a SAAFGate to every smp UnetDecoderBlock that has a skip and
    replace its forward. Returns the number of gates attached. Idempotent."""
    n = 0
    blocks = list(model.decoder.blocks)
    enc_ch = list(model.encoder.out_channels)
    ups = [enc_ch[-1]] + [b.conv2[0].out_channels for b in blocks[:-1]]
    for i, blk in enumerate(blocks):
        total = blk.conv1[0].in_channels
        skip_ch = total - ups[i]
        if skip_ch <= 0 or hasattr(blk, "saaf"):
            continue
        blk.saaf = SAAFGate(ups[i], skip_ch)
        n += 1

        # smp >= 0.4 blocks take (feature_map, target_height, target_width,
        # skip_connection) and interpolate to a size in self.interpolation_mode;
        # older ones take (x, skip) and scale by 2. Match whichever is installed
        # so the same checkpoint loads on the box and here.
        sized = "target_height" in inspect.signature(blk.forward).parameters
        mode = getattr(blk, "interpolation_mode", "nearest")

        def fwd(feature_map, *args, _blk=blk, _sized=sized, _mode=mode, **kwargs):
            if _sized:
                th, tw = args[0], args[1]
                skip = args[2] if len(args) > 2 else kwargs.get("skip_connection")
                x = F.interpolate(feature_map, size=(th, tw), mode=_mode)
            else:
                skip = args[0] if args else kwargs.get("skip")
                x = F.interpolate(feature_map, scale_factor=2.0, mode=_mode)
            if skip is not None:
                x = _blk.saaf(x, skip)
                x = torch.cat([x, skip], dim=1)
                x = _blk.attention1(x)
            x = _blk.conv1(x)
            x = _blk.conv2(x)
            x = _blk.attention2(x)
            return x

        blk.forward = fwd
    return n
