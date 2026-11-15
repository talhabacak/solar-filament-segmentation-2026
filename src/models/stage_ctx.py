"""E174 -- a context module on the stage-4 skip, aimed at the faint population.

E171 measured which encoder stage carries which filament: zeroing the 96 px skip (stage 4)
costs the FAINT copies 0.104 IoU while costing the bright ones nothing (+0.016), and zeroing
the 384 px skip (stage 2) does the opposite. The addressable misses found by E170 are faint
(contrast 0.102 against 0.126 for matched), so the aimed change is more context exactly where
faint objects are carried -- not attention everywhere, which was measured and closed (CoordAtt,
SAAF).

The module is deliberately small: three dilated 3x3 branches (1, 3, 6) plus a channel-wise
gate, added residually, so at initialisation it is the identity and it costs ~0.15 M parameters
at 136 channels. Dilations 3 and 6 at 96 px see 18 and 36 px of the 1536 px frame, i.e. 24 and
48 px at 2048 -- the scale of a filament's neighbourhood rather than its body.
"""
import torch
import torch.nn as nn


class DilatedContext(nn.Module):
    def __init__(self, ch, mid=None, dilations=(1, 3, 6)):
        super().__init__()
        mid = mid or max(ch // 4, 16)
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ch, mid, 3, padding=d, dilation=d, bias=False),
                          nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
            for d in dilations])
        self.fuse = nn.Conv2d(mid * len(dilations), ch, 1, bias=True)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)          # starts as the identity

    def forward(self, x):
        return x + self.fuse(torch.cat([b(x) for b in self.branches], dim=1))


def attach_stage_context(model, stage, dilations=(1, 3, 6)):
    """Wrap model.encoder so that feature[stage] passes through a DilatedContext."""
    enc = model.encoder
    ch = enc.out_channels[stage]
    mod = DilatedContext(ch, dilations=dilations)
    mod = mod.to(next(model.parameters()).device)
    orig = enc.forward

    def fwd(x, _orig=orig, _mod=mod, _s=stage):
        feats = list(_orig(x))
        feats[_s] = _mod(feats[_s])
        return feats

    enc.forward = fwd
    # register so the parameters are trained and saved with the model
    model.add_module(f"stage{stage}_ctx", mod)
    return sum(p.numel() for p in mod.parameters())
