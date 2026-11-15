"""GroupNorm decoder for E103 (docs/chatgpt_deep_search/05 §E103): smp 0.5.0's
decoder_use_norm accepts only batchnorm / instancenorm / layernorm / identity /
inplace, so a GroupNorm decoder is built as a BatchNorm one and every
BatchNorm2d in the decoder is swapped for GroupNorm(groups, C) afterwards.
The trainer records "decoder_gn": groups in the checkpoint and
semantic_candidates.load_model performs the same swap before load_state_dict,
so the state dict keys line up (GroupNorm has weight and bias, BatchNorm's
running statistics are simply absent).
"""

import torch.nn as nn


def swap_decoder_bn_to_gn(model, groups=8):
    """Replace every BatchNorm2d under model.decoder with GroupNorm(groups, C).
    Channel counts in smp decoders are multiples of 16, so 8 groups always
    divide. Returns the number of layers swapped."""
    n = 0
    for name, mod in list(model.decoder.named_modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, nn.BatchNorm2d):
                g = groups if child.num_features % groups == 0 else 1
                setattr(mod, child_name, nn.GroupNorm(g, child.num_features, affine=True))
                n += 1
    return n


def swap_decoder_act(model, act="silu"):
    """Replace every ReLU under model.decoder with the named activation
    (silu or gelu). Returns the number swapped. Talha's question: the encoder
    (EfficientNet) already uses SiLU; only the 20 decoder ReLUs change."""
    new = {"silu": nn.SiLU, "gelu": nn.GELU}[act]
    n = 0
    for name, mod in list(model.decoder.named_modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, nn.ReLU):
                setattr(mod, child_name, new())
                n += 1
    return n
