"""E181 -- anti-aliased downsampling for the production encoder.

Talha, 2026-09-13: the downsampling layers may be losing information. The encoder has no
pooling at all (ledger 29h): EfficientNet-B3 downsamples with five stride-2 convolutions. A
stride-2 convolution subsamples WITHOUT a low-pass filter, so everything above the new Nyquist
frequency folds back as aliasing -- and a filament 1-2 px wide at 2048 (4.5 px at 1536) is
exactly the content that sits near Nyquist after two or three halvings. Zhang (2019, "Making
Convolutional Networks Shift-Invariant Again") fixes this by moving the subsampling after a
blur: conv(stride 1) -> blur -> subsample.

The surgery keeps the pretrained weights valid, because the convolution itself is untouched:
only the sampling changes. Cost is the four-fold compute of those five convolutions at full
stride-1 resolution, which on this network is roughly +50 % step time.

Applied to the production checkpoint's own architecture so the comparison is a single axis.
"""
import torch.nn as nn


def apply_blurpool(model, filt_size=3):
    """Replace every stride-2 conv in the encoder with conv(stride 1) + BlurPool(stride 2)."""
    from timm.layers import BlurPool2d
    n = 0
    for parent_name, parent in list(model.encoder.named_modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Conv2d) and tuple(child.stride) == (2, 2):
                child.stride = (1, 1)
                setattr(parent, name, nn.Sequential(child, BlurPool2d(child.out_channels,
                                                                      filt_size=filt_size, stride=2)))
                n += 1
    return n
