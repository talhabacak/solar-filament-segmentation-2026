"""smp-compatible encoder around TorchGeo's Satlas Sentinel-1 Swin-V2.

Why a wrapper is needed. TorchGeo builds swin_v2_b from torchvision, while smp's
`tu-swinv2_*` encoders come from timm — different module names and a different
feature interface, so the Satlas checkpoint cannot be loaded into an smp encoder
directly. This exposes the torchvision backbone through the interface smp
expects.

Two facts worth recording before using it:

  * Sentinel-1 Satlas weights exist ONLY for Swin_V2_B (88M). Swin_V2_T carries
    Sentinel-2 variants only, so "swin tiny + Sentinel-1" is not available.
  * The stem is natively 2-channel (VH, VV), so feeding the grey frame
    duplicated loads every tensor untouched — no adaptation, and therefore no
    scale bug of the kind that invalidated the first ResNet50 Sentinel-1 run.

Feature pyramid measured on a (1, 2, 256, 256) input:

    features[1] -> (1, 64, 64,  128)   stride 4
    features[3] -> (1, 32, 32,  256)   stride 8
    features[5] -> (1, 16, 16,  512)   stride 16
    features[7] -> (1,  8,  8, 1024)   stride 32

torchvision returns NHWC; smp decoders want NCHW, so each is permuted.

Usage:
    from swin_satlas_encoder import build_swin_satlas_unet
    model = build_swin_satlas_unet(in_ch=2, weights="s1-si")
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

STAGE_IDX = (1, 3, 5, 7)          # blocks whose output is a pyramid level
STAGE_CH = (128, 256, 512, 1024)


class SwinSatlasEncoder(nn.Module):
    """Torchvision Swin-V2 exposed as an smp encoder.

    smp decoders index features from the deepest backwards and expect one entry
    per reduction. Swin starts at stride 4 and has no stride-2 stage, so the two
    shallow slots are filled with the input itself and a pooled copy — the U-Net
    decoder uses them only for the last two upsampling steps, where the skip is
    a low-level detail carrier rather than a semantic one.
    """

    output_stride = 32

    def __init__(self, backbone, in_ch):
        super().__init__()
        self.model = backbone
        self._in_ch = in_ch
        self._depth = 5
        self.out_channels = (in_ch, in_ch, *STAGE_CH)

    def get_stages(self):
        return None

    def forward(self, x):
        feats = [x, nn.functional.avg_pool2d(x, 2)]
        h = x
        for i, blk in enumerate(self.model.features):
            h = blk(h)
            if i in STAGE_IDX:
                feats.append(h.permute(0, 3, 1, 2).contiguous())
        return feats

    def load_state_dict(self, state_dict, **kwargs):    # smp calls this on init
        return self.model.load_state_dict(state_dict, **kwargs)


def build_swin_satlas_unet(in_ch=2, weights="s1-si", classes=1,
                           decoder_channels=(256, 128, 64, 32, 16)):
    """smp U-Net whose encoder is the Satlas Sentinel-1 Swin-V2."""
    import segmentation_models_pytorch as smp
    import torchgeo.models as tgm

    if weights == "imagenet":
        # torchvision's own ImageNet weights, so the SAME architecture can be
        # compared with a different pretraining. Its stem is 3-channel, which is
        # its native configuration exactly as Sentinel-1's is 2 — each side
        # untouched, which is what keeps the comparison about the weights.
        import torchvision
        backbone = torchvision.models.swin_v2_b(weights="IMAGENET1K_V1")
    else:
        enum = {"s1-si": tgm.Swin_V2_B_Weights.SENTINEL1_SI_SATLAS,
                "s1-mi": tgm.Swin_V2_B_Weights.SENTINEL1_MI_SATLAS,
                "none": None, None: None}[weights]
        backbone = tgm.swin_v2_b(weights=enum)
    enc = SwinSatlasEncoder(backbone, in_ch)

    model = smp.Unet(encoder_name="resnet18", encoder_weights=None,
                     in_channels=in_ch, classes=classes,
                     decoder_channels=decoder_channels)
    model.encoder = enc
    # rebuild the decoder for the swin channel widths
    model.decoder = smp.decoders.unet.decoder.UnetDecoder(
        encoder_channels=enc.out_channels,
        decoder_channels=decoder_channels,
        n_blocks=5, use_norm="batchnorm", add_center_block=False,
        attention_type=None)
    model.segmentation_head = smp.base.SegmentationHead(
        in_channels=decoder_channels[-1], out_channels=classes, kernel_size=3)
    return model


if __name__ == "__main__":
    m = build_swin_satlas_unet(in_ch=2, weights="s1-si")
    n = sum(p.numel() for p in m.parameters()) / 1e6
    with torch.no_grad():
        o = m(torch.zeros(1, 2, 256, 256))
    print(f"swin_v2_b + SENTINEL1_SI_SATLAS, in_ch=2: {n:.1f}M params, "
          f"forward -> {tuple(o.shape)}")
