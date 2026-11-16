"""CNN: U-Net with ResNet-34 encoder (timm). Transformer: SegFormer-B0 (HuggingFace, MiT-B0 encoder)."""
import math
import torch, torch.nn as nn, torch.nn.functional as F
from data import NUM_CLASSES


class ConvBR(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Up(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.c1, self.c2 = ConvBR(cin + cskip, cout), ConvBR(cout, cout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.c2(self.c1(torch.cat([x, skip], 1)))


class UNetR34(nn.Module):
    def __init__(self, pretrained, n_cls=NUM_CLASSES):
        super().__init__()
        import timm
        self.encoder = timm.create_model("resnet34.a1_in1k", pretrained=pretrained, features_only=True,
                                         out_indices=(0, 1, 2, 3, 4))  # ch 64,64,128,256,512 ; stride 2,4,8,16,32
        self.d4, self.d3, self.d2, self.d1 = Up(512, 256, 256), Up(256, 128, 128), Up(128, 64, 64), Up(64, 64, 32)
        self.head = nn.Conv2d(32, n_cls, 1)

    def encoder_module(self):
        return self.encoder

    def forward(self, x):
        f0, f1, f2, f3, f4 = self.encoder(x)
        y = self.d1(self.d2(self.d3(self.d4(f4, f3), f2), f1), f0)
        return F.interpolate(self.head(y), size=x.shape[-2:], mode="bilinear", align_corners=False)


class SegFormerB0(nn.Module):
    CKPT = "nvidia/mit-b0"

    def __init__(self, pretrained, n_cls=NUM_CLASSES):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        if pretrained:  # MiT-B0 encoder pretrained on ImageNet-1k, decode head random init
            self.m = SegformerForSemanticSegmentation.from_pretrained(self.CKPT, num_labels=n_cls)
        else:
            self.m = SegformerForSemanticSegmentation(SegformerConfig.from_pretrained(self.CKPT, num_labels=n_cls))

    def encoder_module(self):
        return self.m.segformer

    def forward(self, x):
        lg = self.m(pixel_values=x).logits  # 1/4 resolution
        return F.interpolate(lg, size=x.shape[-2:], mode="bilinear", align_corners=False)


class SegFormerB2(SegFormerB0):
    CKPT = "nvidia/mit-b2"


def official_conv_init(model):
    """Conv init as in the official SegFormer code (fan-out normal); HuggingFace uses N(0, 0.02) for all layers."""
    for m in model.m.segformer.modules():
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


def build_model(arch, pretrained, init_conv="default"):
    m = {"unet_r34": UNetR34, "segformer_b0": SegFormerB0, "segformer_b2": SegFormerB2}[arch](pretrained)
    if arch.startswith("segformer") and not pretrained and init_conv == "official":
        official_conv_init(m)
    return m


def count_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))
