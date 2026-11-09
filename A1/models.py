"""A1: CNN (ResNet-18) và Transformer (ViT-Tiny/16) từ timm + 4 chế độ huấn luyện.

  scratch : khởi tạo ngẫu nhiên, huấn luyện toàn bộ
  head    : pretrained, ĐÓNG BĂNG toàn bộ backbone, chỉ train lớp phân loại
  partial : pretrained, đóng băng nửa đầu (ResNet: stem+layer1+layer2; ViT: patch_embed+pos+6 block đầu)
  full    : pretrained, fine-tune toàn bộ
"""
import timm
import torch.nn as nn

TIMM_NAMES = {"resnet18": "resnet18", "vit_tiny": "vit_tiny_patch16_224"}
HEAD_PREFIX = {"resnet18": ("fc.",), "vit_tiny": ("head.",)}
MODES = ["scratch", "head", "partial", "full"]


def build_model(arch, mode, num_classes, img_size, offline=False):
    """offline=True: không tải trọng số pretrained (chỉ dùng cho kiểm thử)."""
    pretrained = (mode != "scratch") and not offline
    kwargs = {"img_size": img_size} if arch == "vit_tiny" else {}
    return timm.create_model(TIMM_NAMES[arch], pretrained=pretrained, num_classes=num_classes, **kwargs)


def is_trainable(arch, mode, name):
    if mode in ("scratch", "full"):
        return True
    is_head = name.startswith(HEAD_PREFIX[arch])
    if mode == "head":
        return is_head
    # mode == "partial"
    if arch == "resnet18":
        return is_head or name.startswith(("layer3.", "layer4."))
    if name.startswith("blocks."):                       # ViT-Tiny có 12 block: 0-5 đóng băng, 6-11 train
        return int(name.split(".")[1]) >= 6
    return name.startswith(("head.", "norm.", "fc_norm."))


def apply_freeze(model, arch, mode):
    """Đặt requires_grad theo chế độ; trả về danh sách BatchNorm bị đóng băng hoàn toàn
    (phải giữ ở eval() khi train để running stats không bị cập nhật)."""
    for n, p in model.named_parameters():
        p.requires_grad = is_trainable(arch, mode, n)
    frozen_bns = []
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            if all(not p.requires_grad for p in m.parameters(recurse=False)):
                frozen_bns.append(m)
    return frozen_bns


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def param_groups(model, arch, mode, lr_scratch, lr_ft, lr_head):
    head, backbone = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (head if n.startswith(HEAD_PREFIX[arch]) else backbone).append(p)
    if mode == "scratch":
        return [{"params": head + backbone, "lr": lr_scratch}]
    if mode == "head":
        return [{"params": head, "lr": lr_head}]
    groups = [{"params": backbone, "lr": lr_ft}]
    if head:
        groups.append({"params": head, "lr": lr_head})
    return groups
