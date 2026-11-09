"""Kiểm thử A1 bằng DỮ LIỆU GIẢ, KHÔNG tải pretrained (số liệu ở đây KHÔNG dùng cho báo cáo).
    python A1/smoke_test.py
Kiểm tra: (1) số tham số trainable theo chế độ, (2) đóng băng thật sự (trọng số + BatchNorm không đổi),
(3) chạy toàn bộ pipeline.
"""
import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import timm
import torch

from A1.data import make_loaders
from A1.engine import Prep, train_epoch
from A1.models import MODES, apply_freeze, build_model, count_params, param_groups
from A1.run_a1 import run_all
from common.train_utils import get_device

dev = get_device()
K, S = 10, 64

# (1) số tham số trainable: head < partial < full
for arch in ["resnet18", "vit_tiny"]:
    tr = {}
    for mode in MODES:
        m = build_model(arch, mode, K, S, offline=True)
        apply_freeze(m, arch, mode)
        tot, trn = count_params(m)
        tr[mode] = trn
        print(f"[1] {arch:9s} {mode:8s} tổng {tot:,} | trainable {trn:,} ({100 * trn / tot:.1f}%)")
    assert tr["head"] < tr["partial"] < tr["full"] == tr["scratch"], tr

# (2) đóng băng thật sự: sau 1 epoch, tham số/BN bị đóng băng không đổi, tham số train thì đổi
def fake(n):
    y = torch.randint(0, K, (n,))
    x = (torch.rand(n, 3, 32, 32) * 100).to(torch.uint8)
    x[torch.arange(n), :, :8, :] = (y * 25).view(n, 1, 1).to(torch.uint8)
    return x, y

x_tr, y_tr = fake(128); x_va, y_va = fake(64); x_te, y_te = fake(64)
trl, val, tel = make_loaders(x_tr, y_tr, x_va, y_va, x_te, y_te, 32, 1)
for arch, frozen_name, train_name in [("resnet18", "layer1.0.conv1.weight", "fc.weight"),
                                      ("vit_tiny", "blocks.0.attn.qkv.weight", "head.weight")]:
    m = build_model(arch, "head", K, S, offline=True).to(dev)
    bns = apply_freeze(m, arch, "head")
    cfg = timm.data.resolve_data_config({}, model=m)
    prep = Prep(dev, S, cfg["mean"], cfg["std"])
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    bn_before = {n: b.clone() for n, b in m.named_buffers()}
    opt = torch.optim.AdamW(param_groups(m, arch, "head", 1e-3, 1e-4, 1e-3), weight_decay=0.01)
    train_epoch(m, trl, torch.nn.CrossEntropyLoss(), opt, torch.amp.GradScaler("cuda", enabled=False),
                prep, dev, bns, False)
    p = dict(m.named_parameters())
    assert torch.equal(before[frozen_name], p[frozen_name].detach()), f"{frozen_name} bị thay đổi dù đã đóng băng!"
    assert not torch.equal(before[train_name], p[train_name].detach()), f"{train_name} không được cập nhật!"
    for n, b in m.named_buffers():
        assert torch.equal(bn_before[n], b), f"buffer {n} (BatchNorm) bị thay đổi dù đã đóng băng!"
    print(f"[2] {arch}: đóng băng đúng ({frozen_name} không đổi, {train_name} có cập nhật, BatchNorm không đổi)")

# (3) pipeline đầy đủ
info = {"num_classes": K, "class_names": [f"c{i}" for i in range(K)], "n_train": 128, "n_val": 64, "n_test": 64}
args = argparse.Namespace(archs=["resnet18", "vit_tiny"], modes=MODES, epochs=2, img_size=S, batch_size=32,
                          lr_scratch=1e-3, lr_ft=1e-4, lr_head=1e-3, wd=0.01, seed=1, amp=False, offline=True, resume=False,
                          out=os.path.join(tempfile.mkdtemp(), "out"))
run_all(trl, val, tel, info, args, dev)
print("\nCác file tạo ra:", sorted(os.listdir(args.out)))
print("SMOKE TEST A1 OK")
