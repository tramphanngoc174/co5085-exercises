"""Kiểm thử nhanh bằng DỮ LIỆU GIẢ (không cần tải dataset). Số liệu ở đây KHÔNG dùng cho báo cáo.
    python E1/smoke_test.py
"""
import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from torch.utils.data import TensorDataset

from common.data import build_loaders
from common.train_utils import count_params, get_device
from E1.models import MODELS
from E1.run_e1 import run_all

# 1) Kiểm tra hình dạng tensor
x = torch.randn(4, 1, 28, 28)
for name, cls in MODELS.items():
    out = cls()(x)
    assert out.shape == (4, 10), (name, out.shape)
    print(f"{name:8s} output {tuple(out.shape)} | params {count_params(cls()):,}")

# 2) Chạy toàn bộ pipeline với dữ liệu ngẫu nhiên (có nhãn phụ thuộc ảnh để loss giảm)
torch.manual_seed(0)
def fake(n):
    y = torch.randint(0, 10, (n,))
    img = torch.randn(n, 1, 28, 28) * 0.3
    img[torch.arange(n), 0, y, :] += 2.0     # mỗi lớp có một "hàng sáng" riêng
    return TensorDataset(img, y)

tmp = tempfile.mkdtemp()
tr, va, te = build_loaders(fake(1200), fake(300), os.path.join(tmp, "split.json"),
                           val_size=200, seed=1, batch_size=64)
info = {"in_channels": 1, "img_size": 28, "num_classes": 10,
        "class_names": [str(i) for i in range(10)]}
args = argparse.Namespace(models=list(MODELS), epochs=3, lr=1e-3, seed=1,
                          out=os.path.join(tmp, "out"), batch_size=64)
run_all(tr, va, te, info, args, get_device())
print("\nCác file tạo ra:", sorted(os.listdir(args.out)))
print("SMOKE TEST OK")
