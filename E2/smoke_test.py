"""Kiểm thử E2 bằng DỮ LIỆU GIẢ (số liệu ở đây KHÔNG dùng cho báo cáo).
    python E2/smoke_test.py
Gồm 3 kiểm tra: (1) MSA tự viết == nn.MultiheadAttention khi chép cùng trọng số,
(2) hình dạng tensor của mọi tổ hợp tokenizer x MSA, (3) chạy thử toàn bộ pipeline.
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
from E2.models import build_model
from E2.msa import CustomMSA, TorchMSA
from E2.run_e2 import run_all

# (1) Đúng đắn của MSA tự viết
torch.manual_seed(0)
c, t = CustomMSA(64, 4), TorchMSA(64, 4)
with torch.no_grad():
    t.mha.in_proj_weight.copy_(torch.cat([c.q.weight, c.k.weight, c.v.weight], 0))
    t.mha.in_proj_bias.copy_(torch.cat([c.q.bias, c.k.bias, c.v.bias], 0))
    t.mha.out_proj.weight.copy_(c.out.weight)
    t.mha.out_proj.bias.copy_(c.out.bias)
x = torch.randn(3, 10, 64)
diff = (c(x) - t(x)).abs().max().item()
print(f"[1] sai khác lớn nhất giữa MSA tự viết và PyTorch: {diff:.2e}")
assert diff < 1e-5, "MSA tự viết KHÔNG khớp với PyTorch!"
assert count_params(c) == count_params(t), "số tham số hai bản phải bằng nhau"
print(f"    số tham số mỗi bản MSA: {count_params(c):,} (bằng nhau)")

# (2) Hình dạng
img = torch.randn(4, 1, 28, 28)
for tok in ["patch", "row", "cnnstem"]:
    for msa in ["custom", "torch"]:
        m = build_model(tok, msa)
        out = m(img)
        assert out.shape == (4, 10), (tok, msa, out.shape)
        print(f"[2] {tok:8s} {msa:7s} tokens {m.tokenizer.num_tokens:3d} | output {tuple(out.shape)} | params {count_params(m):,}")

# (3) Chạy thử pipeline bằng dữ liệu giả
def fake(n):
    y = torch.randint(0, 10, (n,))
    im = torch.randn(n, 1, 28, 28) * 0.3
    im[torch.arange(n), 0, y, :] += 2.0
    return TensorDataset(im, y)

tmp = tempfile.mkdtemp()
tr, va, te = build_loaders(fake(600), fake(200), os.path.join(tmp, "split.json"),
                           val_size=100, seed=1, batch_size=64)
info = {"in_channels": 1, "img_size": 28, "num_classes": 10}
args = argparse.Namespace(tokenizers=["patch", "row", "cnnstem"], msa=["custom", "torch"], epochs=2, lr=1e-3,
                          seed=1, dim=64, depth=2, heads=4, batch_size=64, out=os.path.join(tmp, "out"))
run_all(tr, va, te, info, args, get_device())
print("\nCác file tạo ra:", sorted(os.listdir(args.out)))
print("SMOKE TEST E2 OK")
