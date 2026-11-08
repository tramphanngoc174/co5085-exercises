"""Kiểm thử E3 bằng DỮ LIỆU GIẢ (số liệu ở đây KHÔNG dùng cho báo cáo).
    python E3/smoke_test.py
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
from E3.models import SeqClassifier
from E3.run_e3 import accuracy, run_all

# (1) Kiểm tra cách tạo chuỗi: so với phép cắt trực tiếp của ảnh
x = torch.randn(2, 1, 28, 28)
m = SeqClassifier("lstm", "row")
assert torch.equal(m.to_seq(x), x[:, 0]), "row sai"
m = SeqClassifier("lstm", "col")
assert torch.equal(m.to_seq(x), x[:, 0].transpose(1, 2)), "col sai"
m = SeqClassifier("lstm", "patch")
s = m.to_seq(x)
assert s.shape == (2, 49, 16)
assert torch.equal(s[0, 0], x[0, 0, 0:4, 0:4].flatten()), "patch 0 sai"
assert torch.equal(s[0, 1], x[0, 0, 0:4, 4:8].flatten()), "patch 1 sai (phải là ô bên phải)"
assert torch.equal(s[0, 7], x[0, 0, 4:8, 0:4].flatten()), "patch 7 sai (phải là ô đầu hàng thứ hai)"
print("[1] cách tạo chuỗi row/col/patch đúng")

# (2) Hình dạng và số tham số
for rep in ["row", "col", "patch"]:
    for rnn in ["lstm", "gru"]:
        m = SeqClassifier(rnn, rep)
        out = m(x)
        assert out.shape == (2, 10)
        print(f"[2] {rnn:5s} {rep:6s} chuỗi {m.seq_len}x{m.feat} | output {tuple(out.shape)} | params {count_params(m):,}")

# (3) Phép xáo hàng: hàng k sau xáo phải bằng hàng perm[k] ban đầu
img = torch.arange(28.0).view(1, 1, 28, 1).expand(1, 1, 28, 28)
perm = torch.randperm(28)
assert torch.equal(img[:, :, perm, :][0, 0, :, 0], perm.float()), "xáo hàng sai"
print("[3] phép xáo hàng đúng")

# (4) Chạy toàn bộ pipeline bằng dữ liệu giả (không có checkpoint E1 -> baseline được huấn luyện lại)
def fake(n):
    y = torch.randint(0, 10, (n,))
    im = torch.randn(n, 1, 28, 28) * 0.3
    im[torch.arange(n), 0, y, :] += 2.0
    return TensorDataset(im, y)

tmp = tempfile.mkdtemp()
tr, va, te = build_loaders(fake(600), fake(200), os.path.join(tmp, "split.json"),
                           val_size=100, seed=1, batch_size=64)
info = {"in_channels": 1, "img_size": 28, "num_classes": 10}
args = argparse.Namespace(representations=["row", "col", "patch"], rnn=["lstm", "gru"], epochs=2, lr=1e-3,
                          seed=1, hidden=32, shuffle_perms=3, batch_size=64,
                          e1_dir=os.path.join(tmp, "khong_co"), out=os.path.join(tmp, "out"))
dev = get_device()
run_all(tr, va, te, info, args, dev)
print("\nCác file tạo ra:", sorted(os.listdir(args.out)))
print("SMOKE TEST E3 OK")
