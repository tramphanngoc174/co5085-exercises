"""E2: Multi-Head Self-Attention.

  - CustomMSA: TỰ VIẾT từ các phép cơ bản (Linear, matmul, softmax, ghép head).
               KHÔNG dùng nn.MultiheadAttention / nn.TransformerEncoder*.
  - TorchMSA : bọc nn.MultiheadAttention của PyTorch để đối chiếu.
Hai lớp có cùng giao diện: nhập x (B, N, D) -> xuất (B, N, D).
"""
import math

import torch
import torch.nn as nn


class CustomMSA(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0, "dim phải chia hết cho num_heads"
        self.h = num_heads
        self.dh = dim // num_heads            # số chiều mỗi head
        self.q = nn.Linear(dim, dim)          # phép chiếu Query
        self.k = nn.Linear(dim, dim)          # phép chiếu Key
        self.v = nn.Linear(dim, dim)          # phép chiếu Value
        self.out = nn.Linear(dim, dim)        # phép chiếu sau khi ghép các head

    def split_heads(self, t):
        B, N, D = t.shape
        # (B, N, D) -> (B, N, h, dh) -> (B, h, N, dh)
        return t.view(B, N, self.h, self.dh).transpose(1, 2)

    def forward(self, x):
        B, N, D = x.shape
        q = self.split_heads(self.q(x))
        k = self.split_heads(self.k(x))
        v = self.split_heads(self.v(x))
        # điểm chú ý: (B, h, N, dh) @ (B, h, dh, N) -> (B, h, N, N), chia sqrt(dh)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
        attn = torch.softmax(scores, dim=-1)  # mỗi hàng cộng lại bằng 1
        ctx = attn @ v                        # (B, h, N, dh)
        # ghép head: (B, h, N, dh) -> (B, N, h, dh) -> (B, N, D)
        ctx = ctx.transpose(1, 2).reshape(B, N, D)
        return self.out(ctx)


class TorchMSA(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x):
        out, _ = self.mha(x, x, x, need_weights=False)
        return out


MSA = {"custom": CustomMSA, "torch": TorchMSA}
