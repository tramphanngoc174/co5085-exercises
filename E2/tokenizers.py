"""E2: ba cách biến ảnh thành chuỗi token. Mỗi tokenizer nhận ảnh (B, C, H, W)
và trả về (B, 1 + N, dim): token CLS ở vị trí 0, theo sau là N token + positional embedding học được."""
import torch
import torch.nn as nn


class TokenizerBase(nn.Module):
    def __init__(self, num_tokens, dim):
        super().__init__()
        self.num_tokens = num_tokens + 1      # đã tính CLS
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, num_tokens + 1, dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def tokens(self, x):                      # lớp con tự định nghĩa: -> (B, N, dim)
        raise NotImplementedError

    def forward(self, x):
        t = self.tokens(x)
        cls = self.cls.expand(t.size(0), -1, -1)
        return torch.cat([cls, t], dim=1) + self.pos


class PatchTokenizer(TokenizerBase):
    """Chia ảnh thành lưới ô patch x patch không chồng lấn, làm phẳng mỗi ô rồi qua Linear."""
    def __init__(self, in_channels, img_size, dim, patch=4):
        assert img_size % patch == 0
        super().__init__((img_size // patch) ** 2, dim)
        self.p = patch
        self.proj = nn.Linear(in_channels * patch * patch, dim)

    def tokens(self, x):
        p = self.p
        x = x.unfold(2, p, p).unfold(3, p, p)            # (B, C, nh, nw, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).flatten(1, 2).flatten(2)   # (B, nh*nw, C*p*p)
        return self.proj(x)


class RowTokenizer(TokenizerBase):
    """Mỗi hàng của ảnh (C x W giá trị) là một token."""
    def __init__(self, in_channels, img_size, dim):
        super().__init__(img_size, dim)
        self.proj = nn.Linear(in_channels * img_size, dim)

    def tokens(self, x):
        x = x.permute(0, 2, 1, 3).flatten(2)             # (B, H, C*W)
        return self.proj(x)


class CNNStemTokenizer(TokenizerBase):
    """CNN nhỏ (2 khối conv-BN-ReLU-pool) rút feature map; mỗi vị trí không gian là một token."""
    def __init__(self, in_channels, img_size, dim):
        super().__init__((img_size // 4) ** 2, dim)
        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                                 nn.ReLU(), nn.MaxPool2d(2))
        self.stem = nn.Sequential(block(in_channels, 32), block(32, 64))
        self.proj = nn.Linear(64, dim)

    def tokens(self, x):
        f = self.stem(x)                                  # (B, 64, H/4, W/4)
        f = f.flatten(2).transpose(1, 2)                  # (B, (H/4)*(W/4), 64)
        return self.proj(f)


TOKENIZERS = {"patch": PatchTokenizer, "row": RowTokenizer, "cnnstem": CNNStemTokenizer}
