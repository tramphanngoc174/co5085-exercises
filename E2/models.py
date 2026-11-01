"""E2: bộ phân loại Transformer nông (pre-LayerNorm): Tokenizer -> [MSA + FFN] x depth -> head trên token CLS."""
import torch.nn as nn

from E2.msa import MSA
from E2.tokenizers import TOKENIZERS


class EncoderBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, msa_cls):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = msa_cls(dim, heads)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(),
                                 nn.Linear(dim * mlp_ratio, dim))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # kết nối tắt quanh MSA
        x = x + self.ffn(self.ln2(x))    # kết nối tắt quanh FFN
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, tokenizer, msa_cls, dim=64, depth=2, heads=4, mlp_ratio=2, num_classes=10):
        super().__init__()
        self.tokenizer = tokenizer
        self.blocks = nn.ModuleList([EncoderBlock(dim, heads, mlp_ratio, msa_cls) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        t = self.tokenizer(x)             # (B, 1+N, dim)
        for blk in self.blocks:
            t = blk(t)
        return self.head(self.norm(t[:, 0]))   # phân loại bằng token CLS


def build_model(tok_name, msa_name, in_channels=1, img_size=28, num_classes=10,
                dim=64, depth=2, heads=4):
    tok = TOKENIZERS[tok_name](in_channels, img_size, dim)
    return TransformerClassifier(tok, MSA[msa_name], dim, depth, heads, 2, num_classes)
