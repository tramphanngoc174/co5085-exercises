"""E3: LSTM / GRU phân loại ảnh sau khi biến ảnh thành CHUỖI.

Ba cách tạo chuỗi từ ảnh (B, C, H, W):
  row   : mỗi HÀNG là một bước thời gian  -> (B, H, C*W)        (28 bước x 28 giá trị)
  col   : mỗi CỘT là một bước thời gian   -> (B, W, C*H)        (28 bước x 28 giá trị)
  patch : ô patch x patch quét theo hàng (trái->phải, trên->dưới) -> (B, nh*nw, C*patch*patch)  (49 bước x 16 giá trị)
Phân loại bằng trạng thái ẩn cuối cùng của RNN.
"""
import torch.nn as nn


class SeqClassifier(nn.Module):
    def __init__(self, rnn_type="lstm", representation="row", in_channels=1, img_size=28,
                 hidden=128, num_layers=1, num_classes=10, patch=4):
        super().__init__()
        self.rep, self.patch = representation, patch
        if representation == "row":
            feat, self.seq_len = in_channels * img_size, img_size
        elif representation == "col":
            feat, self.seq_len = in_channels * img_size, img_size
        elif representation == "patch":
            assert img_size % patch == 0
            feat, self.seq_len = in_channels * patch * patch, (img_size // patch) ** 2
        else:
            raise ValueError(representation)
        self.feat = feat
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[rnn_type]
        self.rnn = rnn_cls(feat, hidden, num_layers, batch_first=True)
        self.head = nn.Linear(hidden, num_classes)

    def to_seq(self, x):
        if self.rep == "row":
            return x.permute(0, 2, 1, 3).flatten(2)          # (B, H, C*W)
        if self.rep == "col":
            return x.permute(0, 3, 1, 2).flatten(2)          # (B, W, C*H)
        p = self.patch                                         # patch
        x = x.unfold(2, p, p).unfold(3, p, p)                  # (B, C, nh, nw, p, p)
        return x.permute(0, 2, 3, 1, 4, 5).flatten(1, 2).flatten(2)   # (B, nh*nw, C*p*p)

    def forward(self, x):
        _, h = self.rnn(self.to_seq(x))
        h_n = h[0] if isinstance(h, tuple) else h              # LSTM trả (h_n, c_n); GRU trả h_n
        return self.head(h_n[-1])                              # trạng thái ẩn cuối của tầng cuối
