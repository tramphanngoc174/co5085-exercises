"""A1: EDA ngắn gọn cho báo cáo: lưới mỗi lớp một ảnh mẫu, thống kê kích thước và mean/std.
    python A1/eda.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from A1.data import load_cifar100_arrays, select_indices

out = "results/A1"
os.makedirs(out, exist_ok=True)
x, y, x_te, y_te, classes = load_cifar100_arrays("data")
tr_idx, va_idx = select_indices(y.numpy(), 450, 50, 42, "splits/cifar100_tr450_val50_seed42.json")
xt = x[torch.tensor(tr_idx)].float() / 255
stats = {"anh_goc": list(x.shape[1:]), "so_lop": len(classes), "train": len(tr_idx), "val": len(va_idx),
         "test": len(y_te), "mean_rgb": xt.mean((0, 2, 3)).tolist(), "std_rgb": xt.std((0, 2, 3)).tolist(),
         "anh_moi_lop": {"train": 450, "val": 50, "test": 100}}
with open(os.path.join(out, "eda_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)
print(json.dumps(stats, indent=2))

fig, axes = plt.subplots(10, 10, figsize=(13, 13.5))
for c, ax in enumerate(axes.ravel()):
    i = int((y == c).nonzero()[0])
    ax.imshow(x[i].permute(1, 2, 0).numpy()); ax.axis("off")
    ax.set_title(classes[c], fontsize=6)
fig.suptitle("CIFAR-100: mỗi lớp một ảnh mẫu (32x32)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(out, "eda_samples.png"), dpi=130)
print("đã lưu", os.path.join(out, "eda_samples.png"))
