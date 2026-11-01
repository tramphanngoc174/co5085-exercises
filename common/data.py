"""Nạp dữ liệu và chia train/val/test CỐ ĐỊNH (dùng chung cho E1, E2, E3).

Quy ước (đề xuất của sinh viên, đề không quy định tỷ lệ):
  - Tập train gốc được chia thành train/val bằng seed cố định.
  - Tập test gốc giữ nguyên, chỉ dùng để báo cáo cuối cùng.
  - Chỉ số chia được lưu vào splits/*.json để E2, E3 dùng đúng cùng split.
"""
import json
import os

import torch
from torch.utils.data import DataLoader, Subset

# (mean, std) dùng để chuẩn hóa ảnh theo từng dataset
STATS = {
    "mnist": (0.1307, 0.3081),
    "fashion_mnist": (0.2860, 0.3530),
}

CLASS_NAMES = {
    "mnist": [str(i) for i in range(10)],
    "fashion_mnist": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                      "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
}


def make_split_indices(n_total, val_size, seed, split_path):
    """Tạo (hoặc đọc lại) chỉ số train/val. Có file rồi thì dùng lại, không tạo mới."""
    if os.path.exists(split_path):
        with open(split_path) as f:
            d = json.load(f)
        assert d["n_total"] == n_total and d["val_size"] == val_size, \
            "File split không khớp cấu hình hiện tại, hãy xóa file hoặc đổi tên."
        return d["train"], d["val"]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    val_idx, train_idx = perm[:val_size], perm[val_size:]
    os.makedirs(os.path.dirname(split_path) or ".", exist_ok=True)
    with open(split_path, "w") as f:
        json.dump({"seed": seed, "n_total": n_total, "val_size": val_size,
                   "train": train_idx, "val": val_idx}, f)
    return train_idx, val_idx


def build_loaders(train_full, test_set, split_path, val_size=6000, seed=42,
                  batch_size=128, num_workers=0):
    """Nhận 2 dataset đã có (tách riêng để dễ kiểm thử bằng dữ liệu giả)."""
    train_idx, val_idx = make_split_indices(len(train_full), val_size, seed, split_path)
    train_set, val_set = Subset(train_full, train_idx), Subset(train_full, val_idx)
    g = torch.Generator().manual_seed(seed)  # shuffle train tái lập được
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              generator=g, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=512, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def get_loaders(dataset="fashion_mnist", data_root="data", split_dir="splits",
                val_size=6000, seed=42, batch_size=128, num_workers=0):
    from torchvision import datasets, transforms
    mean, std = STATS[dataset]
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,), (std,))])
    cls = {"mnist": datasets.MNIST, "fashion_mnist": datasets.FashionMNIST}[dataset]
    train_full = cls(data_root, train=True, download=True, transform=tf)
    test_set = cls(data_root, train=False, download=True, transform=tf)
    split_path = os.path.join(split_dir, f"{dataset}_val{val_size}_seed{seed}.json")
    loaders = build_loaders(train_full, test_set, split_path, val_size, seed,
                            batch_size, num_workers)
    info = {"in_channels": 1, "img_size": 28, "num_classes": 10,
            "class_names": CLASS_NAMES[dataset], "mean": mean, "std": std}
    return (*loaders, info)
