"""A1: CIFAR-100 (subset có kiểm soát), Dataset/DataLoader.
Ảnh gốc 32x32 giữ dạng uint8; việc augmentation + phóng to + chuẩn hóa làm trên GPU (xem A1/engine.py)."""
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TensorImageDataset(Dataset):
    def __init__(self, images, labels):          # images: uint8 (N,3,32,32); labels: long (N,)
        self.images, self.labels = images, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.images[i], self.labels[i]


def select_indices(labels, train_per_class, val_per_class, seed, split_path):
    """Mỗi lớp: lấy ngẫu nhiên (seed cố định) val_per_class ảnh cho validation và train_per_class ảnh
    KHÔNG trùng cho train. Lưu chỉ số vào split_path để tái lập."""
    if os.path.exists(split_path):
        with open(split_path) as f:
            d = json.load(f)
        return d["train"], d["val"]
    rs = np.random.RandomState(seed)
    labels = np.asarray(labels)
    train_idx, val_idx = [], []
    for c in range(int(labels.max()) + 1):
        idx = np.where(labels == c)[0]
        rs.shuffle(idx)
        val_idx += idx[:val_per_class].tolist()
        train_idx += idx[val_per_class:val_per_class + train_per_class].tolist()
    os.makedirs(os.path.dirname(split_path) or ".", exist_ok=True)
    with open(split_path, "w") as f:
        json.dump({"seed": seed, "train_per_class": train_per_class, "val_per_class": val_per_class,
                   "train": train_idx, "val": val_idx}, f)
    return train_idx, val_idx


def make_loaders(x_tr, y_tr, x_va, y_va, x_te, y_te, batch_size, seed):
    g = torch.Generator().manual_seed(seed)
    train = DataLoader(TensorImageDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True, generator=g)
    val = DataLoader(TensorImageDataset(x_va, y_va), batch_size=256, shuffle=False)
    test = DataLoader(TensorImageDataset(x_te, y_te), batch_size=256, shuffle=False)
    return train, val, test


def load_cifar100_arrays(root="data"):
    from torchvision import datasets
    tr = datasets.CIFAR100(root, train=True, download=True)
    te = datasets.CIFAR100(root, train=False, download=True)
    to_t = lambda a: torch.from_numpy(a).permute(0, 3, 1, 2).contiguous()   # (N,32,32,3) -> (N,3,32,32)
    return (to_t(tr.data), torch.tensor(tr.targets), to_t(te.data), torch.tensor(te.targets), tr.classes)


def get_cifar100(root="data", split_dir="splits", train_per_class=200, val_per_class=50,
                 seed=42, batch_size=128):
    x, y, x_te, y_te, classes = load_cifar100_arrays(root)
    split_path = os.path.join(split_dir, f"cifar100_tr{train_per_class}_val{val_per_class}_seed{seed}.json")
    tr_idx, va_idx = select_indices(y.numpy(), train_per_class, val_per_class, seed, split_path)
    tr_idx, va_idx = torch.tensor(tr_idx), torch.tensor(va_idx)
    loaders = make_loaders(x[tr_idx], y[tr_idx], x[va_idx], y[va_idx], x_te, y_te, batch_size, seed)
    info = {"num_classes": 100, "class_names": classes, "n_train": len(tr_idx),
            "n_val": len(va_idx), "n_test": len(y_te)}
    return (*loaders, info)
