"""Pascal VOC 2012 segmentation: split, datasets, colormap."""
import json, os, random
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms.functional as TF

MEAN, STD, IGNORE, NUM_CLASSES = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 255, 21
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
           "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]
EVAL_SIZE = 512  # VOC images are <= 500 px per side -> pad to 512 (pad label = 255, ignored)


def voc_cmap(n=256):
    cmap = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        c, r, g, b = i, 0, 0, 0
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j); g |= ((c >> 1) & 1) << (7 - j); b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        cmap[i] = (r, g, b)
    return cmap


def voc_dir(root):
    return os.path.join(root, "VOCdevkit", "VOC2012")


def ensure_voc(root):
    if not os.path.isdir(voc_dir(root)):
        torchvision.datasets.VOCSegmentation(root, year="2012", image_set="train", download=True)


def read_ids(root, name):
    p = os.path.join(voc_dir(root), "ImageSets", "Segmentation", name + ".txt")
    with open(p) as f:
        return [l.strip() for l in f if l.strip()]


def make_splits(root, seed=42, val_frac=0.1, path="splits/voc2012_seg_seed42.json"):
    """official train (1464) -> train/val (90/10, seed 42); official val (1449) is used as TEST."""
    if os.path.exists(path):
        return json.load(open(path))
    ids = sorted(read_ids(root, "train"))
    random.Random(seed).shuffle(ids)
    nv = int(round(len(ids) * val_frac))
    sp = {"train": sorted(ids[nv:]), "val": sorted(ids[:nv]), "test": sorted(read_ids(root, "val")), "seed": seed}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(sp, open(path, "w"))
    return sp


def _load(root, i):
    d = voc_dir(root)
    return Image.open(f"{d}/JPEGImages/{i}.jpg").convert("RGB"), Image.open(f"{d}/SegmentationClass/{i}.png")


class VOCTrain(Dataset):
    def __init__(self, root, ids, crop=384, scale=(0.5, 2.0)):
        self.root, self.ids, self.crop, self.scale = root, ids, crop, scale

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        img, m = _load(self.root, self.ids[k])
        s = random.uniform(*self.scale)
        w, h = img.size
        nw, nh = int(w * s + .5), int(h * s + .5)
        img, m = img.resize((nw, nh), Image.BILINEAR), m.resize((nw, nh), Image.NEAREST)
        if random.random() < 0.5:
            img, m = TF.hflip(img), TF.hflip(m)
        x = TF.normalize(TF.to_tensor(img), MEAN, STD)
        y = torch.from_numpy(np.array(m, dtype=np.int64))
        ph, pw = max(self.crop - nh, 0), max(self.crop - nw, 0)
        x, y = F.pad(x, (0, pw, 0, ph), value=0.0), F.pad(y, (0, pw, 0, ph), value=IGNORE)
        t, l = random.randint(0, y.shape[0] - self.crop), random.randint(0, y.shape[1] - self.crop)
        return x[:, t:t + self.crop, l:l + self.crop], y[t:t + self.crop, l:l + self.crop]


class VOCEval(Dataset):
    """No augmentation; pad to 512x512 (bottom/right). Metrics are computed at original resolution."""
    def __init__(self, root, ids):
        self.root, self.ids = root, ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        img, m = _load(self.root, self.ids[k])
        x = TF.normalize(TF.to_tensor(img), MEAN, STD)
        y = torch.from_numpy(np.array(m, dtype=np.int64))
        h, w = y.shape
        x = F.pad(x, (0, EVAL_SIZE - w, 0, EVAL_SIZE - h), value=0.0)
        y = F.pad(y, (0, EVAL_SIZE - w, 0, EVAL_SIZE - h), value=IGNORE)
        return x, y
