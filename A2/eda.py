"""EDA of VOC2012 segmentation train/val/test split -> results/A2/eda/*"""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import data

root, out = "data", "results/A2/eda"; os.makedirs(out, exist_ok=True)
data.ensure_voc(root); sp = data.make_splits(root)
stats = {}
for part in ["train", "val", "test"]:
    pix, img_cnt, sizes = np.zeros(21, np.int64), np.zeros(21, np.int64), []
    for i in sp[part]:
        m = np.array(Image.open(f"{data.voc_dir(root)}/SegmentationClass/{i}.png")); sizes.append(m.shape)
        b = np.bincount(m[m != 255].ravel(), minlength=21)[:21]; pix += b; img_cnt += (b > 0)
    stats[part] = {"n": len(sp[part]), "pix_frac": (pix / pix.sum()).tolist(), "img_count": img_cnt.tolist(),
                   "mean_hw": np.mean(sizes, 0).tolist(), "max_hw": np.max(sizes, 0).tolist()}
json.dump(stats, open(f"{out}/eda_stats.json", "w"), indent=1)
fig, ax = plt.subplots(1, 2, figsize=(13, 4)); x = np.arange(21)
ax[0].bar(x, stats["train"]["pix_frac"]); ax[0].set_yscale("log"); ax[0].set_title("Pixel fraction per class (train, log)")
ax[1].bar(x, stats["train"]["img_count"]); ax[1].set_title("Images containing class (train)")
for a in ax: a.set_xticks(x); a.set_xticklabels(data.CLASSES, rotation=90, fontsize=7)
plt.tight_layout(); plt.savefig(f"{out}/class_distribution.png", dpi=150); plt.close()
cm = data.voc_cmap(); fig, ax = plt.subplots(2, 6, figsize=(16, 5.5))
for k, i in enumerate(sp["train"][:6]):
    ax[0, k].imshow(Image.open(f"{data.voc_dir(root)}/JPEGImages/{i}.jpg")); ax[0, k].axis("off")
    ax[1, k].imshow(cm[np.array(Image.open(f"{data.voc_dir(root)}/SegmentationClass/{i}.png"))]); ax[1, k].axis("off")
plt.tight_layout(); plt.savefig(f"{out}/samples.png", dpi=130); plt.close()
print("EDA saved to", out)
