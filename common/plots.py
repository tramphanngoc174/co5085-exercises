"""Vẽ biểu đồ dùng chung: đường cong học, confusion matrix, mẫu dự đoán sai."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
        cm[t, p] += 1          # hàng = nhãn thật, cột = dự đoán
    return cm


def plot_curves(histories, path):
    """histories: {tên_mô_hình: history}. Vẽ loss (trái) và accuracy (phải)."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, h in histories.items():
        ep = range(1, len(h["train_loss"]) + 1)
        ax[0].plot(ep, h["train_loss"], "--", label=f"{name} train")
        ax[0].plot(ep, h["val_loss"], "-", label=f"{name} val")
        ax[1].plot(ep, h["train_acc"], "--", label=f"{name} train")
        ax[1].plot(ep, h["val_acc"], "-", label=f"{name} val")
    ax[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    ax[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    for a in ax:
        a.grid(alpha=.3); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_confusion(cm, class_names, title, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set(title=title, xlabel="Dự đoán", ylabel="Nhãn thật",
           xticks=range(len(class_names)), yticks=range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=60, ha="right", fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    thr = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                    color="white" if cm[i, j] > thr else "black")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_errors(images, y_true, y_pred, class_names, title, path, n=12):
    """images: tensor (N,C,H,W) của các mẫu bị dự đoán sai (đã chuẩn hóa hay chưa đều được)."""
    n = min(n, len(images))
    cols = 6
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.3 * rows), squeeze=False)
    for a in axes.ravel():
        a.axis("off")
    for k in range(n):
        a = axes[k // cols][k % cols]
        img = images[k].squeeze().numpy() if images[k].shape[0] == 1 \
            else images[k].permute(1, 2, 0).numpy()
        a.imshow(img, cmap="gray" if img.ndim == 2 else None)
        a.set_title(f"thật: {class_names[y_true[k]]}\nđoán: {class_names[y_pred[k]]}",
                    fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
