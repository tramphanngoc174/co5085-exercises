"""A1: so sánh CNN (ResNet-18) vs Transformer (ViT-Tiny) với 4 chế độ: scratch / head / partial / full.

Chạy từ thư mục gốc của repo (mặc định: toàn bộ CIFAR-100, ảnh 224x224, 15 epoch, 8 cấu hình):
    python A1/run_a1.py
Mỗi cấu hình xong được lưu vào results/A1/<arch>_<mode>/result.json. Chạy lại cùng lệnh sẽ BỎ QUA
các cấu hình đã xong (an toàn khi Colab bị ngắt); bảng/biểu đồ tổng hợp gộp mọi cấu hình đã có.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch

from A1.data import get_cifar100
from A1.engine import Prep, evaluate, fit, macro_f1
from A1.models import MODES, apply_freeze, build_model, count_params, param_groups
from common.plots import plot_curves
from common.train_utils import get_device, set_seed

ARCH_LABEL = {"resnet18": "ResNet-18 (CNN)", "vit_tiny": "ViT-Tiny/16 (Transformer)"}


def top_confusions(cm, names, k=10):
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(off, axis=None)[::-1][:k]
    return [(names[i // cm.shape[0]], names[i % cm.shape[0]], int(off.flat[i])) for i in flat]


def summarize(out, info, log):
    """Đọc mọi cấu hình đã có trên đĩa, tạo summary.md/json, biểu đồ và phân tích lỗi."""
    rows, hists, cms = [], {}, {}
    for arch in ARCH_LABEL:
        for mode in MODES:
            d = os.path.join(out, f"{arch}_{mode}")
            if os.path.exists(os.path.join(d, "result.json")):
                with open(os.path.join(d, "result.json")) as f:
                    rows.append(json.load(f))
                with open(os.path.join(d, "history.json")) as f:
                    hists[f"{arch}_{mode}"] = json.load(f)
                cms[f"{arch}_{mode}"] = np.load(os.path.join(d, "confusion.npy"))
    if not rows:
        return
    archs = [a for a in ARCH_LABEL if any(r["arch"] == a for r in rows)]
    modes = [m for m in MODES if any(r["mode"] == m for r in rows)]
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    md = ["| Kiến trúc | Chế độ | Tham số tổng | Trainable | Best epoch | Val acc | Test acc | Macro-F1 | s/epoch | Tổng (s) | Bộ nhớ đỉnh (MB) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['arch']} | {r['mode']} | {r['params_total']:,} | {r['params_trainable']:,} | {r['best_epoch']} | "
                  f"{r['val_acc']:.4f} | {r['test_acc']:.4f} | {r['macro_f1']:.4f} | {r['sec_per_epoch']:.1f} | "
                  f"{r['total_sec']:.0f} | {r['peak_mem_mb']:.0f} |")
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    log("\n" + "\n".join(md))

    for arch in archs:
        sub = {m: hists[f"{arch}_{m}"] for m in modes if f"{arch}_{m}" in hists}
        plot_curves(sub, os.path.join(out, f"curves_{arch}.png"))
    xs = np.arange(len(modes))
    w = 0.8 / len(archs)

    def bars(ax, key, title):
        for i, arch in enumerate(archs):
            vals = []
            for m in modes:
                v = [r[key] for r in rows if r["arch"] == arch and r["mode"] == m]
                vals.append(v[0] if v else 0.0)
            ax.bar(xs + i * w - 0.4 + w / 2, vals, w, label=ARCH_LABEL[arch])
        ax.set_xticks(xs); ax.set_xticklabels(modes); ax.set_title(title); ax.grid(axis="y", alpha=.3)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars(ax, "test_acc", "Test accuracy theo chế độ huấn luyện"); ax.set_ylabel("Accuracy"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out, "accuracy_bar.png"), dpi=150); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    bars(ax[0], "sec_per_epoch", "Thời gian mỗi epoch (s)")
    bars(ax[1], "peak_mem_mb", "Bộ nhớ GPU đỉnh (MB)"); ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "cost_bar.png"), dpi=150); plt.close(fig)

    lines = []
    for arch in archs:
        best = max((r for r in rows if r["arch"] == arch), key=lambda r: r["test_acc"])
        cm = cms[f"{arch}_{best['mode']}"]
        lines.append(f"### {ARCH_LABEL[arch]} – chế độ tốt nhất: {best['mode']} (test acc {best['test_acc']:.4f})\n")
        lines.append("| Nhãn thật | Bị đoán là | Số ảnh |\n|---|---|---|")
        for a, b, c in top_confusions(cm, info["class_names"]):
            lines.append(f"| {a} | {b} | {c} |")
        lines.append("")
    with open(os.path.join(out, "top_confusions.md"), "w") as f:
        f.write("\n".join(lines))
    log("\n" + "\n".join(lines))


def run_all(train_loader, val_loader, test_loader, info, args, device):
    out = args.out
    os.makedirs(out, exist_ok=True)
    log_file = open(os.path.join(out, "log.txt"), "a")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n"); log_file.flush()

    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({**vars(args), "device": str(device), "timm": timm.__version__,
                   "torch": torch.__version__}, f, indent=2)
    log(f"train {info['n_train']} | val {info['n_val']} | test {info['n_test']} | lớp {info['num_classes']}")
    use_amp = bool(args.amp and device.type == "cuda")
    K = info["num_classes"]
    loss_fn = torch.nn.CrossEntropyLoss()

    for arch in args.archs:
        for mode in args.modes:
            name = f"{arch}_{mode}"
            mdir = os.path.join(out, name)
            os.makedirs(mdir, exist_ok=True)
            if args.resume and os.path.exists(os.path.join(mdir, "result.json")):
                log(f"\n===== {name}: đã có kết quả, bỏ qua =====")
                continue
            log(f"\n===== {ARCH_LABEL[arch]} | chế độ = {mode} =====")
            set_seed(args.seed)
            model = build_model(arch, mode, K, args.img_size, args.offline).to(device)
            cfg = timm.data.resolve_data_config({}, model=model)
            prep = Prep(device, args.img_size, cfg["mean"], cfg["std"])
            frozen_bns = apply_freeze(model, arch, mode)
            total, trainable = count_params(model)
            log(f"tham số: tổng {total:,} | trainable {trainable:,} ({100 * trainable / total:.1f}%)")
            groups = param_groups(model, arch, mode, args.lr_scratch, args.lr_ft, args.lr_head)
            optimizer = torch.optim.AdamW(groups, weight_decay=args.wd)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            hist = fit(model, train_loader, val_loader, optimizer, scheduler, args.epochs, prep, device,
                       frozen_bns, use_amp, K, log)
            total_sec = time.time() - t0
            peak_mb = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else 0.0
            _, te_acc, cm = evaluate(model, test_loader, loss_fn, prep, device, use_amp, K)
            f1 = macro_f1(cm)
            va_acc = max(hist["val_acc"])
            log(f"best epoch {hist['best_epoch']} | val acc {va_acc:.4f} | test acc {te_acc:.4f} | "
                f"macro-F1 {f1:.4f} | {total_sec:.0f}s | peak mem {peak_mb:.0f} MB")
            row = {"arch": arch, "mode": mode, "params_total": total, "params_trainable": trainable,
                   "best_epoch": hist["best_epoch"], "val_acc": va_acc, "test_acc": float(te_acc),
                   "macro_f1": f1, "sec_per_epoch": float(np.mean(hist["epoch_time"])),
                   "total_sec": total_sec, "peak_mem_mb": peak_mb}
            with open(os.path.join(mdir, "history.json"), "w") as f:
                json.dump(hist, f, indent=2)
            np.save(os.path.join(mdir, "confusion.npy"), cm)
            with open(os.path.join(mdir, "result.json"), "w") as f:    # ghi cuối cùng = đánh dấu đã xong
                json.dump(row, f, indent=2)
            del model, optimizer
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summarize(out, info, log)
    log_file.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archs", nargs="+", default=["resnet18", "vit_tiny"], choices=["resnet18", "vit_tiny"])
    p.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-per-class", type=int, default=450)
    p.add_argument("--val-per-class", type=int, default=50)
    p.add_argument("--lr-scratch", type=float, default=1e-3)
    p.add_argument("--lr-ft", type=float, default=1e-4)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--offline", action="store_true", help="không tải trọng số pretrained (chỉ để kiểm thử)")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", default="results/A1")
    args = p.parse_args()
    device = get_device()
    print("device:", device)
    tr, va, te, info = get_cifar100(args.data_root, "splits", args.train_per_class, args.val_per_class,
                                    args.seed, args.batch_size)
    run_all(tr, va, te, info, args, device)


if __name__ == "__main__":
    main()
