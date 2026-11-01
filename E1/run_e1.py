"""E1: huấn luyện + đánh giá Softmax / MLP / CNN, tạo bảng, biểu đồ, phân tích lỗi.

Chạy từ thư mục gốc của repo:
    python E1/run_e1.py --dataset fashion_mnist --epochs 15
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from common.data import get_loaders
from common.plots import confusion_matrix, plot_confusion, plot_curves, plot_errors
from common.train_utils import (count_params, evaluate, get_device, set_seed,
                                train_model)
from E1.models import MODELS


def run_all(train_loader, val_loader, test_loader, info, args, device):
    out = args.out
    os.makedirs(out, exist_ok=True)
    log_file = open(os.path.join(out, "log.txt"), "w")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n"); log_file.flush()

    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({**vars(args), "device": str(device)}, f, indent=2)

    loss_fn = torch.nn.CrossEntropyLoss()
    histories, rows = {}, []
    for name in args.models:
        log(f"\n===== {name} =====")
        set_seed(args.seed)                      # cùng seed cho mỗi mô hình
        model = MODELS[name](info["in_channels"], info["img_size"],
                             info["num_classes"]).to(device)
        n_params = count_params(model)
        log(f"số tham số: {n_params:,}")
        hist = train_model(model, train_loader, val_loader, args.epochs, args.lr,
                           device, log)
        histories[name] = hist

        # Đánh giá cuối trên TEST bằng trọng số epoch có val accuracy tốt nhất
        _, val_acc, _, _ = evaluate(model, val_loader, loss_fn, device)
        test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, loss_fn, device)
        log(f"best epoch {hist['best_epoch']} | val acc {val_acc:.4f} | "
            f"test acc {test_acc:.4f} | test loss {test_loss:.4f}")

        mdir = os.path.join(out, name)
        os.makedirs(mdir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(mdir, "model.pt"))
        with open(os.path.join(mdir, "history.json"), "w") as f:
            json.dump(hist, f, indent=2)

        cm = confusion_matrix(y_true, y_pred, info["num_classes"])
        plot_confusion(cm, info["class_names"], f"Confusion matrix – {name} (test)",
                       os.path.join(mdir, "confusion_matrix.png"))
        wrong = (y_true != y_pred).nonzero().squeeze(1)[:12]
        if len(wrong) > 0:
            imgs = torch.stack([test_loader.dataset[i][0] for i in wrong.tolist()])
            # (ảnh đã chuẩn hóa nên màu hơi khác ảnh gốc, chỉ để minh họa lỗi)
            plot_errors(imgs, y_true[wrong].tolist(), y_pred[wrong].tolist(),
                        info["class_names"], f"Mẫu dự đoán sai – {name}",
                        os.path.join(mdir, "errors.png"))
        else:
            log("Không có mẫu dự đoán sai, bỏ qua errors.png")

        mean_t = sum(hist["epoch_time"]) / len(hist["epoch_time"])
        rows.append({"model": name, "params": n_params, "best_epoch": hist["best_epoch"],
                     "val_acc": val_acc, "test_acc": test_acc, "test_loss": test_loss,
                     "sec_per_epoch": mean_t})

    plot_curves(histories, os.path.join(out, "curves.png"))
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    md = ["| Mô hình | Tham số | Best epoch | Val acc | Test acc | s/epoch |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['params']:,} | {r['best_epoch']} | "
                  f"{r['val_acc']:.4f} | {r['test_acc']:.4f} | {r['sec_per_epoch']:.1f} |")
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    log("\n" + "\n".join(md))
    log_file.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="fashion_mnist", choices=["mnist", "fashion_mnist"])
    p.add_argument("--models", nargs="+", default=["softmax", "mlp", "cnn"],
                   choices=list(MODELS))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-size", type=int, default=6000)
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", default="results/E1")
    args = p.parse_args()
    device = get_device()
    print("device:", device)
    tr, va, te, info = get_loaders(args.dataset, args.data_root, "splits",
                                   args.val_size, args.seed, args.batch_size)
    run_all(tr, va, te, info, args, device)


if __name__ == "__main__":
    main()
