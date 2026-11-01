"""E2: huấn luyện + so sánh (i) MSA tự viết vs PyTorch (cùng tokenize), (ii) các cách tokenize.

Chạy từ thư mục gốc của repo:
    python E2/run_e2.py --dataset fashion_mnist --epochs 10
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from common.data import get_loaders
from common.plots import plot_curves
from common.train_utils import count_params, evaluate, get_device, set_seed, train_model
from E2.models import build_model


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
    hist_all, rows = {}, []
    for tok in args.tokenizers:
        for msa in args.msa:
            name = f"{tok}_{msa}"
            log(f"\n===== tokenizer={tok} | MSA={msa} =====")
            set_seed(args.seed)
            model = build_model(tok, msa, info["in_channels"], info["img_size"],
                                info["num_classes"], args.dim, args.depth, args.heads).to(device)
            n_params = count_params(model)
            log(f"số tham số: {n_params:,} | số token (kèm CLS): {model.tokenizer.num_tokens}")
            t0 = time.time()
            hist = train_model(model, train_loader, val_loader, args.epochs, args.lr, device, log)
            total = time.time() - t0
            test_loss, test_acc, _, _ = evaluate(model, test_loader, loss_fn, device)
            val_acc = max(hist["val_acc"])
            log(f"best epoch {hist['best_epoch']} | val acc {val_acc:.4f} | test acc {test_acc:.4f} | "
                f"tổng thời gian train {total:.0f}s")
            mdir = os.path.join(out, name)
            os.makedirs(mdir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(mdir, "model.pt"))
            with open(os.path.join(mdir, "history.json"), "w") as f:
                json.dump(hist, f, indent=2)
            hist_all[name] = hist
            rows.append({"tokenizer": tok, "msa": msa, "params": n_params,
                         "tokens": model.tokenizer.num_tokens, "best_epoch": hist["best_epoch"],
                         "val_acc": val_acc, "test_acc": test_acc, "test_loss": test_loss,
                         "sec_per_epoch": sum(hist["epoch_time"]) / len(hist["epoch_time"]),
                         "total_train_sec": total})

    # Biểu đồ: (i) custom vs torch trong từng tokenizer; (ii) các tokenizer với nhau (MSA tự viết)
    for tok in args.tokenizers:
        sub = {m: hist_all[f"{tok}_{m}"] for m in args.msa}
        plot_curves(sub, os.path.join(out, f"curves_{tok}.png"))
    if "custom" in args.msa:
        sub = {t: hist_all[f"{t}_custom"] for t in args.tokenizers}
        plot_curves(sub, os.path.join(out, "curves_tokenizers.png"))

    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    md = ["| Tokenizer | MSA | Tham số | Số token | Best epoch | Val acc | Test acc | s/epoch | Tổng train (s) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['tokenizer']} | {r['msa']} | {r['params']:,} | {r['tokens']} | {r['best_epoch']} | "
                  f"{r['val_acc']:.4f} | {r['test_acc']:.4f} | {r['sec_per_epoch']:.1f} | {r['total_train_sec']:.0f} |")
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    log("\n" + "\n".join(md))
    log_file.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="fashion_mnist", choices=["mnist", "fashion_mnist"])
    p.add_argument("--tokenizers", nargs="+", default=["patch", "row", "cnnstem"],
                   choices=["patch", "row", "cnnstem"])
    p.add_argument("--msa", nargs="+", default=["custom", "torch"], choices=["custom", "torch"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-size", type=int, default=6000)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", default="results/E2")
    args = p.parse_args()
    device = get_device()
    print("device:", device)
    tr, va, te, info = get_loaders(args.dataset, args.data_root, "splits",
                                   args.val_size, args.seed, args.batch_size)
    run_all(tr, va, te, info, args, device)


if __name__ == "__main__":
    main()
