"""E3: LSTM / GRU trên chuỗi ảnh + so sánh với baseline MLP/CNN của E1 + thí nghiệm xáo trộn hàng.

Chạy từ thư mục gốc của repo:
    python E3/run_e3.py --dataset fashion_mnist --epochs 10
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
import torch

from common.data import get_loaders
from common.plots import plot_curves
from common.train_utils import count_params, evaluate, get_device, set_seed, train_model
from E1.models import MODELS as E1_MODELS
from E3.models import SeqClassifier


@torch.no_grad()
def accuracy(model, loader, device, perm=None):
    """Accuracy; nếu có perm thì xáo thứ tự HÀNG của ảnh (chiều cao) trước khi đưa vào mô hình."""
    model.eval()
    correct = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if perm is not None:
            x = x[:, :, perm, :]
        correct += (model(x).argmax(1) == y).sum().item()
        n += y.size(0)
    return correct / n


def get_baseline(name, info, args, loaders, device, log):
    """Nạp checkpoint E1 (cùng split); nếu không có thì huấn luyện lại bằng đúng cấu hình E3."""
    tr, va, _ = loaders
    model = E1_MODELS[name](info["in_channels"], info["img_size"], info["num_classes"]).to(device)
    ckpt = os.path.join(args.e1_dir, name, "model.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        return model, "checkpoint E1"
    log(f"[CẢNH BÁO] Không thấy {ckpt}; huấn luyện lại baseline {name} ({args.epochs} epoch).")
    set_seed(args.seed)
    train_model(model, tr, va, args.epochs, args.lr, device, log)
    return model, f"huấn luyện lại ({args.epochs} epoch)"


def run_all(train_loader, val_loader, test_loader, info, args, device):
    out = args.out
    os.makedirs(out, exist_ok=True)
    log_file = open(os.path.join(out, "log.txt"), "w")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n"); log_file.flush()

    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({**vars(args), "device": str(device)}, f, indent=2)
    loaders = (train_loader, val_loader, test_loader)
    loss_fn = torch.nn.CrossEntropyLoss()
    models, rows, hist_all = {}, [], {}

    # ---- 1) Baseline từ E1 ----
    for name in ["mlp", "cnn"]:
        model, src = get_baseline(name, info, args, loaders, device, log)
        _, va_acc, _, _ = evaluate(model, val_loader, loss_fn, device)
        _, te_acc, _, _ = evaluate(model, test_loader, loss_fn, device)
        log(f"\n===== baseline {name} ({src}) | params {count_params(model):,} | val {va_acc:.4f} | test {te_acc:.4f}")
        models[f"{name} (E1)"] = model
        rows.append({"model": f"{name} (E1)", "seq": "-", "params": count_params(model), "best_epoch": None,
                     "val_acc": va_acc, "test_acc": te_acc, "sec_per_epoch": None, "total_train_sec": None,
                     "note": src})

    # ---- 2) LSTM / GRU trên các cách tạo chuỗi ----
    for rep in args.representations:
        for rnn in args.rnn:
            name = f"{rnn}_{rep}"
            log(f"\n===== {rnn.upper()} | chuỗi = {rep} =====")
            set_seed(args.seed)
            model = SeqClassifier(rnn, rep, info["in_channels"], info["img_size"], args.hidden,
                                  1, info["num_classes"]).to(device)
            n_params = count_params(model)
            log(f"số tham số: {n_params:,} | chuỗi: {model.seq_len} bước x {model.feat} giá trị")
            t0 = time.time()
            hist = train_model(model, train_loader, val_loader, args.epochs, args.lr, device, log)
            total = time.time() - t0
            _, te_acc, _, _ = evaluate(model, test_loader, loss_fn, device)
            va_acc = max(hist["val_acc"])
            log(f"best epoch {hist['best_epoch']} | val acc {va_acc:.4f} | test acc {te_acc:.4f} | train {total:.0f}s")
            mdir = os.path.join(out, name)
            os.makedirs(mdir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(mdir, "model.pt"))
            with open(os.path.join(mdir, "history.json"), "w") as f:
                json.dump(hist, f, indent=2)
            hist_all[name] = hist
            models[name] = model
            rows.append({"model": name, "seq": f"{model.seq_len}x{model.feat}", "params": n_params,
                         "best_epoch": hist["best_epoch"], "val_acc": va_acc, "test_acc": te_acc,
                         "sec_per_epoch": sum(hist["epoch_time"]) / len(hist["epoch_time"]),
                         "total_train_sec": total, "note": ""})

    # Biểu đồ đường cong học: (i) LSTM vs GRU cho từng cách tạo chuỗi, (ii) các cách tạo chuỗi cho từng loại RNN
    for rep in args.representations:
        sub = {r: hist_all[f"{r}_{rep}"] for r in args.rnn}
        plot_curves(sub, os.path.join(out, f"curves_{rep}.png"))
    for rnn in args.rnn:
        sub = {rep: hist_all[f"{rnn}_{rep}"] for rep in args.representations}
        plot_curves(sub, os.path.join(out, f"curves_representations_{rnn}.png"))

    def f4(v):
        return "–" if v is None else f"{v:.4f}"
    def f1(v):
        return "–" if v is None else f"{v:.1f}"
    md = ["| Mô hình | Chuỗi (bước x giá trị) | Tham số | Best epoch | Val acc | Test acc | s/epoch | Ghi chú |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['seq']} | {r['params']:,} | {r['best_epoch'] if r['best_epoch'] else '–'} | "
                  f"{f4(r['val_acc'])} | {f4(r['test_acc'])} | {f1(r['sec_per_epoch'])} | {r['note']} |")
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    log("\n" + "\n".join(md))

    # ---- 3) Thí nghiệm xáo trộn thứ tự hàng (chỉ ở thời điểm TEST, không huấn luyện lại) ----
    H = info["img_size"]
    log(f"\n===== Thí nghiệm xáo hàng ảnh test ({args.shuffle_perms} hoán vị ngẫu nhiên + đảo ngược) =====")
    perms = []
    for i in range(args.shuffle_perms):
        g = torch.Generator().manual_seed(1000 + i)
        perms.append(torch.randperm(H, generator=g).to(device))
    reverse = torch.arange(H - 1, -1, -1).to(device)
    srows = []
    for name, model in models.items():
        orig = accuracy(model, test_loader, device)
        sh = [accuracy(model, test_loader, device, p) for p in perms]
        rev = accuracy(model, test_loader, device, reverse)
        mean = float(np.mean(sh))
        std = float(np.std(sh, ddof=1)) if len(sh) > 1 else 0.0
        srows.append({"model": name, "params": count_params(model), "orig": orig, "shuffled": sh,
                      "shuffled_mean": mean, "shuffled_std": std, "drop_pts": 100 * (orig - mean),
                      "reversed": rev, "reversed_drop_pts": 100 * (orig - rev)})
        log(f"{name:12s} gốc {orig:.4f} | xáo {mean:.4f} ± {std:.4f} (giảm {100*(orig-mean):.2f} điểm) | "
            f"đảo ngược {rev:.4f} (giảm {100*(orig-rev):.2f} điểm)")
    with open(os.path.join(out, "shuffle.json"), "w") as f:
        json.dump(srows, f, indent=2)
    md2 = ["| Mô hình | Tham số | Acc gốc | Acc xáo hàng (TB ± std) | Giảm (điểm %) | Acc đảo ngược hàng | Giảm (điểm %) |",
           "|---|---|---|---|---|---|---|"]
    for r in srows:
        md2.append(f"| {r['model']} | {r['params']:,} | {r['orig']:.4f} | {r['shuffled_mean']:.4f} ± {r['shuffled_std']:.4f} | "
                   f"{r['drop_pts']:.2f} | {r['reversed']:.4f} | {r['reversed_drop_pts']:.2f} |")
    with open(os.path.join(out, "shuffle.md"), "w") as f:
        f.write("\n".join(md2) + "\n")
    log("\n" + "\n".join(md2))

    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(srows) + 3), 4.2))
    xs = np.arange(len(srows))
    ax.bar(xs - 0.2, [r["orig"] for r in srows], 0.4, label="ảnh gốc")
    ax.bar(xs + 0.2, [r["shuffled_mean"] for r in srows], 0.4,
           yerr=[r["shuffled_std"] for r in srows], capsize=3, label="xáo hàng (TB ± std)")
    ax.set_xticks(xs)
    ax.set_xticklabels([r["model"] for r in srows], rotation=40, ha="right", fontsize=8)
    ax.set(ylabel="Test accuracy", title="Ảnh hưởng của việc xáo thứ tự hàng ảnh test")
    ax.legend(); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "shuffle.png"), dpi=150); plt.close(fig)
    log_file.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="fashion_mnist", choices=["mnist", "fashion_mnist"])
    p.add_argument("--representations", nargs="+", default=["row", "col", "patch"],
                   choices=["row", "col", "patch"])
    p.add_argument("--rnn", nargs="+", default=["lstm", "gru"], choices=["lstm", "gru"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-size", type=int, default=6000)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--shuffle-perms", type=int, default=5)
    p.add_argument("--e1-dir", default="results/E1")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", default="results/E3")
    args = p.parse_args()
    device = get_device()
    print("device:", device)
    tr, va, te, info = get_loaders(args.dataset, args.data_root, "splits",
                                   args.val_size, args.seed, args.batch_size)
    run_all(tr, va, te, info, args, device)


if __name__ == "__main__":
    main()
