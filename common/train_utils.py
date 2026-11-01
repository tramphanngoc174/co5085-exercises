"""Training loop TỰ VIẾT (forward -> loss -> backward -> optimizer.step) dùng chung."""
import copy
import random
import time

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()            # 1. xóa gradient cũ
        logits = model(x)                # 2. forward
        loss = loss_fn(logits, y)        # 3. tính loss
        loss.backward()                  # 4. backward: tính gradient
        optimizer.step()                 # 5. cập nhật tham số
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """Trả về loss, accuracy, và (nhãn thật, nhãn dự đoán) để vẽ confusion matrix."""
    model.eval()
    total_loss, n = 0.0, 0
    all_y, all_pred = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += loss_fn(logits, y).item() * x.size(0)
        n += x.size(0)
        all_y.append(y.cpu())
        all_pred.append(logits.argmax(1).cpu())
    y_true, y_pred = torch.cat(all_y), torch.cat(all_pred)
    acc = (y_true == y_pred).float().mean().item()
    return total_loss / n, acc, y_true, y_pred


def train_model(model, train_loader, val_loader, epochs, lr, device, log=print):
    """Lặp qua các epoch, giữ lại trọng số có val accuracy cao nhất."""
    loss_fn = torch.nn.CrossEntropyLoss()   # đã gồm log-softmax bên trong
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "epoch_time": []}
    best_acc, best_state, best_epoch = -1.0, None, -1
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, loss_fn, device)
        dt = time.time() - t0
        for k, v in zip(history, [tr_loss, tr_acc, va_loss, va_acc, dt]):
            history[k].append(v)
        if va_acc > best_acc:
            best_acc, best_epoch = va_acc, ep
            best_state = copy.deepcopy(model.state_dict())
        log(f"epoch {ep:02d}/{epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f} | {dt:.1f}s")
    model.load_state_dict(best_state)   # quay về epoch tốt nhất theo VALIDATION
    history["best_epoch"] = best_epoch
    return history
