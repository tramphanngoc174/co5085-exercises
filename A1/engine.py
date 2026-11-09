"""A1: tiền xử lý trên GPU, vòng lặp huấn luyện/đánh giá TỰ VIẾT (autocast fp16 + GradScaler)."""
import time

import numpy as np
import torch
import torch.nn.functional as F


def random_crop_flip(x, pad=4):
    """Augmentation trên batch (B,3,32,32): random crop (đệm 0 mỗi bên 'pad' pixel) + lật ngang ngẫu nhiên."""
    B, _, H, W = x.shape
    xp = F.pad(x, (pad, pad, pad, pad))
    dy = torch.randint(0, 2 * pad + 1, (B,)).tolist()
    dx = torch.randint(0, 2 * pad + 1, (B,)).tolist()
    x = torch.stack([xp[i, :, dy[i]:dy[i] + H, dx[i]:dx[i] + W] for i in range(B)])
    flip = torch.rand(B, device=x.device) < 0.5
    return torch.where(flip[:, None, None, None], x.flip(3), x)


class Prep:
    """uint8 -> float [0,1] -> (augment nếu train) -> phóng to bilinear -> chuẩn hóa mean/std."""
    def __init__(self, device, size, mean, std):
        self.device, self.size = device, size
        self.mean = torch.tensor(mean, device=device).view(1, -1, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, -1, 1, 1)

    def __call__(self, xb, train):
        x = xb.to(self.device, non_blocking=True).float().div_(255.0)
        if train:
            x = random_crop_flip(x)
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std


def train_epoch(model, loader, loss_fn, optimizer, scaler, prep, device, frozen_bns, use_amp):
    model.train()
    for m in frozen_bns:          # BatchNorm bị đóng băng giữ ở eval() để không cập nhật running stats
        m.eval()
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        x, y = prep(xb, True), yb.to(device)
        optimizer.zero_grad(set_to_none=True)                 # 1. xóa gradient cũ
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)                                 # 2. forward
            loss = loss_fn(logits, y)                         # 3. loss
        scaler.scale(loss).backward()                         # 4. backward
        scaler.step(optimizer)                                # 5. cập nhật tham số
        scaler.update()
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, loss_fn, prep, device, use_amp, num_classes):
    model.eval()
    total_loss, n = 0.0, 0
    cm = torch.zeros(num_classes * num_classes, dtype=torch.long, device=device)
    for xb, yb in loader:
        x, y = prep(xb, False), yb.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            loss = loss_fn(logits.float(), y)
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
        cm += torch.bincount(y * num_classes + logits.argmax(1), minlength=num_classes ** 2)
    cm = cm.view(num_classes, num_classes).cpu().numpy()      # hàng = nhãn thật, cột = dự đoán
    return total_loss / n, cm.trace() / cm.sum(), cm


def macro_f1(cm):
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    return float(f1.mean())


def fit(model, train_loader, val_loader, optimizer, scheduler, epochs, prep, device, frozen_bns,
        use_amp, num_classes, log=print):
    loss_fn = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "epoch_time": []}
    best_acc, best_state, best_epoch = -1.0, None, -1
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, loss_fn, optimizer, scaler, prep, device,
                                      frozen_bns, use_amp)
        va_loss, va_acc, _ = evaluate(model, val_loader, loss_fn, prep, device, use_amp, num_classes)
        scheduler.step()
        dt = time.time() - t0
        for k, v in zip(hist, [tr_loss, tr_acc, va_loss, va_acc, dt]):
            hist[k].append(v)
        if va_acc > best_acc:
            best_acc, best_epoch = va_acc, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        log(f"epoch {ep:02d}/{epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f} | {dt:.1f}s")
    model.load_state_dict(best_state)      # quay về epoch tốt nhất theo VALIDATION
    hist["best_epoch"] = best_epoch
    return hist
