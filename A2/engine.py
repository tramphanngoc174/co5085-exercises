import random, time
import numpy as np, torch, torch.nn.functional as F
from data import IGNORE, NUM_CLASSES

AMP = True  # set by run_a2.py (--no_amp -> fp32)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def dice_loss(logits, y, eps=1.0):
    valid = (y != IGNORE)
    yy = y.clone(); yy[~valid] = 0
    p = logits.softmax(1) * valid.unsqueeze(1)
    oh = F.one_hot(yy, NUM_CLASSES).permute(0, 3, 1, 2) * valid.unsqueeze(1)
    inter, den = (p * oh).sum((0, 2, 3)), p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    return 1 - ((2 * inter + eps) / (den + eps)).mean()


def make_loss(kind):
    def fn(logits, y):
        l = F.cross_entropy(logits, y, ignore_index=IGNORE)
        return l + dice_loss(logits, y) if kind == "ce_dice" else l
    return fn


def train_epoch(model, loader, opt, sched, scaler, loss_fn, dev, freeze_enc, clip=0.0):
    model.train()
    if freeze_enc:
        model.encoder_module().eval()  # keep BN/dropout of frozen encoder fixed
    tot, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
            out = model(x)
        loss = loss_fn(out.float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([q for g in opt.param_groups for q in g["params"]], clip)
        scaler.step(opt); scaler.update(); sched.step()
        tot += loss.item() * x.size(0); n += x.size(0)
    return tot / n


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=dev)
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
            pred = model(x).argmax(1)
        m = y != IGNORE
        cm += torch.bincount(y[m] * NUM_CLASSES + pred[m], minlength=NUM_CLASSES ** 2).view(NUM_CLASSES, NUM_CLASSES)
    return cm.cpu()


def metrics(cm):
    cm = cm.double(); tp = cm.diag(); gt, pr = cm.sum(1), cm.sum(0)
    iou = tp / (gt + pr - tp); dice = 2 * tp / (gt + pr)
    return {"miou": float(np.nanmean(iou.numpy())), "pixacc": float(tp.sum() / cm.sum()),
            "mdice": float(np.nanmean(dice.numpy())), "iou_per_class": [float(v) for v in iou.numpy()]}


@torch.no_grad()
def predict_batch(model, x, dev):
    model.eval()
    with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
        return model(x.to(dev)).argmax(1).cpu()


@torch.no_grad()
def latency_ms(model, dev, size=512, warm=10, iters=50):
    model.eval(); x = torch.randn(1, 3, size, size, device=dev)
    with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
        for _ in range(warm): model(x)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(iters): model(x)
        torch.cuda.synchronize()
    return (time.time() - t) / iters * 1000


@torch.no_grad()
def gflops(model, dev, size=512):
    try:
        from torch.utils.flop_counter import FlopCounterMode
        model.eval(); x = torch.randn(1, 3, size, size, device=dev)
        with FlopCounterMode(display=False) as fc:
            model(x)
        return fc.get_total_flops() / 1e9
    except Exception as e:
        print("flop count failed:", e); return None
