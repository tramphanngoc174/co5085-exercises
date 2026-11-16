"""Run A2 experiments. One run = arch x mode x loss. Resume-able (skips runs with result.json).
python run_a2.py --all            # full grid (8 runs)
python run_a2.py --arch unet_r34 --mode pretrained --loss ce
"""
import argparse, json, os, time, platform
import numpy as np, torch
from torch.utils.data import DataLoader
import data, models, engine

p = argparse.ArgumentParser()
p.add_argument("--arch", default="unet_r34", choices=["unet_r34", "segformer_b0", "segformer_b2"])
p.add_argument("--mode", default="pretrained", choices=["scratch", "pretrained", "frozen"])
p.add_argument("--loss", default="ce", choices=["ce", "ce_dice"])
p.add_argument("--all", action="store_true")
p.add_argument("--tag", default="", help="suffix for the run name, e.g. lr1e-4")
p.add_argument("--epochs", type=int, default=50)
p.add_argument("--bs", type=int, default=16)
p.add_argument("--crop", type=int, default=384)
p.add_argument("--lr_scratch", type=float, default=5e-4)
p.add_argument("--lr_enc", type=float, default=1e-4)
p.add_argument("--lr_dec", type=float, default=1e-3)
p.add_argument("--wd", type=float, default=0.01)
p.add_argument("--seed", type=int, default=42, help="training seed (init, augmentation order)")
p.add_argument("--split_seed", type=int, default=42, help="seed of the train/val split (keep 42)")
p.add_argument("--workers", type=int, default=2)
p.add_argument("--root", default="data")
p.add_argument("--out", default="results/A2")
p.add_argument("--ckpt", default="/content/ckpt_a2")
p.add_argument("--n_vis", type=int, default=16)
p.add_argument("--no_amp", action="store_true", help="train/eval in fp32")
p.add_argument("--clip", type=float, default=0.0, help="grad-norm clipping (0 = off)")
p.add_argument("--init_conv", default="default", choices=["default", "official"], help="official = SegFormer repo conv init (scratch only)")
p.add_argument("--subset", type=int, default=0, help="diagnostic: train on first N train images, validate on the same images")
args = p.parse_args()
engine.AMP = not args.no_amp

GRID = [("unet_r34", "pretrained", "ce"), ("segformer_b0", "pretrained", "ce"),
        ("unet_r34", "scratch", "ce"), ("segformer_b0", "scratch", "ce"),
        ("unet_r34", "frozen", "ce"), ("segformer_b0", "frozen", "ce"),
        ("unet_r34", "pretrained", "ce_dice"), ("segformer_b0", "pretrained", "ce_dice")]


def run(arch, mode, loss_kind):
    name = f"{arch}_{mode}_{loss_kind}" + (f"_{args.tag}" if args.tag else "")
    odir = os.path.join(args.out, name); rpath = os.path.join(odir, "result.json")
    if os.path.exists(rpath):
        print("skip (done):", name); return
    os.makedirs(odir, exist_ok=True); os.makedirs(args.ckpt, exist_ok=True)
    dev = torch.device("cuda")
    engine.seed_all(args.seed)
    data.ensure_voc(args.root)
    sp = data.make_splits(args.root, args.split_seed)
    if args.subset:
        sp = dict(sp); sp["train"] = sp["train"][: args.subset]; sp["val"] = sp["train"]
    tr = DataLoader(data.VOCTrain(args.root, sp["train"], args.crop), args.bs, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    mk = lambda ids: DataLoader(data.VOCEval(args.root, ids), 16, num_workers=args.workers, pin_memory=True)
    va, te = mk(sp["val"]), mk(sp["test"])

    model = models.build_model(arch, mode != "scratch", args.init_conv).to(dev)
    frozen = mode == "frozen"
    if frozen:
        for q in model.encoder_module().parameters():
            q.requires_grad = False
    enc_ids = {id(q) for q in model.encoder_module().parameters()}
    enc = [q for q in model.parameters() if q.requires_grad and id(q) in enc_ids]
    dec = [q for q in model.parameters() if q.requires_grad and id(q) not in enc_ids]
    if mode == "scratch":
        groups = [{"params": enc + dec, "lr": args.lr_scratch}]
    else:
        groups = [{"params": dec, "lr": args.lr_dec}] + ([{"params": enc, "lr": args.lr_enc}] if enc else [])
    opt = torch.optim.AdamW(groups, weight_decay=args.wd)
    total, warm = args.epochs * len(tr), min(200, args.epochs * len(tr) // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: (it + 1) / warm if it < warm else (1 - (it - warm) / (total - warm)) ** 0.9)
    scaler = torch.amp.GradScaler("cuda", enabled=not args.no_amp)
    loss_fn = engine.make_loss(loss_kind)

    n_tot, n_tr = models.count_params(model), models.count_params(model, True)
    print(f"== {name}: params total {n_tot:,} trainable {n_tr:,}")
    torch.cuda.reset_peak_memory_stats(); hist, best, ck = [], -1.0, os.path.join(args.ckpt, name + ".pt")
    for ep in range(1, args.epochs + 1):
        torch.cuda.synchronize(); t0 = time.time()
        l = engine.train_epoch(model, tr, opt, sched, scaler, loss_fn, dev, frozen, args.clip)
        torch.cuda.synchronize(); dt = time.time() - t0
        v = engine.metrics(engine.evaluate(model, va, dev))
        hist.append({"epoch": ep, "train_loss": l, "val_miou": v["miou"], "val_pixacc": v["pixacc"], "sec": dt})
        if v["miou"] > best:
            best = v["miou"]; torch.save(model.state_dict(), ck)
        print(f"ep {ep:3d} loss {l:.4f} val mIoU {v['miou']:.4f} pixAcc {v['pixacc']:.4f} {dt:.1f}s", flush=True)
    peak = torch.cuda.max_memory_allocated() / 2 ** 20

    model.load_state_dict(torch.load(ck)); best_ep = int(np.argmax([h["val_miou"] for h in hist])) + 1
    cm = engine.evaluate(model, te, dev); m = engine.metrics(cm)
    # qualitative predictions on a fixed subset of the test set
    vis_idx = list(range(0, len(sp["test"]), max(1, len(sp["test"]) // args.n_vis)))[: args.n_vis]
    ds = data.VOCEval(args.root, sp["test"])
    xs = torch.stack([ds[i][0] for i in vis_idx]); preds = engine.predict_batch(model, xs, dev).numpy().astype(np.uint8)
    np.savez_compressed(os.path.join(odir, "preds.npz"), preds=preds, ids=np.array([sp["test"][i] for i in vis_idx]))
    res = {"name": name, "arch": arch, "mode": mode, "loss": loss_kind,
           "params_total": n_tot, "params_trainable": n_tr, "best_epoch": best_ep, "best_val_miou": best,
           "test": m, "confusion": cm.tolist(), "sec_per_epoch": float(np.mean([h["sec"] for h in hist])),
           "peak_mem_mb": peak, "latency_ms_512": engine.latency_ms(model, dev), "gflops_512": engine.gflops(model, dev),
           "history": hist, "n_train": len(sp["train"]), "n_val": len(sp["val"]), "n_test": len(sp["test"]),
           "args": vars(args), "env": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "python": platform.python_version()}}
    try:
        import timm, transformers; res["env"].update(timm=timm.__version__, transformers=transformers.__version__)
    except Exception: pass
    json.dump(res, open(rpath, "w"), indent=1)
    print(f"TEST {name}: mIoU {m['miou']:.4f} pixAcc {m['pixacc']:.4f} mDice {m['mdice']:.4f} (best epoch {best_ep})")
    del model, opt; torch.cuda.empty_cache()


for cfg in (GRID if args.all else [(args.arch, args.mode, args.loss)]):
    run(*cfg)
