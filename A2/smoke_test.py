"""Quick check (~1-2 min): shapes, forward/backward, losses, 3 real batches. Run before the full grid."""
import torch, data, models, engine
dev = torch.device("cuda")
for arch in ["unet_r34", "segformer_b0"]:
    for pre in [False, True]:
        m = models.build_model(arch, pre).to(dev)
        x = torch.randn(2, 3, 384, 384, device=dev); y = torch.randint(0, 21, (2, 384, 384), device=dev); y[:, :10] = 255
        for k in ["ce", "ce_dice"]:
            out = m(x); assert out.shape == (2, 21, 384, 384), out.shape
            l = engine.make_loss(k)(out.float(), y); l.backward(); print(arch, "pretrained" if pre else "scratch", k, "loss", round(l.item(), 3), "ok")
        print("  params", f"{models.count_params(m):,}")
data.ensure_voc("data"); sp = data.make_splits("data"); print({k: len(v) for k, v in sp.items() if k != "seed"})
x, y = data.VOCTrain("data", sp["train"])[0]; print("train sample", x.shape, y.shape, y.unique().tolist())
x, y = data.VOCEval("data", sp["test"])[0]; print("eval sample", x.shape, y.shape)
print("SMOKE TEST OK")
