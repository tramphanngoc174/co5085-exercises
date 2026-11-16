"""Sanity check of from-scratch SegFormer initialisation (HF default vs official conv init). ~20 s, CPU is fine."""
import torch, torch.nn as nn, models
for init in ["default", "official"]:
    torch.manual_seed(0)
    m = models.build_model("segformer_b0", False, init).eval()
    st = {}
    for _, mod in m.m.named_modules():
        if isinstance(mod, (nn.Linear, nn.Conv2d)):
            st.setdefault(type(mod).__name__, []).append(mod.weight.detach().std().item())
    bad = [n for n, p in m.named_parameters() if not torch.isfinite(p).all()]
    print(f"[{init}] non-finite params: {len(bad)}")
    for k, v in st.items():
        print(f"   {k:7s} n={len(v):3d} weight std: min {min(v):.4f}  mean {sum(v)/len(v):.4f}  max {max(v):.4f}")
    with torch.no_grad():
        out = m(torch.randn(2, 3, 128, 128))
    print(f"   logits std {out.std().item():.4f}, any NaN: {bool(torch.isnan(out).any())}")
print("CHECK DONE")
