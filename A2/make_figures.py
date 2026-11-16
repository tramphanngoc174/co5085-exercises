"""Read results/A2/*/result.json -> tables (md/csv) + figures in results/A2/figures. Numbers come only from the json files."""
import csv, glob, json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import data

OUT = "results/A2/figures"; os.makedirs(OUT, exist_ok=True)
R = {r["name"]: r for r in (json.load(open(f)) for f in sorted(glob.glob("results/A2/*/result.json")))}
ALL = dict(R)
DIAG = {k: v for k, v in R.items() if v["args"].get("subset")}
R = {k: v for k, v in R.items() if not v["args"].get("subset")}
R = {k: v for k, v in R.items() if v["args"]["seed"] == 42}
print("runs:", list(R), "| diagnostics:", list(DIAG))
order = ["unet_r34_pretrained_ce", "segformer_b0_pretrained_ce", "unet_r34_scratch_ce", "segformer_b0_scratch_ce",
         "unet_r34_frozen_ce", "segformer_b0_frozen_ce", "unet_r34_pretrained_ce_dice", "segformer_b0_pretrained_ce_dice"]
names = [n for n in order if n in R]; names += [n for n in R if n not in names]

rows = [["run", "test mIoU", "pixAcc", "mDice", "best ep", "params total", "params trainable", "s/epoch", "peak MB", "latency ms (512, bs1)", "GFLOPs* (512)"]]
for n in names:
    r = R[n]; t = r["test"]
    rows.append([n, f"{t['miou']:.4f}", f"{t['pixacc']:.4f}", f"{t['mdice']:.4f}", r["best_epoch"], f"{r['params_total']:,}",
                 f"{r['params_trainable']:,}", f"{r['sec_per_epoch']:.1f}", f"{r['peak_mem_mb']:.0f}", f"{r['latency_ms_512']:.1f}",
                 "" if r["gflops_512"] is None else f"{r['gflops_512']:.1f}"])
csv.writer(open(f"{OUT}/results_table.csv", "w", newline="")).writerows(rows)
with open(f"{OUT}/results_table.md", "w") as f:
    f.write("| " + " | ".join(rows[0]) + " |\n|" + "---|" * len(rows[0]) + "\n")
    for r in rows[1:]: f.write("| " + " | ".join(map(str, r)) + " |\n")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for n in names:
    h = R[n]["history"]; ax[0].plot([x["epoch"] for x in h], [x["train_loss"] for x in h], label=n); ax[1].plot([x["epoch"] for x in h], [x["val_miou"] for x in h], label=n)
ax[0].set_title("Train loss"); ax[1].set_title("Validation mIoU (146 images)"); ax[1].legend(fontsize=6)
for a in ax: a.set_xlabel("epoch"); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/curves.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(9, 4)); w = 0.38
def _cands():
    c = {"unet_r34": {}, "segformer_b0": {}}
    for n, r in R.items():
        for a in c:
            if n.startswith(f"{a}_scratch_ce_") and ("_long" in n or "_fair" in n):
                c[a].setdefault(r["args"]["epochs"], []).append(n)
    return c
CANDS = _cands(); COMMON_E = set(CANDS["unet_r34"]) & set(CANDS["segformer_b0"]); LONG_E = max(COMMON_E) if COMMON_E else None
def long_name(a):
    if LONG_E is None:
        allc = [n for v in CANDS[a].values() for n in v]; return allc[0] if allc else None
    return sorted(CANDS[a][LONG_E], key=lambda n: ("_fair" not in n, n))[0]
groups = [("pretrained", lambda a: f"{a}_pretrained_ce"), ("frozen encoder", lambda a: f"{a}_frozen_ce"),
          ("scratch\n(50 epoch)", lambda a: f"{a}_scratch_ce"), (f"scratch\n(dài, {LONG_E} epoch)" if LONG_E else "scratch (dài)", long_name)]
groups = [g for g in groups if any(g[1](a) in R for a in ["unet_r34", "segformer_b0"])]
for k, arch in enumerate(["unet_r34", "segformer_b0"]):
    v = [R.get(g[1](arch), {}).get("test", {}).get("miou", np.nan) for g in groups]
    b = ax.bar(np.arange(len(groups)) + (k - .5) * w, v, w, label=arch)
    for rect, val in zip(b, v):
        if not np.isnan(val): ax.text(rect.get_x() + w / 2, val + .005, f"{val:.3f}", ha="center", fontsize=8)
ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups]); ax.set_ylabel("test mIoU"); ax.legend(); ax.set_title("Test mIoU theo chế độ huấn luyện (loss CE)")
plt.tight_layout(); plt.savefig(f"{OUT}/miou_by_mode.png", dpi=150); plt.close()

pre = [n for n in ["unet_r34_pretrained_ce", "segformer_b0_pretrained_ce"] if n in R]
if pre:
    fig, ax = plt.subplots(figsize=(13, 4)); x = np.arange(21)
    for k, n in enumerate(pre): ax.bar(x + (k - .5) * .4, R[n]["test"]["iou_per_class"], .4, label=n)
    ax.set_xticks(x); ax.set_xticklabels(data.CLASSES, rotation=90); ax.set_ylabel("IoU"); ax.legend(); ax.set_title("Per-class test IoU (pretrained, CE)")
    plt.tight_layout(); plt.savefig(f"{OUT}/per_class_iou.png", dpi=150); plt.close()

PTS = [("U-Net R34", "unet_r34_pretrained_ce", "o"), ("SegFormer-B0", "segformer_b0_pretrained_ce", "s"), ("SegFormer-B2", "segformer_b2_pretrained_ce", "^")]
PTS = [p for p in PTS if p[1] in R]
if PTS:
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for lab, n, mk in PTS:
        vals = [ALL[k]["test"]["miou"] * 100 for k in ALL if k == n or k.startswith(n + "_seed")]
        m = float(np.mean(vals)); sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0; r = R[n]
        for a, x in zip(ax, [r["latency_ms_512"], r["params_total"] / 1e6, r["gflops_512"] if r["gflops_512"] is not None else np.nan]):
            a.errorbar(x, m, yerr=sd, fmt=mk, ms=8, capsize=3, label=lab + (f" (TB ± độ lệch chuẩn, n={len(vals)})" if len(vals) > 1 else " (1 seed)"))
    for a, t in zip(ax, ["Độ trễ (ms, 512×512, bs1, fp16)", "Tổng số tham số (M)", "GFLOPs* (torch đếm, 1 MAC = 2 FLOPs)"]):
        a.set_xlabel(t); a.grid(alpha=.3)
    ax[0].set_ylabel("test mIoU (%)"); ax[0].legend(fontsize=6.5, loc="lower right")
    plt.suptitle("Độ chính xác so với chi phí (pretrained, loss CE)", fontsize=10); plt.tight_layout(); plt.savefig(f"{OUT}/cost_vs_miou.png", dpi=150); plt.close()

if DIAG:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for n, d in sorted(DIAG.items()):
        h = d["history"]; ax.plot([x["epoch"] for x in h], [x["val_miou"] for x in h], label=n.replace("segformer_b0_scratch_ce_", ""))
    ax.set_xlabel("epoch"); ax.set_ylabel("mIoU trên chính 64 ảnh train"); ax.set_title("Chẩn đoán overfit: SegFormer from-scratch"); ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/diag_curves.png", dpi=150); plt.close()

# qualitative: image | GT | pretrained U-Net | pretrained SegFormer
cmap = data.voc_cmap(); root = "data"; show = [n for n in ["unet_r34_pretrained_ce", "segformer_b0_pretrained_ce", long_name("unet_r34"), long_name("segformer_b0")] if n in R]
try: data.ensure_voc(root)
except Exception as ex: print("ensure_voc failed:", ex)
try:
    if show:
        P = {n: np.load(f"results/A2/{n}/preds.npz") for n in show}; ids = list(P[show[0]]["ids"])[:8]
        fig, ax = plt.subplots(len(ids), 2 + len(show), figsize=(2.4 * (2 + len(show)), 2.1 * len(ids)))
        for r, i in enumerate(ids):
            img = Image.open(f"{data.voc_dir(root)}/JPEGImages/{i}.jpg"); gt = np.array(Image.open(f"{data.voc_dir(root)}/SegmentationClass/{i}.png")); h, w = gt.shape
            ax[r, 0].imshow(img); ax[r, 1].imshow(cmap[gt])
            for c, n in enumerate(show):
                p = P[n]["preds"][list(P[n]["ids"]).index(i)][:h, :w]; ax[r, 2 + c].imshow(cmap[p])
                if r == 0: ax[r, 2 + c].set_title(n.replace("_ce", ""), fontsize=6)
            if r == 0: ax[r, 0].set_title("image", fontsize=7); ax[r, 1].set_title("ground truth", fontsize=7)
            for a in ax[r]: a.axis("off")
        plt.tight_layout(); plt.savefig(f"{OUT}/qualitative.png", dpi=130); plt.close()
except FileNotFoundError as ex:
    print('skip qualitative (VOC images not found):', ex)
# confusion matrices + biggest confusions (from stored confusion matrices)
pre = [n for n in ["unet_r34_pretrained_ce", "segformer_b0_pretrained_ce", "segformer_b2_pretrained_ce"] if n in R]
if pre:
    CL = data.CLASSES; lines = []
    fig, ax = plt.subplots(1, len(pre), figsize=(6 * len(pre), 5.6), squeeze=False)
    for k, n in enumerate(pre):
        cm = np.array(R[n]["confusion"], float); rn = cm / cm.sum(1, keepdims=True)
        a = ax[0, k]; a.imshow(rn, vmin=0, vmax=1, cmap="Blues"); a.set_title(n.replace("_ce", ""), fontsize=9)
        a.set_xticks(range(21)); a.set_yticks(range(21)); a.set_xticklabels(CL, rotation=90, fontsize=6); a.set_yticklabels(CL, fontsize=6)
        a.set_xlabel("dự đoán", fontsize=8); a.set_ylabel("lớp thật", fontsize=8)
        off = rn.copy(); np.fill_diagonal(off, 0); idx = np.argsort(-off.ravel())[:5]
        lines.append(f"{n}: " + "; ".join(f"{CL[i // 21]} → {CL[i % 21]} {off.ravel()[i] * 100:.1f}%" for i in idx))
    plt.tight_layout(); plt.savefig(f"{OUT}/confusion.png", dpi=130); plt.close()
    open(f"{OUT}/confusions.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("figures + tables in", OUT)
