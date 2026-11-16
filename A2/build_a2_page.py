"""Build docs/a2.html (+ docs/assets/a2/*.png) from results/A2/*/result.json and reports/a2_discussion.txt.
Run from repo root: python A2/build_a2_page.py   (--draft allows unfinished placeholder text)"""
import glob, html, json, os, re, shutil, sys
DRAFT = "--draft" in sys.argv
e = html.escape
R = {r["name"]: r for r in (json.load(open(f)) for f in sorted(glob.glob("results/A2/*/result.json")))}
DIAG = {k: v for k, v in R.items() if v["args"].get("subset")}; R = {k: v for k, v in R.items() if not v["args"].get("subset")}

import statistics
REP = {k: v for k, v in R.items() if v["args"]["seed"] != 42}
R = {k: v for k, v in R.items() if k not in REP}
SEEDG = []
for _b, _r in R.items():
    _reps = [v for k, v in REP.items() if k.startswith(_b + "_seed")]
    if _reps:
        _runs = [_r] + _reps; _vals = [x["test"]["miou"] * 100 for x in _runs]
        SEEDG.append((_b, [x["args"]["seed"] for x in _runs], _vals, statistics.mean(_vals), statistics.stdev(_vals)))
NSEED = max((len(g[1]) for g in SEEDG), default=1)
HAS_B2 = any(x.startswith("segformer_b2") for x in R)
SEEDLIM = (f"các cấu hình pretrained chính của hai mô hình được chạy {NSEED} seed, các cấu hình còn lại chỉ một seed nên chênh lệch nhỏ ở đó chưa đủ để kết luận; " if SEEDG else "mỗi cấu hình chỉ một seed nên các chênh lệch nhỏ chưa đủ để kết luận; ")
SEEDLIM_C = SEEDLIM[0].upper() + SEEDLIM[1:]
SEEDPROTO = (f"Cấu hình pretrained chính (CE) của hai mô hình được chạy {NSEED} seed ({', '.join(str(x) for x in SEEDG[0][1])}); các cấu hình khác chạy một lần (seed 42). Seed khác nhau chỉ đổi khởi tạo decoder và thứ tự/augmentation, split dữ liệu giữ nguyên. " if SEEDG else "Mỗi cấu hình chạy một lần (một seed). ")
B2TXT = " Thêm SegFormer-B2 (HuggingFace nvidia/mit-b2, pretrain ImageNet-1K) để so sánh với U-Net ở cỡ tham số gần hơn." if HAS_B2 else ""
B0LIM = "chỉ thử SegFormer-B0 và B2, chưa thử các bản lớn hơn của SegFormer; " if HAS_B2 else "chọn B0 thay vì các bản lớn hơn của SegFormer; "
def _pick_pair():
    s_ = [r for n, r in R.items() if n.startswith("segformer_b0_scratch_ce_long")]
    u_ = [r for n, r in R.items() if n.startswith("unet_r34_scratch_ce_fair")] or [r for n, r in R.items() if n.startswith("unet_r34_scratch_ce_long")]
    return (max(s_, key=lambda r: r["args"]["epochs"]) if s_ else None, max(u_, key=lambda r: r["args"]["epochs"]) if u_ else None)
SR, UR = _pick_pair()
SCRATCHLIM, SCRATCHCAP = "", "xem bảng (công thức huấn luyện hai bên có thể khác nhau)."
if SR and UR:
    _same = all(SR["args"].get(k) == UR["args"].get(k) for k in ["lr_scratch", "clip", "no_amp", "epochs"])
    _cv = lambda r: f"{r['best_epoch']}/{r['args']['epochs']}"
    _nc = any(r["best_epoch"] >= 0.9 * r["args"]["epochs"] for r in (SR, UR))
    SCRATCHLIM = (("Hai lần chạy from-scratch dài dùng cùng learning rate, cắt gradient, độ chính xác số học và số epoch nên so sánh được; hai mô hình chỉ khác kiến trúc và cách khởi tạo (SegFormer dùng khởi tạo conv theo mã gốc); " if _same else "Lần chạy from-scratch dài của hai mô hình khác nhau về công thức (độ chính xác số học, cắt gradient, khởi tạo, learning rate) nên không hoàn toàn công bằng; ")
                  + f"epoch tốt nhất là {_cv(SR)} (SegFormer) và {_cv(UR)} (U-Net)" + (", gần sát epoch cuối nên chưa thể coi là đã hội tụ." if _nc else "."))
    SCRATCHCAP = (f"SegFormer {SR['args']['epochs']} epoch, U-Net {UR['args']['epochs']} epoch, " + ("cùng công thức (fp32, cắt gradient, cùng learning rate)." if _same else "khác công thức."))
SCRATCHLIM_P = (SCRATCHLIM[0].lower() + SCRATCHLIM[1:].rstrip(".") + "; ") if SCRATCHLIM else ""
CONF_LINES = open("results/A2/figures/confusions.md", encoding="utf-8").read().strip().split("\n") if os.path.exists("results/A2/figures/confusions.md") else []
if not R: sys.exit("no results")
def desc(r):
    a = r["args"]
    d = f"lr {a['lr_scratch']}, {a['epochs']} epoch" if r["mode"] == "scratch" else f"lr encoder {a['lr_enc']}, decoder {a['lr_dec']}, {a['epochs']} epoch"
    if a.get("clip", 0): d += f", cắt gradient {a['clip']}"
    if a.get("no_amp"): d += ", fp32"
    if a.get("init_conv") == "official": d += ", khởi tạo conv theo mã gốc của SegFormer"
    return d

FP32_KEY = 'segformer_b0_pretrained_ce_fp32'
ORDER = ["unet_r34_pretrained_ce", "segformer_b0_pretrained_ce", "unet_r34_scratch_ce", "segformer_b0_scratch_ce",
         "unet_r34_frozen_ce", "segformer_b0_frozen_ce", "unet_r34_pretrained_ce_dice", "segformer_b0_pretrained_ce_dice"]
NAMES = [n for n in ORDER if n in R] + [n for n in R if n not in ORDER]; any_r = R[NAMES[0]]; A = any_r["args"]
AR = {"unet_r34": "U-Net (ResNet-34)", "segformer_b0": "SegFormer-B0", "segformer_b2": "SegFormer-B2"}
MO = {"pretrained": "pretrained, fine-tune toàn bộ", "scratch": "from-scratch", "frozen": "pretrained, đóng băng encoder"}
LS = {"ce": "CE", "ce_dice": "CE + Dice"}
def label(r):
    extra = f" ({desc(r)})" if r["name"] not in ORDER else ""
    return AR[r["arch"]], MO[r["mode"]] + extra, LS[r["loss"]]
os.makedirs("docs/assets/a2", exist_ok=True)
for src in ["figures/miou_by_mode", "figures/curves", "figures/per_class_iou", "figures/cost_vs_miou", "figures/diag_curves", "figures/confusion", "figures/qualitative", "eda/class_distribution", "eda/samples"]:
    p = f"results/A2/{src}.png"
    if os.path.exists(p): shutil.copy(p, f"docs/assets/a2/{os.path.basename(src)}.png")
fig = lambda n, cap, w="": f'<figure><img src="assets/a2/{n}.png" alt="{e(cap)}"{w}><figcaption>{e(cap)}</figcaption></figure>' if os.path.exists(f"docs/assets/a2/{n}.png") else ""
rows = ""
for n in NAMES:
    r = R[n]; t = r["test"]; a, m, l = label(r); gf = "" if r["gflops_512"] is None else f"{r['gflops_512']:.1f}"
    rows += (f"<tr><td>{e(a)}</td><td>{e(m)}</td><td>{l}</td><td>{r['params_trainable']:,}</td><td>{r['best_val_miou']*100:.2f}</td><td>{t['miou']*100:.2f}</td>"
             f"<td>{t['pixacc']*100:.2f}</td><td>{t['mdice']*100:.2f}</td><td>{r['sec_per_epoch']:.1f}</td><td>{r['latency_ms_512']:.1f}</td><td>{gf}</td></tr>\n")
s = open("reports/a2_discussion.txt", encoding="utf-8").read().strip() if os.path.exists("reports/a2_discussion.txt") else ""
if "[SINH VIÊN" in s and not DRAFT and not os.path.exists("reports/_answers.json"): sys.exit("reports/a2_discussion.txt still has placeholder; edit it first (or --draft)")
disc = ""
_ans = json.load(open("reports/_answers.json", encoding="utf-8")) if os.path.exists("reports/_answers.json") else None
if _ans and _ans.get("p2"):
    for _k, _v in _ans["p2"]:
        if _k in ("h2", "h3"): disc += f"<h3>{e(_v)}</h3>\n"
        elif _k == "p": disc += f"<p>{e(_v)}</p>\n"
        else:
            disc += '<div class="table-wrap"><table>' + "".join("<tr>" + "".join(("<th>" if _i == 0 else "<td>") + e(c) + ("</th>" if _i == 0 else "</td>") for c in r) + "</tr>" for _i, r in enumerate(_v)) + "</table></div>\n"
else:
    for blk in re.split(r"\n\s*\n", s):
        blk = blk.strip()
        if blk.startswith("## "): disc += f"<h3>{e(blk[3:])}</h3>\n"
        elif blk: disc += f"<p>{e(blk).replace(chr(10), '<br>')}</p>\n"
FP32PAGE = (f"các lần chạy chính dùng fp16; chạy lại SegFormer pretrained (CE) bằng fp32 cho {R[FP32_KEY]['test']['miou']*100:.2f}% so với {R['segformer_b0_pretrained_ce']['test']['miou']*100:.2f}% ở fp16, các mô hình khác chưa chạy lại;" if FP32_KEY in R else "các lần chạy chính dùng fp16, chưa kiểm tra lại mô hình pretrained bằng fp32;")

SEEDHTML = ""
if SEEDG:
    SEEDHTML = '<h3>Kết quả nhiều seed</h3><div class="table-wrap"><table><tr><th>Run</th><th>Seed</th><th>Test mIoU từng seed (%)</th><th>Trung bình ± độ lệch chuẩn</th></tr>' + "".join(f"<tr><td>{e(b)}</td><td>{', '.join(map(str, sd))}</td><td>{', '.join(f'{v:.2f}' for v in vals)}</td><td>{m:.2f} ± {sdv:.2f}</td></tr>" for b, sd, vals, m, sdv in SEEDG) + '</table></div><p class="note">Seed chỉ đổi khởi tạo decoder và thứ tự/augmentation; split dữ liệu giữ nguyên.</p>'
CONF_HTML = ("<h3>Nhầm lẫn lớn nhất (lớp thật → lớp dự đoán, tỉ lệ pixel của lớp thật), tập test</h3><ul>" + "".join(f"<li>{e(l)}</li>" for l in CONF_LINES) + "</ul>") if CONF_LINES else ""
B0LIM_P = "chỉ thử SegFormer-B0 và B2; " if HAS_B2 else "chỉ dùng B0; "
B2PAGE = " (và <code>nvidia/mit-b2</code> cho SegFormer-B2)" if HAS_B2 else ""
G = "https://github.com/tramphanngoc174/co5085-exercises"
page = f'''<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>A2 – Phân đoạn ảnh: U-Net vs SegFormer – Phan Ngọc Trâm</title>
<style>
  :root {{ --bg:#ffffff; --fg:#1f2328; --muted:#59636e; --card:#f6f8fa; --line:#d1d9e0; --link:#0969da; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#0d1117; --fg:#e6edf3; --muted:#9198a1; --card:#151b23; --line:#3d444d; --link:#4493f8; }}
  }}
  body {{ font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; background:var(--bg); color:var(--fg);
         max-width:820px; margin:0 auto; padding:2rem 1rem; line-height:1.6; }}
  h1 {{ font-size:1.6rem; margin-bottom:.2rem; }} h2 {{ margin-top:2rem; }} h3 {{ font-size:1.05rem; margin-top:1.2rem; }}
  p.sub {{ color:var(--muted); margin-top:0; }} a {{ color:var(--link); }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:1rem 1.25rem; margin:1rem 0; }}
  table {{ border-collapse:collapse; width:100%; }}
  th, td {{ border:1px solid var(--line); padding:.4rem .5rem; text-align:left; font-size:.85rem; }}
  .table-wrap {{ overflow-x:auto; }} img {{ max-width:100%; height:auto; }}
  figure {{ margin:1rem 0; }} figcaption {{ color:var(--muted); font-size:.88rem; text-align:center; }}
  .note {{ color:var(--muted); font-size:.88rem; }}
</style>
</head>
<body>
  <p><a href="index.html">← Trang chủ</a></p>
  <h1>Bài tập lớn A2 – Phân đoạn ảnh ngữ nghĩa: CNN và Transformer</h1>
  <p class="sub">CO5085 – Học sâu và ứng dụng trong thị giác máy tính – HK 261 (2026–2027)</p>

  <div class="card">
    <strong>Chủ đề:</strong> phân đoạn ảnh ngữ nghĩa (semantic segmentation)<br>
    <strong>Dataset:</strong> Pascal VOC 2012 Segmentation (21 lớp; {any_r['n_train']} train / {any_r['n_val']} validation / {any_r['n_test']} test)<br>
    <strong>CNN:</strong> U-Net với encoder ResNet-34 &nbsp;|&nbsp; <strong>Transformer:</strong> SegFormer-B0<br>
    <strong>Paper phân tích:</strong> Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., Luo, P. <em>SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.</em> Advances in Neural Information Processing Systems 34 (NeurIPS 2021). arXiv:2105.15203. <a href="https://arxiv.org/abs/2105.15203">arxiv.org/abs/2105.15203</a><br>
    <strong>Code:</strong> <a href="{G}/tree/main/A2">thư mục A2 trên GitHub</a> &nbsp;|&nbsp;
    <strong>Báo cáo PDF:</strong> <a href="{G}/blob/main/reports/a2_report.pdf">reports/a2_report.pdf</a><br>
    <strong>A2.1 (chủ đề, paper, kế hoạch):</strong> <a href="{G}/blob/main/reports/a2_1_chu_de_paper_ke_hoach.md">reports/a2_1_chu_de_paper_ke_hoach.md</a><br>
    <strong>Video demo:</strong> chưa có
  </div>

  <h2>1. Bài toán và dữ liệu (EDA)</h2>
  <p>Gán nhãn cho từng pixel (20 lớp vật thể và background). Tập train chính thức 1.464 ảnh được chia (seed {A['seed']}) thành {any_r['n_train']} train và {any_r['n_val']} validation; tập val chính thức ({any_r['n_test']} ảnh) dùng làm test. Không dùng SBD. Viền vật thể có nhãn 255 và bị bỏ qua khi tính loss và metric. Giấy phép: ảnh VOC lấy từ Flickr, mình chỉ dùng cho học tập.</p>
  {fig("class_distribution", "Phân bố lớp trong tập train.")}{fig("samples", "Ảnh và nhãn mẫu.")}

  <h2>2. Dataset, DataLoader và augmentation</h2>
  <p>Huấn luyện: phóng tỉ lệ 0,5–2,0, lật ngang, cắt ngẫu nhiên {A['crop']}×{A['crop']}, chuẩn hóa theo ImageNet, batch {A['bs']}. Đánh giá: không augmentation; ảnh đệm lên 512×512, phần đệm có nhãn 255 nên không tính.</p>

  <h2>3. Phân tích paper</h2>
  <p>SegFormer gồm encoder Transformer phân cấp (MiT) cho đặc trưng ở 4 độ phân giải (1/4, 1/8, 1/16, 1/32), không dùng positional encoding mà dùng Mix-FFN (conv 3×3 trong FFN), self-attention hiệu quả (rút gọn chuỗi Key), và decoder chỉ gồm các lớp MLP. Phân tích đầy đủ (phương pháp, thí nghiệm, hạn chế, đối chiếu) nằm ở mục 3 của <a href="{G}/blob/main/reports/a2_report.pdf">báo cáo PDF</a>.</p>

  <h2>4. Protocol thí nghiệm</h2>
  <p>2 kiến trúc × 3 chế độ (from-scratch, pretrained fine-tune toàn bộ, pretrained đóng băng encoder) với loss CE, thêm ablation CE + Dice cho hai mô hình pretrained. Chung: AdamW (weight decay {A['wd']}), lịch học poly có warm-up, fp16, {A['epochs']} epoch, seed {A['seed']}, chọn epoch theo mIoU trên validation, đo test một lần. Learning rate: from-scratch {A['lr_scratch']}; pretrained/frozen: encoder {A['lr_enc']}, decoder {A['lr_dec']}. Encoder pretrain: <code>resnet34.a1_in1k</code> và <code>nvidia/mit-b0</code>{B2PAGE} (đều ImageNet-1K, công thức huấn luyện khác nhau). {SEEDPROTO}Các dòng có ghi chú trong ngoặc là lần chạy bổ sung: chạy dài hoặc đổi cấu hình cho from-scratch sau khi SegFormer không học được ở cấu hình mặc định, và kiểm tra fp32 cho SegFormer pretrained.{"".join(f" Chẩn đoán overfit ({e(n)}): huấn luyện trên {d['args']['subset']} ảnh, mIoU trên chính các ảnh đó là {d['best_val_miou']*100:.2f}% sau {d['args']['epochs']} epoch." for n, d in DIAG.items())}</p>

  <h2>5. Kết quả</h2>
  <div class="table-wrap"><table>
    <tr><th>Kiến trúc</th><th>Chế độ</th><th>Loss</th><th>Params trainable</th><th>Val mIoU</th><th>Test mIoU</th><th>PixAcc</th><th>mDice</th><th>s/epoch</th><th>ms (512)</th><th>GFLOPs*</th></tr>
{rows}  </table></div>
  {SEEDHTML}
  <p class="note">mIoU, PixAcc, mDice tính theo %, trên {any_r['n_test']} ảnh test. Độ trễ đo với ảnh 512×512, batch 1, fp16, {e(any_r['env']['gpu'])}; cùng một kiến trúc dao động khoảng 2 ms giữa các lần đo. *GFLOPs do torch FlopCounterMode đếm (một phép nhân-cộng tính là 2 FLOPs), chưa so trực tiếp được với con số trong paper SegFormer.</p>
  {fig("miou_by_mode", "Test mIoU theo chế độ huấn luyện (loss CE).", ' style="max-width:560px"')}{fig("curves", "Loss huấn luyện và mIoU validation theo epoch.")}{fig("per_class_iou", "IoU từng lớp trên test (pretrained, CE).")}{fig("cost_vs_miou", "Độ chính xác so với độ trễ, số tham số và GFLOPs (pretrained, CE). Thanh sai số là độ lệch chuẩn khi có nhiều seed.")}{fig("confusion", "Confusion matrix (chuẩn hóa theo hàng) trên test, các mô hình pretrained (CE).")}{CONF_HTML}{fig("diag_curves", "Chẩn đoán: mIoU trên chính 64 ảnh train của SegFormer from-scratch (A: fp16; B: fp16 + khởi tạo gốc + cắt gradient; C: như B nhưng fp32).")}{fig("qualitative", "Ảnh, nhãn thật và dự đoán trên một số ảnh test cố định.", ' style="max-width:620px"')}

  <h2>6. Thảo luận</h2>
{disc}
  <h2>7. Hạn chế</h2>
  <p>{SEEDLIM_C}không dùng SBD (chỉ {any_r['n_train']} ảnh train); không tinh chỉnh siêu tham số riêng cho từng mô hình; {FP32PAGE} SegFormer from-scratch ở fp16 không học được (50 epoch) còn ở fp32 thì học được, nên so sánh from-scratch dùng các lần chạy dài; {SCRATCHLIM_P}{B0LIM_P}hai encoder pretrain bằng công thức khác nhau.</p>
</body>
</html>
'''
open("docs/a2.html", "w", encoding="utf-8").write(page); print("built docs/a2.html")
