"""Build reports/a2_report.pdf (+ results/A2/figures/highlights.md) from results/A2/*/result.json.
Run from repo root:  python A2/build_a2_report.py   (add --draft to allow unfinished placeholder text)"""
import glob, json, os, re, sys
from docx import Document
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib.utils import ImageReader

DRAFT = "--draft" in sys.argv
FD = "/usr/share/fonts/truetype/dejavu/"
for n, f in [("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSans-Oblique.ttf"), ("DV-BI", "DejaVuSans-BoldOblique.ttf")]:
    pdfmetrics.registerFont(TTFont(n, FD + f))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")
B = ParagraphStyle("b", fontName="DV", fontSize=9.2, leading=13.5, spaceAfter=5)
H1 = ParagraphStyle("h1", parent=B, fontName="DV-B", fontSize=13.5, leading=17, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#1a3a6b"))
H2 = ParagraphStyle("h2", parent=B, fontName="DV-B", fontSize=10.3, leading=14, spaceBefore=6, spaceAfter=3)
H3 = ParagraphStyle("h3", parent=B, fontName="DV-B", fontSize=9.3, leading=13, spaceBefore=4, spaceAfter=2, textColor=colors.HexColor("#333333"))
SM = ParagraphStyle("sm", parent=B, fontSize=7.4, leading=9.5, spaceAfter=2)
CAP = ParagraphStyle("cap", parent=B, fontName="DV-I", fontSize=8, leading=10, textColor=colors.HexColor("#444444"))
esc = lambda s: s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

R = {r["name"]: r for r in (json.load(open(f)) for f in sorted(glob.glob("results/A2/*/result.json")))}
FP32_KEY = 'segformer_b0_pretrained_ce_fp32'
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
if not R: sys.exit("no results/A2/*/result.json found")
def desc(r):
    a = r["args"]
    d = f"lr {a['lr_scratch']}, {a['epochs']} epoch" if r["mode"] == "scratch" else f"lr encoder {a['lr_enc']}, decoder {a['lr_dec']}, {a['epochs']} epoch"
    if a.get("clip", 0): d += f", cắt gradient {a['clip']}"
    if a.get("no_amp"): d += ", fp32"
    if a.get("init_conv") == "official": d += ", khởi tạo conv theo mã gốc của SegFormer"
    return d

ORDER = ["unet_r34_pretrained_ce", "segformer_b0_pretrained_ce", "unet_r34_scratch_ce", "segformer_b0_scratch_ce",
         "unet_r34_frozen_ce", "segformer_b0_frozen_ce", "unet_r34_pretrained_ce_dice", "segformer_b0_pretrained_ce_dice"]
NAMES = [n for n in ORDER if n in R]; EXTRA = [n for n in R if n not in NAMES]; NAMES += EXTRA; any_r = R[NAMES[0]]; A = any_r["args"]
pct = lambda v: f"{v * 100:.2f}"
if FP32_KEY in R and 'segformer_b0_pretrained_ce' in R:
    FP32TXT = (f"Các lần chạy chính dùng fp16 (AMP); chạy lại SegFormer pretrained (CE) bằng fp32 cho {R[FP32_KEY]['test']['miou']*100:.2f}% mIoU so với {R['segformer_b0_pretrained_ce']['test']['miou']*100:.2f}% ở fp16 (chênh nhỏ, cùng cỡ mức nhiễu giữa các lần chạy); các mô hình khác chưa chạy lại bằng fp32.")
else:
    FP32TXT = "Các lần chạy chính dùng fp16 (AMP); chưa kiểm tra lại các mô hình pretrained bằng fp32."
def reason(r):
    if r["mode"] != "scratch": return "so sánh SegFormer lớn hơn (B2) với U-Net và B0" if r["arch"] == "segformer_b2" else "kiểm tra ảnh hưởng của fp16 lên kết quả pretrained"
    if "fair" in r["name"]: return "chạy lại U-Net from-scratch đúng công thức của SegFormer (fp32, cắt gradient, cùng learning rate) để so sánh công bằng"
    if "long" in r["name"]: return "chạy dài hơn cho from-scratch" + (" (dùng fp32 vì thử nghiệm chẩn đoán cho thấy fp16 làm SegFormer from-scratch không học được)" if r["arch"] == "segformer_b0" else ", để so sánh với SegFormer")
    return "thử learning rate nhỏ hơn sau khi SegFormer from-scratch không học được ở cấu hình mặc định"


# ---------- highlights (facts only, for the student to comment on) ----------
def highlights():
    L = ["# Số liệu nổi bật (chỉ là sự kiện, chưa phải nhận xét). Dùng để trả lời câu 8–11.\n",
         "Lưu ý: phần lớn cấu hình chỉ chạy 1 seed. Ở A1, hai lần chạy cùng seed từng lệch 2,9 điểm accuracy (val) ở epoch 10, nên chênh lệch nhỏ (dưới ~1 điểm mIoU) chưa chắc là thật.\n"]
    rk = sorted(R.values(), key=lambda r: -r["test"]["miou"])
    L.append("**Xếp hạng test mIoU (%):** " + "; ".join(f"{r['name']} {pct(r['test']['miou'])}" for r in rk) + "\n")
    g = lambda n: R[n]["test"]["miou"] * 100 if n in R else None
    for arch in ["unet_r34", "segformer_b0"]:
        p, s, f = g(f"{arch}_pretrained_ce"), g(f"{arch}_scratch_ce"), g(f"{arch}_frozen_ce")
        if p is not None and s is not None: L.append(f"- {arch}: pretrained − scratch = {p - s:+.2f} điểm mIoU")
        if p is not None and f is not None: L.append(f"- {arch}: pretrained(full fine-tune) − frozen = {p - f:+.2f} điểm mIoU")
        d = g(f"{arch}_pretrained_ce_dice")
        if p is not None and d is not None: L.append(f"- {arch}: CE+Dice − CE = {d - p:+.2f} điểm mIoU")
    for k in ["pretrained", "scratch", "frozen"]:
        u, s = g(f"unet_r34_{k}_ce"), g(f"segformer_b0_{k}_ce")
        if u is not None and s is not None: L.append(f"- Chế độ {k}: SegFormer − U-Net = {s - u:+.2f} điểm mIoU")
    for a in ["unet_r34", "segformer_b0"]:
        for n in [x for x in R if x.startswith((f"{a}_scratch_ce_long", f"{a}_scratch_ce_fair"))]:
            L.append(f"- {a}: scratch dài ({R[n]['args']['epochs']} epoch) = {g(n):.2f} mIoU, so với scratch 50 epoch {g(a + '_scratch_ce'):.2f}; best epoch {R[n]['best_epoch']}/{R[n]['args']['epochs']}")
    ls = sorted([x for x in R if x.startswith("segformer_b0_scratch_ce_long")], key=lambda x: -R[x]["args"]["epochs"])
    lu = sorted([x for x in R if x.startswith("unet_r34_scratch_ce_fair")] or [x for x in R if x.startswith("unet_r34_scratch_ce_long")], key=lambda x: -R[x]["args"]["epochs"])
    if ls and lu: L.append(f"- Scratch dài: SegFormer − U-Net = {g(ls[0]) - g(lu[0]):+.2f} điểm mIoU (công thức huấn luyện hai bên khác nhau)")
    if "segformer_b0_pretrained_ce_fp32" in R: L.append(f"- SegFormer pretrained: fp32 − fp16 = {g('segformer_b0_pretrained_ce_fp32') - g('segformer_b0_pretrained_ce'):+.2f} điểm mIoU")
    for n, d in sorted(DIAG.items()): L.append(f"- Chẩn đoán {n} ({desc(d)}): mIoU trên chính {d['args']['subset']} ảnh train = {d['best_val_miou']*100:.2f}%")
    for b, sd, vals, m, sdv in SEEDG: L.append(f"- Nhiều seed {b}: seeds {sd}, test mIoU {', '.join(f'{v:.2f}' for v in vals)}; trung bình {m:.2f} ± {sdv:.2f}")
    gu = [x for x in SEEDG if x[0] == "unet_r34_pretrained_ce"]; gs = [x for x in SEEDG if x[0] == "segformer_b0_pretrained_ce"]
    if gu and gs: L.append(f"- Pretrained, trung bình các seed: SegFormer-B0 − U-Net = {gs[0][3] - gu[0][3]:+.2f} điểm mIoU (độ lệch chuẩn: U-Net {gu[0][4]:.2f}, SegFormer {gs[0][4]:.2f})")
    for n in [x for x in R if x.startswith("segformer_b2")]:
        L.append(f"- {n}: test mIoU {g(n):.2f}; so với U-Net pretrained {g(n) - g('unet_r34_pretrained_ce'):+.2f}, so với SegFormer-B0 pretrained {g(n) - g('segformer_b0_pretrained_ce'):+.2f} điểm; params {R[n]['params_total']:,}")
    L.append("\n**Chi phí:**")
    for n in NAMES:
        r = R[n]; L.append(f"- {n}: params {r['params_total']:,} (trainable {r['params_trainable']:,}), {r['sec_per_epoch']:.1f}s/epoch, "
                           f"bộ nhớ đỉnh {r['peak_mem_mb']:.0f} MB, độ trễ {r['latency_ms_512']:.1f} ms, GFLOPs {r['gflops_512'] if r['gflops_512'] is None else round(r['gflops_512'], 1)}, best epoch {r['best_epoch']}/{r['args']['epochs']}")
    CL = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
    L.append("\n**3 lớp IoU thấp nhất / 3 lớp cao nhất (test), các run pretrained CE:**")
    for n in [x for x in NAMES if x.endswith("pretrained_ce")]:
        iou = R[n]["test"]["iou_per_class"]; o = sorted(range(21), key=lambda i: iou[i])
        L.append(f"- {n}: thấp nhất " + ", ".join(f"{CL[i]} {iou[i]*100:.1f}" for i in o[:3]) + "; cao nhất " + ", ".join(f"{CL[i]} {iou[i]*100:.1f}" for i in o[-3:]))
    os.makedirs("results/A2/figures", exist_ok=True); open("results/A2/figures/highlights.md", "w").write("\n".join(L) + "\n")
highlights()

# ---------- student's docx -> flowables (verbatim, plus verified numeric fixes) ----------
FIX_CELL = {"51,8%": "51,8% (đa tỉ lệ; một tỉ lệ: 51,0%)", "84,0%": "84,0% (đa tỉ lệ; một tỉ lệ: 82,4%)"}
FIX_TXT = [("Trong khi đó, SegFormer-B5 đạt 51,8% mIoU.", "Trong khi đó, SegFormer-B5 đạt 51,8% mIoU (kiểm thử đa tỉ lệ; với một tỉ lệ là 51,0%).")]
def tbl(rows, widths=None, fs=8):
    t = Table([[Paragraph(esc(c), ParagraphStyle("c", parent=B, fontSize=fs, leading=fs + 2.5, spaceAfter=0)) for c in r] for r in rows], colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .4, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return t
QGROUP = {1: "3.1 Tóm tắt phương pháp (ý tưởng và kiến trúc)", 5: "3.2 Thí nghiệm của paper", 7: "3.3 Hạn chế khi áp dụng cho tập của bài làm"}
QTITLE = {1: "Encoder phân cấp (Hierarchical Transformer Encoder)", 2: "Mix-FFN và việc bỏ positional encoding", 3: "Efficient Self-Attention", 4: "Decoder All-MLP",
          5: "Tập dữ liệu, metric và kết luận chính", 6: "Kết quả ablation đáng chú ý", 7: "Hạn chế khi áp dụng cho tập nhỏ như Pascal VOC"}
ANS_DOCX = "reports/a2_paper_analysis_final.docx"
QTITLE.update({8: "Pretrained, from-scratch và frozen encoder", 9: "Chi phí tính toán: tham số, GFLOPs, độ trễ, bộ nhớ", 10: "Các lớp có IoU thấp nhất",
               11: "Cấu hình thực nghiệm, khác biệt so với paper và bài học"})
_f = lambda x: f"{x:.2f}".replace(".", ",")
def _seed_para():
    gu = [g for g in SEEDG if g[0] == "unet_r34_pretrained_ce"]; gs = [g for g in SEEDG if g[0] == "segformer_b0_pretrained_ce"]
    if not (gu and gs): return None
    u, s_ = gu[0], gs[0]
    return (f"Với 3 seed (split dữ liệu giữ nguyên), U-Net ResNet34 pretrained đạt test mIoU {', '.join(_f(v) for v in u[2])} (trung bình {_f(u[3])} ± {_f(u[4])}) "
            f"và SegFormer-B0 pretrained đạt {', '.join(_f(v) for v in s_[2])} (trung bình {_f(s_[3])} ± {_f(s_[4])}). Chênh lệch giữa hai trung bình ({_f(abs(u[3] - s_[3]))} điểm) nhỏ hơn độ lệch chuẩn của mỗi bên, "
            "nên chưa thể kết luận bên nào tốt hơn về mIoU tổng. Giá trị nổi bật của B0 nằm ở số tham số và FLOPs thấp hơn, không phải độ chính xác cao hơn.")
def _scratch_para():
    b, c = DIAG.get("segformer_b0_scratch_ce_diagB"), DIAG.get("segformer_b0_scratch_ce_diagC")
    diag = (f"Thử nghiệm chẩn đoán cho mô hình học thuộc 64 ảnh cho thấy cùng một cấu hình (learning rate 2e-4, khởi tạo conv theo mã gốc, cắt gradient 1,0) chỉ đạt {_f(b['best_val_miou']*100)}% khi tính bằng fp16 nhưng đạt {_f(c['best_val_miou']*100)}% khi tính bằng fp32; "
            "vì vậy nguyên nhân chính của thất bại là độ chính xác số học fp16 (cơ chế cụ thể chưa được điều tra), không phải số epoch hay lượng dữ liệu. ") if (b and c) else ""
    return ("Ở chế độ from-scratch, hai lần chạy không chỉ khác nhau về số epoch. Lần chạy 50 epoch (fp16) không học được: mô hình gần như chỉ dự đoán background (mIoU 3,87%, epoch tốt nhất là epoch 1). "
            + diag + "Lần chạy 100 epoch dùng fp32 cùng cắt gradient, khởi tạo theo mã gốc và learning rate 2e-4, đạt 23,14% (epoch tốt nhất 90/100, nên chưa hội tụ). "
            "U-Net from-scratch đạt 11,11% sau 50 epoch và 18,43% sau 100 epoch (cùng dùng fp16; learning rate 5e-4 cho lần chạy 100 epoch). Do công thức huấn luyện của hai bên khác nhau, chưa thể kết luận kiến trúc nào học từ đầu tốt hơn. "
            "Dù vậy, kết quả from-scratch vẫn thấp hơn rất nhiều so với pretrained, nên số liệu hiện tại không ủng hộ ý cho rằng chỉ cần tăng số epoch là thay được pretraining.")
def _frozen_para():
    s_, u = R.get("segformer_b0_frozen_ce"), R.get("unet_r34_frozen_ce")
    if not (s_ and u): return None
    return (f"Lưu ý khi so sánh chế độ frozen: ở SegFormer-B0 chỉ decoder (khoảng {_f(s_['params_trainable']/1e6)}M tham số) được huấn luyện, còn ở U-Net là khoảng {_f(u['params_trainable']/1e6)}M; "
            "mức giảm so với fine-tune toàn bộ (8,0 điểm ở SegFormer-B0, 3,5 điểm ở U-Net) vì vậy chưa so sánh công bằng giữa hai kiến trúc.")
REPLACE_PARA = {
    "Lưu ý khi đối chiếu tài liệu:": "Lưu ý: mục 3.1 của paper nêu R = [64, 16, 4, 1], còn Bảng 6 ở phụ lục ghi R = [8, 4, 2, 1]. Hai giá trị này không trùng nhau (có thể Bảng 6 là tỉ lệ theo từng chiều, vì 8², 4², 2², 1² = 64, 16, 4, 1, nhưng đây chỉ là suy luận của mình, paper không giải thích). Bài làm dùng [64, 16, 4, 1] theo phần mô tả phương pháp.",
    "Về loss, khi trình bày nên mô tả đúng": "Về loss, paper không mô tả riêng hàm loss; mục 4.1 chỉ nêu tác giả không dùng các thủ thuật như OHEM, auxiliary loss hay class-balance loss. Đóng góp của SegFormer nằm ở thiết kế encoder–decoder, không phải một loss mới.",
    "Decoder có thể giữ cấu trúc gọn vì": "Theo paper (mục 3.2), decoder có thể giữ cấu trúc gọn vì encoder Transformer có vùng tiếp nhận hiệu dụng (effective receptive field) lớn hơn encoder CNN: attention ở các stage thấp thiên về cục bộ, còn ở Stage 4 mang tính phi cục bộ, nên khi hợp nhất các stage, decoder nhận được cả hai loại thông tin. Decoder chủ yếu cần chuẩn hóa, đồng nhất độ phân giải và hợp nhất các biểu diễn này, thay vì xây dựng thêm một mạng giải mã nặng.",
    "Thí nghiệm from-scratch 100 epoch đạt 23,14%": _scratch_para,
    "Để đặt kết quả trong bối cảnh, U-Net ResNet34 pretrained": _seed_para,
}
INSERT_AFTER = {
    "Khác với ViT thuần thường tạo ra": ["Để tạo các feature này, paper chia ảnh thành patch 4×4 (nhỏ hơn patch 16×16 của ViT, có lợi cho dự đoán dày) và dùng Overlapped Patch Merging (K = 7, S = 4, P = 3 ở tầng đầu; K = 3, S = 2, P = 1 ở các tầng sau) để giữ tính liên tục cục bộ quanh các patch. Cột \"Vai trò đặc trưng\" trong bảng trên là diễn giải của mình; paper (Hình 3) chỉ nêu rằng attention ở các stage thấp thiên về cục bộ còn ở Stage 4 mang tính phi cục bộ."],
    "Decoder có thể giữ cấu trúc gọn vì": ["Paper kiểm chứng điều này ở Bảng 1d (ADE20K): ghép decoder MLP với encoder CNN cho mIoU thấp hơn nhiều (ResNet-50 34,7%; ResNet-101 38,7%; ResNeXt-101 39,8%) so với MiT-B2 (45,4%)."],
    "Kết luận cốt lõi của paper": ["Paper còn đánh giá độ bền trên Cityscapes-C (Bảng 5): với nhiễu Gaussian, SegFormer-B5 đạt 57,8% mIoU, trong khi DeepLabV3+ (ResNet-101) đạt 13,2%."],
    "Từ các phân tích trên, hướng thực nghiệm": ["Về hạn chế do chính paper nêu (mục 5): chưa rõ mô hình nhỏ nhất (khoảng 3,7M tham số) có chạy tốt trên chip thiết bị biên chỉ có 100K bộ nhớ hay không; paper cũng không có phần phân tích riêng các trường hợp thất bại, chỉ có hình minh họa định tính. Về giả thiết huấn luyện, paper dùng encoder pretrain ImageNet-1K, AdamW với learning rate 6e-5, crop 512 và 160K bước (ablation 40K bước) trên các tập có hàng chục nghìn ảnh; thiết lập của bài làm (1.318 ảnh train, khoảng 4.100 bước) nhỏ hơn nhiều."],
    "Pretrained + fine-tuning đạt 66,46% mIoU": [_frozen_para],
    "SegFormer-B2 đạt 78,31% mIoU, cao hơn B0": [],
    "Một điểm cần báo cáo minh bạch là peak memory": ["Ghi chú đo: peak memory là đỉnh trong cả huấn luyện và validation (batch 16, ảnh 512×512), không phải bộ nhớ huấn luyện thuần. GFLOPs do torch FlopCounterMode đếm (một phép nhân-cộng tính là 2 FLOPs), nên con số 15,7 GFLOPs của B0 chưa so trực tiếp được với 8,4G trong paper, vốn có thể đếm theo quy ước khác. Độ trễ đo ở fp16, ảnh 512×512, batch 1 trên GPU Tesla T4 và dao động khoảng 2 ms giữa các lần đo."],
    "Thứ hai, paper nghiên cứu hệ MiT": ["Về cách huấn luyện, paper dùng mmsegmentation, AdamW với learning rate 6e-5, crop 512, 160K bước (ablation 40K bước) và batch 16; bài làm dùng PyTorch/HuggingFace, crop 384, 50 epoch (khoảng 4.100 bước, batch 16), learning rate 1e-4 cho encoder và 1e-3 cho decoder, với augmentation tương tự paper (scale 0,5–2,0, lật ngang, cắt ngẫu nhiên). Kết quả của SegFormer-B2 chỉ chạy một seed, còn U-Net và SegFormer-B0 pretrained có 3 seed (Bảng 2)."],
}
PATCH = [("Tôi lựa chọn", "Mình lựa chọn"), ("của nhóm", "của mình"), ("nhóm thực nghiệm trên Pascal VOC", "mình thực nghiệm trên Pascal VOC"),
         ("So với paper, nhóm có", "So với paper, bài làm có"), ("nhóm tập trung vào B0 và B2", "mình tập trung vào B0 và B2"), ("nhóm mở rộng phân tích", "mình mở rộng phân tích"),
         ("tổng hợp ba cấu hình SegFormer-B0", "tổng hợp bốn cấu hình SegFormer-B0"), ("20.210 ảnh huấn luyện, 150 lớp", "20.210 ảnh, 150 lớp"),
         ("vừa ít suy giảm hơn khi tăng độ phân giải kiểm thử:", "vừa ít suy giảm hơn khi tăng độ phân giải kiểm thử (768×768 dùng cửa sổ trượt, 1024×2048 dùng cả ảnh):"),
         ("nhưng chair và diningtable vẫn là những lớp tương đối khó.", "nhưng chair, sofa và diningtable vẫn là ba lớp có IoU thấp nhất."),
         ("Tuy nhiên, latency đo được của B0 là 12,4 ms, cao hơn U-Net (10,3 ms). Điều này cho thấy Params và FLOPs thấp không tự động đồng nghĩa với thời gian suy luận thực tế thấp hơn; latency còn phụ thuộc vào cấu trúc phép toán, khả năng tối ưu kernel và phần cứng.",
          "Latency đo được của B0 là 12,4 ms, của U-Net là 10,3 ms. Tuy nhiên, cùng kiến trúc B0 nhưng cấu hình khác đo ra 10,4–12,5 ms (ví dụ lần chạy from-scratch: 10,4 ms), nên chênh lệch khoảng 2 ms nằm trong dao động đo và chưa kết luận được B0 chậm hơn U-Net. Điều có thể nói là Params và FLOPs thấp không tự động đồng nghĩa với latency thấp: B0 có ít tham số hơn khoảng 6,6 lần và ít FLOPs hơn khoảng 3,8 lần nhưng latency không thấp hơn."),
         ("U-Net có latency thấp nhất trong ba cấu hình, còn B0", "U-Net có latency đo thấp nhất trong ba cấu hình (chênh lệch với B0 nằm trong dao động đo), còn B0")]
DROP = ("Tài liệu này trả lời lần lượt", "Lưu ý về dữ liệu:", "Các chỉ số mIoU, Pixel Accuracy", "Nguồn số liệu thực nghiệm:", "Phân tích SegFormer và đối chiếu", "Xie và cộng sự (2021)", "SegFormer: Simple and Efficient", "PHÂN TÍCH PAPER")
_used = set()
def _patch(t):
    for a, b in PATCH:
        if a in t: t = t.replace(a, b); _used.add(a)
    return t
def student_items():
    """-> (part1_items, part2_items); item = (kind, payload) kind in h2,h3,p,table"""
    from docx.table import Table as DT; from docx.text.paragraph import Paragraph as DP
    d = Document(ANS_DOCX); P1, P2 = [], []; part = 0; cur = None
    for el in d.element.body.iterchildren():
        if el.tag.endswith("}p"):
            p = DP(el, d); t = p.text.strip(); st = p.style.name
            if not t: continue
            if st.startswith("Heading 1"):
                if t.startswith("Phần II"): part, cur = 2, P2
                elif t.startswith("Phần I"): part, cur = 1, P1
                elif t.startswith("Kết luận chung"): part, cur = 2, P2; P2.append(("h3", "Kết luận"))
                else: part = 0
                continue
            if part == 0: continue
            if st.startswith("Heading 2"):
                m = re.match(r"Câu (\d+)\.", t)
                if m:
                    n = int(m.group(1))
                    if n in QGROUP: cur.append(("h2", QGROUP[n]))
                    cur.append(("h3", QTITLE.get(n, t)))
                else: cur.append(("h2", t))
                continue
            if t.startswith(DROP): continue
            orig = t; hit = None
            for k, v in REPLACE_PARA.items():
                if t.startswith(k):
                    new = v() if callable(v) else v
                    if new: t = new
                    hit = k; break
            t = _patch(t); cur.append(("p", t))
            for k, v in INSERT_AFTER.items():
                if orig.startswith(k):
                    for x in v:
                        x = x() if callable(x) else x
                        if x: cur.append(("p", _patch(x)))
        elif el.tag.endswith("}tbl") and part:
            T = DT(el, d); rows = [[_patch(FIX_CELL.get(c.text.strip(), c.text.strip())) for c in r.cells] for r in T.rows]
            for r in rows:
                if r and r[0] == "tvmonitor" and len(r) > 2 and r[2] in ("—", "-", ""): r[2] = "71,54%"
            cur.append(("table", rows))
    return P1, P2
def render_items(items):
    out = []
    for k, v in items:
        if k == "h2": out.append(Paragraph(esc(v), H2))
        elif k == "h3": out.append(Paragraph(esc(v), H3))
        elif k == "p": out.append(Paragraph(esc(v), B))
        else: out += [tbl(v, fs=7.5 if len(v[0]) > 5 else 8.2), Spacer(1, 5)]
    return out
def dump_items(P1, P2, path):
    L = ["# Mục 3. Phân tích paper"] + [(f"\n## {v}" if k == "h2" else f"\n### {v}" if k == "h3" else (v if k == "p" else "\n".join(" | ".join(r) for r in v))) for k, v in P1]
    L += ["\n# Mục 6. Thảo luận kết quả và đối chiếu với paper"] + [(f"\n### {v}" if k in ("h2", "h3") else (v if k == "p" else "\n".join(" | ".join(r) for r in v))) for k, v in P2]
    open(path, "w", encoding="utf-8").write("\n\n".join(L))
P1, P2 = student_items()
dump_items(P1, P2, "reports/a2_phan_tich_paper_da_chinh_sua.txt")
import json as _json; _json.dump({"p1": P1, "p2": P2}, open("reports/_answers.json", "w", encoding="utf-8"), ensure_ascii=False)
_miss = [a[:40] for a, b in PATCH if a not in _used]
if _miss: print("WARNING patches not applied:", _miss)
_left = [t for k, t in P1 + P2 if k == "p" and "nhóm" in t]
if _left: print("WARNING 'nhóm' left in:", [t[:60] for t in _left])

NOTE_PAPER = ("Ghi chú đối chiếu với bản PDF của paper (arXiv v2): (1) số tham số của SegFormer-B0 được paper ghi là 3,7M ở Hình 1 và phần Kết luận, "
              "còn Bảng 2 ghi 3,8M (Bảng 1a: encoder 3,4M + decoder 0,4M); (2) các con số 51,8% (ADE20K) và 84,0% (Cityscapes) của B5 là kết quả kiểm thử đa tỉ lệ (MS), "
              "kết quả một tỉ lệ (SS) là 51,0% và 82,4%; (3) ở bản này của paper, mục 4.1 nêu pretrain ImageNet-1K, AdamW, learning rate ban đầu 6e-5, crop 512 trên ADE20K.")

def text_file(path, marker):
    s = open(path, encoding="utf-8").read().strip(); fl = []
    if marker in s and not DRAFT: sys.exit(f"{path} still has placeholder '{marker}'. Edit it first (or use --draft).")
    for blk in re.split(r"\n\s*\n", s):
        blk = blk.strip()
        if blk.startswith("## "): fl.append(Paragraph(esc(blk[3:]), H2))
        elif blk: fl.append(Paragraph(esc(blk).replace("\n", "<br/>"), B))
    return fl

def img(path, w=16.5 * cm, cap=None):
    if not os.path.exists(path): return []
    iw, ih = ImageReader(path).getSize(); o = [Image(path, w, w * ih / iw)]
    return o + ([Paragraph(cap, CAP)] if cap else []) + [Spacer(1, 6)]

S = []; add = S.append
add(Paragraph("A2 — Phân đoạn ảnh ngữ nghĩa: U-Net (ResNet-34) và SegFormer-B0 trên Pascal VOC 2012", ParagraphStyle("t", parent=H1, fontSize=16, leading=20)))
add(Paragraph("Học sâu và ứng dụng trong thị giác máy tính (CO5085), HK 261. Sinh viên: Phan Ngọc Trâm (2570790), làm cá nhân. Giảng viên: Lê Thành Sách, Nguyễn Quốc Minh.", B))
add(Paragraph("1. Bài toán và dữ liệu", H1))
add(Paragraph(f"Bài toán: gán nhãn cho từng pixel (21 lớp: 20 vật thể và background). Dữ liệu: Pascal VOC 2012 Segmentation. Tập train chính thức 1.464 ảnh được chia (seed {A['seed']}) thành "
              f"{any_r['n_train']} ảnh train và {any_r['n_val']} ảnh validation; tập val chính thức ({any_r['n_test']} ảnh) được dùng làm test, không dùng để chọn mô hình. "
              "Không dùng SBD (bộ nhãn mở rộng), đây là một hạn chế có chủ đích. Viền vật thể có nhãn 255 và bị bỏ qua khi tính loss và metric. "
              "Giấy phép: ảnh VOC lấy từ Flickr, dùng cho nghiên cứu/học tập; không có giấy phép mở rõ ràng.", B))
S += img("results/A2/eda/class_distribution.png", cap="Hình 1. Phân bố lớp trong tập train (số liệu từ eda.py).")
S += img("results/A2/eda/samples.png", cap="Hình 2. Một số ảnh và nhãn trong tập train.")
add(Paragraph("2. Dataset, DataLoader và augmentation", H1))
add(Paragraph(f"Huấn luyện: phóng ngẫu nhiên tỉ lệ 0,5–2,0, lật ngang ngẫu nhiên, cắt ngẫu nhiên {A['crop']}×{A['crop']} (đệm bằng giá trị 0 với ảnh và 255 với nhãn), chuẩn hóa theo thống kê ImageNet, batch {A['bs']}. "
              "Đánh giá: không augmentation; ảnh được đệm lên 512×512 (phần đệm có nhãn 255 nên bị bỏ qua), nên số liệu tính trên đúng pixel của ảnh gốc.", B))
add(Paragraph("3. Phân tích paper SegFormer", H1))
add(Paragraph("Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., Luo, P. SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. Advances in Neural Information Processing Systems 34 (NeurIPS 2021). arXiv:2105.15203.", B))
add(Paragraph("Các mục 3 và 6 tách ba loại thông tin: (i) điều paper công bố, (ii) số liệu thực nghiệm của bài làm, (iii) nhận định rút ra từ việc đối chiếu hai nguồn.", B))
S += render_items(P1); add(Paragraph(NOTE_PAPER, SM))
add(Paragraph("4. Pipeline và protocol thí nghiệm", H1))
add(Paragraph("Hai kiến trúc: U-Net với encoder ResNet-34 (timm, trọng số resnet34.a1_in1k) làm đại diện CNN; SegFormer-B0 (HuggingFace, encoder MiT-B0 pretrain ImageNet-1K, decoder khởi tạo ngẫu nhiên) làm đại diện Transformer." + B2TXT + " "
              "Ba chế độ: from-scratch, pretrained (fine-tune toàn bộ), pretrained và đóng băng encoder (chỉ train decoder). Thêm một ablation loss: cross-entropy so với cross-entropy + Dice (cho hai mô hình pretrained). "
              f"Chung: AdamW (weight decay {A['wd']}), lịch học poly có warm-up, fp16 (các lần ghi fp32 thì dùng fp32), {A['epochs']} epoch, seed {A['seed']}. Learning rate: from-scratch {A['lr_scratch']}; pretrained/frozen: encoder {A['lr_enc']}, decoder {A['lr_dec']}. "
              "Không tinh chỉnh siêu tham số riêng cho từng mô hình." + ("".join(f" Chạy thêm {n} ({desc(R[n])}): {reason(R[n])}." for n in EXTRA)) + ("".join(f" Chẩn đoán overfit {n}: huấn luyện trên {d['args']['subset']} ảnh, mIoU trên chính các ảnh đó là {d['best_val_miou']*100:.2f}% sau {d['args']['epochs']} epoch." for n, d in DIAG.items())) + " Mô hình tốt nhất theo mIoU trên tập validation được dùng để đo trên test. " + SEEDPROTO + ""
              "Lưu ý: hai encoder đều pretrain trên ImageNet-1K nhưng bằng công thức huấn luyện khác nhau, nên khác biệt CNN–Transformer còn lẫn yếu tố này.", B))
add(Paragraph("5. Kết quả", H1))
rows = [["Run", "mIoU", "PixAcc", "mDice", "Params train.", "s/epoch", "Bộ nhớ MB", "ms (512)", "GFLOPs*"]]
for n in NAMES:
    r = R[n]; t = r["test"]
    rows.append([n + (f" [{r['args']['epochs']}ep]" if r["args"]["epochs"] != A["epochs"] else ""), pct(t["miou"]), pct(t["pixacc"]), pct(t["mdice"]), f"{r['params_trainable']:,}", f"{r['sec_per_epoch']:.1f}", f"{r['peak_mem_mb']:.0f}", f"{r['latency_ms_512']:.1f}", "" if r["gflops_512"] is None else f"{r['gflops_512']:.1f}"])
add(tbl(rows, [4.6 * cm, 1.4 * cm, 1.4 * cm, 1.4 * cm, 2.3 * cm, 1.4 * cm, 1.6 * cm, 1.4 * cm, 1.4 * cm], fs=7))
add(Paragraph("Bảng 1. Kết quả trên tập test (1.449 ảnh). mIoU, PixAcc, mDice tính theo %. Độ trễ đo với ảnh 512×512, batch 1, fp16, GPU: " + any_r["env"]["gpu"] + ". Bộ nhớ là đỉnh trong cả huấn luyện và validation (batch 16 ảnh 512×512). *GFLOPs do torch FlopCounterMode đếm, một phép nhân-cộng tính là 2 FLOPs; nhiều paper đếm là 1 (có thể cả SegFormer), nên chưa so trực tiếp với con số 8,4G trong paper. Độ trễ của cùng một kiến trúc dao động khoảng 2 ms giữa các lần đo.", CAP))
if SEEDG:
    rows2 = [["Run", "Seed", "Test mIoU từng seed (%)", "Trung bình ± độ lệch chuẩn"]] + [[b, ", ".join(map(str, sd)), ", ".join(f"{v:.2f}" for v in vals), f"{m:.2f} ± {sdv:.2f}"] for b, sd, vals, m, sdv in SEEDG]
    add(Spacer(1, 6)); add(tbl(rows2, [5.4 * cm, 2.2 * cm, 5 * cm, 4 * cm], fs=8))
    add(Paragraph("Bảng 2. Test mIoU với nhiều seed (độ lệch chuẩn mẫu). Seed chỉ đổi khởi tạo decoder và thứ tự/augmentation; split dữ liệu giữ nguyên.", CAP))
for f, c in [("miou_by_mode", "Hình 3. Test mIoU theo chế độ huấn luyện (loss CE). Cột scratch dài: " + SCRATCHCAP + ""), ("curves", "Hình 4. Loss huấn luyện và mIoU trên validation theo epoch."),
             ("per_class_iou", "Hình 5. IoU từng lớp trên test (pretrained, CE)."), ("cost_vs_miou", "Hình 6. Độ chính xác so với độ trễ, số tham số và GFLOPs (pretrained, CE). Thanh sai số là độ lệch chuẩn khi có nhiều seed."), ("diag_curves", "Hình 7. Chẩn đoán: mIoU trên chính 64 ảnh train của SegFormer from-scratch (A: fp16; B: fp16 + khởi tạo gốc + cắt gradient; C: như B nhưng fp32)."),
             ("qualitative", "Hình 8. Ảnh, nhãn thật và dự đoán của các mô hình trên một số ảnh test cố định.")]:
    S += img(f"results/A2/figures/{f}.png", cap=c)
S += img("results/A2/figures/confusion.png", cap="Hình 9. Confusion matrix (chuẩn hóa theo hàng, lớp thật) trên test cho các mô hình pretrained, loss CE.")
if CONF_LINES:
    add(Paragraph("Nhầm lẫn lớn nhất (lớp thật → lớp dự đoán, tỉ lệ pixel của lớp thật bị gán sang lớp đó), tập test:", H2))
    for _l in CONF_LINES: add(Paragraph(esc(_l), B))
add(Paragraph("6. Thảo luận kết quả và đối chiếu với paper", H1)); S += (render_items(P2) if P2 else text_file("reports/a2_discussion.txt", "[SINH VIÊN"))
add(Paragraph("7. Hạn chế", H1))
add(Paragraph(SEEDLIM_C + "không dùng SBD (chỉ 1.318 ảnh train); không tinh chỉnh siêu tham số cho từng mô hình; " + B0LIM + "số epoch và crop nhỏ hơn thiết lập trong paper; "
              "so sánh CNN–Transformer bị lẫn công thức pretrain khác nhau. " + FP32TXT + " "
              "SegFormer from-scratch ở fp16 (50 epoch, lr 5e-4 và 1e-4) chỉ dự đoán gần như toàn background; trong thử nghiệm học thuộc 64 ảnh, cấu hình B (fp16) và C (fp32) chỉ khác nhau ở độ chính xác số học nhưng chỉ C học được, vì vậy so sánh from-scratch dùng các lần chạy dài. "
              + SCRATCHLIM, B))
add(Paragraph("8. Ghi nhận việc dùng AI", H1)); S += text_file("reports/a2_ai_disclosure.txt", "[SINH VIÊN")
add(Paragraph("Tài liệu tham khảo", H1))
add(Paragraph("[1] Xie et al., SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers, NeurIPS 2021 (arXiv:2105.15203).<br/>[2] Ronneberger et al., U-Net: Convolutional Networks for Biomedical Image Segmentation, arXiv:1505.04597, 2015.<br/>[3] Everingham et al., The PASCAL Visual Object Classes (VOC) Challenge 2012.", B))
os.makedirs("reports", exist_ok=True)
SimpleDocTemplate("reports/a2_report.pdf", pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm, title="A2 report").build(S)
print("built reports/a2_report.pdf and results/A2/figures/highlights.md")
