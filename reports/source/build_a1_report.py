"""Dựng reports/a1_report.pdf (báo cáo riêng cho bài tập lớn A1).

Chạy từ thư mục gốc của repo:   python reports/source/build_a1_report.py
Cần:  pip install reportlab pillow
Số liệu được chép từ results/A1/summary.md, log.txt, top_confusions.md, eda_stats.json, config.json.
Các bảng suy ra (chênh lệch, tỷ lệ) được tính từ chính các số này trong code bên dưới.
"""
import os
import sys

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
OUT = os.path.join(ROOT, "reports", "a1_report.pdf")

FONT_DIRS = ["/usr/share/fonts/truetype/dejavu/", "C:/Windows/Fonts/"]
def reg(name, files):
    for d in FONT_DIRS:
        for f in files:
            p = os.path.join(d, f)
            if os.path.exists(p):
                pdfmetrics.registerFont(TTFont(name, p)); return
    raise SystemExit(f"Không tìm thấy font cho {name}")
reg("VN", ["DejaVuSans.ttf", "arial.ttf"])
reg("VNB", ["DejaVuSans-Bold.ttf", "arialbd.ttf"])
pdfmetrics.registerFontFamily("VN", normal="VN", bold="VNB", italic="VN", boldItalic="VNB")

S = {
    "title": ParagraphStyle("title", fontName="VNB", fontSize=18, leading=24, spaceAfter=4),
    "h1": ParagraphStyle("h1", fontName="VNB", fontSize=14, leading=19, spaceBefore=14, spaceAfter=6, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="VNB", fontSize=11.5, leading=16, spaceBefore=10, spaceAfter=4, keepWithNext=1),
    "p": ParagraphStyle("p", fontName="VN", fontSize=10, leading=14.5, spaceAfter=5),
    "b": ParagraphStyle("b", fontName="VN", fontSize=10, leading=14.5, leftIndent=14, bulletIndent=2, spaceAfter=3),
    "c": ParagraphStyle("c", fontName="VN", fontSize=8.5, leading=11.5),
    "cb": ParagraphStyle("cb", fontName="VNB", fontSize=8.5, leading=11.5),
    "cap": ParagraphStyle("cap", fontName="VN", fontSize=8.5, leading=11.5, textColor=colors.HexColor("#444444"),
                          spaceAfter=8, alignment=1),
}
P = lambda t, s="p": Paragraph(t, S[s])
def bullets(items):
    return [Paragraph(t, S["b"], bulletText="•") for t in items]
def code(t):
    return f"<font name='VN'>{t}</font>"
def f4(x):
    return f"{x:.4f}".replace(".", ",")
def f1(x):
    return f"{x:.1f}".replace(".", ",")
def ap(x):        # giá trị tuyệt đối, điểm phần trăm, 1 chữ số thập phân
    return f"{100 * abs(x):.1f}".replace(".", ",")
def pts(x):       # x là hiệu accuracy (0-1) -> điểm phần trăm
    return f"{100 * x:+.2f}".replace(".", ",")

def img(rel, width, caption):
    path = os.path.join(ROOT, rel)
    w, h = PILImage.open(path).size
    return KeepTogether([Image(path, width=width, height=width * h / w), P(caption, "cap")])

def table(rows, widths, header=True):
    data = [[Paragraph(str(c), S["cb"] if (header and i == 0) else S["c"]) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")))
    t.setStyle(TableStyle(st))
    return t

# ---------------------------------------------------------------- số liệu (results/A1/summary.md)
# (arch, mode): params_total, trainable, best_epoch, val, test, macro_f1, s_per_epoch, total_s, mem_mb, train_acc_ep15
R = {
    ("resnet18", "scratch"): (11227812, 11227812, 15, 0.6098, 0.6038, 0.6030, 76.3, 1144, 1779),
    ("resnet18", "head"):    (11227812, 51300,    14, 0.6290, 0.6233, 0.6206, 24.6, 369, 1135),
    ("resnet18", "partial"): (11227812, 10544740, 13, 0.7736, 0.7780, 0.7773, 37.3, 559, 1256),
    ("resnet18", "full"):    (11227812, 11227812, 13, 0.7924, 0.7982, 0.7981, 76.6, 1150, 1778),
    ("vit_tiny", "scratch"): (5543716, 5543716, 15, 0.3802, 0.3911, 0.3829, 67.6, 1013, 2309),
    ("vit_tiny", "head"):    (5543716, 19300,   12, 0.7668, 0.7675, 0.7673, 24.0, 360, 651),
    ("vit_tiny", "partial"): (5543716, 2688868, 12, 0.8438, 0.8478, 0.8475, 44.9, 674, 1241),
    ("vit_tiny", "full"):    (5543716, 5543716, 14, 0.8730, 0.8754, 0.8753, 67.2, 1008, 2309),
}
TOTAL, TRAIN, BEST, VAL, TEST, F1, SPE, TOT, MEM = range(9)
NAME = {"resnet18": "ResNet-18 (CNN)", "vit_tiny": "ViT-Tiny/16 (Transformer)"}
MODES = ["scratch", "head", "partial", "full"]
# train acc / val acc tại epoch 15 (log.txt) để nhận xét overfitting
GAP = {("resnet18", "partial"): (0.8582, 0.7734), ("resnet18", "full"): (0.8721, 0.7902),
       ("vit_tiny", "partial"): (0.9938, 0.8426), ("vit_tiny", "full"): (0.9993, 0.8728)}


def sections():
    s = [P("Bài tập lớn A1 – CNN và Transformer trên CIFAR-100", "title"),
         P("Học sâu và ứng dụng trong thị giác máy tính (CO5085) – Học kỳ 261 (2026–2027)"),
         P("Sinh viên: Phan Ngọc Trâm – MSSV 2570790 (làm cá nhân)"),
         P("Mã nguồn: https://github.com/tramphanngoc174/co5085-exercises (thư mục A1)"),
         Spacer(1, 6)]

    # ---------------- 1
    s += [P("1. Tóm tắt", "h1"),
          P("Báo cáo so sánh một CNN (ResNet-18) và một Transformer (ViT-Tiny/16) trên CIFAR-100 (100 lớp), với bốn chế độ huấn luyện: "
            "<b>scratch</b> (khởi tạo ngẫu nhiên), và ba chế độ trên trọng số pretrained: <b>head</b> (đóng băng toàn bộ backbone, chỉ train lớp phân loại), "
            "<b>partial</b> (đóng băng nửa đầu mạng) và <b>full</b> (fine-tune toàn bộ). Tổng cộng 8 cấu hình, mỗi cấu hình chạy một lần (seed 42, 15 epoch).")]
    s += bullets([
        f"Tốt nhất: <b>ViT-Tiny pretrained + fine-tune toàn bộ</b> với test accuracy {f4(R['vit_tiny','full'][TEST])}; "
        f"ResNet-18 tốt nhất là full với {f4(R['resnet18','full'][TEST])}.",
        f"<b>Từ đầu, CNN hơn hẳn Transformer</b>: ResNet-18 {f4(R['resnet18','scratch'][TEST])} so với ViT-Tiny {f4(R['vit_tiny','scratch'][TEST])} "
        f"(chênh {ap(R['resnet18','scratch'][TEST]-R['vit_tiny','scratch'][TEST])} điểm).",
        f"<b>Pretrain tăng mạnh</b>: full so với scratch tăng {ap(R['resnet18','full'][TEST]-R['resnet18','scratch'][TEST])} điểm với ResNet-18 "
        f"và {ap(R['vit_tiny','full'][TEST]-R['vit_tiny','scratch'][TEST])} điểm với ViT-Tiny.",
        "<b>Đóng băng đổi accuracy lấy chi phí</b>: head-only chạy nhanh hơn khoảng 3 lần và dùng ít bộ nhớ hơn nhưng kém full nhiều; "
        "partial lấy lại phần lớn accuracy với chi phí thấp hơn full (mục 6).",
        "<b>Lưu ý quan trọng khi so sánh CNN và Transformer</b>: hai mô hình pretrained được pretrain trên dữ liệu khác nhau "
        "(ResNet-18 trên ImageNet-1k, ViT-Tiny trên ImageNet-21k rồi ImageNet-1k) và khác số tham số; do đó kết quả không tách riêng được ảnh hưởng của kiến trúc (mục 6.3, mục 8)."])

    # ---------------- 2
    s += [P("2. Dataset", "h1"),
          P("<b>Tập dữ liệu:</b> CIFAR-100 (Krizhevsky, 2009): 60.000 ảnh màu 32×32, 100 lớp (600 ảnh mỗi lớp), gồm 50.000 ảnh train và 10.000 ảnh test gốc. "
            "Lý do chọn: nằm trong danh sách đề cho phép; nhiều lớp (100) và khó hơn hẳn các tập của E1; đủ nhỏ để chạy 8 cấu hình trên GPU Colab; "
            "có nhiều cặp lớp giống nhau (các loại cây, các nhóm người, đồ dùng) nên phân tích lỗi có ý nghĩa."),
          P("<b>License:</b> đây là tập dữ liệu công khai do Krizhevsky công bố, trích dẫn bài báo “Learning Multiple Layers of Features from Tiny Images” (2009). "
            "Thẻ dataset chính thức trên Hugging Face ghi license là “unknown”; một số bản sao của bên thứ ba ghi MIT hoặc CC BY 4.0 nhưng đó không phải nguồn gốc. "
            "Nhóm chỉ dùng cho học tập, phi thương mại; người đọc nên đối chiếu thêm tại trang chính thức (cs.toronto.edu/~kriz/cifar.html)."),
          P("<b>Chia dữ liệu</b> (seed 42, chỉ số lưu trong splits/cifar100_tr450_val50_seed42.json):"),
          table([["Tập", "Số ảnh", "Mỗi lớp", "Nguồn"],
                 ["Train", "45.000", "450", "tập train gốc (50.000 ảnh)"],
                 ["Validation", "5.000", "50", "tập train gốc, không trùng train; dùng để chọn epoch"],
                 ["Test", "10.000", "100", "tập test gốc, chỉ dùng để báo cáo cuối"]],
                [70, 60, 60, 305]), Spacer(1, 4),
          P("Cả 50.000 ảnh train gốc đều được dùng (không lấy subset). Các lớp cân bằng hoàn toàn nên accuracy và macro-F1 gần nhau."),
          P("<b>EDA:</b> ảnh gốc 3×32×32; trung bình RGB trên tập train (0,5068; 0,4865; 0,4412), độ lệch chuẩn (0,2674; 0,2565; 0,2761). "
            "Hình 1 là một ảnh mẫu mỗi lớp. Ảnh rất nhỏ nên nhiều lớp (ví dụ các loại cây, boy/girl/man/woman) khó phân biệt kể cả bằng mắt."),
          img("results/A1/eda_samples.png", 330, "Hình 1. CIFAR-100: mỗi lớp một ảnh mẫu (32×32).")]

    # ---------------- 3
    s += [P("3. DataLoader và augmentation", "h1")] + bullets([
        "Dataset: lớp <font name='VN'>TensorImageDataset</font> giữ toàn bộ ảnh dưới dạng tensor uint8 (N×3×32×32) trong bộ nhớ; "
        "<font name='VN'>DataLoader</font> batch size 128, tập train xáo trộn bằng generator có seed (file A1/data.py).",
        "Augmentation chỉ ở tập train, chạy trên GPU cho từng batch: <b>random crop</b> 32×32 sau khi đệm 4 pixel giá trị 0 mỗi bên (độ lệch ngẫu nhiên riêng cho từng ảnh) "
        "và <b>lật ngang</b> với xác suất 0,5. Tập validation và test không augmentation.",
        "Sau augmentation, ảnh được <b>phóng to bằng nội suy bilinear từ 32×32 lên 224×224</b> (đúng kích thước pretrain của hai mô hình), rồi chuẩn hóa mean/std "
        "theo cấu hình pretrained của timm cho từng mô hình. Phóng to không thêm thông tin mới cho ảnh; đây là một lựa chọn để dùng trọng số pretrained.",
        "Huấn luyện dùng mixed precision (fp16 autocast + GradScaler) để tiết kiệm thời gian và bộ nhớ."])

    # ---------------- 4
    s += [P("4. Protocol thí nghiệm", "h1"),
          P("<b>Mô hình</b> (thư viện timm 1.0.31, torch 2.14.1+cu130 theo config.json):"),
          table([["Kiến trúc", "Họ", "Tham số (100 lớp)", "Trọng số pretrained (timm)", "Pretrain trên"],
                 ["ResNet-18", "CNN", "11.227.812", "resnet18.a1_in1k", "ImageNet-1k"],
                 ["ViT-Tiny/16", "Transformer", "5.543.716", "vit_tiny_patch16_224.augreg_in21k_ft_in1k", "ImageNet-21k, rồi fine-tune ImageNet-1k"]],
                [65, 65, 85, 170, 110]), Spacer(1, 4),
          P("<b>Bốn chế độ huấn luyện</b> (trên từng kiến trúc):")]
    s += [table([["Chế độ", "Trọng số khởi tạo", "Phần được train", "ResNet-18 trainable", "ViT-Tiny trainable"],
                 ["scratch", "ngẫu nhiên", "toàn bộ", "11.227.812 (100%)", "5.543.716 (100%)"],
                 ["head", "pretrained", "chỉ lớp phân loại (backbone đóng băng)", "51.300 (0,5%)", "19.300 (0,3%)"],
                 ["partial", "pretrained",
                  "ResNet: layer3, layer4, fc (đóng băng conv1, bn1, layer1, layer2). ViT: block 6–11, norm, head (đóng băng patch_embed, cls_token, pos_embed, block 0–5)",
                  "10.544.740 (93,9%)", "2.688.868 (48,5%)"],
                 ["full", "pretrained", "toàn bộ", "11.227.812 (100%)", "5.543.716 (100%)"]],
                [45, 65, 190, 100, 95]), Spacer(1, 4)]
    s += bullets([
        "Tối ưu: AdamW (weight decay 0,01), lịch cosine theo epoch trong 15 epoch, loss Cross-Entropy. Learning rate: scratch 1e-3; "
        "head 1e-3 cho lớp phân loại; partial và full dùng 1e-4 cho backbone và 1e-3 cho lớp phân loại. Chưa tinh chỉnh các siêu tham số này.",
        "Các lớp BatchNorm của phần bị đóng băng được giữ ở chế độ eval để thống kê chạy không bị cập nhật; mã kiểm thử A1/smoke_test.py có bước kiểm tra trọng số và BatchNorm đóng băng không đổi sau một epoch.",
        "Vòng lặp huấn luyện tự viết (zero_grad → forward → loss → backward → step). Chọn epoch có validation accuracy cao nhất, rồi đo test accuracy một lần. "
        "Chỉ số: accuracy, macro-F1, số tham số trainable, thời gian mỗi epoch (gồm cả lượt validation), bộ nhớ GPU đỉnh. Mỗi cấu hình chạy một lần, seed 42."])

    # ---------------- 5
    rows = [["Kiến trúc", "Chế độ", "Trainable", "Best ep.", "Val acc", "Test acc", "Macro F1", "s/epoch", "Tổng (s)", "Bộ nhớ (MB)"]]
    for a in ("resnet18", "vit_tiny"):
        for m in MODES:
            r = R[a, m]
            rows.append([NAME[a].split(" ")[0].split("/")[0], m, f"{r[TRAIN]:,}".replace(",", "."), r[BEST], f4(r[VAL]), f4(r[TEST]),
                         f4(r[F1]), f1(r[SPE]), r[TOT], r[MEM]])
    s += [P("5. Kết quả", "h1"), table(rows, [64, 48, 62, 38, 42, 42, 46, 50, 40, 63]),
          P("Nguồn: results/A1/summary.md (một lần chạy cho mỗi cấu hình; test accuracy đo tại epoch có validation accuracy cao nhất).", "cap"),
          img("results/A1/accuracy_bar.png", 330, "Hình 2. Test accuracy theo chế độ huấn luyện và kiến trúc."),
          img("results/A1/cost_bar.png", 440, "Hình 3. Thời gian mỗi epoch và bộ nhớ GPU đỉnh."),
          img("results/A1/curves_resnet18.png", 440, "Hình 4. Đường cong loss/accuracy của ResNet-18 (nét đứt: train, nét liền: validation)."),
          img("results/A1/curves_vit_tiny.png", 440, "Hình 5. Đường cong loss/accuracy của ViT-Tiny.")]

    # ---------------- 6
    s += [P("6. So sánh và thảo luận", "h1"), P("6.1 Train từ đầu và pretrained", "h2")]
    s += [table([["Kiến trúc", "scratch (test acc)", "full (test acc)", "Chênh lệch"],
                 ["ResNet-18", f4(R['resnet18','scratch'][TEST]), f4(R['resnet18','full'][TEST]),
                  pts(R['resnet18','full'][TEST]-R['resnet18','scratch'][TEST]) + " điểm"],
                 ["ViT-Tiny", f4(R['vit_tiny','scratch'][TEST]), f4(R['vit_tiny','full'][TEST]),
                  pts(R['vit_tiny','full'][TEST]-R['vit_tiny','scratch'][TEST]) + " điểm"]],
                [100, 120, 120, 155]), Spacer(1, 4)]
    s += bullets([
        "Pretrain giúp cả hai mô hình, nhưng ViT-Tiny được lợi nhiều hơn rất nhiều (khoảng 48 điểm so với 19 điểm).",
        "<b>ViT cần pretrain trong thiết lập này.</b> Từ đầu, ViT-Tiny chỉ đạt 0,3911 trên 45.000 ảnh, thấp hơn ResNet-18 từ đầu 21,3 điểm; "
        "sau pretrain thì ViT-Tiny lại vượt ResNet-18. Điều này phù hợp với nhận định thường gặp rằng Transformer cần nhiều dữ liệu hơn khi không có tri thức đã học, "
        "nhưng thí nghiệm này chưa kiểm chứng riêng (chẳng hạn bằng cách thay đổi lượng dữ liệu).",
        "Cả hai mô hình scratch đều chưa hội tụ sau 15 epoch: best epoch là 15 và validation accuracy vẫn tăng (Hình 4–5). "
        "Do đó khoảng cách scratch và pretrained có thể bị phóng đại so với khi huấn luyện lâu hơn."])

    s += [P("6.2 Các kiểu đóng băng", "h2")]
    rows = [["Kiến trúc", "Chế độ", "Test acc", "So với full", "s/epoch (% của full)", "Bộ nhớ đỉnh (% của full)"]]
    for a in ("resnet18", "vit_tiny"):
        full = R[a, "full"]
        for m in ("head", "partial", "full"):
            r = R[a, m]
            rows.append([NAME[a].split(" ")[0], m, f4(r[TEST]), pts(r[TEST]-full[TEST]) + " điểm" if m != "full" else "–",
                         f"{f1(r[SPE])} ({100*r[SPE]/full[SPE]:.0f}%)", f"{r[MEM]} ({100*r[MEM]/full[MEM]:.0f}%)"])
    s += [table(rows, [65, 55, 60, 80, 115, 120]), Spacer(1, 4)]
    s += bullets([
        f"<b>Head-only</b> rẻ nhất: khoảng 24 giây/epoch (so với 67–77 giây của full) và ViT-Tiny chỉ cần 651 MB. "
        f"Với ViT-Tiny, head-only vẫn đạt {f4(R['vit_tiny','head'][TEST])} chỉ với 19.300 tham số trainable, cao hơn ResNet-18 scratch và gần ResNet-18 partial "
        f"({f4(R['resnet18','partial'][TEST])}). Với ResNet-18, head-only chỉ đạt {f4(R['resnet18','head'][TEST])}, hơn scratch có {ap(R['resnet18','head'][TEST]-R['resnet18','scratch'][TEST])} điểm: "
        "đặc trưng ImageNet đóng băng của ResNet-18 trên ảnh CIFAR phóng to dùng được kém hơn đặc trưng của ViT-Tiny.",
        "<b>Partial</b> lấy lại gần hết accuracy của full (kém 2,0 điểm ở ResNet-18 và 2,8 điểm ở ViT-Tiny) với chi phí thấp hơn: ResNet-18 chỉ mất khoảng một nửa thời gian "
        "(dù vẫn train 93,9% số tham số, vì phần đóng băng là các tầng đầu), ViT-Tiny mất khoảng 2/3 thời gian và dùng khoảng 54% bộ nhớ của full.",
        "<b>Full</b> cho accuracy cao nhất ở cả hai kiến trúc nhưng đắt nhất. Freeze có lợi khi ưu tiên chi phí hoặc khi dữ liệu/tài nguyên hạn chế "
        "(partial là điểm cân bằng tốt trong thí nghiệm này); khi cần accuracy tối đa và đủ tài nguyên thì fine-tune toàn bộ.",
        "Chênh lệch giữa partial và full chỉ 2–3 điểm, có thể gần với độ dao động giữa các lần chạy (mục 8), nên không nên xem là khác biệt chắc chắn."])

    s += [P("6.3 CNN và Transformer", "h2")]
    rows = [["Chế độ", "ResNet-18", "ViT-Tiny", "ViT − ResNet"]]
    for m in MODES:
        rows.append([m, f4(R['resnet18', m][TEST]), f4(R['vit_tiny', m][TEST]), pts(R['vit_tiny', m][TEST]-R['resnet18', m][TEST]) + " điểm"])
    s += [table(rows, [90, 110, 110, 185]), Spacer(1, 4)]
    s += bullets([
        "Ở cả ba chế độ pretrained, ViT-Tiny cao hơn ResNet-18 (từ khoảng 7 đến 14 điểm); ở scratch thì ngược lại, ResNet-18 cao hơn nhiều.",
        "<b>Không thể quy chênh lệch pretrained cho riêng kiến trúc.</b> ViT-Tiny được pretrain trên ImageNet-21k rồi ImageNet-1k, còn ResNet-18 chỉ trên ImageNet-1k, "
        "và hai mô hình khác công thức pretrain (augreg và a1) cũng như số tham số (5,5 triệu so với 11,2 triệu). Kết quả scratch là phép so sánh sạch hơn "
        "(cả hai không có pretrain) và cho thấy CNN dễ huấn luyện hơn trong điều kiện dữ liệu và thời gian của bài này.",
        "Về chi phí: ở scratch và full, ViT-Tiny nhanh hơn một chút (khoảng 67 so với 76 giây/epoch) dù chỉ có khoảng một nửa số tham số, "
        "nhưng dùng nhiều bộ nhớ đỉnh hơn (2309 MB so với 1778 MB); ở head-only ViT-Tiny dùng ít bộ nhớ hơn (651 MB so với 1135 MB)."])

    s += [P("6.4 Overfitting", "h2")] + bullets([
        "Các mô hình pretrained của ViT-Tiny khớp tập train gần như hoàn toàn (train accuracy 0,9993 với full và 0,9938 với partial ở epoch 15) trong khi validation accuracy là "
        "0,8728 và 0,8426, tức khoảng cách khoảng 12,7 và 15,1 điểm. Validation loss của partial thấp nhất ở epoch 2 (0,6197) rồi tăng lên 0,6860, trong khi validation accuracy vẫn nhích lên.",
        "ResNet-18 pretrained có khoảng cách nhỏ hơn (8,5 điểm với partial, 8,2 điểm với full). Các biện pháp chống overfitting (augmentation mạnh hơn, dropout, weight decay lớn hơn, dừng sớm) chưa được thử."])

    # ---------------- 7
    s += [P("7. Phân tích lỗi", "h1"),
          P("Dưới đây là các cặp lớp bị nhầm nhiều nhất của cấu hình tốt nhất mỗi kiến trúc (full), tính trên 100 ảnh test mỗi lớp (nguồn: results/A1/top_confusions.md).")]
    s += [table([["ResNet-18 full: thật → đoán", "Số ảnh", "ViT-Tiny full: thật → đoán", "Số ảnh"],
                 ["seal → otter", 24, "oak_tree → maple_tree", 21],
                 ["maple_tree → oak_tree", 17, "maple_tree → oak_tree", 15],
                 ["pine_tree → oak_tree", 16, "girl → baby", 11],
                 ["girl → woman", 16, "girl → boy", 11],
                 ["woman → girl", 16, "woman → man", 11],
                 ["dolphin → shark", 15, "boy → baby", 10],
                 ["bowl → plate", 15, "dolphin → whale", 10],
                 ["boy → man", 15, "plate → bowl", 10],
                 ["woman → man", 14, "willow_tree → maple_tree", 10],
                 ["boy → baby", 13, "boy → man", 9]],
                [150, 50, 150, 50 + 95]), Spacer(1, 4)]
    s += bullets([
        "Các lỗi tập trung vào cặp lớp giống nhau về hình dạng hoặc ngữ nghĩa: các loại cây (maple, oak, pine, willow), nhóm người (boy, girl, man, woman, baby), "
        "đồ dùng (bowl và plate), động vật sống nước (dolphin với shark hoặc whale, seal với otter). Với ảnh 32×32, chi tiết phân biệt (hình dạng lá, tuổi, giới tính) gần như bị mất.",
        "Hai mô hình có chung nhiều cặp nhầm (maple_tree và oak_tree, girl và boy/woman, boy và man, bowl và plate, dolphin với cá voi hoặc cá mập), "
        "nên khó khăn đến từ dữ liệu hơn là từ riêng một kiến trúc. Nhầm lẫn lớn nhất của ResNet-18 là seal → otter (24 trên 100 ảnh seal)."])

    # ---------------- 8
    s += [P("8. Hạn chế và tính minh bạch", "h1")] + bullets([
        "<b>Mỗi cấu hình chỉ chạy một lần.</b> Lần chạy đầu của vit_tiny scratch bị gián đoạn (Colab ngắt) ở epoch 10 và được chạy lại từ đầu với cùng seed 42. "
        "Ở epoch 10, validation accuracy của lần bị gián đoạn là 0,3054 còn lần chạy lại là 0,3342 (chênh 2,9 điểm). "
        "Điều này cho thấy kết quả không hoàn toàn lặp lại giữa các lần chạy dù cùng seed (có thể do tính không tất định của GPU), nên các chênh lệch dưới khoảng 3 điểm cần được xem thận trọng. "
        "Bảng kết quả dùng lần chạy hoàn chỉnh.",
        "<b>Yếu tố gây nhiễu giữa CNN và Transformer</b>: nguồn pretrain khác nhau (ImageNet-1k so với ImageNet-21k rồi 1k), số tham số khác nhau, công thức pretrain khác nhau. "
        "Báo cáo không thực hiện ablation để tách các yếu tố này.",
        "Huấn luyện 15 epoch cho mọi cấu hình; các mô hình scratch chưa hội tụ nên accuracy của chúng là cận dưới. Chưa tinh chỉnh learning rate hay siêu tham số khác.",
        "Ảnh 32×32 được phóng lên 224×224 để dùng mô hình pretrained; kết quả có thể khác nếu dùng ảnh độ phân giải thật.",
        "Chỉ so sánh một CNN và một Transformer cỡ nhỏ; kết luận không nên khái quát sang các kiến trúc hoặc kích thước khác.",
        "Chưa thử các biện pháp chống overfitting và chưa đo thời gian suy luận."])

    # ---------------- 9
    s += [P("9. Tái lập", "h1"),
          P("Từ thư mục gốc của repo (trên Colab, cài thêm timm): " + code("pip install timm") + ", " + code("python A1/eda.py") + ", " + code("python A1/run_a1.py") +
            ". Mặc định chạy 8 cấu hình, 15 epoch, ảnh 224×224. Mỗi cấu hình lưu vào results/A1/&lt;kiến trúc&gt;_&lt;chế độ&gt;/ "
            "(history.json, confusion.npy, result.json); chạy lại cùng lệnh sẽ bỏ qua các cấu hình đã xong. "
            "Cấu hình, log và bảng nằm ở results/A1/ (config.json, log.txt, summary.md, top_confusions.md)."),
          P("10. Ghi nhận việc sử dụng công cụ AI", "h1"),
          P("Công cụ AI Claude (Anthropic) được dùng để hỗ trợ phân tích đề, soạn khung mã nguồn A1 (thư mục A1) và bản nháp báo cáo, trang GitHub Pages từ số liệu thật do sinh viên "
            "cung cấp. Sinh viên chạy thí nghiệm trên Google Colab, cung cấp log và hình kết quả, và chịu trách nhiệm kiểm tra, giải thích mã nguồn và số liệu đã nộp.")]
    return s


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("VN", 8)
    canvas.drawCentredString(A4[0] / 2, 25, f"CO5085 – A1 – Phan Ngọc Trâm – 2570790 – trang {doc.page}")
    canvas.restoreState()


def build():
    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=50, rightMargin=50, topMargin=48, bottomMargin=48,
                            title="CO5085 - Báo cáo A1", author="Phan Ngọc Trâm")
    doc.build(sections(), onFirstPage=footer, onLaterPages=footer)
    print("Đã tạo", os.path.abspath(OUT))


if __name__ == "__main__":
    build()
