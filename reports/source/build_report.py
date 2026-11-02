"""Dựng reports/exercises.pdf (báo cáo chung E1-E3).

Chạy từ thư mục gốc của repo:   python reports/source/build_report.py
Cần:  pip install reportlab pillow
Số liệu bên dưới được chép từ results/E1/summary.md và results/E1/log.txt (lần chạy 15 epoch).
Khi làm E2/E3: thêm hàm section_e2(), section_e3() rồi gọi trong build().
"""
import os
import sys

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..")
OUT = os.path.join(ROOT, "reports", "exercises.pdf")

# ---- Font hỗ trợ tiếng Việt (DejaVu có sẵn trên Linux/Colab; trên Windows đổi đường dẫn) ----
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
    "b": ParagraphStyle("b", fontName="VN", fontSize=10, leading=14.5, leftIndent=14,
                        bulletIndent=2, spaceAfter=3),
    "c": ParagraphStyle("c", fontName="VN", fontSize=8.8, leading=12),
    "cb": ParagraphStyle("cb", fontName="VNB", fontSize=8.8, leading=12),
    "cap": ParagraphStyle("cap", fontName="VN", fontSize=8.5, leading=11.5,
                          textColor=colors.HexColor("#444444"), spaceAfter=8, alignment=1),
}
P = lambda t, s="p": Paragraph(t, S[s])
def bullets(items):
    return [Paragraph(t, S["b"], bulletText="•") for t in items]

def img(rel, width, caption):
    path = os.path.join(ROOT, rel)
    w, h = PILImage.open(path).size
    return KeepTogether([Image(path, width=width, height=width * h / w), P(caption, "cap")])

def table(rows, widths, header=True, align_right_from=None):
    data = [[Paragraph(str(c), S["cb"] if (header and i == 0) else S["c"]) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")))
    t.setStyle(TableStyle(st))
    return t

# ------------------------------------------------------------------ nội dung
def front():
    return [
        P("Báo cáo bài tập E1–E3", "title"),
        P("Học sâu và ứng dụng trong thị giác máy tính (CO5085) – Học kỳ 261 (2026–2027)"),
        P("Sinh viên: Phan Ngọc Trâm – MSSV 2570790 (làm cá nhân)"),
        P("Repository: https://github.com/tramphanngoc174/co5085-exercises"),
        Spacer(1, 6),
        table([["Bài", "Hạn nộp", "Trạng thái trong bản báo cáo này"],
               ["E1 – Softmax / MLP / CNN", "02/11/2026", "Đã có kết quả (mục 1)"],
               ["E2 – MSA + tokenization", "02/11/2026", "Đã có kết quả (mục 2)"],
               ["E3 – LSTM / GRU", "09/11/2026", "Chưa có trong bản này (mục 3)"]],
              [170, 80, 245]),
        P("Báo cáo này là một file PDF chung, được cập nhật dần theo từng hạn nộp.", "p"),
    ]

def section_data():
    return [
        P("0. Dữ liệu và thiết lập chung", "h1"),
        P("<b>Tập dữ liệu:</b> Fashion-MNIST (ảnh xám 28×28, 10 lớp quần áo/giày dép), dùng chung cho E1–E3. "
          "Lý do chọn: tập nhỏ, chạy nhanh; khó hơn MNIST nên các mô hình phân biệt được với nhau, "
          "và các lớp dễ nhầm (áo) cho phép phân tích lỗi có ý nghĩa."),
        P("<b>Chia dữ liệu:</b> 60.000 ảnh train gốc được chia ngẫu nhiên với seed 42 thành "
          "<b>54.000 train</b> và <b>6.000 validation</b>; <b>10.000 ảnh test</b> gốc giữ nguyên. "
          "Validation dùng để chọn epoch; test chỉ dùng để báo cáo kết quả cuối. "
          "Chỉ số chia được lưu trong <font name='VN'>splits/fashion_mnist_val6000_seed42.json</font> "
          "để E2, E3 dùng đúng cùng split. Tỷ lệ chia là lựa chọn của sinh viên (đề không quy định)."),
        P("<b>Tiền xử lý:</b> chuẩn hóa theo mean 0,2860 và std 0,3530; không dùng data augmentation."),
        P("<b>Môi trường:</b> PyTorch, chạy trên Google Colab (GPU CUDA). "
          "Seed, cấu hình và log của từng lần chạy được lưu trong <font name='VN'>results/E1/</font> "
          "(<font name='VN'>config.json</font>, <font name='VN'>log.txt</font>)."),
    ]

def section_e1():
    out = [P("1. E1 – Phân loại ảnh: Softmax, MLP, CNN", "h1")]

    out += [P("1.1 Kiến trúc", "h2"),
        table([["Mô hình", "Cấu trúc", "Số tham số"],
               ["Softmax", "Flatten (784) → Linear (784→10); softmax nằm trong hàm loss", "7.850"],
               ["MLP", "Flatten (784) → Linear 512 + ReLU → Linear 256 + ReLU → Linear 10", "535.818"],
               ["CNN", "2 khối [Conv 3×3 + BatchNorm + ReLU + MaxPool 2×2] với 32 và 64 kênh; "
                       "Flatten (64·7·7 = 3136) → Linear 128 + ReLU + Dropout 0,3 → Linear 10", "421.834"]],
              [60, 360, 75]),
        Spacer(1, 4),
        P("Cả ba mô hình trả về logit; softmax được tính bên trong <font name='VN'>CrossEntropyLoss</font>. "
          "Softmax và MLP nhận đầu vào là vector flatten (đúng yêu cầu của đề); chỉ CNN khai thác cấu trúc không gian của ảnh.")]

    out += [P("1.2 Thiết lập huấn luyện", "h2")] + bullets([
        "Vòng lặp huấn luyện tự viết: <font name='VN'>zero_grad() → forward → loss → backward() → optimizer.step()</font> "
        "(file <font name='VN'>common/train_utils.py</font>).",
        "Optimizer Adam, learning rate 0,001, batch size 128, 15 epoch, loss Cross-Entropy; cả ba mô hình dùng cùng cấu hình.",
        "Mỗi epoch đánh giá trên validation; trọng số của epoch có <b>validation accuracy cao nhất</b> được giữ lại "
        "và dùng để đo test accuracy một lần duy nhất.",
        "Seed 42 được đặt lại trước khi huấn luyện từng mô hình. Mỗi mô hình chỉ được huấn luyện <b>một lần</b>."])

    out += [P("1.3 Kết quả so sánh", "h2"),
        table([["Mô hình", "Tham số", "Best epoch", "Train acc (tại best epoch)", "Val acc", "Test acc", "Giây/epoch"],
               ["Softmax", "7.850", "15", "0,8660", "0,8498", "0,8442", "12,4"],
               ["MLP", "535.818", "12", "0,9373", "0,8983", "0,8896", "12,8"],
               ["CNN", "421.834", "13", "0,9572", "0,9292", "0,9244", "14,6"]],
              [62, 65, 55, 115, 60, 60, 78]),
        P("Nguồn: <font name='VN'>results/E1/summary.md</font> và <font name='VN'>results/E1/log.txt</font> "
          "(một lần chạy, seed 42, 15 epoch).", "cap")]
    out += bullets([
        "CNN đạt test accuracy cao nhất (0,9244), hơn MLP 3,5 điểm phần trăm (0,8896) và hơn Softmax 8,0 điểm (0,8442). "
        "MLP hơn Softmax 4,5 điểm.",
        "CNN có <b>ít tham số hơn MLP</b> (421.834 so với 535.818, ít hơn khoảng 21%) nhưng chính xác hơn; "
        "trong thí nghiệm này, số tham số không quyết định chất lượng mà cách mô hình khai thác cấu trúc ảnh quan trọng hơn.",
        "Test accuracy thấp hơn validation accuracy từ 0,5 đến 0,9 điểm ở cả ba mô hình, điều này là bình thường vì epoch được chọn theo validation.",
        "Thời gian mỗi epoch chênh lệch nhỏ (12,4–14,6 giây); do đó thời gian không phải yếu tố phân biệt trong thí nghiệm này."])

    out += [P("1.4 Đường cong học", "h2"),
        img("results/E1/curves.png", 495,
            "Hình 1. Loss và accuracy theo epoch trên tập train (nét đứt) và validation (nét liền) của ba mô hình.")]
    out += bullets([
        "<b>Softmax</b>: train accuracy chỉ đạt khoảng 0,866 và train loss giảm rất chậm sau vài epoch đầu; "
        "validation accuracy dao động trong khoảng 0,83–0,85. Mô hình tuyến tính bị giới hạn bởi capacity thấp, chưa phải do thiếu epoch.",
        "<b>MLP</b>: train loss giảm liên tục (đến 0,1419) nhưng validation loss thấp nhất ở epoch 5 (0,3204) rồi tăng lên 0,3820 ở epoch 15, "
        "trong khi train accuracy (0,9452) vượt xa validation accuracy (0,8940). Đây là dấu hiệu overfitting.",
        "<b>CNN</b>: validation loss thấp nhất khoảng 0,2249 (epoch 10 và 14) và tăng nhẹ lên 0,2412 ở epoch 15, "
        "train accuracy 0,9636. Overfitting có xuất hiện nhưng nhẹ hơn MLP (khoảng cách train–validation tại best epoch là 2,8 điểm so với 3,9 điểm của MLP).",
        "Validation accuracy của CNN dao động giữa các epoch (từ 0,9143 đến 0,9292 trong epoch 11–15), "
        "nên việc chọn đúng một epoch tốt nhất có yếu tố ngẫu nhiên."])

    out += [P("1.5 Phân tích lỗi", "h2"),
        P("Số mẫu đúng trên 1.000 ảnh test của mỗi lớp (đường chéo của confusion matrix), nên cũng bằng recall tính theo phần nghìn:")]
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    sm = [848, 964, 727, 834, 804, 938, 527, 911, 936, 953]
    ml = [905, 966, 806, 921, 813, 941, 647, 981, 973, 943]
    cn = [901, 991, 881, 924, 893, 977, 749, 978, 985, 965]
    rows = [["Lớp", "Softmax", "MLP", "CNN"]] + [[c, a, b, d] for c, a, b, d in zip(classes, sm, ml, cn)]
    rows.append(["Tổng (đúng/10.000)", sum(sm), sum(ml), sum(cn)])
    out += [table(rows, [140, 80, 80, 80]), Spacer(1, 6)]
    out += [img("results/E1/cnn/confusion_matrix.png", 285, "Hình 2. Confusion matrix của CNN trên tập test."),
            img("results/E1/mlp/confusion_matrix.png", 285, "Hình 3. Confusion matrix của MLP trên tập test."),
            img("results/E1/softmax/confusion_matrix.png", 285, "Hình 4. Confusion matrix của Softmax trên tập test.")]
    out += bullets([
        "<b>Lớp khó nhất ở cả ba mô hình là Shirt</b>: đúng 527 (Softmax), 647 (MLP), 749 (CNN) trên 1.000 ảnh. "
        "Shirt bị nhầm nhiều nhất sang T-shirt/top (170, 183 và 106 ảnh), rồi sang Pullover và Coat.",
        "Các lỗi tập trung vào nhóm áo: T-shirt/top, Shirt, Pullover, Coat, Dress. Ví dụ Pullover bị nhầm sang Coat 148 (Softmax), 84 (MLP), 47 (CNN) ảnh.",
        "Nhóm giày dép nhầm lẫn lẫn nhau: Ankle boot bị đoán là Sneaker 27 (Softmax), 49 (MLP), 30 (CNN) ảnh; Sneaker bị đoán là Ankle boot 52, 12, 19 ảnh.",
        "Trouser, Bag, Sneaker, Sandal và Ankle boot được nhận dạng tốt (từ 911 đến 991 trên 1.000 ảnh) ở mọi mô hình, "
        "nên phần lớn sự khác biệt giữa các mô hình nằm ở nhóm áo.",
        "So với Softmax, CNN tăng recall của Shirt thêm 222 ảnh trên 1.000 (từ 527 lên 749) và so với MLP thêm 102 ảnh (từ 647 lên 749)."])

    out += [img("results/E1/cnn/errors.png", 495, "Hình 5. Mười hai mẫu test đầu tiên bị CNN dự đoán sai (nhãn thật và nhãn dự đoán ghi trên mỗi ảnh)."),
            img("results/E1/mlp/errors.png", 495, "Hình 6. Mười hai mẫu test đầu tiên bị MLP dự đoán sai."),
            img("results/E1/softmax/errors.png", 495, "Hình 7. Mười hai mẫu test đầu tiên bị Softmax dự đoán sai.")]
    out += [P("Lưu ý: các hình 5–7 là 12 mẫu sai đầu tiên theo thứ tự trong tập test, không phải lấy ngẫu nhiên, "
              "nên một số ảnh xuất hiện ở nhiều mô hình. Hai dòng tiêu đề ở hàng dưới của các hình bị chồng lên ảnh phía trên (lỗi hiển thị nhỏ, không ảnh hưởng nhãn).", "cap")]
    out += bullets([
        "Một số mẫu khó phân biệt kể cả khi nhìn bằng mắt, ví dụ áo sơ mi ngắn tay có cúc (nhãn Shirt) bị cả ba mô hình đoán là T-shirt/top, "
        "hoặc đôi giày có quai, gót cao (nhãn Ankle boot) bị CNN và MLP đoán là Sandal.",
        "Ảnh 28×28 xám bỏ mất nhiều chi tiết (cổ áo, cúc, độ dài tay), nên ranh giới giữa các lớp áo vốn mờ."])

    out += [P("1.6 Nhận xét và hạn chế", "h2")] + bullets([
        "Thứ hạng CNN &gt; MLP &gt; Softmax nhất quán trên cả validation và test. Softmax bị giới hạn bởi capacity (underfitting); "
        "MLP có đủ capacity nhưng overfit sau khoảng epoch 5; CNN đạt kết quả tốt nhất với ít tham số hơn MLP.",
        "Giải thích thường gặp cho việc CNN tốt hơn là convolution dùng chung trọng số và bắt đặc trưng cục bộ (cạnh, hình dạng) "
        "bất kể vị trí trong ảnh, trong khi MLP và Softmax xem mỗi pixel là một đầu vào độc lập. "
        "Báo cáo này không có thí nghiệm riêng (ablation) để kiểm chứng giải thích đó.",
        "Hạn chế: mỗi mô hình chỉ chạy một lần với một seed nên chưa đo được độ dao động giữa các lần chạy; "
        "chưa tinh chỉnh siêu tham số, chưa dùng data augmentation hay dừng sớm; "
        "CNN có thêm BatchNorm và Dropout còn MLP thì không, nên phần chênh lệch giữa hai mô hình không chỉ do convolution."])
    return out

def section_e2():
    out = [P("2. E2 – Transformer: MSA và tokenization", "h1")]

    out += [P("2.1 Thiết lập chung", "h2")] + bullets([
        "Bộ phân loại: Tokenizer → 2 khối encoder [LayerNorm → MSA → cộng tắt, LayerNorm → FFN → cộng tắt] → LayerNorm → Linear 10 trên token CLS. "
        "Kích thước ẩn 64, 4 head, FFN có hệ số mở rộng 2 (GELU). Mọi tokenizer đều thêm token CLS và positional embedding học được.",
        "Dữ liệu, split, seed 42 và vòng lặp huấn luyện tự viết giống E1; Adam, learning rate 0,001, batch 128, loss Cross-Entropy, "
        "chọn epoch theo validation accuracy. Khác E1: <b>10 epoch</b> (thay vì 15) để tiết kiệm thời gian chạy.",
        "Có 3 cách tokenize × 2 bản MSA = 6 lần huấn luyện, mỗi lần một seed. Các lần chạy dùng chung cấu hình (xem <font name='VN'>results/E2/config.json</font>)."])

    out += [P("2.2 MSA tự viết và MSA PyTorch", "h2"),
        P("<b>MSA tự viết</b> (<font name='VN'>E2/msa.py</font>, lớp <font name='VN'>CustomMSA</font>) chỉ dùng Linear, nhân ma trận, softmax và ghép head: "
          "ba lớp Linear tạo Q, K, V; tách thành 4 head (mỗi head 16 chiều); điểm chú ý = Q·Kᵀ chia căn bậc hai của 16; softmax theo chiều token; "
          "nhân với V; ghép các head lại và qua một lớp Linear đầu ra. Bản này không dùng <font name='VN'>nn.MultiheadAttention</font> hay "
          "<font name='VN'>nn.TransformerEncoder</font>. <b>MSA PyTorch</b> là <font name='VN'>nn.MultiheadAttention</font> (<font name='VN'>TorchMSA</font>) với cùng giao diện."),
        P("<b>Kiểm tra tính đúng:</b> chép trọng số của bản tự viết sang bản PyTorch rồi so đầu ra trên một batch ngẫu nhiên (3 mẫu × 10 token × 64 chiều). "
          "Sai khác tuyệt đối lớn nhất là <b>1,19×10⁻⁷</b> (ngưỡng kiểm tra 10⁻⁵). Cả hai bản có cùng <b>16.640</b> tham số mỗi khối MSA. "
          "(Kiểm tra này nằm trong <font name='VN'>E2/smoke_test.py</font>.) Khi huấn luyện, hai bản khởi tạo trọng số theo cách khác nhau, "
          "nên kết quả huấn luyện của chúng không giống hệt nhau dù cùng seed.")]

    out += [P("2.3 Ba cách tokenize", "h2"),
        table([["Tokenizer", "Cách lấy token", "Số token (kèm CLS)"],
               ["patch", "Chia ảnh thành lưới ô 4×4 không chồng lấn (49 ô); mỗi ô 16 giá trị qua Linear 16→64", "50"],
               ["row", "Mỗi hàng của ảnh (28 giá trị) là một token (28 token); qua Linear 28→64", "29"],
               ["cnnstem", "CNN 2 khối [Conv 3×3 + BatchNorm + ReLU + MaxPool] (32 và 64 kênh) cho feature map 7×7; "
                           "mỗi vị trí là một token 64 chiều (49 token), qua Linear 64→64", "50"]],
              [65, 355, 75])]

    out += [P("2.4 Kết quả", "h2"),
        table([["Tokenizer", "MSA", "Tham số", "Best epoch", "Val acc", "Test acc", "Giây/epoch", "Tổng train (s)"],
               ["patch", "tự viết", "72.074", "10", "0,8800", "0,8753", "15,5", "155"],
               ["patch", "PyTorch", "72.074", "9", "0,8788", "0,8736", "15,7", "157"],
               ["row", "tự viết", "71.498", "9", "0,8888", "0,8816", "14,8", "148"],
               ["row", "PyTorch", "71.498", "6", "0,8878", "0,8784", "14,8", "148"],
               ["cnnstem", "tự viết", "94.154", "10", "0,9138", "0,9077", "17,7", "177"],
               ["cnnstem", "PyTorch", "94.154", "8", "0,9060", "0,8986", "18,1", "181"]],
              [66, 56, 56, 48, 50, 50, 72, 97]),
        P("Nguồn: <font name='VN'>results/E2/summary.md</font> và <font name='VN'>log.txt</font> (10 epoch, seed 42, GPU CUDA).", "cap")]

    out += [P("2.5 So sánh (i): MSA tự viết và MSA PyTorch (cùng cách tokenize)", "h2"),
        table([["Tokenizer", "Test acc tự viết", "Test acc PyTorch", "Chênh lệch (tự viết − PyTorch)"],
               ["patch", "0,8753", "0,8736", "+0,17 điểm"],
               ["row", "0,8816", "0,8784", "+0,32 điểm"],
               ["cnnstem", "0,9077", "0,8986", "+0,91 điểm"]],
              [80, 110, 110, 195]), Spacer(1, 6),
        img("results/E2/curves_patch.png", 440, "Hình 8. Tokenizer patch: loss và accuracy theo epoch của MSA tự viết (custom) và MSA PyTorch (torch)."),
        img("results/E2/curves_row.png", 440, "Hình 9. Tokenizer row: custom và torch."),
        img("results/E2/curves_cnnstem.png", 440, "Hình 10. Tokenizer cnnstem: custom và torch.")]
    out += bullets([
        "Hai bản cho kết quả rất gần nhau: chênh lệch test accuracy là 0,17 điểm (patch), 0,32 điểm (row) và 0,91 điểm (cnnstem), "
        "và đường train loss/accuracy của hai bản gần như chồng lên nhau ở cả ba tokenizer (train accuracy epoch 10: 0,8968 và 0,8957 với patch; "
        "0,9097 và 0,9097 với row; 0,9261 và 0,9248 với cnnstem). Điều này phù hợp với việc MSA tự viết tính cùng hàm với bản PyTorch.",
        "Bản tự viết có test accuracy nhỉnh hơn ở cả ba tokenizer, nhưng chưa thể kết luận bản nào tốt hơn: mỗi cấu hình chỉ chạy một lần, "
        "và validation accuracy dao động giữa các epoch (ví dụ cnnstem–PyTorch giảm từ 0,9028 ở epoch 9 xuống 0,8883 ở epoch 10, tức 1,45 điểm, "
        "lớn hơn khoảng chênh 0,91 điểm giữa hai bản). Epoch tốt nhất cũng khác nhau giữa hai bản (ví dụ 6 so với 9 với row).",
        "Thời gian huấn luyện gần như bằng nhau (155 và 157 s; 148 và 148 s; 177 và 181 s), nên trong thiết lập này bản PyTorch không nhanh hơn rõ rệt. "
        "Chuỗi chỉ có 29–50 token và mô hình nhỏ, nên thời gian có thể bị chi phối bởi các phần khác (nạp dữ liệu, các lớp còn lại); báo cáo không có phép đo riêng để khẳng định."])

    out += [P("2.6 So sánh (ii): các cách tokenize (dùng MSA tự viết)", "h2"),
        img("results/E2/curves_tokenizers.png", 440, "Hình 11. So sánh ba tokenizer với MSA tự viết (nét đứt: train, nét liền: validation).")]
    out += bullets([
        "<b>cnnstem cho kết quả tốt nhất</b>: test accuracy 0,9077, cao hơn row (0,8816) 2,6 điểm và cao hơn patch (0,8753) 3,2 điểm; "
        "đường validation của cnnstem nằm trên hai tokenizer còn lại ở hầu hết các epoch. Đổi lại cnnstem có nhiều tham số hơn (94.154 so với khoảng 72.000, nhiều hơn khoảng 31%) "
        "và chậm hơn (17,7 s/epoch so với 14,8 s của row và 15,5 s của patch).",
        "<b>row</b> (0,8816) nhỉnh hơn <b>patch</b> (0,8753) 0,6 điểm, nhưng chênh lệch nhỏ và validation accuracy của patch dao động nhiều "
        "(0,8720 ở epoch 6, rồi 0,8610 và 0,8582 ở epoch 7–8), nên chưa nên kết luận chắc về thứ tự của hai tokenizer này. "
        "Row dùng ít token nhất (29) và nhanh nhất (14,8 s/epoch).",
        "patch học chậm hơn ở những epoch đầu (train accuracy epoch 1 là 0,6948 so với 0,7883 của row và 0,7769 của cnnstem).",
        "Ở cả ba cấu hình, train accuracy vẫn còn tăng ở epoch 10 và khoảng cách train–validation nhỏ (1,2 đến 2,1 điểm), "
        "nên mô hình có thể chưa hội tụ hết sau 10 epoch; báo cáo chưa thử huấn luyện lâu hơn.",
        "Tham khảo (không cùng điều kiện): CNN của E1 đạt test accuracy 0,9244 với 15 epoch và 421.834 tham số; các Transformer nhỏ ở E2 "
        "(10 epoch, 71.000–94.000 tham số) đạt 0,8736–0,9077."])

    out += [P("2.7 Nhận xét và hạn chế", "h2")] + bullets([
        "MSA tự viết đúng về số học (sai khác 1,19×10⁻⁷ so với PyTorch khi cùng trọng số) và cho kết quả huấn luyện gần bản PyTorch ở cả ba cách tokenize.",
        "Trong các lần chạy này, cách lấy token ảnh hưởng nhiều hơn việc chọn bản MSA: chênh lệch giữa các tokenizer (tới 3,2 điểm) lớn hơn chênh lệch giữa hai bản MSA (tới 0,9 điểm).",
        "Một giải thích có thể cho ưu thế của cnnstem là tầng convolution đã rút đặc trưng cục bộ trước khi attention làm việc, "
        "trong khi patch và row đưa giá trị pixel thô qua một phép Linear. Báo cáo không có thí nghiệm riêng (ablation) để kiểm chứng; "
        "cnnstem cũng có nhiều tham số hơn nên phần cải thiện không chỉ do cách tokenize.",
        "Hạn chế: mỗi cấu hình chỉ chạy một lần với một seed nên chưa đo được độ dao động giữa các lần chạy; chỉ 10 epoch; chưa tinh chỉnh siêu tham số "
        "(kích thước ẩn, số tầng, số head, patch size), chưa dùng dropout, data augmentation hay lịch learning rate; "
        "chỉ thử patch 4×4 không chồng lấn và chưa thử các biến thể khác (patch chồng lấn, patch size khác, token theo kênh)."])
    return out

def section_pending(n, title):
    return [P(f"{n}. {title}", "h1"), P("Chưa thực hiện trong bản báo cáo này. Nội dung sẽ được bổ sung vào chính file này khi đến hạn.")]

def section_ai():
    return [
        P("4. Ghi nhận việc sử dụng công cụ AI", "h1"),
        P("Công cụ AI Claude (Anthropic) được dùng để hỗ trợ: phân tích đề bài và lập kế hoạch nộp bài; soạn khung mã nguồn E1 và E2 "
          "(các file trong <font name='VN'>common/</font>, <font name='VN'>E1/</font> và <font name='VN'>E2/</font>), tệp khung README, trang GitHub Pages; "
          "và soạn bản nháp báo cáo từ số liệu thật do sinh viên cung cấp. "
          "Sinh viên chạy thí nghiệm trên Google Colab, cung cấp log và hình kết quả, và chịu trách nhiệm kiểm tra và giải thích mã nguồn cũng như số liệu đã nộp."),
    ]

def section_repro():
    return [
        P("5. Tái lập kết quả E1 và E2", "h1"),
        P("Từ thư mục gốc của repo: <font name='VN'>python E1/run_e1.py --dataset fashion_mnist --epochs 15</font>. "
          "Kết quả nằm trong <font name='VN'>results/E1/</font>. E2: <font name='VN'>python E2/run_e2.py --dataset fashion_mnist --epochs 10</font>, kết quả nằm trong <font name='VN'>results/E2/</font> (log, cấu hình, bảng, biểu đồ, checkpoint). "
          "Hướng dẫn môi trường và lệnh chạy chi tiết: xem README của repository."),
    ]

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("VN", 8)
    canvas.drawCentredString(A4[0] / 2, 25, f"CO5085 – Phan Ngọc Trâm – 2570790 – trang {doc.page}")
    canvas.restoreState()

def build():
    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=50, rightMargin=50, topMargin=48, bottomMargin=48,
                            title="CO5085 - Báo cáo bài tập E1-E3", author="Phan Ngọc Trâm")
    story = front() + section_data() + section_e1() + \
        section_e2() + section_pending(3, "E3 – LSTM / GRU trên chuỗi ảnh") + \
        section_ai() + section_repro()
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("Đã tạo", os.path.abspath(OUT))

if __name__ == "__main__":
    build()
