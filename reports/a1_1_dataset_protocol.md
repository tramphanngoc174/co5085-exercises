# A1.1 – Dataset và protocol

**CO5085 – Học sâu và ứng dụng trong thị giác máy tính – HK 261 (2026–2027)**
**Sinh viên:** Phan Ngọc Trâm – MSSV 2570790 (làm cá nhân)

> **Lưu ý về nội dung:** tài liệu này mô tả protocol đã được thực hiện trong mục A1.2 và không chứa số liệu kết quả (kết quả nằm trong `reports/a1_report.pdf`).

## 1. Chủ đề

So sánh CNN và Transformer trên một tập ảnh lớn hơn các tập của E1, với các chế độ huấn luyện: từ đầu (from-scratch) và pretrained; đóng băng toàn bộ backbone, đóng băng một phần, và fine-tune toàn bộ.

## 2. Dataset

| Mục | Nội dung |
|---|---|
| Tập dữ liệu | **CIFAR-100** (Krizhevsky, 2009) |
| Quy mô | 60.000 ảnh màu 32×32, 100 lớp (600 ảnh mỗi lớp); 50.000 ảnh train gốc và 10.000 ảnh test gốc |
| Lý do chọn | Nằm trong danh sách đề cho phép; không phải MNIST / Fashion-MNIST / CIFAR-10 (đã dùng cho E1); 100 lớp (đề ưu tiên ≥ 10 lớp); đủ lớn (50.000 ảnh train) để so sánh CNN và ViT, nhưng đủ nhỏ để chạy 8 cấu hình trên GPU Colab; có nhiều cặp lớp giống nhau (các loại cây, nhóm người, đồ dùng) nên phân tích lỗi có ý nghĩa |
| License | Tập dữ liệu công khai do Krizhevsky công bố, trích dẫn bài báo “Learning Multiple Layers of Features from Tiny Images” (2009). Thẻ dataset chính thức trên Hugging Face ghi license là “unknown”; một số bản sao của bên thứ ba ghi MIT hoặc CC BY 4.0 nhưng đó không phải nguồn gốc. Chỉ dùng cho học tập, phi thương mại. Trang chính thức: https://www.cs.toronto.edu/~kriz/cifar.html |
| Cách lấy | `torchvision.datasets.CIFAR100` (tải tự động) |

### 2.1 Chia dữ liệu (cố định, seed 42)

| Tập | Số ảnh | Mỗi lớp | Nguồn |
|---|---|---|---|
| Train | 45.000 | 450 | tập train gốc |
| Validation | 5.000 | 50 | tập train gốc, không trùng với train; dùng để chọn epoch |
| Test | 10.000 | 100 | tập test gốc; chỉ dùng để báo cáo kết quả cuối |

Dùng toàn bộ 50.000 ảnh train gốc, không lấy subset. Chỉ số train/validation được lưu trong `splits/cifar100_tr450_val50_seed42.json` để tái lập.

### 2.2 DataLoader và augmentation

- Ảnh giữ dạng tensor uint8 (N×3×32×32) trong bộ nhớ, đưa vào `DataLoader` (batch 128; tập train xáo trộn bằng generator có seed). Mã: `A1/data.py`.
- Augmentation (chỉ tập train, thực hiện trên GPU): random crop 32×32 sau khi đệm 4 pixel giá trị 0 mỗi bên; lật ngang với xác suất 0,5. Tập validation và test không augmentation.
- Sau augmentation: phóng to bilinear từ 32×32 lên **224×224** (kích thước pretrain của hai mô hình), rồi chuẩn hóa mean/std theo cấu hình pretrained của timm cho từng mô hình. Mã: `A1/engine.py`.

## 3. Kiến trúc

| Họ | Mô hình | Số tham số (100 lớp) | Trọng số pretrained (timm) | Pretrain trên |
|---|---|---|---|---|
| CNN | **ResNet-18** | 11.227.812 | `resnet18.a1_in1k` | ImageNet-1k |
| Transformer | **ViT-Tiny/16** (`vit_tiny_patch16_224`) | 5.543.716 | `vit_tiny_patch16_224.augreg_in21k_ft_in1k` | ImageNet-21k, rồi fine-tune ImageNet-1k |

Hai mô hình khác nhau về nguồn pretrain và số tham số; điều này được ghi nhận như một hạn chế khi so sánh (xem báo cáo A1, mục 6.3 và 8).

## 4. Kế hoạch thí nghiệm: from-scratch, pretrained và đóng băng

Mỗi kiến trúc chạy 4 chế độ (tổng cộng **8 cấu hình**) trên cùng một split:

| Chế độ | Khởi tạo | Phần được train | ResNet-18 trainable | ViT-Tiny trainable |
|---|---|---|---|---|
| `scratch` | ngẫu nhiên | toàn bộ | 100% | 100% |
| `head` | pretrained | chỉ lớp phân loại; **đóng băng toàn bộ backbone** | 51.300 (0,5%) | 19.300 (0,3%) |
| `partial` | pretrained | ResNet: layer3, layer4, fc (đóng băng conv1, bn1, layer1, layer2). ViT: block 6–11, norm, head (đóng băng patch_embed, cls_token, pos_embed, block 0–5) | 10.544.740 (93,9%) | 2.688.868 (48,5%) |
| `full` | pretrained | toàn bộ (fine-tune toàn bộ) | 100% | 100% |

Các lớp BatchNorm của phần bị đóng băng được giữ ở chế độ eval.

### 4.1 Thiết lập huấn luyện (giống nhau cho mọi cấu hình)

- Vòng lặp huấn luyện tự viết (zero_grad → forward → loss → backward → step); loss Cross-Entropy; mixed precision fp16.
- AdamW (weight decay 0,01), lịch cosine theo epoch, **15 epoch**, batch 128, seed 42.
- Learning rate: scratch 1e-3; head 1e-3 (lớp phân loại); partial và full dùng 1e-4 cho backbone và 1e-3 cho lớp phân loại.
- Chọn epoch có validation accuracy cao nhất, rồi đo test một lần.

### 4.2 Chỉ số và phân tích

- Accuracy (validation và test), macro-F1, số tham số trainable, thời gian mỗi epoch, bộ nhớ GPU đỉnh.
- Phân tích lỗi: các cặp lớp bị nhầm nhiều nhất của cấu hình tốt nhất mỗi kiến trúc.
- Mục tiêu so sánh: (i) từ đầu và pretrained; (ii) các kiểu đóng băng; (iii) CNN và Transformer.

## 5. Môi trường và tái lập

- Google Colab (GPU), PyTorch và timm; cấu hình chạy lưu trong `results/A1/config.json`.
- Mã nguồn: thư mục `A1/` (`data.py`, `models.py`, `engine.py`, `run_a1.py`, `eda.py`, `smoke_test.py`); lệnh chạy xem `README.md`.
- Dữ liệu và kết quả: `splits/`, `results/A1/`.

## 6. Rủi ro đã nhận diện

- ViT huấn luyện từ đầu trên dữ liệu cỡ này có thể đạt accuracy thấp; được xem là một kết quả cần báo cáo chứ không phải lỗi.
- Hai mô hình pretrained trên dữ liệu khác nhau, nên khi so sánh bản pretrained cần thận trọng khi quy kết quả cho kiến trúc.
- Mỗi cấu hình chỉ chạy một lần (một seed), nên độ dao động giữa các lần chạy chưa được đo.
