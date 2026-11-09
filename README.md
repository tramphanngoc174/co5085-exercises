# CO5085 – Bài tập E1–E3: Học sâu và ứng dụng trong thị giác máy tính

- **Sinh viên:** Phan Ngọc Trâm – MSSV 2570790 (làm cá nhân)
- **Môn học:** Học sâu và ứng dụng trong thị giác máy tính (CO5085), HK 261 (2026–2027)
- **GitHub Pages:** https://tramphanngoc174.github.io/co5085-exercises/
- **Báo cáo PDF chung (E1–E3):** [`reports/exercises.pdf`](reports/exercises.pdf)

## Trạng thái nộp bài

| Hạng mục | Hạn (23:59 GMT+7) | Trạng thái | Tag |
|---|---|---|---|
| GitHub Pages (khung trang) | 19/10/2026 | Hoàn thành | `pages-w7` |
| E1 – Softmax / MLP / CNN | 02/11/2026 | Đã có kết quả và báo cáo | `submit-e1` |
| E2 – MSA + tokenization | 02/11/2026 | Đã có kết quả và báo cáo | `submit-e2` |
| E3 – LSTM / GRU | 09/11/2026 | Đã có kết quả và báo cáo | `submit-e3` |
| A1.2 – CNN vs Transformer (CIFAR-100) | 09/11/2026 | Đã có kết quả và báo cáo (`reports/a1_report.pdf`) | `submit-a1` |

## Cấu trúc thư mục

```
.
├── README.md
├── requirements.txt
├── docs/          # nguồn GitHub Pages
├── reports/       # exercises.pdf (báo cáo chung) + nguồn báo cáo
├── common/        # code dùng chung (data, split, training loop, vẽ biểu đồ)
├── splits/        # chỉ số train/val cố định để tái lập
├── E1/            # Softmax, MLP, CNN
├── E2/            # MSA tự viết, MSA PyTorch, tokenization
├── E3/            # LSTM, GRU
└── results/       # log, config, biểu đồ, bảng (mỗi bài một thư mục)
```

## Môi trường

- Python 3.10+ và PyTorch (ghi lại phiên bản thực tế sau khi chạy: `python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"`).
- Cài thư viện: `pip install -r requirements.txt`
- Dataset: Fashion-MNIST (tải tự động bằng torchvision vào `data/`). Chia train/val = 54.000/6.000 bằng seed 42, test = 10.000 ảnh gốc; chỉ số chia lưu trong `splits/`.

## Lệnh chạy

Chạy tất cả từ thư mục gốc của repo.

**E1** (Softmax / MLP / CNN):
```bash
python E1/smoke_test.py                 # kiểm thử nhanh bằng dữ liệu giả (không dùng cho báo cáo)
python E1/run_e1.py --dataset fashion_mnist --epochs 15
```
Kết quả (log, config, bảng, biểu đồ, checkpoint) nằm trong `results/E1/`.

Dựng lại báo cáo PDF từ số liệu E1: `pip install reportlab pillow` rồi `python reports/source/build_report.py`.

**E2** (MSA tự viết vs PyTorch, 3 cách tokenize):
```bash
python E2/smoke_test.py                 # kiểm thử MSA, hình dạng tensor, pipeline bằng dữ liệu giả
python E2/run_e2.py --dataset fashion_mnist --epochs 10
```
Kết quả nằm trong `results/E2/`.

**E3** (LSTM / GRU trên chuỗi ảnh, baseline E1, thí nghiệm xáo hàng):
```bash
python E3/smoke_test.py
python E3/run_e3.py --dataset fashion_mnist --epochs 10
```
Baseline MLP/CNN được nạp từ checkpoint `results/E1/*/model.pt`. Kết quả nằm trong `results/E3/`.

**A1** (CNN vs Transformer trên CIFAR-100, nhiều chế độ huấn luyện):
```bash
pip install timm
python A1/smoke_test.py                 # kiểm thử bằng dữ liệu giả, không tải pretrained
python A1/eda.py                        # thống kê dataset + ảnh mẫu
python A1/run_a1.py                      # mặc định: CIFAR-100 đầy đủ, 224x224, 15 epoch, 2 kiến trúc x 4 chế độ = 8 lần chạy
                                        # chạy lại cùng lệnh sẽ bỏ qua cấu hình đã xong (an toàn khi Colab bị ngắt)
```
Kết quả nằm trong `results/A1/`.

## Ghi nhận việc sử dụng công cụ AI

Công cụ AI (Claude, Anthropic) được dùng để hỗ trợ: phân tích đề và lập kế hoạch; soạn khung mã nguồn E1–E3 và A1 (`common/`, `E1/`, `E2/`, `E3/`, `A1/`), tệp khung README, trang GitHub Pages; soạn bản nháp báo cáo từ số liệu thật do sinh viên cung cấp. Sinh viên chạy thí nghiệm trên Google Colab và chịu trách nhiệm kiểm tra, giải thích mã nguồn và số liệu.
