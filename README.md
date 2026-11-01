# CO5085 – Bài tập E1–E3: Học sâu và ứng dụng trong thị giác máy tính

- **Sinh viên:** Phan Ngọc Trâm – MSSV 2570790 (làm cá nhân)
- **Môn học:** Học sâu và ứng dụng trong thị giác máy tính (CO5085), HK 261 (2026–2027)
- **GitHub Pages:** https://tramphanngoc174.github.io/co5085-exercises/
- **Báo cáo PDF chung (E1–E3):** [`reports/exercises.pdf`](reports/exercises.pdf)

## Trạng thái nộp bài

| Hạng mục | Hạn (23:59 GMT+7) | Trạng thái | Tag |
|---|---|---|---|
| GitHub Pages (khung trang) | 19/10/2026 | Đang thực hiện | `pages-w7` |
| E1 – Softmax / MLP / CNN | 02/11/2026 | Chưa thực hiện | `submit-e1-e2` |
| E2 – MSA + tokenization | 02/11/2026 | Chưa thực hiện | `submit-e1-e2` |
| E3 – LSTM / GRU | 09/11/2026 | Chưa thực hiện | `submit-e3` |

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

*(E2, E3: sẽ bổ sung.)*

## Ghi nhận việc sử dụng công cụ AI

Công cụ AI (Claude) được dùng để hỗ trợ phân tích đề, lập kế hoạch, gợi ý cấu trúc dự án và soạn các tệp khung (README, trang GitHub Pages). *(Sẽ cập nhật chi tiết ở các mốc E1–E3.)* Sinh viên tự chịu trách nhiệm về nội dung nộp và có thể giải thích code cũng như số liệu.
