# A2.1 — Chủ đề, paper, dữ liệu, kế hoạch

**Môn:** CO5085 · **Sinh viên:** Phan Ngọc Trâm (2570790), làm cá nhân

## 1. Chủ đề
Phân đoạn ảnh ngữ nghĩa (semantic segmentation), so sánh **CNN (U-Net, encoder ResNet-34)** và **Transformer (SegFormer-B0)**. Bài này khác A1 (phân loại ảnh): đầu ra là nhãn cho từng pixel.

## 2. Paper phân tích
Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., Luo, P. *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.* *Advances in Neural Information Processing Systems 34* (NeurIPS 2021). arXiv:2105.15203.
Paper tham khảo thêm: Ronneberger et al., *U-Net*, MICCAI 2015 (arXiv:1505.04597).
(Sinh viên cần tự kiểm tra lại citation và link trước khi nộp.)

## 3. Dữ liệu
- **Pascal VOC 2012 Segmentation**: 21 lớp (20 vật thể + background), nhãn pixel; vùng viền vật thể có nhãn 255 (bỏ qua khi tính loss và metric).
- Tải bằng `torchvision.datasets.VOCSegmentation`. Về giấy phép: ảnh VOC lấy từ Flickr, bộ dữ liệu công bố cho nghiên cứu; không có giấy phép mở rõ ràng nên chỉ dùng cho học tập (cần đối chiếu điều khoản trên trang VOC trước khi nộp).
- Split (seed 42): `train.txt` chính thức (1.464 ảnh) → **1.318 train / 146 val**; **test = `val.txt` chính thức (1.449 ảnh)**, không dùng để chọn mô hình.
- Không dùng SBD (augmented set, ~10k ảnh); đây là một hạn chế có chủ đích để bài chạy được trong thời gian và GPU Colab.

## 4. Metric
mIoU (chính), pixel accuracy, mean Dice, IoU từng lớp, confusion matrix; chi phí: số tham số (tổng/trainable), s/epoch, bộ nhớ GPU đỉnh, độ trễ suy luận 512×512, GFLOPs.

## 5. Kế hoạch thí nghiệm
| # | Kiến trúc | Chế độ | Loss |
|---|---|---|---|
| 1,2 | U-Net R34 / SegFormer-B0 | pretrained (ImageNet-1k), fine-tune toàn bộ | CE |
| 3,4 | như trên | from-scratch | CE |
| 5,6 | như trên | đóng băng encoder, chỉ train decoder | CE |
| 7,8 | như trên | pretrained | CE + Dice |

Cấu hình chung: seed 42, AdamW (wd 0,01), poly LR có warm-up, fp16, 50 epoch, crop 384, augmentation: scale 0,5–2,0, lật ngang, random crop. Đánh giá trên ảnh gốc (pad 512, phần pad bị bỏ qua).
Lưu ý: cả hai encoder đều pretrain trên ImageNet-1k nhưng không cùng công thức huấn luyện nên so sánh CNN và Transformer còn bị nhiễu bởi yếu tố này.

## 6. Kế hoạch phân tích paper
Tóm tắt phương pháp (Mix Transformer encoder phân cấp, không positional encoding, decoder MLP), thí nghiệm của paper, hạn chế, rồi đối chiếu với kết quả của mình (xem `reports/a2_paper_notes_TEMPLATE.md`).
