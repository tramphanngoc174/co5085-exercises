# Số liệu nổi bật (chỉ là sự kiện, chưa phải nhận xét). Dùng để trả lời câu 8–11.

Lưu ý: phần lớn cấu hình chỉ chạy 1 seed. Ở A1, hai lần chạy cùng seed từng lệch 2,9 điểm accuracy (val) ở epoch 10, nên chênh lệch nhỏ (dưới ~1 điểm mIoU) chưa chắc là thật.

**Xếp hạng test mIoU (%):** segformer_b2_pretrained_ce 78.31; unet_r34_pretrained_ce_dice 67.48; unet_r34_pretrained_ce 67.18; segformer_b0_pretrained_ce_dice 67.02; segformer_b0_pretrained_ce_fp32 66.68; segformer_b0_pretrained_ce 66.46; unet_r34_frozen_ce 63.70; segformer_b0_frozen_ce 58.44; segformer_b0_scratch_ce_long100 23.14; unet_r34_scratch_ce_long100 18.43; unet_r34_scratch_ce 11.11; segformer_b0_scratch_ce_lr1e-4 4.04; segformer_b0_scratch_ce 3.87

- unet_r34: pretrained − scratch = +56.06 điểm mIoU
- unet_r34: pretrained(full fine-tune) − frozen = +3.47 điểm mIoU
- unet_r34: CE+Dice − CE = +0.30 điểm mIoU
- segformer_b0: pretrained − scratch = +62.58 điểm mIoU
- segformer_b0: pretrained(full fine-tune) − frozen = +8.01 điểm mIoU
- segformer_b0: CE+Dice − CE = +0.57 điểm mIoU
- Chế độ pretrained: SegFormer − U-Net = -0.72 điểm mIoU
- Chế độ scratch: SegFormer − U-Net = -7.24 điểm mIoU
- Chế độ frozen: SegFormer − U-Net = -5.26 điểm mIoU
- unet_r34: scratch dài (100 epoch) = 18.43 mIoU, so với scratch 50 epoch 11.11; best epoch 97/100
- segformer_b0: scratch dài (100 epoch) = 23.14 mIoU, so với scratch 50 epoch 3.87; best epoch 90/100
- Scratch dài: SegFormer − U-Net = +4.71 điểm mIoU (công thức huấn luyện hai bên khác nhau)
- SegFormer pretrained: fp32 − fp16 = +0.22 điểm mIoU
- Chẩn đoán segformer_b0_scratch_ce_diagA (lr 0.0002, 200 epoch): mIoU trên chính 64 ảnh train = 4.49%
- Chẩn đoán segformer_b0_scratch_ce_diagB (lr 0.0002, 200 epoch, cắt gradient 1.0, khởi tạo conv theo mã gốc của SegFormer): mIoU trên chính 64 ảnh train = 4.63%
- Chẩn đoán segformer_b0_scratch_ce_diagC (lr 0.0002, 200 epoch, cắt gradient 1.0, fp32, khởi tạo conv theo mã gốc của SegFormer): mIoU trên chính 64 ảnh train = 47.68%
- Nhiều seed segformer_b0_pretrained_ce: seeds [42, 1, 2], test mIoU 66.46, 67.52, 66.08; trung bình 66.68 ± 0.75
- Nhiều seed unet_r34_pretrained_ce: seeds [42, 1, 2], test mIoU 67.18, 67.29, 66.43; trung bình 66.97 ± 0.47
- Pretrained, trung bình các seed: SegFormer-B0 − U-Net = -0.28 điểm mIoU (độ lệch chuẩn: U-Net 0.47, SegFormer 0.75)
- segformer_b2_pretrained_ce: test mIoU 78.31; so với U-Net pretrained +11.13, so với SegFormer-B0 pretrained +11.85 điểm; params 27,362,773

**Chi phí:**
- unet_r34_pretrained_ce: params 24,429,941 (trainable 24,429,941), 24.5s/epoch, bộ nhớ đỉnh 2451 MB, độ trễ 10.3 ms, GFLOPs 58.9, best epoch 50/50
- segformer_b0_pretrained_ce: params 3,719,541 (trainable 3,719,541), 23.7s/epoch, bộ nhớ đỉnh 2948 MB, độ trễ 12.4 ms, GFLOPs 15.7, best epoch 29/50
- unet_r34_scratch_ce: params 24,429,941 (trainable 24,429,941), 24.8s/epoch, bộ nhớ đỉnh 2470 MB, độ trễ 10.3 ms, GFLOPs 58.9, best epoch 50/50
- segformer_b0_scratch_ce: params 3,719,541 (trainable 3,719,541), 23.7s/epoch, bộ nhớ đỉnh 2948 MB, độ trễ 10.4 ms, GFLOPs 15.7, best epoch 1/50
- unet_r34_frozen_ce: params 24,429,941 (trainable 3,145,269), 22.3s/epoch, bộ nhớ đỉnh 1899 MB, độ trễ 10.8 ms, GFLOPs 58.9, best epoch 36/50
- segformer_b0_frozen_ce: params 3,719,541 (trainable 400,149), 22.3s/epoch, bộ nhớ đỉnh 2903 MB, độ trễ 12.4 ms, GFLOPs 15.7, best epoch 43/50
- unet_r34_pretrained_ce_dice: params 24,429,941 (trainable 24,429,941), 27.3s/epoch, bộ nhớ đỉnh 3312 MB, độ trễ 10.8 ms, GFLOPs 58.9, best epoch 37/50
- segformer_b0_pretrained_ce_dice: params 3,719,541 (trainable 3,719,541), 25.9s/epoch, bộ nhớ đỉnh 3105 MB, độ trễ 10.5 ms, GFLOPs 15.7, best epoch 45/50
- segformer_b0_pretrained_ce_fp32: params 3,719,541 (trainable 3,719,541), 34.9s/epoch, bộ nhớ đỉnh 3835 MB, độ trễ 12.6 ms, GFLOPs 15.7, best epoch 45/50
- segformer_b0_scratch_ce_long100: params 3,719,541 (trainable 3,719,541), 35.5s/epoch, bộ nhớ đỉnh 3835 MB, độ trễ 12.4 ms, GFLOPs 15.7, best epoch 90/100
- segformer_b0_scratch_ce_lr1e-4: params 3,719,541 (trainable 3,719,541), 24.4s/epoch, bộ nhớ đỉnh 2947 MB, độ trễ 11.2 ms, GFLOPs 15.7, best epoch 1/50
- segformer_b2_pretrained_ce: params 27,362,773 (trainable 27,362,773), 44.3s/epoch, bộ nhớ đỉnh 8805 MB, độ trễ 28.0 ms, GFLOPs 121.4, best epoch 35/50
- unet_r34_scratch_ce_long100: params 24,429,941 (trainable 24,429,941), 24.9s/epoch, bộ nhớ đỉnh 2449 MB, độ trễ 10.6 ms, GFLOPs 58.9, best epoch 97/100

**3 lớp IoU thấp nhất / 3 lớp cao nhất (test), các run pretrained CE:**
- unet_r34_pretrained_ce: thấp nhất chair 22.5, sofa 37.0, pottedplant 45.1; cao nhất aeroplane 85.6, bird 85.8, background 92.9
- segformer_b0_pretrained_ce: thấp nhất chair 22.7, sofa 35.7, diningtable 40.8; cao nhất bird 83.6, aeroplane 84.5, background 92.1
- segformer_b2_pretrained_ce: thấp nhất chair 34.7, sofa 51.0, diningtable 57.4; cao nhất aeroplane 91.4, bird 91.6, background 94.9
