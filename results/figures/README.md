# results/figures

Chạy `python main.py visualize-tv1` để tạo biểu đồ TV1 trong `tv1/`:

- đường accuracy/loss 300 epoch;
- phân bố lớp train/validation/test;
- ma trận nhầm lẫn FP32 trên test;
- sơ đồ kiến trúc ResNetv1-6 1D.

Chạy `python main.py plot` để tạo `quantization/`:

- accuracy và macro-F1 theo calibration size (mean ± SD qua ba seed);
- dung lượng file FP32/INT8;
- ma trận nhầm lẫn test của FP32 và model INT8 đại diện đã chọn.
- `pipeline/algorithm_flow.png`: sơ đồ trình bày dữ liệu, baseline, PTQ INT8,
  đánh giá và forward pass ResNet 1D.
- `pipeline/quantization_har_archify.html`: sơ đồ Archify tương tác, thể hiện
  chi tiết hai nhánh residual, nhánh FP32 đối chứng, calibration PTQ và nhánh
  chọn/đánh giá INT8.
- `pipeline/quantization_har_archify.png`: ảnh sáng 1440×900 để chèn báo cáo;
  `pipeline/quantization_har_archify_dark.png`: ảnh tối 2048×1320 để trình chiếu.

Mọi hình đều đọc từ CSV thực nghiệm, không hard-code số liệu trong mã vẽ.
