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

Mọi hình đều đọc từ CSV thực nghiệm, không hard-code số liệu trong mã vẽ.
