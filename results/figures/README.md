# results/figures

Chạy `python main.py visualize-tv1` để tạo biểu đồ TV1 trong `tv1/`:

- đường accuracy/loss 300 epoch;
- phân bố lớp train/validation/test;
- ma trận nhầm lẫn FP32 trên test;
- sơ đồ kiến trúc ResNetv1-6 1D.

Các biểu đồ tổng hợp FP32–INT8 của TV3 sẽ được tạo bởi `python main.py plot` sau khi phần lượng tử hóa hoàn tất.
