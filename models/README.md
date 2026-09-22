# models

Artifact TV1:

- `baseline.keras`: checkpoint FP32 tốt nhất theo validation accuracy;
- `baseline_metadata.json`: SHA-256, kiến trúc, cấu hình, phiên bản và metric.

Artifact TV2:

- `baseline_fp32.tflite`: baseline cùng định dạng TFLite để so sánh;
- `int8_n{size}_s{seed}.tflite`: chín model PTQ full-integer INT8;
- `model_metadata.json`: model hash, calibration indices, dtype,
  scale/zero-point, operator và kiểm tra float fallback.

Tất cả artifact TV2 dùng đúng checkpoint ID của TV1. Các file TFLite nhỏ hơn
52 KiB nên được lưu cùng repository; sinh lại bằng `python main.py quantize`.
