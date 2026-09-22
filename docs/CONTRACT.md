# Giao diện bàn giao — đọc trước khi code

Đây là quy ước đề xuất cần cả nhóm chốt trong Phase 1. Đổi schema phải báo trước, không âm thầm đổi tên kênh, nhãn hay file.

## TV1 → TV2/TV3

- `data/processed/har.npz`: x_train, y_train, x_val, y_val, x_test, y_test.
- x: float32, dạng (số cửa sổ, số thời điểm, số kênh), **đã chuẩn hóa**. Số kênh và thứ tự phải xác nhận trước khi làm.
- y: số nguyên từ 0 đến K-1; labels.json ánh xạ rõ, không nhầm nhãn gốc của dataset.
- `data/processed/preprocess.json`: mean/std, thứ tự kênh, shape, cách xử lý std=0, labels và xác nhận normalization_applied=true.
- `data/processed/split.json`: danh sách subject mỗi split để kiểm tra không giao nhau.
- `models/baseline.keras`: checkpoint chính thức.
- `models/baseline_metadata.json`: checkpoint_id (hash), seed, cấu hình kiến trúc/train, phiên bản thư viện, đường dẫn metadata dữ liệu.
- `data/processed/sanity_samples.npz`: vài cửa sổ, nhãn và output baseline để đối chiếu.

Chuẩn hóa chỉ fit trên train. Không chuẩn hóa thêm trong quantize/evaluate. Các artifact trên đã được TV1 tạo và kiểm tra.

## TV2 → TV3

- `models/baseline_fp32.tflite`.
- `models/int8_n{size}_s{seed}.tflite` cho từng cấu hình.
- `models/model_metadata.json`: danh sách model_file, checkpoint_id, calibration_size, calibration_seed, calibration_indices, input/output dtype và quantization parameters, converter version.
- Đọc dtype/scales/zero-points từ model thực tế, không giả định tất cả đầu vào luôn cùng scale.
- Xuất mô hình chưa đồng nghĩa chứng minh full-integer: kiểm tra toán tử và báo float fallback nếu có.

## TV3 → cả nhóm

- `results/metrics.csv`: một dòng cho từng model/seed/split, cùng schema ở file mẫu.
- accuracy_pct: 0–100; macro_f1: 0–1; model_size_bytes: số byte thực tế.
- Baseline không có calibration: để trống calibration_size và calibration_seed.
- checkpoint_id gắn với model FP32 gốc; model_id xác định artifact thực tế.
- `results/summary.csv`, `results/figures/`: dữ liệu và biểu đồ tổng hợp.
- Chỉ chạy test sau khi nhóm chốt cấu hình bằng validation; không dùng kết quả test để chọn tiếp.

## Kiểm tra bàn giao

1. TV1 và TV2 cùng chạy một mẫu qua baseline đã lưu.
2. TV2 và TV3 cùng đối chiếu input/output INT8 trên một mẫu.
3. TV3 xác nhận không trộn split, checkpoint hoặc nhãn trong bảng.
4. Mỗi PR ghi lệnh đã chạy thực tế và kết quả, không chỉ ghi “code chạy được”.
