# Phương pháp lượng tử hóa sau huấn luyện INT8

Phần này triển khai tại `src/calibration.py` và `src/quantize.py`. Nguồn mô
hình duy nhất là checkpoint FP32 có SHA256
`aa323b3e55d1f904edbb05fb6d2a817b748b6e0ad26b7eb0166777c94535c74e`.

## Quan hệ với paper

Novac et al. (2021), DOI `10.3390/s21092984`, mô tả ResNetv1-6 cho UCI-HAR
ở Hình 4 và khảo sát lượng tử hóa phục vụ vi điều khiển. Kiến trúc baseline
của nhóm bám theo hình này nhưng dùng Conv1D cho tín hiệu `(128, 9)`.

Điểm khác cần nói rõ: paper dùng luồng MicroAI/QAT trong thí nghiệm chính,
còn đề tài này chủ động khảo sát **TFLite PTQ full-integer INT8**. Vì vậy đây
là thí nghiệm dựa trên paper, không tuyên bố tái hiện nguyên bản kết quả paper.

## Thuật toán PTQ INT8

1. Nạp `models/baseline.keras`; kiểm tra SHA256 với metadata.
2. Xuất một baseline `baseline_fp32.tflite` để so sánh cùng định dạng file.
3. Với mỗi calibration size `50`, `200`, `500` và seed `11`, `22`, `33`, lấy
   mẫu **không lặp** từ `x_train`. Tensor này đã được z-score ở TV1; không
   chuẩn hóa lần hai và không dùng validation/test để calibration.
4. Representative dataset chạy qua model để TFLite tìm miền giá trị của
   activation và suy ra `scale`, `zero_point`.
5. Ép tập toán tử `TFLITE_BUILTINS_INT8`, input `int8`, output `int8`.
6. Mở lại từng model bằng interpreter, kiểm tra input/output và mọi tensor;
   nếu còn tensor float thì dừng thay vì gọi nhầm là full-integer.

Công thức lượng tử hóa affine:

```text
q = clip(round(r / scale + zero_point), -128, 127)
r_xấp_xỉ = scale × (q - zero_point)
```

Trong đó `r` là số thực FP32 và `q` là số nguyên INT8. Trọng số và activation
dùng ít bit hơn nên file nhỏ hơn, đổi lại việc làm tròn/cắt biên có thể làm
thay đổi dự đoán.

## Cách chạy và đầu ra

```bash
python main.py quantize
```

Đầu ra gồm một FP32 TFLite, chín INT8 TFLite và
`models/model_metadata.json`. Metadata lưu chính xác calibration indices,
dtype, scale, zero-point, danh sách operator, model SHA256 và dung lượng.

Đã kiểm tra trên Python 3.10.12, TensorFlow 2.16.1 và NumPy 1.26.4. Cả chín
model đều có input/output `int8`, không có tensor float fallback. Mỗi model
INT8 là 25.104 byte; baseline FP32 TFLite là 51.728 byte.

## Giới hạn

- Đây là PTQ, không fine-tune lại sau lượng tử hóa.
- Calibration ngẫu nhiên được lặp ba seed, nhưng ba seed vẫn là cỡ mẫu nhỏ.
- Dung lượng là số byte của file TFLite, không phải Flash/RAM thực đo trên MCU.
- Không báo tốc độ hoặc năng lượng vì đề tài chỉ chạy trên laptop.
