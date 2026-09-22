# firmware/ — Suy luận (inference) ResNetv1-6 1D bằng C99 thuần

Cài đặt C99 độc lập (không phụ thuộc thư viện ngoài, không malloc) cho model
`uci_har_resnetv1_6_fp32` (`models/baseline.keras`). Mục tiêu: build và kiểm
tra đúng đắn trên PC trước, có cấu trúc sẵn sàng port sang ESP-IDF/ESP32.

Xem giải thích đầy đủ về thuật toán, layout dữ liệu và cách trích tham số
tại [`docs/model_architecture_and_c_inference.md`](../docs/model_architecture_and_c_inference.md).

## Cấu trúc thư mục

```
firmware/
├── include/
│   ├── har_resnet.h          viết tay — API công khai
│   ├── har_resnet_params.h   SINH TỰ ĐỘNG bởi src/export_c.py — không sửa tay
│   └── har_golden.h          SINH TỰ ĐỘNG bởi src/export_c.py — không sửa tay
├── src/
│   ├── har_resnet.c          viết tay — thuật toán forward pass
│   ├── har_resnet_params.c   SINH TỰ ĐỘNG — trọng số const float
│   └── har_golden.c          SINH TỰ ĐỘNG — 12 golden vector + logits tham chiếu
├── test/
│   └── test_inference.c      viết tay — chương trình main() kiểm thử
├── Makefile
└── .gitignore                bỏ qua build/
```

Các file đánh dấu "SINH TỰ ĐỘNG" được `python main.py export-c` (chạy từ
gốc repo) ghi đè hoàn toàn mỗi lần chạy — không sửa tay, mọi thay đổi sẽ mất.

## Sinh lại file tham số / golden vector

```bash
cd ..                 # về gốc repo
python -m pip install -r requirements.txt   # cần TensorFlow + NumPy
python main.py export-c
```

Lệnh trên đọc `models/baseline.keras`, kiểm tra sha256 khớp
`models/baseline_metadata.json`, rồi sinh lại 4 file C ở trên cùng
`models/params_fp32.npz` và `models/params_manifest.json`.

## Build và chạy kiểm thử

```bash
cd firmware
make clean && make        # biên dịch: cc -std=c99 -O2 -Wall -Wextra -Wpedantic
make test                 # chạy build/test_inference
```

`make test` chạy 12 golden vector (2 mẫu/lớp, lấy từ
`data/processed/sanity_samples.npz`) qua `har_resnet_predict`, in bảng đối
chiếu với logits Keras, kiểm tra `har_normalize`, rồi in `PASS`/`FAIL` và
trả exit code 0/1 tương ứng.

Kết quả thực đo khi triển khai: 12/12 argmax khớp, `max_abs_diff` trên
logits ≈ 1e-6 (ngưỡng chấp nhận 1e-3), build 0 warning. Chi tiết ở mục 5
của tài liệu kiến trúc.

## Layout bộ nhớ (RAM tĩnh)

```c
static float buf_a[128 * 32];   /* 16 KB */
static float buf_b[128 * 32];   /* 16 KB */
static float buf_c[64 * 32];    /*  8 KB */
static float buf_d[64 * 32];    /*  8 KB */
static float buf_e[64 * 32];    /*  8 KB */
```

Tổng buffer trung gian ≈ 56 KB (đo bằng `size build/test_inference`:
`.bss ≈ 57 376` byte), cộng input 128×9 float (~4,5 KB) và vài biến cục bộ
nhỏ → tổng RAM hoạt động ≈ 62 KB. Bảng trọng số 10 726 tham số (≈ 41,9 KB)
nằm trong mảng `const float`, thuộc `.rodata` — trên ESP32 sẽ nằm ở **flash**,
không tốn DRAM. ESP32 có ~320 KB DRAM nội bộ nên đáp ứng thoải mái.

## Ghi chú port sang ESP-IDF

- Toàn bộ mã trong `include/` và `src/` là C99 chuẩn, không dùng API POSIX
  hay `stdio.h` (trừ `test/test_inference.c`, chỉ dùng khi build test trên
  PC) → có thể copy nguyên `include/` + `src/*.c` (trừ `test/`) vào một
  ESP-IDF component.
- Thêm `firmware/CMakeLists.txt` với
  `idf_component_register(SRCS "src/har_resnet.c" "src/har_resnet_params.c"
  INCLUDE_DIRS "include")` khi tạo component thật (chưa làm trong bản này).
- Các mảng `const float har_*` sẽ tự động được trình biên dịch ESP-IDF đặt
  vào flash (`.rodata`), không cần đổi cách khai báo.
- Đo thời gian suy luận trên ESP32 bằng `esp_timer_get_time()` bao quanh lời
  gọi `har_resnet_predict`.
- `har_golden.c`/`har_golden.h` chỉ cần khi test trên PC/CI; có thể loại bỏ
  khỏi ảnh flash cuối cùng để tiết kiệm ~200 KB nếu cần.

## Giới hạn hiện tại

- Chỉ có FP32. INT8 chưa triển khai vì phụ thuộc `src/quantize.py` (TV2)
  hoàn thiện trước — xem mục 6 của tài liệu kiến trúc.
- Chưa đo accuracy trên toàn bộ 2947 cửa sổ test bằng C (chỉ 12 golden
  vector) — nằm ngoài phạm vi lần triển khai này.
