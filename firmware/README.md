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

## Cấu trúc hàm theo khối kiến trúc

`har_resnet.c` tách thuật toán thành đúng 3 khối kiến trúc thật của model
(không có stem conv — input đi thẳng vào Block 1), mỗi khối một hàm công
khai riêng, khai báo trong `har_resnet.h`:

- `har_block1_forward()` — residual block projection shortcut, hạ mẫu thời
  gian 128 → 64 (nhánh A: Conv→ReLU→Conv→MaxPool/2; nhánh B: Conv k=1→MaxPool/2).
- `har_block2_forward()` — residual block identity shortcut, giữ nguyên
  (64, 32) (nhánh A: Conv→ReLU→Conv; nhánh B: identity, không phép toán).
- `har_head_forward()` — GlobalMaxPooling1D + Dense(32→6), không softmax.

`har_resnet_predict()` chỉ còn 3 lời gọi liên tiếp của 3 hàm trên, dùng lại
đúng 5 buffer `static` ở mục "Layout bộ nhớ" bên dưới (không đổi `.bss`).
Mỗi hàm khối có block comment 5 mục (NGUỒN INPUT / CẤU TRÚC / CÔNG THỨC /
SHAPE / THAM SỐ) đối chiếu trực tiếp với sơ đồ nhánh và công thức toán ở
[`docs/model_architecture_and_c_inference.md`](../docs/model_architecture_and_c_inference.md)
(mục 1.2 và mục 2). Xem thêm 4 hàm nội bộ `static` theo nhánh/hợp nhánh
(`har_block1_branch_main`, `har_block1_branch_shortcut`,
`har_block2_branch_main`, `har_residual_merge_relu`) trong `har_resnet.c`.

CẢNH BÁO: **Không thread-safe**: 5 buffer `static` dùng chung nên chỉ được có tối
đa MỘT lời gọi `har_resnet_predict` đang chạy tại một thời điểm; không gọi
từ nhiều task/thread hoặc ISR cùng lúc. Xem cảnh báo đầy đủ trong
`firmware/include/har_resnet.h`.

## Layout bộ nhớ (RAM tĩnh)

```c
static float buf_a[128 * 32];   /* 16 KB */
static float buf_b[128 * 32];   /* 16 KB */
static float buf_c[64 * 32];    /*  8 KB */
static float buf_d[64 * 32];    /*  8 KB */
static float buf_e[64 * 32];    /*  8 KB */
```

Tổng buffer trung gian của **thư viện** ≈ 56 KB. Đo đúng con số này bằng
cách biên dịch `har_resnet.c` thành object file RIÊNG (không lẫn buffer của
test), vì đây là con số phản ánh đúng RAM mà firmware ESP32 thật (không
link `test/`) sẽ dùng:

```bash
cd firmware && cc -std=c99 -O2 -Wall -Wextra -Wpedantic -Iinclude \
  -c src/har_resnet.c -o /tmp/har_resnet_lib.o
size /tmp/har_resnet_lib.o
# .bss = 57 344 byte
```

Cộng thêm input 128×9 float (~4,5 KB) và vài biến cục bộ nhỏ → tổng RAM
hoạt động ước lượng ≈ 62 KB. Bảng trọng số 10 726 tham số (≈ 41,9 KB) nằm
trong mảng `const float`, thuộc `.rodata` — trên ESP32 sẽ nằm ở **flash**,
không tốn DRAM. ESP32 có ~320 KB DRAM nội bộ nên đáp ứng thoải mái.

**Lưu ý:** `size build/test_inference` (binary test đầy đủ, sau `make`) cho
ra `.bss = 114 720` byte, KHÔNG phải 57 344 — vì `test/test_inference.c`
cấp thêm 5 buffer `static` RIÊNG (57 344 byte, cộng dồn với 57 344 byte của
thư viện) để chạy `check_block_decomposition()` độc lập với buffer nội bộ
của `har_resnet.c`. Đây là chi phí RAM của RIÊNG chương trình test trên PC,
không phải chi phí RAM của thư viện suy luận khi tích hợp vào firmware
ESP32 thật (không biên dịch cùng `test/`). Xem giải thích đầy đủ tại
[`docs/model_architecture_and_c_inference.md`](../docs/model_architecture_and_c_inference.md)
mục 4.3.

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
