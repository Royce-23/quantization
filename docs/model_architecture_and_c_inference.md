# Kiến trúc mô hình, trích xuất tham số và suy luận (inference) bằng C

Tài liệu này là đầu ra chính cho 3 yêu cầu: (1) mô tả flow/thuật toán, (2)
trích xuất parameters, (3) cài đặt inference bằng C thuần, cho mô hình FP32
`uci_har_resnetv1_6_fp32` (`src/model.py`) đã bàn giao tại
`models/baseline.keras`.

## 1. Kiến trúc và luồng dữ liệu (flow)

Mô hình là **ResNetv1-6 1D**, điều chỉnh Conv/Pool 1D từ Hình 4, mục 6.1.1
của Novac et al. (2021), *Sensors*, https://doi.org/10.3390/s21092984, gồm 2
residual block với `filters = 32`. Sơ đồ kiến trúc do Keras vẽ:
[`results/figures/tv1/model_architecture.png`](../results/figures/tv1/model_architecture.png).

### 1.1 Bảng layer (tên Keras thật — dùng làm khoá khi trích params)

| # | Layer (tên Keras) | Phép toán | Input shape | Output shape | Params |
| --- | --- | --- | --- | --- | ---: |
| 0 | `sensor_window` | Input | — | (128, 9) | 0 |
| 1 | `block1_conv1` | Conv1D k=3, same, bias | (128, 9) | (128, 32) | 896 |
| 2 | `block1_relu1` | ReLU | (128, 32) | (128, 32) | 0 |
| 3 | `block1_conv2` | Conv1D k=3, same, bias | (128, 32) | (128, 32) | 3104 |
| 4 | `block1_main_pool` | MaxPool1D pool=2 stride=2 valid | (128, 32) | (64, 32) | 0 |
| 5 | `block1_projection` | Conv1D k=1, valid, bias — trên `sensor_window` | (128, 9) | (128, 32) | 320 |
| 6 | `block1_shortcut_pool` | MaxPool1D pool=2 | (128, 32) | (64, 32) | 0 |
| 7 | `block1_add` | Add(#4, #6) | (64, 32)×2 | (64, 32) | 0 |
| 8 | `block1_output` | ReLU | (64, 32) | (64, 32) | 0 |
| 9 | `block2_conv1` | Conv1D k=3, same, bias | (64, 32) | (64, 32) | 3104 |
| 10 | `block2_relu1` | ReLU | (64, 32) | (64, 32) | 0 |
| 11 | `block2_conv2` | Conv1D k=3, same, bias | (64, 32) | (64, 32) | 3104 |
| 12 | `block2_add` | Add(#11, #8) — shortcut identity | (64, 32)×2 | (64, 32) | 0 |
| 13 | `block2_output` | ReLU | (64, 32) | (64, 32) | 0 |
| 14 | `global_max_pool` | GlobalMaxPooling1D (max theo trục thời gian) | (64, 32) | (32,) | 0 |
| 15 | `activity_logits` | Dense(6), bias, **không activation → logits thô** | (32,) | (6,) | 198 |
| | | | | **Tổng** | **10 726** |

Dự đoán lớp = `argmax(logits)`. Softmax **không nằm trong model** (huấn
luyện dùng `CategoricalCrossentropy(from_logits=True)`); trong C chỉ cần
`argmax`, có `har_softmax` cho demo/UI nếu cần xác suất.

### 1.2 Sơ đồ ASCII

```text
        sensor_window (128,9)  [đã z-score]
              |
      +-------+--------------------------------+
      |                                        |
  block1_conv1 k3 (128,32)                block1_projection k1 (128,32)
      | ReLU                                   |
  block1_conv2 k3 (128,32)                block1_shortcut_pool /2 (64,32)
      |                                        |
  block1_main_pool /2 (64,32)                  |
      +------------------ Add -----------------+
                          | ReLU  -> x (64,32) -------+
                          |                           |
                    block2_conv1 k3 (64,32)           | identity
                          | ReLU                      |
                    block2_conv2 k3 (64,32)           |
                          +--------- Add -------------+
                                     | ReLU (64,32)
                            global_max_pool -> (32,)
                            activity_logits Dense(6) -> logits(6)
                                     | argmax
                                   class 0..5
```

**Ghi chú quan trọng**: mô hình **không có BatchNormalization** ở bất kỳ
đâu (xem mục 7). Conv1D dùng `use_bias=True` mặc định của Keras.

## 2. Thuật toán từng phép toán (pseudocode)

Ký hiệu: `in[t*Cin + c]` (layout `[time][channel]`),
`w[(ki*Cin + ci)*Cout + co]` (layout kernel `[k][cin][cout]`, đúng C-order
flatten của Keras — không transpose), `out[t*Cout + co]`.

### 2.1 Conv1D "same", kernel lẻ K, stride 1

Dùng chung cho cả `padding="same"` (K=3) lẫn `padding="valid"` với K=1
(khi đó `pad = K/2 = 0`, tức không đệm gì — đúng là "valid"):

```
pad = K / 2                      // chia nguyên, K lẻ
for t in 0 .. L-1:
    for co in 0 .. Cout-1:
        acc = bias[co]                       // kiểu float (32-bit)
        for ki in 0 .. K-1:
            ts = t + ki - pad
            if ts < 0 or ts >= L: continue   // zero padding
            for ci in 0 .. Cin-1:
                acc += in[ts*Cin + ci] * w[(ki*Cin + ci)*Cout + co]
        out[t*Cout + co] = acc
```

**Giải thích same-padding của TensorFlow**: với stride 1 và K lẻ,
`padding="same"` đệm đối xứng `pad_left = pad_right = (K-1)/2 = K/2`
(chia nguyên), giữ nguyên độ dài chuỗi (`L_out = L_in`). Công thức trên
khớp chính xác — không có trường hợp lệch 1 vị trí như hay gặp với K chẵn.

### 2.2 ReLU

`x[i] = x[i] > 0 ? x[i] : 0` (làm tại chỗ được).

### 2.3 MaxPool1D pool=2, stride=2, padding valid

Dùng cho L chẵn (128 → 64):

```
for t in 0 .. L/2 - 1:
    for c in 0 .. C-1:
        out[t*C + c] = max(in[(2t)*C + c], in[(2t+1)*C + c])
```

### 2.4 Add

`out[i] = a[i] + b[i]`.

### 2.5 GlobalMaxPool1D

```
for c in 0 .. C-1:
    m = -FLT_MAX
    for t in 0 .. L-1: m = max(m, in[t*C + c])
    out[c] = m
```

### 2.6 Dense

`out[o] = bias[o] + sum_i in[i] * w[i*Dout + o]`.

### 2.7 Normalize (z-score, áp dụng trước layer 1)

`xn[t*9 + c] = (x_raw[t*9 + c] - mean[c]) / std[c]`, `mean`/`std` lấy từ
`data/processed/preprocess.json` (fit trên tập train của model, không rò rỉ
dữ liệu validation/test).

Kiểu tích luỹ: **`float` 32-bit** ở cả C và NumPy reference (không dùng
`double`), để sát nhất với kernel float32 của TensorFlow.

## 3. Trích xuất tham số (parameters)

Lệnh `python main.py export-c` (cài đặt tại `src/export_c.py`) thực hiện:

1. Tính sha256 của `models/baseline.keras`, đối chiếu
   `models/baseline_metadata.json["checkpoint_id"]`; lệch → báo lỗi, dừng
   (bảo vệ baseline FP32 đã công bố khỏi bị export nhầm checkpoint khác).
2. Nạp model, lấy trọng số **theo tên layer** (`model.get_layer(name)`),
   kiểm tra đúng shape ở bảng 1.1, kiểm tra tổng tham số = 10 726 và khớp
   `model.count_params()`.
3. Đọc `mean`/`std` từ `data/processed/preprocess.json`.
4. Ghi `models/params_fp32.npz` (`savez_compressed`, 14 key: 5 lớp Conv
   ×`_w`/`_b`, `activity_logits_w`/`_b`, `input_mean`, `input_std`) — bản
   canonical để Python/NumPy dùng lại **không cần TensorFlow**.
5. Sinh mã C tĩnh (mục 4) và `models/params_manifest.json` (checkpoint_id,
   danh sách layer/shape/số tham số, sha256 + số byte từng file C sinh ra,
   môi trường Python/NumPy/TensorFlow).

### 3.1 Layout dữ liệu (đã chốt, không tự suy luận lại)

- Tensor kích hoạt: **row-major `[time][channel]`** — giống NHWC 1D của
  Keras, `in[t*Cin + c]`.
- Kernel Conv1D: **`[k][cin][cout]`**, đúng thứ tự C-order khi
  `kernel.reshape(-1)` trên mảng Keras gốc shape `(k, cin, cout)` —
  **không transpose**. Cho phép export chỉ bằng flatten trực tiếp và dễ
  đối chiếu bằng mắt với `model.get_layer(...).get_weights()`.
- Dense: kernel Keras shape `(din, dout)`, flatten C-order → `w[i*dout+o]`.

### 3.2 Định dạng file tham số

`models/params_fp32.npz` — key đặt theo tên layer + hậu tố `_w`/`_b`, dtype
`float32`, giữ nguyên shape Keras:

```
block1_conv1_w      (3, 9, 32)      block1_conv1_b      (32,)
block1_conv2_w      (3, 32, 32)     block1_conv2_b      (32,)
block1_projection_w (1, 9, 32)      block1_projection_b (32,)
block2_conv1_w      (3, 32, 32)     block2_conv1_b      (32,)
block2_conv2_w      (3, 32, 32)     block2_conv2_b      (32,)
activity_logits_w   (32, 6)         activity_logits_b   (6,)
input_mean          (9,)            input_std           (9,)
```

### 3.3 Quy tắc in số sang C (round-trip float32 không mất bit)

- `"{:.9g}f".format(float(np.float32(v)))`, ví dụ `-0.0123456791f`,
  `1.5e-09f`.
- Nếu chuỗi ra `inf`/`nan` → `export_c.py` raise lỗi, không được sinh ra
  file lỗi.
- 8 giá trị mỗi dòng, thụt lề 4 space, dấu phẩy sau mỗi dòng trừ dòng cuối
  của mảng.
- Comment shape phía trên mỗi mảng, ví dụ
  `/* shape (3, 9, 32) -> flatten [k][cin][cout] */`.

## 4. Cài đặt suy luận bằng C (`firmware/`)

### 4.1 API (`firmware/include/har_resnet.h`, viết tay)

```c
void har_normalize(const float *raw, float *normalized);
void har_resnet_predict(const float *x_norm, float *logits);
int  har_argmax(const float *values, int count);
void har_softmax(const float *logits, int count, float *probs);
const char *har_label_name(int class_id);

void har_conv1d_same(const float *in, int len, int cin,
                      const float *w, const float *b,
                      int ksize, int cout, float *out);
void har_relu(float *x, int count);
void har_maxpool1d_2(const float *in, int len, int ch, float *out);
void har_add(const float *a, const float *b, int count, float *out);
void har_global_max_pool(const float *in, int len, int ch, float *out);
void har_dense(const float *in, int din, const float *w, const float *b,
               int dout, float *out);
```

`firmware/src/har_resnet.c` cài đặt thuật toán **viết tay** (không sinh tự
động), có comment tiếng Việt giải thích từng bước. `har_resnet_params.{h,c}`
và `har_golden.{h,c}` **sinh tự động** bởi `src/export_c.py` (có dòng cảnh
báo "TỰ ĐỘNG SINH — KHÔNG SỬA TAY" ở đầu file).

### 4.2 Lịch trình buffer tĩnh (không malloc, không VLA)

```c
static float buf_a[128 * 32];   /* 16 KB */
static float buf_b[128 * 32];   /* 16 KB */
static float buf_c[64 * 32];    /*  8 KB */
static float buf_d[64 * 32];    /*  8 KB */
static float buf_e[64 * 32];    /*  8 KB */
```

Trình tự trong `har_resnet_predict` (13 bước, cố định theo kế hoạch):

```
1.  conv1d_same(x_norm, 128, 9,  block1_conv1_w/b,  3, 32) -> buf_a
2.  relu(buf_a, 128*32)
3.  conv1d_same(buf_a, 128, 32, block1_conv2_w/b,  3, 32) -> buf_b
4.  maxpool1d_2(buf_b, 128, 32) -> buf_c
5.  conv1d_same(x_norm, 128, 9, block1_projection_w/b, 1, 32) -> buf_a   (ghi đè, hợp lệ)
6.  maxpool1d_2(buf_a, 128, 32) -> buf_d
7.  add(buf_c, buf_d, 64*32) -> buf_c ; relu(buf_c)            // buf_c = x sau block1
8.  conv1d_same(buf_c, 64, 32, block2_conv1_w/b, 3, 32) -> buf_e
9.  relu(buf_e, 64*32)
10. conv1d_same(buf_e, 64, 32, block2_conv2_w/b, 3, 32) -> buf_d
11. add(buf_d, buf_c, 64*32) -> buf_c ; relu(buf_c)
12. global_max_pool(buf_c, 64, 32) -> pooled[32]  (mảng local, 128 byte)
13. dense(pooled, 32, activity_logits_w/b, 6) -> logits
```

### 4.3 RAM/độ phức tạp

**RAM tĩnh** (đo bằng `size firmware/build/test_inference`, khớp tính tay):

| Thành phần | Byte |
| --- | ---: |
| `buf_a` + `buf_b` (128×32 float ×2) | 32 768 |
| `buf_c` + `buf_d` + `buf_e` (64×32 float ×3) | 24 576 |
| **Tổng buffer trung gian (.bss)** | **57 344 (~56 KB)** |
| Input `x_norm` (128×9 float, do caller cấp) | 4 608 (~4,5 KB) |
| `pooled` (biến local trong hàm) | 128 |
| `logits` (do caller cấp) | 24 |
| **Tổng RAM hoạt động ước lượng** | **≈ 62 KB** |

ESP32 (dòng phổ biến) có ~320 KB DRAM nội bộ → RAM hoạt động của model này
chiếm khoảng 19% DRAM, khả thi khi port sang ESP-IDF.

Bảng trọng số 10 726 tham số × 4 byte ≈ **41,9 KB** nằm trong mảng `const
float` (`.rodata`) — trên ESP32 phần này nằm ở **flash**, không tốn DRAM.

**Số phép nhân-cộng (MAC)** mỗi cửa sổ suy luận (ước lượng theo kích thước
kernel đầy đủ, chưa trừ các vị trí bị zero-padding ở biên — số thực tế nhỏ
hơn một chút):

| Layer | Công thức `L_out × Cout × K × Cin` | MAC |
| --- | --- | ---: |
| `block1_conv1` | 128×32×3×9 | 110 592 |
| `block1_conv2` | 128×32×3×32 | 393 216 |
| `block1_projection` | 128×32×1×9 | 36 864 |
| `block2_conv1` | 64×32×3×32 | 196 608 |
| `block2_conv2` | 64×32×3×32 | 196 608 |
| `activity_logits` (dense) | 32×6 | 192 |
| **Tổng** | | **934 080 (~0,93 MMAC/cửa sổ)** |

### 4.4 Ràng buộc code C

- C99, biên dịch sạch: `cc -std=c99 -O2 -Wall -Wextra -Wpedantic -c
  src/har_resnet.c` → **0 warning** (đã kiểm tra thực tế, xem mục 5).
- Không `malloc`, không VLA, không biến toàn cục non-static ngoài mảng
  tham số sinh tự động (`har_*` trong `har_resnet_params.c`).
- `har_conv1d_same` giả định `in` và `out` **không trùng nhau** (không
  in-place) — ghi trong comment của hàm và header.
- Toàn bộ đường tính toán chính dùng `float`, không dùng `double`.

## 5. Build/chạy và kết quả đối chiếu thực tế

```bash
python main.py --check
python main.py export-c
cd firmware && make clean && make test
cd .. && python -m pytest -q
```

Kết quả thực tế đã đo trên máy dùng để triển khai (Python 3.10.12,
TensorFlow 2.16.1, NumPy 1.26.4, `cc`/gcc trên Linux):

- `src/reference_forward.py::forward_batch` so với `baseline_logits` của
  Keras trên 12 mẫu sanity: **max |diff| = 1.9073486e-06**, argmax khớp
  **12/12**. (Ngưỡng chặn theo kế hoạch: < 1e-4 — đạt rất xa ngưỡng.)
- `firmware/build/test_inference` (C, so với cùng 12 golden vector): tất cả
  12 mẫu có **argmax khớp Keras**, **max_abs_diff trên logits ≈ 1e-6**
  (< ngưỡng chấp nhận 1e-3), test `har_normalize` đạt (sai số ≤ 6e-8), in
  `PASS`, **exit code 0**.
- `cc -std=c99 -Wall -Wextra -Wpedantic -Iinclude -c src/har_resnet.c -o
  /dev/null`: **0 warning**.
- `size firmware/build/test_inference`: `bss = 57376` byte, khớp ước lượng
  56 KB buffer tĩnh ở mục 4.3 (chênh lệch nhỏ do các biến `bss` khác của
  runtime C).

Sai số C so với Python-NumPy-reference nhỏ hơn nhiều so với sai số giữa
NumPy-reference và Keras — vì cả hai (C và NumPy reference) cùng dùng
`float32` và cùng thứ tự cộng dồn theo pseudocode mục 2, còn Keras/TensorFlow
dùng kernel BLAS tối ưu với thứ tự cộng khác (nhưng vẫn nằm trong sai số
tích luỹ float32 chấp nhận được).

## 6. Giới hạn và hướng phát triển tiếp (INT8/ESP32)

Xem chi tiết tại mục 9 của `.bangiao/ke-hoach.md`. Tóm tắt:

1. **INT8 trong C**: chưa làm. TV2 đã tạo và đánh giá `.tflite` full-integer
   INT8 cùng `scale`/`zero_point`, nhưng firmware viết tay hiện vẫn là FP32.
   Nếu mở rộng, cần thêm conv int8 → int32 accumulator → requantize
   (multiplier + shift) và mở rộng `export_c.py` với `--dtype int8`.
2. **ESP-IDF component**: thêm `firmware/CMakeLists.txt`
   (`idf_component_register`), đo thời gian bằng `esp_timer_get_time()`.
   Trọng số `const` đã tự nằm ở `.rodata` → tự vào flash khi build ESP-IDF,
   không cần thay đổi cách khai báo.
3. **Đo accuracy toàn tập test (2947 cửa sổ) bằng C**: chưa làm, cần xuất
   `x_test` ra file nhị phân và viết `firmware/test/eval_testset.c`.
4. **Tối ưu tốc độ**: loop tiling, im2col, `-ffast-math`, SIMD của
   ESP32-S3 — ngoài phạm vi bản FP32 tham chiếu này.

## 7. Ghi chú tên gọi

Góp ý "Resnet16" của giảng viên trong kế hoạch này được hiểu là
**ResNetv1-6** — kiến trúc duy nhất tồn tại trong repo
(`configs/project.json["model"]["architecture"] =
"ResNetv1-6_1D_adapted_from_Figure_4"`), khớp Hình 4 và mục 6.1.1 của paper
Novac et al. (2021). Nếu ý là một kiến trúc ResNet 16 lớp khác, đó là đổi
kiến trúc + huấn luyện lại — nằm ngoài phạm vi công việc này vì sẽ làm sai
lệch baseline FP32 (accuracy 92,60%, macro-F1 0,9271) đã bàn giao và công
bố.

Mô hình **không có BatchNormalization** ở bất kỳ layer nào (xác nhận trực
tiếp trong `src/model.py`, và số tham số 10 726 chỉ khớp phép tính tay khi
không có BN — xem `.bangiao/ke-hoach.md` mục 1.1). Vì vậy code C trong
`firmware/` **không cài đặt op BatchNorm**; nếu về sau huấn luyện lại một
kiến trúc có BN, cần thêm bước BN riêng vào cả `reference_forward.py` lẫn
`har_resnet.c` — hiện tại là ngoài phạm vi để không phá baseline đã công bố.
