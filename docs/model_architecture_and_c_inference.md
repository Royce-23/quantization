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

### 1.2 Sơ đồ luồng NHIỀU NHÁNH (đánh số node N0–N15, nhánh A1/B1/A2/B2)

```text
                     xraw (128, 9)  [dữ liệu cảm biến thô]
                              |
                      [N0] har_normalize : z-score theo kênh
                              |
                     xn (128, 9)  ─────────────── điểm RẼ NHÁNH số 1
                        /                              \
        ┌──────────────┘                                └──────────────┐
        │  NHÁNH A1 (main path — Block 1)               NHÁNH B1 (shortcut — Block 1)
        │  xuất phát: xn                                 xuất phát: xn (CÙNG nguồn với A1)
        │                                                │
   [N1] block1_conv1   Conv1D K=3 same  (128,9)->(128,32)│
        │  công thức 2.3                                 │
   [N2] block1_relu1   ReLU             (128,32)         │
        │  công thức 2.4                            [N5] block1_projection
   [N3] block1_conv2   Conv1D K=3 same  (128,32)->(128,32)    Conv1D K=1 valid
        │  công thức 2.3                                 │    (128,9)->(128,32)
   [N4] block1_main_pool  MaxPool/2     (128,32)->(64,32)│    công thức 2.3 với p=0
        │  công thức 2.5                            [N6] block1_shortcut_pool
        │                                                │    MaxPool/2 ->(64,32)
        └──────────────┐                ┌────────────────┘    công thức 2.5
                       ▼                ▼
                  [N7] block1_add   ⊕  (cộng phần tử, công thức 2.6)
                       │
                  [N8] block1_output  ReLU        ──> x1 (64, 32)
                       │
                       ├──────────────────────────────── điểm RẼ NHÁNH số 2
                       │                                          \
        ┌──────────────┘                                           └──────────┐
        │  NHÁNH A2 (main path — Block 2)                 NHÁNH B2 (identity shortcut)
        │  xuất phát: x1                                   xuất phát: x1 (KHÔNG phép toán,
        │                                                  chỉ dẫn thẳng tensor sang ⊕)
   [N9] block2_conv1   Conv1D K=3 same  (64,32)->(64,32)          │
        │                                                         │
  [N10] block2_relu1   ReLU                                       │
        │                                                         │
  [N11] block2_conv2   Conv1D K=3 same  (64,32)->(64,32)          │
        │                                                         │
        └──────────────┐                         ┌────────────────┘
                       ▼                         ▼
                  [N12] block2_add   ⊕  (công thức 2.6)
                       │
                  [N13] block2_output ReLU       ──> x2 (64, 32)
                       │
                  [N14] global_max_pool  (64,32) -> (32,)   công thức 2.7
                       │
                  [N15] activity_logits  Dense(32->6)       công thức 2.8
                       │
                     z (6,)  logits thô
                       │
                     argmax                                  công thức 2.9
                       │
                  lớp 0..5  (WALKING ... LAYING)
```

**Ghi chú quan trọng**: mô hình **không có BatchNormalization** ở bất kỳ
đâu (xem mục 7). Conv1D dùng `use_bias=True` mặc định của Keras. Model
**không có stem conv riêng**: input đi thẳng vào Block 1 (không có bước xử
lý nào giữa `har_normalize` và `block1_conv1`/`block1_projection`).

### 1.3 Bảng mô tả nhánh

| Nhánh | Xuất phát từ | Đi qua node | Thuật toán trong nhánh | Hợp lại tại | Shape khi hợp |
| --- | --- | --- | --- | --- | --- |
| A1 (main, Block 1) | `xn` (128,9) | N1→N2→N3→N4 | Conv k3 → ReLU → Conv k3 → MaxPool/2 | `block1_add` [N7] | (64,32) |
| B1 (projection shortcut, Block 1) | `xn` (128,9) — **cùng nguồn A1** | N5→N6 | Conv k1 (đổi 9→32 kênh) → MaxPool/2 | `block1_add` [N7] | (64,32) |
| A2 (main, Block 2) | `x1` (64,32) | N9→N10→N11 | Conv k3 → ReLU → Conv k3 | `block2_add` [N12] | (64,32) |
| B2 (identity shortcut, Block 2) | `x1` (64,32) — **cùng nguồn A2** | (không node) | không phép toán, truyền thẳng | `block2_add` [N12] | (64,32) |

### 1.4 Vì sao B1 cần Conv k=1 còn B2 là identity

Nhánh chính (main) của Block 1 đổi số kênh 9 → 32 **và** độ dài trục thời
gian 128 → 64. Muốn cộng được (`block1_add`) thì nhánh shortcut phải khớp
cả 2 chiều với nhánh main, nên B1 bắt buộc phải có `Conv1D k=1` (đổi số
kênh 9 → 32) rồi `MaxPool/2` (đổi độ dài 128 → 64). Block 2 không đổi
shape giữa input và output nhánh main (vẫn (64, 32) → (64, 32)) nên B2 để
**identity** (không có lớp trọng số nào) — rẻ hơn về tham số/tính toán và
giữ đường gradient sạch (đặc trưng đúng tinh thần residual connection gốc
của ResNet).

### 1.5 Ánh xạ 3 tầng: Keras ↔ NumPy ↔ C

| Khối | Layer Keras | Hàm NumPy (`src/reference_forward.py`) | Hàm C (`firmware/src/har_resnet.c`) |
| --- | --- | --- | --- |
| Block 1 | `block1_*` (5 layer) | `block1_forward()` | `har_block1_forward()` |
| Block 2 | `block2_*` (4 layer) | `block2_forward()` | `har_block2_forward()` |
| Head | `global_max_pool`, `activity_logits` | `head_forward()` | `har_head_forward()` |

### 1.6 Bản đồ tra cứu từng node: công thức → thuật toán → code

Bảng này dùng khi trình bày sơ đồ: mỗi node N0–N15 không phải là một hộp
vẽ thêm cho đẹp mà đều truy ngược được về (i) công thức ở mục 2, (ii) layer
Keras thật trong `src/model.py`, và (iii) hàm NumPy/C thực thi cùng phép
tính. Tên hàm C không kèm số dòng để tài liệu không lỗi thời khi code được
chỉnh sửa; dùng tìm kiếm theo tên hàm trong file đã nêu.

| Node | Input → output | Công thức / thuật toán | Keras hoặc NumPy tham chiếu | C tương ứng | Tham số thật |
| --- | --- | --- | --- | --- | --- |
| N0 `Z-score` | `(128,9) → (128,9)` | Mục 2.1: `(x−μ)/σ` theo từng kênh | `prepare_data.py::split_and_normalize`; `reference_forward.py::normalize` | `har_normalize` | `har_input_mean[9]`, `har_input_std[9]` |
| N1 `block1_conv1` | `(128,9) → (128,32)` | Mục 2.3: Conv1D K=3, same, bias | `model.py::build_model`; `conv1d_same` | `har_conv1d_same`, gọi bởi `har_block1_branch_main` | `block1_conv1_w (3,9,32)`, `b (32,)`: 896 |
| N2 `block1_relu1` | `(128,32) → (128,32)` | Mục 2.4: `max(0,x)` | `relu` | `har_relu`, trong `har_block1_branch_main` | 0 |
| N3 `block1_conv2` | `(128,32) → (128,32)` | Mục 2.3: Conv1D K=3, same, bias | `conv1d_same` | `har_conv1d_same`, trong `har_block1_branch_main` | `block1_conv2_w (3,32,32)`, `b (32,)`: 3.104 |
| N4 `main_pool` | `(128,32) → (64,32)` | Mục 2.5: `max(x[2t],x[2t+1])` | `maxpool1d_2` | `har_maxpool1d_2`, trong `har_block1_branch_main` | 0 |
| N5 `projection` | `(128,9) → (128,32)` | Mục 2.3 với K=1, nên không padding; đổi kênh 9→32 | `conv1d_same` | `har_conv1d_same`, gọi bởi `har_block1_branch_shortcut` | `block1_projection_w (1,9,32)`, `b (32,)`: 320 |
| N6 `shortcut_pool` | `(128,32) → (64,32)` | Mục 2.5: MaxPool/2; làm shortcut khớp shape nhánh main | `maxpool1d_2` | `har_maxpool1d_2`, trong `har_block1_branch_shortcut` | 0 |
| N7 `block1_add` | hai `(64,32) → (64,32)` | Mục 2.6: `ReLU(F(x)+S(x))` | `add` rồi `relu` trong `block1_forward` | `har_residual_merge_relu` → `har_add` + `har_relu` | 0 |
| N8 `block1_output` | `(64,32) → (64,32)` | Chính là output ReLU của N7, không phải layer có thêm trọng số | giá trị trả về `block1_forward` | output `har_block1_forward` | 0 |
| N9 `block2_conv1` | `(64,32) → (64,32)` | Mục 2.3: Conv1D K=3, same, bias | `conv1d_same` | `har_conv1d_same`, trong `har_block2_branch_main` | `block2_conv1_w (3,32,32)`, `b (32,)`: 3.104 |
| N10 `block2_relu1` | `(64,32) → (64,32)` | Mục 2.4: `max(0,x)` | `relu` | `har_relu`, trong `har_block2_branch_main` | 0 |
| N11 `block2_conv2` | `(64,32) → (64,32)` | Mục 2.3: Conv1D K=3, same, bias | `conv1d_same` | `har_conv1d_same`, trong `har_block2_branch_main` | `block2_conv2_w (3,32,32)`, `b (32,)`: 3.104 |
| N12 `block2_add` | hai `(64,32) → (64,32)` | Mục 2.6: `ReLU(F(x)+x)`; shortcut là identity | `add` rồi `relu` trong `block2_forward` | `har_residual_merge_relu(s64_1, x64, ...)` | 0 |
| N13 `block2_output` | `(64,32) → (64,32)` | Chính là output ReLU của N12 | giá trị trả về `block2_forward` | output `har_block2_forward` | 0 |
| N14 `global_max_pool` | `(64,32) → (32,)` | Mục 2.7: lấy max theo 64 thời điểm, cho mỗi channel | `global_max_pool` | `har_global_max_pool`, trong `har_head_forward` | 0 |
| N15 `Dense + argmax` | `(32,) → (6,) → class` | Mục 2.8 Dense tạo logits; mục 2.9 chọn `argmax` | `dense`; `numpy.argmax` | `har_dense`, `har_argmax`, trong/sau `har_head_forward` | `activity_logits_w (32,6)`, `b (6,)`: 198 |

Tổng tham số có trọng số là `896 + 3.104 + 320 + 3.104 + 3.104 + 198 =
10.726`; ReLU, Pool, Add, identity, GlobalMaxPool và argmax là các phép
toán không có tham số học.

### 1.7 Bản đồ các node PTQ: công thức → code → phạm vi C

Các node sau `FP32 classifier` trong sơ đồ là **pipeline thí nghiệm sau
huấn luyện**, không phải các layer của ResNet. Do đó chúng chạy bằng
Python/TensorFlow Lite, không có hàm C99 tương ứng trong `firmware/`. Đây là
phân biệt bắt buộc khi trình bày để không tuyên bố sai rằng firmware hiện đã
chạy INT8.

| Node PTQ | Thuật toán / công thức | Mã triển khai | Có C99 trong `firmware/`? |
| --- | --- | --- | --- |
| Chọn mẫu calibration | lấy `n` chỉ số không lặp từ `x_train` với seed; `n ∈ {50,200,500}` | `src/calibration.py::select_calibration` | Không — bước chuẩn bị thí nghiệm |
| Representative dataset | đưa các tensor train đã Z-score qua converter để quan sát miền activation | `src/quantize.py::_representative_dataset` | Không — do TFLite Converter thực hiện |
| Quantize affine | Mục 2.12: `q=clip(round(r/S+Z),-128,127)` | cấu hình `TFLiteConverter` trong `src/quantize.py::run` | Không — tạo `.tflite`; C hiện chỉ FP32 |
| Kiểm tra full-integer | xác minh input/output INT8 và không có tensor float fallback | `src/quantize.py::_inspect_tflite` | Không |
| INT8 inference để đo | lượng tử hóa input, chạy `Interpreter`, giải lượng tử logits | `src/evaluate.py::infer_tflite` | Không — interpreter TFLite trên Python |
| Chọn cấu hình | gom 3 seed theo calibration size, ưu tiên mean accuracy rồi macro-F1 | `src/evaluate.py::_select_calibration_size` | Không — logic đánh giá |
| Test/report | accuracy, macro-F1, bytes, confusion matrix trên test chỉ sau khi chọn | `src/evaluate.py::run` | Không — đánh giá thí nghiệm |

Nếu mở rộng sau này sang firmware INT8, cần cài riêng Conv INT8 tích luỹ
`int32`, requantize theo `scale/zero_point` (mục 2.12), và export trọng số
INT8. Việc đó được ghi rõ là hướng phát triển, không thuộc firmware FP32
hiện tại.

### 1.8 Sơ đồ trực quan

Bản vẽ matplotlib có nhánh thật (2 điểm rẽ nhánh, 2 node ⊕, nhãn "Nhánh A
(main)"/"Nhánh B (shortcut)"/"Nhánh B (identity)") tại
[`results/figures/pipeline/algorithm_flow.png`](../results/figures/pipeline/algorithm_flow.png)
(bản SVG cùng thư mục để phóng to), sinh bởi `python main.py visualize-pipeline`
(`src/visualize_pipeline.py`).

## 2. Công thức toán của từng phép toán

Ký hiệu chung:

```text
L      : độ dài trục thời gian (128 hoặc 64)
Cin    : số kênh vào ; Cout : số kênh ra ; K : kích thước kernel (lẻ)
p      = floor(K / 2)                       (số ô đệm mỗi bên, same-padding)
x[t][c]    : activation, layout [time][channel], phẳng -> x_flat[t*Cin + c]
W[k][ci][co] : kernel Conv1D, phẳng -> w_flat[(k*Cin + ci)*Cout + co]
b[co]      : bias, phẳng -> b_flat[co]
```

### 2.1 Chuẩn hoá z-score theo kênh (tiền xử lý, ngoài model Keras)

```text
xn[t][c] = ( xraw[t][c] - mu[c] ) / sigma[c] ,   t = 0..127 ,  c = 0..8
```
`mu`, `sigma` fit **chỉ trên tập train**, lưu ở `data/processed/preprocess.json`,
export sang `har_input_mean[9]`, `har_input_std[9]`.

### 2.2 Zero-padding (định nghĩa phụ, dùng cho 2.3)

```text
xtilde[s][c] = x[s][c]   nếu 0 <= s < L
xtilde[s][c] = 0         nếu s < 0  hoặc  s >= L
```

### 2.3 Conv1D "same", stride 1, K lẻ, có bias

```text
y[t][co] = b[co] + SUM_{k=0}^{K-1} SUM_{ci=0}^{Cin-1} W[k][ci][co] * xtilde[t + k - p][ci]

với t = 0 .. L-1 ,  co = 0 .. Cout-1 ,  p = floor(K/2)
L_out = L      (same-padding, stride 1, K lẻ  =>  giữ nguyên độ dài)
```

**Giải thích same-padding TensorFlow:** với stride 1 và K **lẻ**, TF đệm đối xứng
`pad_left = pad_right = (K-1)/2 = floor(K/2)`, nên `L_out = L`. Trường hợp `K = 1`
cho `p = 0` → không đệm gì → **trùng khít với `padding="valid"`**; vì vậy layer
`block1_projection` (k=1, valid) dùng chung đúng công thức này.

Dạng đã thay chỉ số phẳng (chính xác như code C):
```text
out[t*Cout + co] = b[co] + SUM_k SUM_ci  in[(t+k-p)*Cin + ci] * w[(k*Cin + ci)*Cout + co]
                   (bỏ qua các số hạng có  t+k-p < 0  hoặc  t+k-p >= L)
```

Kiểu tích luỹ: **float 32-bit**, cộng dồn theo thứ tự `k` ngoài, `ci` trong
(khớp chính xác `firmware/src/har_resnet.c::har_conv1d_same` và
`src/reference_forward.py::conv1d_same` — không dùng `double`).

### 2.4 ReLU

```text
y[i] = max(0, x[i])
```

### 2.5 MaxPool1D, pool = 2, stride = 2, padding "valid"

```text
y[t][c] = max( x[2t][c] , x[2t+1][c] ) ,   t = 0 .. floor(L/2) - 1
L_out = floor(L / 2)        (ở đây L luôn chẵn: 128 -> 64)
```

### 2.6 Hợp nhánh residual (Add + ReLU)

```text
y[t][c] = max( 0 , F[t][c] + S[t][c] )

F = output nhánh chính (main path)
S = output nhánh tắt (shortcut):
      Block 1: S = MaxPool2( Conv1D_{K=1}( xn ) )      (projection shortcut)
      Block 2: S = x                                    (identity shortcut)
```

### 2.7 GlobalMaxPooling1D (max theo trục thời gian)

```text
g[c] = max_{0 <= t < L} x[t][c] ,   c = 0 .. C-1
(L, C) -> (C,)      ở đây: (64, 32) -> (32,)
```

### 2.8 Dense (fully connected), không activation

```text
z[o] = b[o] + SUM_{i=0}^{Din-1} g[i] * W[i][o] ,   o = 0 .. Dout-1
phẳng: out[o] = b[o] + SUM_i in[i] * w[i*Dout + o]
(32,) -> (6,)   — đây là LOGITS THÔ
```

### 2.9 Dự đoán và softmax (softmax KHÔNG nằm trong model)

```text
yhat = argmax_{o} z[o]

(chỉ cho demo/UI, ổn định số:)
m    = max_j z[j]
p[o] = exp( z[o] - m ) / SUM_j exp( z[j] - m )
```
Huấn luyện dùng `CategoricalCrossentropy(from_logits=True)` nên softmax nằm ngoài
đồ thị model; `argmax(z) == argmax(softmax(z))` nên C chỉ cần `har_argmax`.

### 2.10 Công thức đếm tham số

```text
Conv1D : n_param = K * Cin * Cout + Cout
Dense  : n_param = Din * Dout + Dout
```
| Layer | Thay số | Tham số |
| --- | --- | ---: |
| `block1_conv1` | 3·9·32 + 32 | 896 |
| `block1_conv2` | 3·32·32 + 32 | 3 104 |
| `block1_projection` | 1·9·32 + 32 | 320 |
| `block2_conv1` | 3·32·32 + 32 | 3 104 |
| `block2_conv2` | 3·32·32 + 32 | 3 104 |
| `activity_logits` | 32·6 + 6 | 198 |
| **Tổng** | | **10 726** |

### 2.11 Công thức đếm phép nhân–cộng (MAC) — chính xác và chặn trên

Chặn trên đơn giản `L·Cout·K·Cin` không trừ phần zero-padding ở biên nên
**cao hơn thực tế**. Công thức chính xác trừ đúng số tap bị cắt ở biên:

```text
số tap hợp lệ  = K*L - p*(p+1)          (stride 1, same-padding, K lẻ, p = floor(K/2))
MAC_chính_xác  = ( K*L - p*(p+1) ) * Cin * Cout
MAC_chặn_trên  =        K*L         * Cin * Cout
Số phép cộng bias = L * Cout
```
Kiểm tra: K=3 → p=1 → tap = 3L−2. K=1 → p=0 → tap = L (không có biên bị cắt). ✔

| Layer | Chặn trên | Chính xác (trừ zero-pad) |
| --- | ---: | ---: |
| `block1_conv1` (K=3, L=128, 9→32) | 110 592 | (3·128−2)·9·32 = **110 016** |
| `block1_conv2` (K=3, L=128, 32→32) | 393 216 | 382·1024 = **391 168** |
| `block1_projection` (K=1, L=128, 9→32) | 36 864 | 128·9·32 = **36 864** |
| `block2_conv1` (K=3, L=64, 32→32) | 196 608 | (3·64−2)·1024 = **194 560** |
| `block2_conv2` (K=3, L=64, 32→32) | 196 608 | **194 560** |
| `activity_logits` | 192 | **192** |
| **Tổng** | **934 080** | **927 360** |

Con số **927 360** (≈0,93 MMAC/cửa sổ) là số phép nhân–cộng **chính xác**
mỗi lần suy luận (1 cửa sổ 128×9); **934 080** chỉ là chặn trên lý thuyết
nếu không trừ các vị trí kernel rơi ra ngoài biên chuỗi thời gian.

### 2.12 Công thức lượng tử hoá affine INT8 (CHỈ MÔ TẢ — CHƯA triển khai trong `firmware/`)

Đưa vào đây để nối mạch với `docs/quantization.md`; nhắc lại rõ: đây **chỉ
là công thức tham khảo**, `firmware/` hiện chỉ có suy luận FP32 (xem mục 6.1).

```text
Lượng tử hoá   : q = clip( round( r / S + Z ), -128, 127 )
Giải lượng tử  : r_xapxi = S * ( q - Z )
S = scale (float32 > 0) ,  Z = zero_point (int8)

Conv INT8 (dạng tổng quát của TFLite, per-channel cho weight, Zw = 0):
  acc_int32[t][co] = SUM_k SUM_ci ( q_x[t+k-p][ci] - Zx ) * q_w[k][ci][co]  + q_b[co]
  q_y[t][co] = clip( Z_y + round( M[co] * acc_int32[t][co] ), -128, 127 )
  với M[co] = (Sx * Sw[co]) / Sy , hiện thực bằng (multiplier int32, shift) thay cho float.
```

Lưu ý: khối INT8 ở trên tích luỹ bằng `int32` (`acc_int32`), KHÁC với op FP32
ở mục 2.1–2.9 (tích luỹ `float` 32-bit, xem câu cuối mục 2.3) — hai kiểu
tích luỹ này không dùng chung, không nhầm lẫn.

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

Lệnh `python main.py describe-params` (cài đặt tại `src/describe_params.py`,
chạy sau `export-c`) đọc `models/params_fp32.npz` và sinh báo cáo tham số
CHI TIẾT (vai trò, nguồn input, ý nghĩa từng chiều shape, công thức toán,
tên biến C, layout, thống kê min/max/mean/std từng tensor) tại
[`results/parameters_reference.md`](../results/parameters_reference.md).
File này bổ sung cho `models/params_manifest.json` (vốn chỉ có shape/số
lượng, không có ngữ nghĩa) — dùng khi cần tra cứu nhanh ý nghĩa một tham số
cụ thể mà không phải đọc lại `src/model.py`.

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

/* API cấp KHỐI — mỗi khối kiến trúc một hàm riêng (Y1: "phần thuật toán ở
 * mỗi khối viết bằng C"). har_resnet_predict() chỉ còn 3 lời gọi liên tiếp
 * của 3 hàm dưới đây, dùng chung 5 buffer static ở mục 4.2. */
void har_block1_forward(const float *x_norm, float *y64,
                        float *s128_0, float *s128_1, float *s64_0);
void har_block2_forward(const float *x64, float *y64,
                        float *s64_0, float *s64_1);
void har_head_forward(const float *x64, float *logits);

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

Mỗi hàm khối (`har_block1_forward`, `har_block2_forward`, `har_head_forward`,
kể cả các hàm `static` nội bộ theo nhánh) có block comment 5 mục **NGUỒN
INPUT / CẤU TRÚC / CÔNG THỨC / SHAPE / THAM SỐ** ngay phía trên định nghĩa —
đối chiếu trực tiếp với sơ đồ nhánh ở mục 1.2 và công thức ở mục 2.

CẢNH BÁO: **Không thread-safe**: `har_resnet_predict` (và các hàm khối khi gọi qua
nó) dùng chung 5 buffer `static` ở mục 4.2 — chỉ được có tối đa 1 lời gọi
đang chạy tại một thời điểm, không gọi từ nhiều task/thread hay ISR cùng lúc
(chi tiết xem comment trong `firmware/include/har_resnet.h`).

### 4.2 Cấu trúc theo khối + lịch trình buffer tĩnh (không malloc, không VLA)

```c
static float buf_a[128 * 32];   /* 16 KB */
static float buf_b[128 * 32];   /* 16 KB */
static float buf_c[64 * 32];    /*  8 KB */
static float buf_d[64 * 32];    /*  8 KB */
static float buf_e[64 * 32];    /*  8 KB */
```

`har_resnet_predict` — sau tái cấu trúc: chỉ còn 3 lời gọi khối, dùng lại
đúng 5 buffer trên (không thêm/bớt, `.bss` không đổi):

```c
void har_resnet_predict(const float *x_norm, float *logits) {
    /* Khối 1: (128,9) -> (64,32). scratch: buf_a, buf_b, buf_d. */
    har_block1_forward(x_norm, buf_c, buf_a, buf_b, buf_d);
    /* Khối 2: (64,32) -> (64,32), ghi đè tại chỗ lên buf_c. scratch: buf_e, buf_d. */
    har_block2_forward(buf_c, buf_c, buf_e, buf_d);
    /* Head: (64,32) -> logits(6). */
    har_head_forward(buf_c, logits);
}
```

Bên trong từng hàm khối (thứ tự op và tham số giữ **nguyên xi** so với bản
trước khi tách hàm — kiểm chứng bit-exact ở mục 5):

```
har_block1_branch_main(x_norm, s128_0, s128_1, out64):      NHÁNH A1
    1. conv1d_same(x_norm, 128, 9,  block1_conv1_w/b, K=3, 32) -> s128_0
    2. relu(s128_0, 128*32)
    3. conv1d_same(s128_0, 128, 32, block1_conv2_w/b, K=3, 32) -> s128_1
    4. maxpool1d_2(s128_1, 128, 32) -> out64

har_block1_branch_shortcut(x_norm, s128_0, out64):           NHÁNH B1
    5. conv1d_same(x_norm, 128, 9, block1_projection_w/b, K=1, 32) -> s128_0
    6. maxpool1d_2(s128_0, 128, 32) -> out64

har_residual_merge_relu(main, shortcut, count, out):         HỢP NHÁNH (⊕)
    add(main, shortcut, count, out); relu(out, count)

har_block1_forward = branch_main(-> y64) ; branch_shortcut(-> s64_0) ;
                     residual_merge_relu(y64, s64_0 -> y64)

har_block2_branch_main(x64, s64_0, out64):                   NHÁNH A2
    8.  conv1d_same(x64,   64, 32, block2_conv1_w/b, K=3, 32) -> s64_0
    9.  relu(s64_0, 64*32)
    10. conv1d_same(s64_0, 64, 32, block2_conv2_w/b, K=3, 32) -> out64

har_block2_forward = branch_main(-> s64_1) ;                  NHÁNH B2 = identity (x64)
                     residual_merge_relu(s64_1, x64 -> y64)

har_head_forward(x64, logits):
    12. global_max_pool(x64, 64, 32) -> pooled[32]  (mảng local, 128 byte)
    13. dense(pooled, 32, activity_logits_w/b, 6) -> logits
```

Bảng ánh xạ buffer khi `har_resnet_predict` gọi các hàm khối:

| Lời gọi | x_norm/x64 (input) | y64 (output) | scratch |
| --- | --- | --- | --- |
| `har_block1_forward` | `x_norm` (do caller cấp) | `buf_c` | `buf_a`, `buf_b`, `buf_d` |
| `har_block2_forward` | `buf_c` | `buf_c` (ghi đè tại chỗ) | `buf_e`, `buf_d` |
| `har_head_forward` | `buf_c` | `logits` (do caller cấp) | — |

### 4.3 RAM/độ phức tạp

**RAM tĩnh của THƯ VIỆN suy luận** (đây là con số dùng để tính RAM thực tế
trên ESP32 — production firmware KHÔNG link `test/test_inference.c`, chỉ
link `har_resnet.c` + `har_resnet_params.c`). Đo bằng cách biên dịch
`har_resnet.c` thành object file riêng, tách khỏi mọi buffer chỉ phục vụ
test:

```bash
cd firmware && cc -std=c99 -O2 -Wall -Wextra -Wpedantic -Iinclude \
  -c src/har_resnet.c -o /tmp/har_resnet_lib.o
size /tmp/har_resnet_lib.o
```

| Thành phần | Byte |
| --- | ---: |
| `buf_a` + `buf_b` (128×32 float ×2) | 32 768 |
| `buf_c` + `buf_d` + `buf_e` (64×32 float ×3) | 24 576 |
| **Tổng buffer trung gian (.bss) — chỉ thư viện** | **57 344 (~56 KB)** |
| Input `x_norm` (128×9 float, do caller cấp) | 4 608 (~4,5 KB) |
| `pooled` (biến local trong hàm) | 128 |
| `logits` (do caller cấp) | 24 |
| **Tổng RAM hoạt động ước lượng (production, không có test)** | **≈ 62 KB** |

**Lưu ý về `.bss` của binary test `firmware/build/test_inference`:** binary
này KHÔNG dùng để ước lượng RAM ESP32, vì `firmware/test/test_inference.c`
cấp thêm **5 buffer `static` RIÊNG** (57 344 byte, độc lập với 5 buffer của
thư viện ở trên) chỉ để chạy `check_block_decomposition()` — hàm này cố
tình gọi `har_block1_forward → har_block2_forward → har_head_forward` bằng
buffer riêng của test, tách biệt khỏi buffer nội bộ `har_resnet.c`, để so
sánh `==` tuyệt đối với `har_resnet_predict()`. Vì vậy `size
firmware/build/test_inference` cho ra **`.bss = 114 720` byte** (= 57 344
thư viện + 57 344 buffer test-only), gần gấp đôi con số 57 344 ở trên — đây
là chi phí RAM CỦA RIÊNG BỘ TEST TRÊN PC, KHÔNG PHẢI chi phí RAM của thư
viện suy luận khi tích hợp vào firmware ESP32 thật (không biên dịch cùng
`test/test_inference.c`). Xem số đo thực tế và lệnh kiểm chứng ở mục 5.

ESP32 (dòng phổ biến) có ~320 KB DRAM nội bộ → RAM hoạt động của model này
chiếm khoảng 19% DRAM, khả thi khi port sang ESP-IDF.

Bảng trọng số 10 726 tham số × 4 byte ≈ **41,9 KB** nằm trong mảng `const
float` (`.rodata`) — trên ESP32 phần này nằm ở **flash**, không tốn DRAM.

**Số phép nhân-cộng (MAC)** mỗi cửa sổ suy luận — xem công thức đầy đủ và
giải thích tại mục 2.11. Nhắc lại bảng 2 cột (chặn trên vs chính xác đã trừ
zero-padding ở biên):

| Layer | Chặn trên | Chính xác (trừ zero-pad) |
| --- | ---: | ---: |
| `block1_conv1` (K=3, L=128, 9→32) | 110 592 | 110 016 |
| `block1_conv2` (K=3, L=128, 32→32) | 393 216 | 391 168 |
| `block1_projection` (K=1, L=128, 9→32) | 36 864 | 36 864 |
| `block2_conv1` (K=3, L=64, 32→32) | 196 608 | 194 560 |
| `block2_conv2` (K=3, L=64, 32→32) | 196 608 | 194 560 |
| `activity_logits` (dense) | 192 | 192 |
| **Tổng** | **934 080** | **927 360 (~0,93 MMAC/cửa sổ)** |

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
  **12/12**. (Ngưỡng chặn dùng trong `tests/test_c_export.py`: < 1e-4 — đạt rất xa ngưỡng.)
- `firmware/build/test_inference` (C, so với cùng 12 golden vector): tất cả
  12 mẫu có **argmax khớp Keras**, **max_abs_diff trên logits ≈ 1e-6**
  (< ngưỡng chấp nhận 1e-3), test `har_normalize` đạt (sai số ≤ 6e-8),
  `check_block_decomposition` (`har_block1_forward` → `har_block2_forward` →
  `har_head_forward` so với `har_resnet_predict` nguyên khối, so sánh `==`
  tuyệt đối) đạt **OK**, in `PASS`, **exit code 0**.
- `cc -std=c99 -Wall -Wextra -Wpedantic -Iinclude -c src/har_resnet.c -o
  /dev/null`: **0 warning**.
- RAM thực tế của **thư viện** (con số dùng cho ESP32, xem giải thích đầy đủ
  ở mục 4.3): `cc -std=c99 -O2 -Wall -Wextra -Wpedantic -Iinclude -c
  src/har_resnet.c -o /tmp/har_resnet_lib.o && size /tmp/har_resnet_lib.o`
  → `bss = 57 344` byte, khớp đúng ước lượng 56 KB buffer tĩnh ở mục 4.3.
- `size firmware/build/test_inference` (binary TEST đầy đủ, có thêm 5
  buffer test-only của `check_block_decomposition`): `bss = 114 720` byte
  (= 57 344 thư viện + 57 344 buffer riêng của `test/test_inference.c`) —
  **đây KHÔNG phải chi phí RAM của thư viện suy luận**, chỉ là `.bss` của
  chương trình test trên PC; xem giải thích ở mục 4.3.

Sai số C so với Python-NumPy-reference nhỏ hơn nhiều so với sai số giữa
NumPy-reference và Keras — vì cả hai (C và NumPy reference) cùng dùng
`float32` và cùng thứ tự cộng dồn theo công thức mục 2, còn Keras/TensorFlow
dùng kernel BLAS tối ưu với thứ tự cộng khác (nhưng vẫn nằm trong sai số
tích luỹ float32 chấp nhận được).

## 6. Giới hạn và hướng phát triển tiếp (INT8/ESP32)

### 6.1 Công thức lượng tử hoá INT8 — CHƯA triển khai trong `firmware/`

Công thức affine đầy đủ (lượng tử hoá/giải lượng tử, conv INT8 dạng TFLite
per-channel) đã trình bày ở mục 2.12. Nhắc lại: đây **chỉ là mô tả toán
học**, `firmware/` hiện chỉ có suy luận FP32 (xem tóm tắt bên dưới).

Tóm tắt các hướng mở rộng còn lại:

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
không có BN — xem bảng tham số ở mục 1.1 và mục 2.10). Vì vậy code C trong
`firmware/` **không cài đặt op BatchNorm**; nếu về sau huấn luyện lại một
kiến trúc có BN, cần thêm bước BN riêng vào cả `reference_forward.py` lẫn
`har_resnet.c` — hiện tại là ngoài phạm vi để không phá baseline đã công bố.
