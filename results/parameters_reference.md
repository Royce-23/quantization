<!-- TỰ ĐỘNG SINH bởi src/describe_params.py — KHÔNG SỬA TAY. -->

# Tham số ResNetv1-6 1D — tài liệu chi tiết

## 1. Tiêu đề và nguồn

- Nguồn tham số: `models/params_fp32.npz`
- checkpoint_id: `aa323b3e55d1f904edbb05fb6d2a817b748b6e0ad26b7eb0166777c94535c74e`
- Tổng tham số: **10726**

## 2. Bảng tổng quan

| # | layer | khối | Nhánh | op | kernel shape | bias shape | params | % tổng |
| ---: | --- | --- | --- | --- | --- | --- | ---: | ---: |
| 1 | `block1_conv1` | Block 1 | A (main path) | Conv1D K=3, padding=same, stride=1, use_bias=True | (3, 9, 32) | (32,) | 896 | 8.35% |
| 2 | `block1_conv2` | Block 1 | A (main path) | Conv1D K=3, padding=same, stride=1, use_bias=True | (3, 32, 32) | (32,) | 3104 | 28.94% |
| 3 | `block1_projection` | Block 1 | B (projection shortcut) | Conv1D K=1, padding=valid, stride=1, use_bias=True | (1, 9, 32) | (32,) | 320 | 2.98% |
| 4 | `block2_conv1` | Block 2 | A (main path) | Conv1D K=3, padding=same, stride=1, use_bias=True | (3, 32, 32) | (32,) | 3104 | 28.94% |
| 5 | `block2_conv2` | Block 2 | A (main path) | Conv1D K=3, padding=same, stride=1, use_bias=True | (3, 32, 32) | (32,) | 3104 | 28.94% |
| 6 | `activity_logits` | Head (bộ phân loại) | — (không có nhánh, sau điểm hợp nhánh cuối cùng) | Dense (fully connected), use_bias=True, KHÔNG activation | (32, 6) | (6,) | 198 | 1.85% |
| | | | | | | **Tổng** | **10726** | **100.00%** |

## 3. Chi tiết từng layer

### block1_conv1

- **Vai trò**: Trích đặc trưng thời gian đầu tiên, nâng 9 kênh cảm biến lên 32 filter.
- **Nguồn input**: sensor_window (128, 9) đã z-score
- **Khối / Nhánh**: Block 1 / A (main path)
- **Phép toán**: Conv1D K=3, padding=same, stride=1, use_bias=True
- **Shape kernel**: `(3, 9, 32)` — ý nghĩa từng chiều:
  - K: vị trí trong kernel thời gian (3 bước, chỉ số 0..2)
  - Cin: kênh vào (9 kênh cảm biến: total_acc x/y/z, body_gyro x/y/z, body_acc x/y/z)
  - Cout: kênh ra (32 filter học được)
- **Shape bias**: `(32,)` — ý nghĩa từng chiều:
  - Cout: một bias cho mỗi filter
- **Số phần tử**: kernel = 864, bias = 32, tổng = 896
- **Công thức toán**: `y[t][co] = b[co] + SUM_k SUM_ci W[k][ci][co] * xtilde[t+k-1][ci]`
- **Tên biến C**: `har_block1_conv1_w / har_block1_conv1_b`
- **Layout C**: `w[(k*Cin + ci)*Cout + co]  (flatten C-order, KHÔNG transpose)`
- **Shape output**: `(128, 32)`
- **Thống kê kernel**: min=-0.766590, max=0.782176, mean=0.001405, std=0.250928, max|w|=0.782176, %0=0.000000
- **Thống kê bias**: min=-0.868884, max=0.586917, mean=-0.182203, std=0.339612, max|w|=0.868884, %0=0.000000

### block1_conv2

- **Vai trò**: Lớp Conv1D thứ hai của nhánh main Block 1, trước khi MaxPool/2.
- **Nguồn input**: output của block1_relu1 (128, 32) — ReLU(block1_conv1(x_norm))
- **Khối / Nhánh**: Block 1 / A (main path)
- **Phép toán**: Conv1D K=3, padding=same, stride=1, use_bias=True
- **Shape kernel**: `(3, 32, 32)` — ý nghĩa từng chiều:
  - K: vị trí trong kernel thời gian (3 bước, chỉ số 0..2)
  - Cin: kênh vào (32 filter của block1_conv1 sau ReLU)
  - Cout: kênh ra (32 filter)
- **Shape bias**: `(32,)` — ý nghĩa từng chiều:
  - Cout: một bias cho mỗi filter
- **Số phần tử**: kernel = 3072, bias = 32, tổng = 3104
- **Công thức toán**: `y[t][co] = b[co] + SUM_k SUM_ci W[k][ci][co] * xtilde[t+k-1][ci]`
- **Tên biến C**: `har_block1_conv2_w / har_block1_conv2_b`
- **Layout C**: `w[(k*Cin + ci)*Cout + co]  (flatten C-order, KHÔNG transpose)`
- **Shape output**: `(128, 32)`
- **Thống kê kernel**: min=-0.678215, max=0.460098, mean=-0.037635, std=0.142518, max|w|=0.678215, %0=0.000000
- **Thống kê bias**: min=-0.861934, max=0.670135, mean=0.070841, std=0.280552, max|w|=0.861934, %0=0.000000

### block1_projection

- **Vai trò**: Đổi số kênh 9 -> 32 cho nhánh shortcut (projection shortcut kiểu ResNet) để cộng được với nhánh main sau khi cả hai cùng MaxPool/2.
- **Nguồn input**: sensor_window (128, 9) đã z-score — CÙNG nguồn với nhánh A
- **Khối / Nhánh**: Block 1 / B (projection shortcut)
- **Phép toán**: Conv1D K=1, padding=valid, stride=1, use_bias=True
- **Shape kernel**: `(1, 9, 32)` — ý nghĩa từng chiều:
  - K: vị trí trong kernel thời gian (K=1, chỉ 1 bước — tương đương Conv1x1, không trộn thông tin giữa các bước thời gian)
  - Cin: kênh vào (9 kênh cảm biến)
  - Cout: kênh ra (32 filter, khớp số kênh của nhánh main để cộng được)
- **Shape bias**: `(32,)` — ý nghĩa từng chiều:
  - Cout: một bias cho mỗi filter
- **Số phần tử**: kernel = 288, bias = 32, tổng = 320
- **Công thức toán**: `p[t][co] = b[co] + SUM_ci W[0][ci][co] * x_norm[t][ci]`
- **Tên biến C**: `har_block1_projection_w / har_block1_projection_b`
- **Layout C**: `w[(k*Cin + ci)*Cout + co]  (flatten C-order, K=1 nên chỉ số k luôn 0)`
- **Shape output**: `(128, 32)`
- **Thống kê kernel**: min=-1.026587, max=1.240301, mean=0.016881, std=0.302107, max|w|=1.240301, %0=0.000000
- **Thống kê bias**: min=-0.861934, max=0.670135, mean=0.070841, std=0.280552, max|w|=0.861934, %0=0.000000

### block2_conv1

- **Vai trò**: Lớp Conv1D đầu tiên của nhánh main Block 2 (không hạ mẫu thời gian).
- **Nguồn input**: output của Block 1 — x1 (64, 32)
- **Khối / Nhánh**: Block 2 / A (main path)
- **Phép toán**: Conv1D K=3, padding=same, stride=1, use_bias=True
- **Shape kernel**: `(3, 32, 32)` — ý nghĩa từng chiều:
  - K: vị trí trong kernel thời gian (3 bước, chỉ số 0..2)
  - Cin: kênh vào (32, output Block 1)
  - Cout: kênh ra (32 filter)
- **Shape bias**: `(32,)` — ý nghĩa từng chiều:
  - Cout: một bias cho mỗi filter
- **Số phần tử**: kernel = 3072, bias = 32, tổng = 3104
- **Công thức toán**: `y[t][co] = b[co] + SUM_k SUM_ci W[k][ci][co] * xtilde[t+k-1][ci]`
- **Tên biến C**: `har_block2_conv1_w / har_block2_conv1_b`
- **Layout C**: `w[(k*Cin + ci)*Cout + co]  (flatten C-order, KHÔNG transpose)`
- **Shape output**: `(64, 32)`
- **Thống kê kernel**: min=-0.597979, max=0.574085, mean=-0.024372, std=0.143854, max|w|=0.597979, %0=0.000000
- **Thống kê bias**: min=-0.539987, max=0.551610, mean=-0.028938, std=0.289266, max|w|=0.551610, %0=0.000000

### block2_conv2

- **Vai trò**: Lớp Conv1D thứ hai của nhánh main Block 2, kết quả được cộng trực tiếp với nhánh identity (không có lớp weight nào ở nhánh B).
- **Nguồn input**: output của block2_relu1 (64, 32) — ReLU(block2_conv1(x1))
- **Khối / Nhánh**: Block 2 / A (main path)
- **Phép toán**: Conv1D K=3, padding=same, stride=1, use_bias=True
- **Shape kernel**: `(3, 32, 32)` — ý nghĩa từng chiều:
  - K: vị trí trong kernel thời gian (3 bước, chỉ số 0..2)
  - Cin: kênh vào (32 filter của block2_conv1 sau ReLU)
  - Cout: kênh ra (32 filter)
- **Shape bias**: `(32,)` — ý nghĩa từng chiều:
  - Cout: một bias cho mỗi filter
- **Số phần tử**: kernel = 3072, bias = 32, tổng = 3104
- **Công thức toán**: `y[t][co] = b[co] + SUM_k SUM_ci W[k][ci][co] * xtilde[t+k-1][ci]`
- **Tên biến C**: `har_block2_conv2_w / har_block2_conv2_b`
- **Layout C**: `w[(k*Cin + ci)*Cout + co]  (flatten C-order, KHÔNG transpose)`
- **Shape output**: `(64, 32)`
- **Thống kê kernel**: min=-0.575388, max=0.514672, mean=-0.025668, std=0.133464, max|w|=0.575388, %0=0.000000
- **Thống kê bias**: min=-0.541010, max=1.214255, mean=-0.041209, std=0.312437, max|w|=1.214255, %0=0.000000

### activity_logits

- **Vai trò**: Lớp phân loại cuối, xuất logits thô (softmax không nằm trong model).
- **Nguồn input**: global_max_pool(output Block 2) — vector (32,)
- **Khối / Nhánh**: Head (bộ phân loại) / — (không có nhánh, sau điểm hợp nhánh cuối cùng)
- **Phép toán**: Dense (fully connected), use_bias=True, KHÔNG activation
- **Shape kernel**: `(32, 6)` — ý nghĩa từng chiều:
  - Din: 32 đặc trưng sau GlobalMaxPooling1D
  - Dout: 6 lớp hoạt động (WALKING..LAYING)
- **Shape bias**: `(6,)` — ý nghĩa từng chiều:
  - Dout: một bias cho mỗi lớp
- **Số phần tử**: kernel = 192, bias = 6, tổng = 198
- **Công thức toán**: `z[o] = b[o] + SUM_i g[i] * W[i][o]`
- **Tên biến C**: `har_activity_logits_w / har_activity_logits_b`
- **Layout C**: `w[i*Dout + o]  (flatten C-order theo shape Keras (Din, Dout))`
- **Shape output**: `(6,)`
- **Thống kê kernel**: min=-1.368823, max=0.911315, mean=0.018618, std=0.329044, max|w|=1.368823, %0=0.000000
- **Thống kê bias**: min=-1.090670, max=1.319516, mean=-0.000001, std=1.022859, max|w|=1.319516, %0=0.000000

## 4. Thống kê số học (đã trình bày trong mục 3, kèm lại ở đây để dễ tra cứu và nối mạch sang phần INT8)

| layer | tensor | min | max | mean | std | max\|w\| | %0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `block1_conv1` | kernel | -0.766590 | 0.782176 | 0.001405 | 0.250928 | 0.782176 | 0.000000 |
| `block1_conv1` | bias | -0.868884 | 0.586917 | -0.182203 | 0.339612 | 0.868884 | 0.000000 |
| `block1_conv2` | kernel | -0.678215 | 0.460098 | -0.037635 | 0.142518 | 0.678215 | 0.000000 |
| `block1_conv2` | bias | -0.861934 | 0.670135 | 0.070841 | 0.280552 | 0.861934 | 0.000000 |
| `block1_projection` | kernel | -1.026587 | 1.240301 | 0.016881 | 0.302107 | 1.240301 | 0.000000 |
| `block1_projection` | bias | -0.861934 | 0.670135 | 0.070841 | 0.280552 | 0.861934 | 0.000000 |
| `block2_conv1` | kernel | -0.597979 | 0.574085 | -0.024372 | 0.143854 | 0.597979 | 0.000000 |
| `block2_conv1` | bias | -0.539987 | 0.551610 | -0.028938 | 0.289266 | 0.551610 | 0.000000 |
| `block2_conv2` | kernel | -0.575388 | 0.514672 | -0.025668 | 0.133464 | 0.575388 | 0.000000 |
| `block2_conv2` | bias | -0.541010 | 1.214255 | -0.041209 | 0.312437 | 1.214255 | 0.000000 |
| `activity_logits` | kernel | -1.368823 | 0.911315 | 0.018618 | 0.329044 | 1.368823 | 0.000000 |
| `activity_logits` | bias | -1.090670 | 1.319516 | -0.000001 | 1.022859 | 1.319516 | 0.000000 |

## 5. Chuẩn hoá đầu vào (z-score theo kênh)

- **Công thức**: `xn[t][c] = (xraw[t][c] - mu[c]) / sigma[c]`
- **Tên biến C**: `har_input_mean` (mean), `har_input_std` (std)

| # | kênh | mu[c] | sigma[c] |
| ---: | --- | ---: | ---: |
| 0 | `total_acc_x` | 0.805391 | 0.410322 |
| 1 | `total_acc_y` | 0.029006 | 0.399567 |
| 2 | `total_acc_z` | 0.080120 | 0.354005 |
| 3 | `body_gyro_x` | 0.004766 | 0.408725 |
| 4 | `body_gyro_y` | -0.002092 | 0.382862 |
| 5 | `body_gyro_z` | -0.000882 | 0.258296 |
| 6 | `body_acc_x` | -0.000794 | 0.197001 |
| 7 | `body_acc_y` | -0.000252 | 0.126466 |
| 8 | `body_acc_z` | -0.000306 | 0.108309 |

## 6. Ước lượng bộ nhớ tham số

- FP32 = 10726 × 4 B = 42904 B (41.90 KiB)
- INT8 giả định = 10726 × 1 B = 10726 B (10.47 KiB)
- Đây là phép tính lý thuyết trên số tham số, KHÁC với dung lượng file `.tflite` đo thực tế ở `docs/quantization.md` (51 728 -> 25 104 byte) vì file TFLite còn chứa metadata/flatbuffer.

## 7. Đối chiếu Keras <-> NumPy <-> C

| Khối | Layer Keras | Hàm NumPy (`src/reference_forward.py`) | Hàm C (`firmware/src/har_resnet.c`) |
| --- | --- | --- | --- |
| Block 1 | `block1_*` (5 layer) | `block1_forward()` | `har_block1_forward()` |
| Block 2 | `block2_*` (4 layer) | `block2_forward()` | `har_block2_forward()` |
| Head | `global_max_pool`, `activity_logits` | `head_forward()` | `har_head_forward()` |

