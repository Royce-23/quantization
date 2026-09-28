/* API công khai của bộ suy luận (inference) ResNetv1-6 1D cho UCI-HAR.
 *
 * File này viết TAY (không sinh tự động). Cài đặt tương ứng nằm ở
 * src/har_resnet.c. Các hằng số kích thước và tham số trọng số nằm ở
 * har_resnet_params.h / har_resnet_params.c (SINH TỰ ĐỘNG).
 */
#ifndef HAR_RESNET_H
#define HAR_RESNET_H

#include "har_resnet_params.h"

/* ---- API cấp cao ------------------------------------------------------ */

/* Chuẩn hoá z-score theo kênh: normalized[t*9+c] = (raw[t*9+c]-mean[c])/std[c].
 * raw, normalized: mảng phẳng độ dài HAR_INPUT_TIMESTEPS*HAR_INPUT_CHANNELS.
 * Cho phép raw == normalized (ghi đè tại chỗ). */
void har_normalize(const float *raw, float *normalized);

/* CẢNH BÁO: KHÔNG THREAD-SAFE / KHÔNG REENTRANT.
 * har_resnet_predict (và har_block1_forward/har_block2_forward khi được gọi qua nó)
 * dùng chung các buffer trung gian `static` cấp phát tĩnh trong har_resnet.c
 * (~56 KB .bss) để tránh malloc/VLA khi port sang ESP32. Hệ quả:
 *   - chỉ được có TỐI ĐA MỘT lời gọi đang chạy tại một thời điểm;
 *   - không gọi từ nhiều task/thread FreeRTOS cùng lúc;
 *   - không gọi từ ngữ cảnh ISR hoặc lồng nhau (re-entrant).
 * Nếu cần chạy song song: bọc bằng mutex, hoặc sửa để caller tự cấp buffer
 * (các hàm har_block*_forward đã nhận scratch buffer qua tham số nên đủ điều kiện). */

/* Chạy toàn bộ forward pass ResNetv1-6.
 * x_norm: mảng phẳng đã chuẩn hoá, độ dài 128*9, layout [time][channel].
 * logits: mảng ra, độ dài HAR_NUM_CLASSES (logits thô, KHÔNG softmax). */
void har_resnet_predict(const float *x_norm, float *logits);

/* ---- API cấp KHỐI (mỗi khối kiến trúc một hàm riêng) ------------------
 * Dùng để test/đối chiếu từng khối độc lập, hoặc để tái sử dụng khi ghép
 * pipeline khác. har_resnet_predict() ở trên chỉ là 3 lời gọi liên tiếp
 * của các hàm dưới đây, dùng chung 5 buffer static hiện có.
 * Xem docs/model_architecture_and_c_inference.md (mục 2 — công thức toán,
 * mục 4 — cài đặt C) để đối chiếu công thức/sơ đồ nhánh với các hàm này. */

/* Block 1 — residual block projection shortcut + hạ mẫu trục thời gian 128 -> 64.
 * x_norm : (128, 9) phẳng [time][channel], ĐÃ z-score.
 * y64    : (64, 32) phẳng — output khối, do caller cấp.
 * s128_0, s128_1 : 2 buffer scratch (128*32 float) do caller cấp.
 * s64_0          : 1 buffer scratch (64*32 float) do caller cấp.
 * Yêu cầu: 5 con trỏ trên KHÔNG được trùng nhau. */
void har_block1_forward(const float *x_norm, float *y64,
                        float *s128_0, float *s128_1, float *s64_0);

/* Block 2 — residual block identity shortcut, giữ nguyên (64, 32).
 * x64 : (64, 32) phẳng — output của Block 1.
 * y64 : (64, 32) phẳng — output khối. CHO PHÉP y64 == x64 (ghi đè tại chỗ,
 *       an toàn vì phép hợp nhánh là element-wise).
 * s64_0, s64_1 : 2 buffer scratch (64*32 float), phải khác x64 và khác nhau. */
void har_block2_forward(const float *x64, float *y64,
                        float *s64_0, float *s64_1);

/* Head — GlobalMaxPooling1D + Dense(32 -> 6), KHÔNG softmax.
 * x64 : (64, 32) phẳng — output Block 2.
 * logits : (HAR_NUM_CLASSES,) do caller cấp. */
void har_head_forward(const float *x64, float *logits);

/* Chỉ số phần tử lớn nhất trong values[0..count-1]. */
int har_argmax(const float *values, int count);

/* Softmax ổn định số (trừ max trước khi exp) — chỉ dùng cho demo/UI,
 * KHÔNG dùng trong har_resnet_predict (model chỉ xuất logits thô). */
void har_softmax(const float *logits, int count, float *probs);

/* Tên nhãn ứng với class_id, hoặc NULL nếu class_id ngoài [0, HAR_NUM_CLASSES). */
const char *har_label_name(int class_id);

/* ---- Các phép toán cơ bản (public để test riêng từng op) -------------- */

/* Conv1D "same", stride 1, kernel lẻ ksize (ksize=1 => pad=0, dùng được cho
 * lớp block1_projection kiểu "valid").
 * in:  (len, cin) phẳng, layout [time][channel]
 * w:   (ksize, cin, cout) phẳng, layout [k][cin][cout] (đúng thứ tự Keras)
 * b:   (cout,)
 * out: (len, cout) phẳng
 * GIẢ ĐỊNH: con trỏ in và out KHÔNG được trùng nhau (không in-place). */
void har_conv1d_same(const float *in, int len, int cin,
                      const float *w, const float *b,
                      int ksize, int cout, float *out);

/* ReLU tại chỗ: x[i] = max(x[i], 0). */
void har_relu(float *x, int count);

/* MaxPool1D pool=2, stride=2, padding valid. len phải chẵn.
 * in: (len, ch) phẳng -> out: (len/2, ch) phẳng. */
void har_maxpool1d_2(const float *in, int len, int ch, float *out);

/* out[i] = a[i] + b[i], i = 0..count-1. */
void har_add(const float *a, const float *b, int count, float *out);

/* GlobalMaxPooling1D: max theo trục thời gian.
 * in: (len, ch) phẳng -> out: (ch,). */
void har_global_max_pool(const float *in, int len, int ch, float *out);

/* Dense: out[o] = bias[o] + sum_i in[i] * w[i*dout + o].
 * in: (din,), w: (din, dout) phẳng, b: (dout,), out: (dout,). */
void har_dense(const float *in, int din, const float *w, const float *b,
               int dout, float *out);

#endif /* HAR_RESNET_H */
