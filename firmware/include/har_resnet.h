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

/* Chạy toàn bộ forward pass ResNetv1-6.
 * x_norm: mảng phẳng đã chuẩn hoá, độ dài 128*9, layout [time][channel].
 * logits: mảng ra, độ dài HAR_NUM_CLASSES (logits thô, KHÔNG softmax). */
void har_resnet_predict(const float *x_norm, float *logits);

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
