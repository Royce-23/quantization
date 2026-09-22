/* Cài đặt thuật toán forward pass ResNetv1-6 1D — VIẾT TAY (không sinh tự
 * động). Đối chiếu với src/reference_forward.py (NumPy) và mục 5.2/5.4 của
 * .bangiao/ke-hoach.md.
 *
 * Chuẩn C99, không malloc/VLA, chỉ dùng buffer tĩnh (static) để sẵn sàng
 * port sang ESP32 (không có heap lớn / không cấp phát động trong vòng lặp
 * suy luận).
 */
#include "har_resnet.h"

#include <float.h>
#include <math.h>
#include <stddef.h>

/* ------------------------------------------------------------------------
 * Lịch trình buffer tĩnh (mục 5.4 kế hoạch) — cố định, không tự sáng tạo.
 * Tổng RAM tĩnh cho các buffer trung gian: 2*(128*32) + 3*(64*32) floats
 * = (8192 + 6144) * 4 byte = 57 344 byte (~56 KB). ESP32 có ~320 KB DRAM
 * nên thoả mái.
 * ------------------------------------------------------------------------ */
static float buf_a[128 * 32]; /* 16 KB */
static float buf_b[128 * 32]; /* 16 KB */
static float buf_c[64 * 32];  /*  8 KB */
static float buf_d[64 * 32];  /*  8 KB */
static float buf_e[64 * 32];  /*  8 KB */

/* ------------------------------------------------------------------------
 * Op cơ bản
 * ------------------------------------------------------------------------ */

void har_conv1d_same(const float *in, int len, int cin,
                      const float *w, const float *b,
                      int ksize, int cout, float *out) {
    /* pad = ksize/2 (chia nguyên). Với ksize lẻ, TensorFlow padding="same"
     * và stride=1 luôn đệm đối xứng pad_left = pad_right = (ksize-1)/2,
     * đúng bằng ksize/2 (chia nguyên C). ksize=1 -> pad=0 -> chính là
     * padding "valid" của lớp block1_projection, nên hàm này dùng chung
     * được cho cả hai trường hợp. */
    int pad = ksize / 2;
    int t;

    for (t = 0; t < len; t++) {
        int co;
        for (co = 0; co < cout; co++) {
            float acc = b[co];
            int ki;
            for (ki = 0; ki < ksize; ki++) {
                int ts = t + ki - pad;
                int ci;
                if (ts < 0 || ts >= len) {
                    continue; /* zero padding ngoài biên: không cộng gì */
                }
                for (ci = 0; ci < cin; ci++) {
                    acc += in[ts * cin + ci] * w[(ki * cin + ci) * cout + co];
                }
            }
            out[t * cout + co] = acc;
        }
    }
}

void har_relu(float *x, int count) {
    int i;
    for (i = 0; i < count; i++) {
        if (x[i] < 0.0f) {
            x[i] = 0.0f;
        }
    }
}

void har_maxpool1d_2(const float *in, int len, int ch, float *out) {
    int t;
    for (t = 0; t < len / 2; t++) {
        int c;
        for (c = 0; c < ch; c++) {
            float v0 = in[(2 * t) * ch + c];
            float v1 = in[(2 * t + 1) * ch + c];
            out[t * ch + c] = (v0 > v1) ? v0 : v1;
        }
    }
}

void har_add(const float *a, const float *b, int count, float *out) {
    int i;
    for (i = 0; i < count; i++) {
        out[i] = a[i] + b[i];
    }
}

void har_global_max_pool(const float *in, int len, int ch, float *out) {
    int c;
    for (c = 0; c < ch; c++) {
        float m = -FLT_MAX;
        int t;
        for (t = 0; t < len; t++) {
            float v = in[t * ch + c];
            if (v > m) {
                m = v;
            }
        }
        out[c] = m;
    }
}

void har_dense(const float *in, int din, const float *w, const float *b,
               int dout, float *out) {
    int o;
    for (o = 0; o < dout; o++) {
        float acc = b[o];
        int i;
        for (i = 0; i < din; i++) {
            acc += in[i] * w[i * dout + o];
        }
        out[o] = acc;
    }
}

/* ------------------------------------------------------------------------
 * API cấp cao
 * ------------------------------------------------------------------------ */

void har_normalize(const float *raw, float *normalized) {
    int t;
    for (t = 0; t < HAR_INPUT_TIMESTEPS; t++) {
        int c;
        for (c = 0; c < HAR_INPUT_CHANNELS; c++) {
            int idx = t * HAR_INPUT_CHANNELS + c;
            normalized[idx] = (raw[idx] - har_input_mean[c]) / har_input_std[c];
        }
    }
}

void har_resnet_predict(const float *x_norm, float *logits) {
    float pooled[HAR_FILTERS]; /* 32 float = 128 byte, biến local nhỏ */

    /* 1. Nhánh chính block1: conv k3 (128,9)->(128,32) */
    har_conv1d_same(x_norm, HAR_INPUT_TIMESTEPS, HAR_INPUT_CHANNELS,
                     har_block1_conv1_w, har_block1_conv1_b,
                     HAR_BLOCK1_CONV1_K, HAR_BLOCK1_CONV1_COUT, buf_a);
    /* 2. ReLU tại chỗ trên buf_a */
    har_relu(buf_a, HAR_INPUT_TIMESTEPS * HAR_FILTERS);
    /* 3. conv k3 (128,32)->(128,32) */
    har_conv1d_same(buf_a, HAR_INPUT_TIMESTEPS, HAR_FILTERS,
                     har_block1_conv2_w, har_block1_conv2_b,
                     HAR_BLOCK1_CONV2_K, HAR_BLOCK1_CONV2_COUT, buf_b);
    /* 4. maxpool /2 (128,32)->(64,32): buf_c = nhánh chính sau pool */
    har_maxpool1d_2(buf_b, HAR_INPUT_TIMESTEPS, HAR_FILTERS, buf_c);

    /* 5. Nhánh tắt (shortcut): conv k1 "valid" trên x_norm gốc, ghi đè buf_a
     *    (hợp lệ vì nội dung cũ của buf_a đã được dùng xong ở bước 3). */
    har_conv1d_same(x_norm, HAR_INPUT_TIMESTEPS, HAR_INPUT_CHANNELS,
                     har_block1_projection_w, har_block1_projection_b,
                     HAR_BLOCK1_PROJECTION_K, HAR_BLOCK1_PROJECTION_COUT, buf_a);
    /* 6. maxpool /2 (128,32)->(64,32): buf_d = nhánh tắt sau pool */
    har_maxpool1d_2(buf_a, HAR_INPUT_TIMESTEPS, HAR_FILTERS, buf_d);

    /* 7. Add(main, shortcut) rồi ReLU -> buf_c = x sau block1 (64,32) */
    har_add(buf_c, buf_d, 64 * HAR_FILTERS, buf_c);
    har_relu(buf_c, 64 * HAR_FILTERS);

    /* 8. block2_conv1 k3 (64,32)->(64,32) */
    har_conv1d_same(buf_c, 64, HAR_FILTERS,
                     har_block2_conv1_w, har_block2_conv1_b,
                     HAR_BLOCK2_CONV1_K, HAR_BLOCK2_CONV1_COUT, buf_e);
    /* 9. ReLU tại chỗ trên buf_e */
    har_relu(buf_e, 64 * HAR_FILTERS);
    /* 10. block2_conv2 k3 (64,32)->(64,32) */
    har_conv1d_same(buf_e, 64, HAR_FILTERS,
                     har_block2_conv2_w, har_block2_conv2_b,
                     HAR_BLOCK2_CONV2_K, HAR_BLOCK2_CONV2_COUT, buf_d);

    /* 11. Add(main, identity=buf_c) rồi ReLU -> buf_c = x sau block2 (64,32) */
    har_add(buf_d, buf_c, 64 * HAR_FILTERS, buf_c);
    har_relu(buf_c, 64 * HAR_FILTERS);

    /* 12. GlobalMaxPool theo trục thời gian -> pooled[32] */
    har_global_max_pool(buf_c, 64, HAR_FILTERS, pooled);

    /* 13. Dense(32 -> 6) -> logits thô (không activation) */
    har_dense(pooled, HAR_DENSE_IN, har_activity_logits_w, har_activity_logits_b,
              HAR_DENSE_OUT, logits);
}

int har_argmax(const float *values, int count) {
    int best_index = 0;
    float best_value = values[0];
    int i;
    for (i = 1; i < count; i++) {
        if (values[i] > best_value) {
            best_value = values[i];
            best_index = i;
        }
    }
    return best_index;
}

void har_softmax(const float *logits, int count, float *probs) {
    float max_value = logits[0];
    float sum = 0.0f;
    int i;
    for (i = 1; i < count; i++) {
        if (logits[i] > max_value) {
            max_value = logits[i];
        }
    }
    for (i = 0; i < count; i++) {
        probs[i] = expf(logits[i] - max_value);
        sum += probs[i];
    }
    for (i = 0; i < count; i++) {
        probs[i] /= sum;
    }
}

const char *har_label_name(int class_id) {
    if (class_id < 0 || class_id >= HAR_NUM_CLASSES) {
        return NULL;
    }
    return har_label_names[class_id];
}
