/* Cài đặt thuật toán forward pass ResNetv1-6 1D — VIẾT TAY (không sinh tự
 * động). Đối chiếu với src/reference_forward.py (NumPy) và
 * docs/model_architecture_and_c_inference.md (mục 2 — công thức, mục 4 — cài đặt C).
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
 * Lịch trình buffer tĩnh (xem docs/model_architecture_and_c_inference.md mục 4.2)
 * — cố định, không tự sáng tạo.
 * Tổng RAM tĩnh cho các buffer trung gian: 2*(128*32) + 3*(64*32) floats
 * = (8192 + 6144) * 4 byte = 57 344 byte (~56 KB). ESP32 có ~320 KB DRAM
 * nên thoả mái.
 *
 * CẢNH BÁO: KHÔNG THREAD-SAFE / KHÔNG REENTRANT: 5 buffer dưới đây được TOÀN BỘ các
 * hàm khối (har_block1_forward/har_block2_forward qua har_resnet_predict)
 * dùng chung. Chỉ được có tối đa MỘT lời gọi har_resnet_predict đang chạy
 * tại một thời điểm; không gọi từ nhiều task/thread hoặc ISR cùng lúc.
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

/* ==========================================================================
 * Hàm nhánh/hợp nhánh nội bộ (static) — dùng chung bởi Block 1 và Block 2.
 * ========================================================================== */

/* har_residual_merge_relu — điểm HỢP NHÁNH dùng chung cho cả 2 khối:
 * y[i] = max(0, main[i] + shortcut[i])  (công thức 2.6 của tài liệu kiến trúc).
 * Cho phép out trùng với main hoặc shortcut (ghi đè tại chỗ, an toàn vì phép
 * toán là element-wise: đọc a[i], b[i] rồi ghi out[i] cùng chỉ số). */
static void har_residual_merge_relu(const float *main_path, const float *shortcut,
                                     int count, float *out) {
    har_add(main_path, shortcut, count, out);
    har_relu(out, count);
}

/* har_block1_branch_main — NHÁNH A (main path) của Block 1.
 * NGUỒN INPUT : x_norm (128, 9) — trực tiếp từ sensor_window đã z-score.
 * CẤU TRÚC    : Conv1D k=3 same -> ReLU -> Conv1D k=3 same -> MaxPool/2.
 * CÔNG THỨC   :
 *   h1[t][co] = b1[co] + sum_{k=0..2} sum_{ci=0..8}  W1[k][ci][co] * xt[t+k-1][ci]
 *   h2[t][co] = max(0, h1[t][co])
 *   h3[t][co] = b2[co] + sum_{k=0..2} sum_{ci=0..31} W2[k][ci][co] * h2t[t+k-1][ci]
 *   out64[t][co] = max(h3[2t][co], h3[2t+1][co])
 * SHAPE       : (128, 9) -> (64, 32).
 * THAM SỐ     : block1_conv1 896 + block1_conv2 3104 = 4000 tham số. */
static void har_block1_branch_main(const float *x_norm, float *s128_0,
                                    float *s128_1, float *out64) {
    har_conv1d_same(x_norm, HAR_INPUT_TIMESTEPS, HAR_INPUT_CHANNELS,
                     har_block1_conv1_w, har_block1_conv1_b,
                     HAR_BLOCK1_CONV1_K, HAR_BLOCK1_CONV1_COUT, s128_0);
    har_relu(s128_0, HAR_INPUT_TIMESTEPS * HAR_FILTERS);
    har_conv1d_same(s128_0, HAR_INPUT_TIMESTEPS, HAR_FILTERS,
                     har_block1_conv2_w, har_block1_conv2_b,
                     HAR_BLOCK1_CONV2_K, HAR_BLOCK1_CONV2_COUT, s128_1);
    har_maxpool1d_2(s128_1, HAR_INPUT_TIMESTEPS, HAR_FILTERS, out64);
}

/* har_block1_branch_shortcut — NHÁNH B (projection shortcut) của Block 1.
 * NGUỒN INPUT : x_norm (128, 9) — CÙNG nguồn với nhánh A ở trên (không phải
 *               output của nhánh A).
 * CẤU TRÚC    : Conv1D k=1 "valid" (đổi 9 -> 32 kênh) -> MaxPool/2.
 * CÔNG THỨC   :
 *   p[t][co] = bp[co] + sum_{ci=0..8} Wp[0][ci][co] * x_norm[t][ci]
 *   out64[t][co] = max(p[2t][co], p[2t+1][co])
 * SHAPE       : (128, 9) -> (64, 32).
 * THAM SỐ     : block1_projection 320 tham số. */
static void har_block1_branch_shortcut(const float *x_norm, float *s128_0,
                                        float *out64) {
    har_conv1d_same(x_norm, HAR_INPUT_TIMESTEPS, HAR_INPUT_CHANNELS,
                     har_block1_projection_w, har_block1_projection_b,
                     HAR_BLOCK1_PROJECTION_K, HAR_BLOCK1_PROJECTION_COUT, s128_0);
    har_maxpool1d_2(s128_0, HAR_INPUT_TIMESTEPS, HAR_FILTERS, out64);
}

/* har_block2_branch_main — NHÁNH A (main path) của Block 2.
 * NGUỒN INPUT : x64 (64, 32) — output của KHỐI 1.
 * CẤU TRÚC    : Conv1D k=3 same -> ReLU -> Conv1D k=3 same (không pool, giữ độ dài).
 * CÔNG THỨC   :
 *   h1[t][co] = b1[co] + sum_{k=0..2} sum_{ci=0..31} W1[k][ci][co] * x64t[t+k-1][ci]
 *   h2[t][co] = max(0, h1[t][co])
 *   out64[t][co] = b2[co] + sum_{k=0..2} sum_{ci=0..31} W2[k][ci][co] * h2t[t+k-1][ci]
 * SHAPE       : (64, 32) -> (64, 32).
 * THAM SỐ     : block2_conv1 3104 + block2_conv2 3104 = 6208 tham số. */
static void har_block2_branch_main(const float *x64, float *s64_0, float *out64) {
    har_conv1d_same(x64, 64, HAR_FILTERS,
                     har_block2_conv1_w, har_block2_conv1_b,
                     HAR_BLOCK2_CONV1_K, HAR_BLOCK2_CONV1_COUT, s64_0);
    har_relu(s64_0, 64 * HAR_FILTERS);
    har_conv1d_same(s64_0, 64, HAR_FILTERS,
                     har_block2_conv2_w, har_block2_conv2_b,
                     HAR_BLOCK2_CONV2_K, HAR_BLOCK2_CONV2_COUT, out64);
}

/* ============================ KHỐI 1 (BLOCK 1) ============================
 * NGUỒN INPUT : trực tiếp từ sensor_window (128, 9) đã z-score — đây là khối
 *               ĐẦU TIÊN, không có stem conv trước nó.
 * CẤU TRÚC    : residual block kiểu ResNet v1 với PROJECTION shortcut, đồng
 *               thời hạ mẫu trục thời gian 128 -> 64.
 *   Nhánh A (main)     : Conv1D k=3 same -> ReLU -> Conv1D k=3 same -> MaxPool/2
 *   Nhánh B (shortcut) : Conv1D k=1 valid -> MaxPool/2
 *   Hợp nhánh          : y = ReLU(A + B)
 * CÔNG THỨC   :
 *   h1[t][co] = b1[co] + sum_{k=0..2} sum_{ci=0..8}  W1[k][ci][co] * xt[t+k-1][ci]
 *   h2[t][co] = max(0, h1[t][co])
 *   h3[t][co] = b2[co] + sum_{k=0..2} sum_{ci=0..31} W2[k][ci][co] * h2t[t+k-1][ci]
 *   A[t][co]  = max(h3[2t][co], h3[2t+1][co])
 *   p[t][co]  = bp[co] + sum_{ci=0..8} Wp[0][ci][co] * xt[t][ci]
 *   B[t][co]  = max(p[2t][co], p[2t+1][co])
 *   y[t][co]  = max(0, A[t][co] + B[t][co])
 *   (xt, h2t = phiên bản zero-padding: xt[s]=x[s] nếu 0<=s<L, ngược lại = 0)
 * SHAPE       : (128, 9) -> (64, 32)
 * THAM SỐ     : block1_conv1 896 + block1_conv2 3104 + block1_projection 320
 *               = 4320 tham số.
 * ========================================================================== */
void har_block1_forward(const float *x_norm, float *y64,
                        float *s128_0, float *s128_1, float *s64_0) {
    har_block1_branch_main(x_norm, s128_0, s128_1, y64);       /* A -> y64   */
    har_block1_branch_shortcut(x_norm, s128_0, s64_0);         /* B -> s64_0 */
    har_residual_merge_relu(y64, s64_0, 64 * HAR_FILTERS, y64); /* add(A,B)   */
}

/* ============================ KHỐI 2 (BLOCK 2) ============================
 * NGUỒN INPUT : output của KHỐI 1 — x64 (64, 32).
 * CẤU TRÚC    : residual block kiểu ResNet v1 với IDENTITY shortcut, giữ
 *               nguyên độ phân giải trục thời gian (không pool).
 *   Nhánh A (main)     : Conv1D k=3 same -> ReLU -> Conv1D k=3 same
 *   Nhánh B (shortcut) : identity — KHÔNG có phép toán, dùng thẳng x64
 *   Hợp nhánh          : y = ReLU(A + B)
 * CÔNG THỨC   :
 *   h1[t][co] = b1[co] + sum_{k=0..2} sum_{ci=0..31} W1[k][ci][co] * x64t[t+k-1][ci]
 *   h2[t][co] = max(0, h1[t][co])
 *   A[t][co]  = b2[co] + sum_{k=0..2} sum_{ci=0..31} W2[k][ci][co] * h2t[t+k-1][ci]
 *   B[t][co]  = x64[t][co]                          (identity, không phép toán)
 *   y[t][co]  = max(0, A[t][co] + B[t][co])
 * SHAPE       : (64, 32) -> (64, 32)
 * THAM SỐ     : block2_conv1 3104 + block2_conv2 3104 = 6208 tham số.
 * ========================================================================== */
void har_block2_forward(const float *x64, float *y64,
                        float *s64_0, float *s64_1) {
    har_block2_branch_main(x64, s64_0, s64_1);                    /* A -> s64_1 */
    /* NHÁNH B = identity: KHÔNG phép toán, dùng thẳng x64. */
    har_residual_merge_relu(s64_1, x64, 64 * HAR_FILTERS, y64);   /* add(A,B)   */
}

/* ============================== HEAD (bộ phân loại) ========================
 * NGUỒN INPUT : output của KHỐI 2 — x64 (64, 32).
 * CẤU TRÚC    : GlobalMaxPooling1D (max theo trục thời gian) -> Dense(32 -> 6),
 *               KHÔNG có activation (logits thô; softmax không nằm trong model).
 * CÔNG THỨC   :
 *   g[c] = max_{0<=t<64} x64[t][c]                    , c = 0..31
 *   z[o] = bd[o] + sum_{i=0..31} g[i] * Wd[i][o]       , o = 0..5
 * SHAPE       : (64, 32) -> (32,) -> (6,)
 * THAM SỐ     : activity_logits 198 tham số.
 * ========================================================================== */
void har_head_forward(const float *x64, float *logits) {
    float pooled[HAR_FILTERS]; /* 32 float = 128 byte, biến local nhỏ */

    har_global_max_pool(x64, 64, HAR_FILTERS, pooled);
    har_dense(pooled, HAR_DENSE_IN, har_activity_logits_w, har_activity_logits_b,
              HAR_DENSE_OUT, logits);
}

void har_resnet_predict(const float *x_norm, float *logits) {
    /* Khối 1: (128,9) -> (64,32). scratch: buf_a, buf_b, buf_d. */
    har_block1_forward(x_norm, buf_c, buf_a, buf_b, buf_d);
    /* Khối 2: (64,32) -> (64,32), ghi đè tại chỗ lên buf_c. scratch: buf_e, buf_d. */
    har_block2_forward(buf_c, buf_c, buf_e, buf_d);
    /* Head: (64,32) -> logits(6). */
    har_head_forward(buf_c, logits);
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
