/* Chương trình kiểm thử C: chạy 12 golden vector qua har_resnet_predict và
 * đối chiếu với logits do models/baseline.keras (TensorFlow) sinh ra.
 * Viết tay — xem docs/model_architecture_and_c_inference.md mục 5. */
#include <math.h>
#include <stdio.h>

#include "har_golden.h"
#include "har_resnet.h"

#define LOGIT_TOLERANCE 1e-3f
#define NORMALIZE_TOLERANCE 1e-5f

static int check_normalize(void) {
    float raw[HAR_INPUT_TIMESTEPS * HAR_INPUT_CHANNELS];
    float normalized[HAR_INPUT_TIMESTEPS * HAR_INPUT_CHANNELS];
    int t, c, ok;
    float max_abs_diff;

    /* raw = mean cho mọi bước thời gian -> kỳ vọng normalize ra toàn 0. */
    for (t = 0; t < HAR_INPUT_TIMESTEPS; t++) {
        for (c = 0; c < HAR_INPUT_CHANNELS; c++) {
            raw[t * HAR_INPUT_CHANNELS + c] = har_input_mean[c];
        }
    }
    har_normalize(raw, normalized);
    max_abs_diff = 0.0f;
    for (t = 0; t < HAR_INPUT_TIMESTEPS * HAR_INPUT_CHANNELS; t++) {
        float diff = fabsf(normalized[t] - 0.0f);
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
        }
    }
    ok = (max_abs_diff <= NORMALIZE_TOLERANCE);
    printf("normalize(mean) -> 0: max_abs_diff=%.8f %s\n", (double)max_abs_diff,
           ok ? "OK" : "FAIL");

    /* raw = mean + std -> kỳ vọng normalize ra toàn 1. */
    for (t = 0; t < HAR_INPUT_TIMESTEPS; t++) {
        for (c = 0; c < HAR_INPUT_CHANNELS; c++) {
            raw[t * HAR_INPUT_CHANNELS + c] = har_input_mean[c] + har_input_std[c];
        }
    }
    har_normalize(raw, normalized);
    max_abs_diff = 0.0f;
    for (t = 0; t < HAR_INPUT_TIMESTEPS * HAR_INPUT_CHANNELS; t++) {
        float diff = fabsf(normalized[t] - 1.0f);
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
        }
    }
    if (max_abs_diff > NORMALIZE_TOLERANCE) {
        ok = 0;
    }
    printf("normalize(mean+std) -> 1: max_abs_diff=%.8f %s\n", (double)max_abs_diff,
           (max_abs_diff <= NORMALIZE_TOLERANCE) ? "OK" : "FAIL");

    return ok;
}

/* check_block_decomposition — xác nhận phân rã theo khối (har_block1_forward
 * -> har_block2_forward -> har_head_forward) cho ra kết quả BẰNG TUYỆT ĐỐI
 * (không sai số) so với har_resnet_predict() nguyên khối, trên cùng 1 mẫu
 * golden input. Dùng buffer static RIÊNG (không tái sử dụng buffer nội bộ
 * của har_resnet.c) để không phụ thuộc chi tiết cài đặt. */
static float block_test_s128_0[128 * 32];
static float block_test_s128_1[128 * 32];
static float block_test_s64_0[64 * 32];
static float block_test_s64_1[64 * 32];
static float block_test_x64[64 * 32];

static int check_block_decomposition(void) {
    float logits_a[HAR_NUM_CLASSES];
    float logits_b[HAR_NUM_CLASSES];
    int c;
    int ok = 1;

    /* (a) phân rã theo khối: block1 -> block2 -> head. */
    har_block1_forward(har_golden_input[0], block_test_x64,
                        block_test_s128_0, block_test_s128_1, block_test_s64_0);
    har_block2_forward(block_test_x64, block_test_x64,
                        block_test_s64_0, block_test_s64_1);
    har_head_forward(block_test_x64, logits_a);

    /* (b) hàm nguyên khối hiện có. */
    har_resnet_predict(har_golden_input[0], logits_b);

    for (c = 0; c < HAR_NUM_CLASSES; c++) {
        if (logits_a[c] != logits_b[c]) {
            ok = 0;
        }
    }
    printf("block_decomposition == har_resnet_predict: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

int main(void) {
    int i;
    int all_argmax_ok = 1;
    int all_logit_ok = 1;
    int normalize_ok;
    int block_decomposition_ok;

    printf("HAR_CHECKPOINT_ID=%s\n", HAR_CHECKPOINT_ID);
    printf("idx | label              | expected_pred | got_pred | max_abs_diff\n");
    printf("----+--------------------+----------------+----------+-------------\n");

    for (i = 0; i < HAR_GOLDEN_COUNT; i++) {
        float logits[HAR_NUM_CLASSES];
        int got_pred, expected_pred;
        float max_abs_diff = 0.0f;
        int c;
        const char *label_name;

        har_resnet_predict(har_golden_input[i], logits);
        got_pred = har_argmax(logits, HAR_NUM_CLASSES);
        expected_pred = har_golden_prediction[i];

        for (c = 0; c < HAR_NUM_CLASSES; c++) {
            float diff = fabsf(logits[c] - har_golden_logits[i][c]);
            if (diff > max_abs_diff) {
                max_abs_diff = diff;
            }
        }

        label_name = har_label_name(har_golden_label[i]);
        printf("%3d | %-18s | %14d | %8d | %.6f\n", i,
               label_name != NULL ? label_name : "?", expected_pred, got_pred,
               (double)max_abs_diff);

        if (got_pred != expected_pred) {
            all_argmax_ok = 0;
        }
        if (max_abs_diff > LOGIT_TOLERANCE) {
            all_logit_ok = 0;
        }
    }

    normalize_ok = check_normalize();
    block_decomposition_ok = check_block_decomposition();

    if (all_argmax_ok && all_logit_ok && normalize_ok && block_decomposition_ok) {
        printf("PASS\n");
        return 0;
    }
    printf("FAIL (argmax_ok=%d logit_ok=%d normalize_ok=%d block_decomposition_ok=%d)\n",
           all_argmax_ok, all_logit_ok, normalize_ok, block_decomposition_ok);
    return 1;
}
