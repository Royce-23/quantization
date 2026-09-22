/* Chương trình kiểm thử C: chạy 12 golden vector qua har_resnet_predict và
 * đối chiếu với logits do models/baseline.keras (TensorFlow) sinh ra.
 * Viết tay — mục 5.6 của .bangiao/ke-hoach.md. */
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

int main(void) {
    int i;
    int all_argmax_ok = 1;
    int all_logit_ok = 1;
    int normalize_ok;

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

    if (all_argmax_ok && all_logit_ok && normalize_ok) {
        printf("PASS\n");
        return 0;
    }
    printf("FAIL (argmax_ok=%d logit_ok=%d normalize_ok=%d)\n", all_argmax_ok,
           all_logit_ok, normalize_ok);
    return 1;
}
