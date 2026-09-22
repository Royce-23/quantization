/* TỰ ĐỘNG SINH bởi src/export_c.py — KHÔNG SỬA TAY. */
#ifndef HAR_GOLDEN_H
#define HAR_GOLDEN_H

#include "har_resnet_params.h"

#define HAR_GOLDEN_COUNT 12

extern const float har_golden_input[HAR_GOLDEN_COUNT][HAR_INPUT_TIMESTEPS * HAR_INPUT_CHANNELS];
extern const float har_golden_logits[HAR_GOLDEN_COUNT][HAR_NUM_CLASSES];
extern const int   har_golden_label[HAR_GOLDEN_COUNT];       /* nhãn thật y */
extern const int   har_golden_prediction[HAR_GOLDEN_COUNT];  /* argmax của Keras */

#endif /* HAR_GOLDEN_H */
