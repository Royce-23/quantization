/* TỰ ĐỘNG SINH bởi src/export_c.py — KHÔNG SỬA TAY. */
#ifndef HAR_RESNET_PARAMS_H
#define HAR_RESNET_PARAMS_H

#define HAR_CHECKPOINT_ID "aa323b3e55d1f904edbb05fb6d2a817b748b6e0ad26b7eb0166777c94535c74e"
#define HAR_PARAM_COUNT   10726
#define HAR_INPUT_TIMESTEPS 128
#define HAR_INPUT_CHANNELS  9
#define HAR_FILTERS         32
#define HAR_NUM_CLASSES     6

#define HAR_BLOCK1_CONV1_K 3
#define HAR_BLOCK1_CONV1_CIN 9
#define HAR_BLOCK1_CONV1_COUT 32
extern const float har_block1_conv1_w[HAR_BLOCK1_CONV1_K * HAR_BLOCK1_CONV1_CIN * HAR_BLOCK1_CONV1_COUT];
extern const float har_block1_conv1_b[HAR_BLOCK1_CONV1_COUT];

#define HAR_BLOCK1_CONV2_K 3
#define HAR_BLOCK1_CONV2_CIN 32
#define HAR_BLOCK1_CONV2_COUT 32
extern const float har_block1_conv2_w[HAR_BLOCK1_CONV2_K * HAR_BLOCK1_CONV2_CIN * HAR_BLOCK1_CONV2_COUT];
extern const float har_block1_conv2_b[HAR_BLOCK1_CONV2_COUT];

#define HAR_BLOCK1_PROJECTION_K 1
#define HAR_BLOCK1_PROJECTION_CIN 9
#define HAR_BLOCK1_PROJECTION_COUT 32
extern const float har_block1_projection_w[HAR_BLOCK1_PROJECTION_K * HAR_BLOCK1_PROJECTION_CIN * HAR_BLOCK1_PROJECTION_COUT];
extern const float har_block1_projection_b[HAR_BLOCK1_PROJECTION_COUT];

#define HAR_BLOCK2_CONV1_K 3
#define HAR_BLOCK2_CONV1_CIN 32
#define HAR_BLOCK2_CONV1_COUT 32
extern const float har_block2_conv1_w[HAR_BLOCK2_CONV1_K * HAR_BLOCK2_CONV1_CIN * HAR_BLOCK2_CONV1_COUT];
extern const float har_block2_conv1_b[HAR_BLOCK2_CONV1_COUT];

#define HAR_BLOCK2_CONV2_K 3
#define HAR_BLOCK2_CONV2_CIN 32
#define HAR_BLOCK2_CONV2_COUT 32
extern const float har_block2_conv2_w[HAR_BLOCK2_CONV2_K * HAR_BLOCK2_CONV2_CIN * HAR_BLOCK2_CONV2_COUT];
extern const float har_block2_conv2_b[HAR_BLOCK2_CONV2_COUT];

#define HAR_DENSE_IN  32
#define HAR_DENSE_OUT 6
extern const float har_activity_logits_w[HAR_DENSE_IN * HAR_DENSE_OUT];
extern const float har_activity_logits_b[HAR_DENSE_OUT];

extern const float har_input_mean[HAR_INPUT_CHANNELS];
extern const float har_input_std[HAR_INPUT_CHANNELS];
extern const char *const har_label_names[HAR_NUM_CLASSES];

#endif /* HAR_RESNET_PARAMS_H */
