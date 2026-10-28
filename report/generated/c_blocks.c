void har_block1_forward(const float *x_norm, float *y64,
                        float *s128_0, float *s128_1, float *s64_0) {
    har_block1_branch_main(x_norm, s128_0, s128_1, y64);       /* A -> y64   */
    har_block1_branch_shortcut(x_norm, s128_0, s64_0);         /* B -> s64_0 */
    har_residual_merge_relu(y64, s64_0, 64 * HAR_FILTERS, y64); /* add(A,B)   */
}

void har_block2_forward(const float *x64, float *y64,
                        float *s64_0, float *s64_1) {
    har_block2_branch_main(x64, s64_0, s64_1);                    /* A -> s64_1 */
    har_residual_merge_relu(s64_1, x64, 64 * HAR_FILTERS, y64);   /* add(A,B)   */
}

void har_head_forward(const float *x64, float *logits) {
    float pooled[HAR_FILTERS];

    har_global_max_pool(x64, 64, HAR_FILTERS, pooled);
    har_dense(pooled, HAR_DENSE_IN, har_activity_logits_w, har_activity_logits_b,
              HAR_DENSE_OUT, logits);
}

void har_resnet_predict(const float *x_norm, float *logits) {
    har_block1_forward(x_norm, buf_c, buf_a, buf_b, buf_d);
    har_block2_forward(buf_c, buf_c, buf_e, buf_d);
    /* Head: (64,32) -> logits(6). */
    har_head_forward(buf_c, logits);
}
