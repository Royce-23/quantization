void har_conv1d_same(const float *in, int len, int cin,
                      const float *w, const float *b,
                      int ksize, int cout, float *out) {
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
                    continue;
                }
                for (ci = 0; ci < cin; ci++) {
                    acc += in[ts * cin + ci] * w[(ki * cin + ci) * cout + co];
                }
            }
            out[t * cout + co] = acc;
        }
    }
}
