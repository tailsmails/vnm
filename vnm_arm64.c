// .vmodules/vnm/vnm_arm64.c

#ifdef __ARM_NEON
#include <arm_neon.h>
#include <string.h>
#include <math.h>

float neon_dot_product_arm64(const float* a, const float* b, int len) {
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i < len - 3; i += 4) {
        float32x4_t a_vec = vld1q_f32(&a[i]);
        float32x4_t b_vec = vld1q_f32(&b[i]);
        sum_vec = vmlaq_f32(sum_vec, a_vec, b_vec);
    }
    float sum = vaddvq_f32(sum_vec);
    for (; i < len; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

static inline float fast_exp_c(float x) {
    if (x < -88.0f) x = -88.0f;
    if (x > 88.0f) x = 88.0f;
    float fb = x * 12102203.0f;
    int bits = (int)fb + 1065353216;
    float res;
    memcpy(&res, &bits, sizeof(float));
    return res;
}

float approx_sigmoid_neon(float x) {
    return 1.0f / (1.0f + fast_exp_c(-x));
}

float approx_tanh_neon(float x) {
    float exp_2x = fast_exp_c(2.0f * x);
    return (exp_2x - 1.0f) / (exp_2x + 1.0f);
}

float approx_inv_sqrt_neon(float x) {
    float xhalf = 0.5f * x;
    int i;
    memcpy(&i, &x, sizeof(float));
    i = 0x5f3759df - (i >> 1);
    float y;
    memcpy(&y, &i, sizeof(float));
    y = y * (1.5f - xhalf * y * y);
    return y;
}

float fast_max_neon(float a, float b) {
    return fmaxf(a, b);
}
#else
float neon_dot_product_arm64(const float* a, const float* b, int len) {
    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

float approx_sigmoid_neon(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float approx_tanh_neon(float x) {
    return tanhf(x);
}

float approx_inv_sqrt_neon(float x) {
    float xhalf = 0.5f * x;
    int i;
    memcpy(&i, &x, sizeof(float));
    i = 0x5f3759df - (i >> 1);
    float y;
    memcpy(&y, &i, sizeof(float));
    y = y * (1.5f - xhalf * y * y);
    return y;
}

float fast_max_neon(float a, float b) {
    return a > b ? a : b;
}
#endif
