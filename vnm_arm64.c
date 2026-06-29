// .vmodules/vnm/vnm_arm64.c

#include <string.h>
#include <math.h>

#ifdef __ARM_NEON
#include <arm_neon.h>

static inline float fast_reciprocal(float x) {
    float est = vrecpes_f32(x);
    float step1 = vrecpss_f32(x, est);
    float est1 = est * step1;
    float step2 = vrecpss_f32(x, est1);
    return est1 * step2;
}

float approx_tanh_neon(float x) {
    float clamped_x = fmaxf(fminf(x, 3.0f), -3.0f);
    float x2 = clamped_x * clamped_x;
    float num = clamped_x * (x2 + 15.0f);
    float den = 6.0f * x2 + 15.0f;
    float res = num * fast_reciprocal(den);
    return fmaxf(fminf(res, 1.0f), -1.0f);
}

float approx_sigmoid_neon(float x) {
    return 0.5f + 0.5f * approx_tanh_neon(0.5f * x);
}

float neon_dot_product_arm64(const float* __restrict__ a, const float* __restrict__ b, int len) {
    #if defined(__GNUC__) || defined(__clang__)
    a = (const float*)__builtin_assume_aligned(a, 16);
    b = (const float*)__builtin_assume_aligned(b, 16);
    #endif
    
    float32x4_t s0 = vdupq_n_f32(0.0f);
    float32x4_t s1 = vdupq_n_f32(0.0f);
    float32x4_t s2 = vdupq_n_f32(0.0f);
    float32x4_t s3 = vdupq_n_f32(0.0f);
    
    int i = 0;
    for (; i <= len - 16; i += 16) {
        #if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(a + i + 32, 0, 0);
        __builtin_prefetch(b + i + 32, 0, 0);
        #endif

        s0 = vfmaq_f32(s0, vld1q_f32(a + i),      vld1q_f32(b + i));
        s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4),  vld1q_f32(b + i + 4));
        s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8),  vld1q_f32(b + i + 8));
        s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    
    float32x4_t sum_vec = vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3));
    for (; i <= len - 4; i += 4) {
        sum_vec = vfmaq_f32(sum_vec, vld1q_f32(a + i), vld1q_f32(b + i));
    }
    
    float sum = vaddvq_f32(sum_vec);
    for (; i < len; i++) {
        sum += a[i] * b[i];
    }
    return sum;
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
