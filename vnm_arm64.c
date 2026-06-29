// .vmodules/vnm/vnm_arm64.c

#ifdef __ARM_NEON
#include <arm_neon.h>
#include <string.h>
#include <math.h>

#if defined(VNM_F16) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#include <arm_fp16.h>

float neon_dot_product_arm64(const float* __restrict__ a, const float* __restrict__ b, int len) {
    #if defined(__GNUC__) || defined(__clang__)
    a = (const float*)__builtin_assume_aligned(a, 16);
    b = (const float*)__builtin_assume_aligned(b, 16);
    #endif
    
    float16x8_t s0 = vdupq_n_f16(0.0f);
    float16x8_t s1 = vdupq_n_f16(0.0f);
    
    int i = 0;
    for (; i <= len - 16; i += 16) {
        #if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(a + i + 32, 0, 0);
        __builtin_prefetch(b + i + 32, 0, 0);
        #endif
        
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);
        
        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        float32x4_t b3 = vld1q_f32(b + i + 12);
        
        float16x4_t ha0 = vcvt_f16_f32(a0);
        float16x4_t ha1 = vcvt_f16_f32(a1);
        float16x4_t ha2 = vcvt_f16_f32(a2);
        float16x4_t ha3 = vcvt_f16_f32(a3);
        
        float16x4_t hb0 = vcvt_f16_f32(b0);
        float16x4_t hb1 = vcvt_f16_f32(b1);
        float16x4_t hb2 = vcvt_f16_f32(b2);
        float16x4_t hb3 = vcvt_f16_f32(b3);
        
        float16x8_t va0 = vcombine_f16(ha0, ha1);
        float16x8_t va1 = vcombine_f16(ha2, ha3);
        
        float16x8_t vb0 = vcombine_f16(hb0, hb1);
        float16x8_t vb1 = vcombine_f16(hb2, hb3);
        
        s0 = vfmaq_f16(s0, va0, vb0);
        s1 = vfmaq_f16(s1, va1, vb1);
    }
    
    float16x8_t sum_vec = vaddq_f16(s0, s1);
    float sum = (float)vaddvq_f16(sum_vec);
    
    for (; i < len; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

#else

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
#endif

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
