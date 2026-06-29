// .vmodules/vnm/vnm_arm64.c

float #ifdef __ARM_NEON
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
    float16x8_t s2 = vdupq_n_f16(0.0f);
    float16x8_t s3 = vdupq_n_f16(0.0f);
    
    int i = 0;
    if (len >= 32) {
        for (; i <= len - 32; i += 32) {
            #if defined(__GNUC__) || defined(__clang__)
            __builtin_prefetch(a + i + 64, 0, 0);
            __builtin_prefetch(b + i + 64, 0, 0);
            #endif
            
            float32x4_t a0 = vld1q_f32(a + i);
            float32x4_t a1 = vld1q_f32(a + i + 4);
            float32x4_t a2 = vld1q_f32(a + i + 8);
            float32x4_t a3 = vld1q_f32(a + i + 12);
            float32x4_t a4 = vld1q_f32(a + i + 16);
            float32x4_t a5 = vld1q_f32(a + i + 20);
            float32x4_t a6 = vld1q_f32(a + i + 24);
            float32x4_t a7 = vld1q_f32(a + i + 28);
            
            float32x4_t b0 = vld1q_f32(b + i);
            float32x4_t b1 = vld1q_f32(b + i + 4);
            float32x4_t b2 = vld1q_f32(b + i + 8);
            float32x4_t b3 = vld1q_f32(b + i + 12);
            float32x4_t b4 = vld1q_f32(b + i + 16);
            float32x4_t b5 = vld1q_f32(b + i + 20);
            float32x4_t b6 = vld1q_f32(b + i + 24);
            float32x4_t b7 = vld1q_f32(b + i + 28);
            
            float16x8_t va0 = vcvt_high_f16_f32(vcvt_f16_f32(a0), a1);
            float16x8_t va1 = vcvt_high_f16_f32(vcvt_f16_f32(a2), a3);
            float16x8_t va2 = vcvt_high_f16_f32(vcvt_f16_f32(a4), a5);
            float16x8_t va3 = vcvt_high_f16_f32(vcvt_f16_f32(a6), a7);
            
            float16x8_t vb0 = vcvt_high_f16_f32(vcvt_f16_f32(b0), b1);
            float16x8_t vb1 = vcvt_high_f16_f32(vcvt_f16_f32(b2), b3);
            float16x8_t vb2 = vcvt_high_f16_f32(vcvt_f16_f32(b4), b5);
            float16x8_t vb3 = vcvt_high_f16_f32(vcvt_f16_f32(b6), b7);
            
            s0 = vfmaq_f16(s0, va0, vb0);
            s1 = vfmaq_f16(s1, va1, vb1);
            s2 = vfmaq_f16(s2, va2, vb2);
            s3 = vfmaq_f16(s3, va3, vb3);
        }
        s0 = vaddq_f16(s0, s2);
        s1 = vaddq_f16(s1, s3);
    }
    
    for (; i <= len - 16; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);
        
        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        float32x4_t b3 = vld1q_f32(b + i + 12);
        
        float16x8_t va0 = vcvt_high_f16_f32(vcvt_f16_f32(a0), a1);
        float16x8_t va1 = vcvt_high_f16_f32(vcvt_f16_f32(a2), a3);
        
        float16x8_t vb0 = vcvt_high_f16_f32(vcvt_f16_f32(b0), b1);
        float16x8_t vb1 = vcvt_high_f16_f32(vcvt_f16_f32(b2), b3);
        
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
    float32x4_t s4 = vdupq_n_f32(0.0f);
    float32x4_t s5 = vdupq_n_f32(0.0f);
    float32x4_t s6 = vdupq_n_f32(0.0f);
    float32x4_t s7 = vdupq_n_f32(0.0f);
    
    int i = 0;
    if (len >= 32) {
        for (; i <= len - 32; i += 32) {
            #if defined(__GNUC__) || defined(__clang__)
            __builtin_prefetch(a + i + 64, 0, 0);
            __builtin_prefetch(b + i + 64, 0, 0);
            #endif

            s0 = vfmaq_f32(s0, vld1q_f32(a + i),      vld1q_f32(b + i));
            s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4),  vld1q_f32(b + i + 4));
            s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8),  vld1q_f32(b + i + 8));
            s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
            s4 = vfmaq_f32(s4, vld1q_f32(a + i + 16), vld1q_f32(b + i + 16));
            s5 = vfmaq_f32(s5, vld1q_f32(a + i + 20), vld1q_f32(b + i + 20));
            s6 = vfmaq_f32(s6, vld1q_f32(a + i + 24), vld1q_f32(b + i + 24));
            s7 = vfmaq_f32(s7, vld1q_f32(a + i + 28), vld1q_f32(b + i + 28));
        }
        s0 = vaddq_f32(s0, s4);
        s1 = vaddq_f32(s1, s5);
        s2 = vaddq_f32(s2, s6);
        s3 = vaddq_f32(s3, s7);
    }
    
    for (; i <= len - 16; i += 16) {
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
