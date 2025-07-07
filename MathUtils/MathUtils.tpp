#include "MathUtils.hpp"

// TODO: actually implement clear/accumulate behaviour right now it's just accumulate for MatrixColumnSum

#if defined(__AVX512F__)
template <bool clear> void MathUtils::MatrixColumnSum(const float* a, float* b, size_t a_r, size_t a_c) {
    assert(__builtin_cpu_supports("avx512f"));

    // compute sum
    for (size_t i = 0; i < a_r; i++) {

        size_t j = 0;
        for (; j+16 <= a_c; j += 16) {
            const __m512 _a = _mm512_loadu_ps(&a[i*a_c+j]);
            const __m512 _b = _mm512_loadu_ps(&b[j]);
            const __m512 _c = _mm512_add_ps(_a, _b);

            _mm512_storeu_ps(&b[j], _c);
        }

        for (size_t j = a_c-(a_c%16); j < a_c; j++) {
            b[j] += a[i*a_c+j];
        }
    }
}
void MathUtils::Normalize(float* a, float sum, size_t n) {
    assert(__builtin_cpu_supports("avx512f"));

    const __m512 _sum = _mm512_set1_ps(sum);

    size_t i = 0;
    for (; i+16 < n; i += 16) {
        const __m512 _a = _mm512_loadu_ps(&a[i]);
        const __m512 _res = _mm512_div_ps(_a, _sum);

        _mm512_storeu_ps(&a[i], _res);
    }

    for (; i < n; i++) {
        a[i] /= sum;
    }
}
void MathUtils::Scale(float* a, float scale, size_t n) {
    assert(__builtin_cpu_supports("avx512f"));

    const __m512 _scale = _mm512_set1_ps(scale);

    size_t i = 0;
    for (; i+16 < n; i += 16) {
        const __m512 _a = _mm512_loadu_ps(&a[i]);
        const __m512 _res = _mm512_mul_ps(_a, _scale);

        _mm512_storeu_ps(&a[i], _res);
    }

    for (; i < n; i++) {
        a[i] *= scale;
    }
}
#elif defined(__AVX2__) && defined(__FMA__)
template <bool clear> void MathUtils::MatrixColumnSum(const float* a, float* b, size_t a_r, size_t a_c) {
    assert(__builtin_cpu_supports("avx2"));
    assert(__builtin_cpu_supports("fma"));

    // compute sum
    for (size_t i = 0; i < a_r; i++) {

        size_t j = 0;
        for (; j+8 <= a_c; j += 8) {
            const __m256 _a = _mm256_loadu_ps(&a[i*a_c+j]);
            const __m256 _b = _mm256_loadu_ps(&b[j]);
            const __m256 _c = _mm256_add_ps(_a, _b);

            _mm256_storeu_ps(&b[j], _c);
        }

        for (size_t j = a_c-(a_c%8); j < a_c; j++) {
            b[j] += a[i*a_c+j];
        }
    }
}
void MathUtils::Normalize(float* a, float sum, size_t n) {
    assert(__builtin_cpu_supports("avx2"));
    assert(__builtin_cpu_supports("fma"));

    const __m256 _sum = _mm256_set1_ps(sum);

    size_t i = 0;
    for (; i+8 < n; i += 8) {
        const __m256 _a = _mm256_loadu_ps(&a[i]);
        const __m256 _res = _mm256_div_ps(_a, _sum);

        _mm256_storeu_ps(&a[i], _res);
    }

    for (; i < n; i++) {
        a[i] /= sum;
    }
}
void MathUtils::Scale(float* a, float scale, size_t n) {
    assert(__builtin_cpu_supports("avx2"));
    assert(__builtin_cpu_supports("fma"));

    const __m256 _scale = _mm256_set1_ps(scale);

    size_t i = 0;
    for (; i+8 < n; i += 8) {
        const __m256 _a = _mm256_loadu_ps(&a[i]);
        const __m256 _res = _mm256_mul_ps(_a, _scale);

        _mm256_storeu_ps(&a[i], _res);
    }

    for (; i < n; i++) {
        a[i] *= scale;
    }
}
#else
template <bool clear> void MathUtils::MatrixColumnSum(const float* a, float* b, size_t a_r, size_t a_c) {
    // compute sum
    for (size_t i = 0; i < a_r; i++) {

        #pragma omp simd
        for (size_t j = 0; j < a_c; j++) {
            b[j] += a[i*a_c+j];
        }
    }
}
void MathUtils::Normalize(float* a, float sum, size_t n) {

    #pragma omp simd
    for (size_t i = 0; i < n; i++) {
        a[i] /= sum;
    }
}
void MathUtils::Scale(float* a, float scale, size_t n) {

    #pragma omp simd
    for (size_t i = 0; i < n; i++) {
        a[i] *= scale;
    }
}
#endif
