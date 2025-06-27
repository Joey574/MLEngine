#include "LossMetric.hpp"

void LossMetric::OneHotLoss(const float* __restrict x, const float* __restrict y, float* __restrict c, size_t rows, size_t cols) {
    std::memcpy(c, x, rows*cols*sizeof(float));

    #pragma omp parallel for simd
    for (size_t i = 0; i < rows; i++) {
        c[i*cols+(int)y[i]]--;
    }
}

#if defined(__AVX512F__)
void LossMetric::MaeLoss(const float* __restrict x, const float* __restrict y, float* __restrict c, size_t rows, size_t cols) {
    const __m512 _zero = _mm512_setzero_ps();
    const __m512 _one = _mm512_set1_ps(1.0f);
    const __m512 _none = _mm512_set1_ps(-1.0f);

    #pragma omp parallel for
    for (ssize_t i = 0; i <= (ssize_t)(rows*cols)-16; i += 16) {
        const __m512 _x = _mm512_loadu_ps(&x[i]);
        const __m512 _y = _mm512_loadu_ps(&y[i]);

        const __m512 _diff = _mm512_sub_ps(_x, _y);
        const __mmask16 _cmp = _mm512_cmp_ps_mask(_diff, _zero, _CMP_GT_OQ);
        const __m512 _res = _mm512_mask_blend_ps(_cmp, _none, _one);

        _mm512_storeu_ps(&c[i], _res);
    }

    for (size_t i = (rows*cols)-((rows*cols)%16); i < rows*cols; i++) {
        c[i] = (x[i] - y[i]) > 0.0f ? 1.0f : -1.0f;
    }
}
void LossMetric::MseLoss(const float* __restrict x, const float* __restrict y, float* __restrict c, size_t rows, size_t cols) {
    const __m512 _two = _mm512_set1_ps(2.0f);

    #pragma omp parallel for
    for (ssize_t i = 0; i <= (ssize_t)(rows*cols)-16; i += 16) {
        const __m512 _x = _mm512_loadu_ps(&x[i]);
        const __m512 _y = _mm512_loadu_ps(&y[i]);

        const __m512 _x2 = _mm512_sub_ps(_x, _y);
        const __m512 _res = _mm512_mul_ps(_two, _x2);
        _mm512_storeu_ps(&c[i], _res);
    }

    for (size_t i = (rows*cols)-((rows*cols)%16); i < rows*cols; i++) {
        c[i] = 2.0f * (x[i] - y[i]);
    }
}
#elif defined(__AVX2__) && defined(__FMA__)
void LossMetric::MaeLoss(const float* __restrict x, const float* __restrict y, float* __restrict c, size_t rows, size_t cols) {
    const __m256 _zero = _mm256_setzero_ps();
    const __m256 _one = _mm256_set1_ps(1.0f);
    const __m256 _none = _mm256_set1_ps(-1.0f);

    #pragma omp parallel for
    for (ssize_t i = 0; i <= (ssize_t)(rows*cols)-8; i += 8) {
        const __m256 _x = _mm256_loadu_ps(&x[i]);
        const __m256 _y = _mm256_loadu_ps(&y[i]);

        const __m256 _diff = _mm256_sub_ps(_x, _y);
        const __m256 _cmp = _mm256_cmp_ps(_diff, _zero, _CMP_GT_OQ);
        const __m256 _res = _mm256_blendv_ps(_none, _one, _cmp);

        _mm256_storeu_ps(&c[i], _res);
    }

    for (size_t i = (rows*cols)-((rows*cols)%8); i < rows*cols; i++) {
        c[i] = (x[i] - y[i]) > 0.0f ? 1.0f : -1.0f;
    }

}
void LossMetric::MseLoss(const float* __restrict x, const float* __restrict y, float* __restrict c, size_t rows, size_t cols) {
    const __m256 _two = _mm256_set1_ps(2.0f);

    #pragma omp parallel for
    for (ssize_t i = 0; i <= (ssize_t)(rows*cols)-8; i += 8) {
        const __m256 _x = _mm256_loadu_ps(&x[i]);
        const __m256 _y = _mm256_loadu_ps(&y[i]);

        const __m256 _x2 = _mm256_sub_ps(_x, _y);
        const __m256 _res = _mm256_mul_ps(_two, _x2);
        _mm256_storeu_ps(&c[i], _res);
    }

    for (size_t i = (rows*cols)-((rows*cols)%8); i < rows*cols; i++) {
        c[i] = 2.0f * (x[i] - y[i]);
    }
}
#else
void LossMetric::MaeLoss(const float* __restrict x, const float* __restrict y, float* __restrict c, size_t rows, size_t cols) {
    #pragma omp parallel for simd
    for (size_t i = 0; i < rows*cols; i++) {
        c[i] = (x[i] - y[i]) > 0.0f ? 1.0f : -1.0f;
    }

}
void LossMetric::MseLoss(const float* __restrict x, const float* __restrict y, float* __restrict c, size_t rows, size_t cols) {
    #pragma omp parallel for simd
    for (size_t i = 0; i < rows*cols; i++) {
        c[i] = 2.0f * (x[i] - y[i]);
    }
}
#endif
