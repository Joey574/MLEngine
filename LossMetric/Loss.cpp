#include "LossMetric.hpp"

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
void LossMetric::OneHotLoss(const float* __restrict x, const float* __restrict y, float* __restrict c, size_t rows, size_t cols) {
    std::memcpy(c, x, rows*cols*sizeof(float));

    #pragma omp parallel for simd
    for (size_t i = 0; i < rows; i++) {
        c[i*cols+(int)y[i]]--;
    }
}
