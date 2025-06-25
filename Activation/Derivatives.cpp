#include "Activation.hpp"
#include "../MathUtils/MathUtils.hpp"

void Activation::LinearDerivative(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    return;
}
void Activation::SigmoidDerivative(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    const __m256 _zero = _mm256_setzero_ps();
    const __m256 _one = _mm256_set1_ps(1.0f);
    const size_t n = r*c;

    #pragma omp parallel for
    for (size_t i = 0; i <= ((ssize_t)n)-8; i+= 8) {
        const __m256 _x = _mm256_load_ps(&x[i]);
        const __m256 _y = _mm256_load_ps(&y[i]);
        
        const __m256 _nx = _mm256_sub_ps(_zero, _x);

        const __m256 _ex = MathUtils::Exp256(_nx);

        const __m256 _x2 = _mm256_add_ps(_one, _ex);
        const __m256 _ires = _mm256_rcp_ps(_x2);

        const __m256 _nires = _mm256_sub_ps(_one, _ires);
        const __m256 _x3 = _mm256_mul_ps(_ires, _nires);
        const __m256 _res = _mm256_mul_ps(_x3, _y);

        _mm256_store_ps(&y[i], _res);
    }

    for (size_t i = n-(n%8); i < n; i++) {
        float s = 1.0f / (1.0f + std::exp(-x[i]));
        y[i] *= s * (1.0f - s);
    }
}
void Activation::ReLUDerivative(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    const size_t n = r*c;
 
    #pragma omp parallel for simd
    for (size_t i = 0; i < n; i++) {
        y[i] = x[i] > 0.0f ? y[i] : 0.0f;
    }
}
void Activation::LeakyReLUDerivative(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    const size_t n = r*c;

    #pragma omp parallel for simd
    for (size_t i = 0; i < n; i++) {
        y[i] = x[i] > 0.0f ? y[i] : (y[i] * 0.1f);
    }
}
void Activation::ELUDerivative(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    const __m256 _zero = _mm256_setzero_ps();
    const __m256 _one = _mm256_set1_ps(1.0f);
    const size_t n = r*c;

    #pragma omp parallel for
    for (ssize_t i = 0; i <= ((ssize_t)n)-8; i+= 8) {
        const __m256 _x = _mm256_load_ps(&x[i]);
        const __m256 _y = _mm256_load_ps(&y[i]);

        const __m256 _ex = MathUtils::Exp256(_x);

        const __m256 _mask = _mm256_cmp_ps(_x, _zero, _CMP_GT_OS);
        const __m256 _x2 = _mm256_blendv_ps(_ex, _one, _mask);
        const __m256 _res = _mm256_mul_ps(_x2, _y);
        
        _mm256_store_ps(&y[i], _res);
    }

    for (size_t i = n-(n%8); i < n; i++) {
        y[i] = x[i] > 0.0f ? y[i] : (y[i] * std::exp(x[i]));
    }
}
