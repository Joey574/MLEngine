#include "NeuralNetwork.hpp"

void NeuralNetwork::SigmoidDerivative(const float* __restrict x, float* __restrict y, size_t n) {
    const __m256 _one = _mm256_set1_ps(1.0f);
    const __m256 _zero = _mm256_setzero_ps();

    #pragma omp parallel for
    for (size_t i = 0; i <= n-8; i+= 8) {
        const __m256 _x = _mm256_loadu_ps(&x[i]);
        const __m256 _nx = _mm256_sub_ps(_zero, _x);

        const __m256 _ex = Exp256(_nx);

        const __m256 _x2 = _mm256_add_ps(_one, _ex);
        const __m256 _ires = _mm256_rcp_ps(_x2);

        const __m256 _nires = _mm256_sub_ps(_one, _ires);
        const __m256 _res = _mm256_mul_ps(_ires, _nires);

        _mm256_storeu_ps(&y[i], _res);
    }

    for (size_t i = n-(n%8); i < n; i++) {
        float s = 1.0f / (1.0f + std::exp(-x[i]));
        y[i] = s * (1.0f - s);
    }
}

void NeuralNetwork::ReLUDerivative(const float* __restrict x, float* __restrict y, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        y[i] = x[i] > 0.0f ? y[i] : 0.0f;
    }
}

void NeuralNetwork::LeakyReLUDerivative(const float* __restrict x, float* __restrict y, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        y[i] = x[i] > 0.0f ? y[i] : (y[i] * 0.1f);
    }
}

void NeuralNetwork::ELUDerivative(const float* __restrict x, float* __restrict y, size_t n) {
    const __m256 _one = _mm256_set1_ps(1.0f);
    const __m256 _zero = _mm256_setzero_ps();

    #pragma omp parallel for
    for (size_t i = 0; i <= n-8; i += 8) {
        const __m256 _x = _mm256_loadu_ps(&x[i]);
        const __m256 _ex = Exp256(_x);

        const __m256 _mask = _mm256_cmp_ps(_x, _zero, _CMP_GT_OS);
        const __m256 _res = _mm256_blendv_ps(_ex, _one, _mask);

        _mm256_storeu_ps(&y[i], _res);
    }

    for (size_t i = n-(n%8); i < n; i++) {
        y[i] = x[i] > 0.0f ? 1.0f : std::exp(x[i]);
    }
}