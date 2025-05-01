#include "NeuralNetwork.hpp"

void NeuralNetwork::Sigmoid(float* __restrict x, float* __restrict y, size_t n) {
    const __m256 _one = _mm256_set1_ps(1.0f);
    const __m256 _zero = _mm256_setzero_ps();

    #pragma omp parallel for
    for (size_t i = 0; i <= n-8; i += 8) {
        const __m256 _x = _mm256_loadu_ps(&x[i]);
        const __m256 _nx = _mm256_sub_ps(_zero, _x);
        
        const __m256 _ex = Exp256(_nx);

        const __m256 _x2 = _mm256_add_ps(_one, _ex);
        const __m256 _res = _mm256_rcp_ps(_x2);

        _mm256_storeu_ps(&y[i], _res);
    }

    for (size_t i = n-(n%8); i < n; i++) {
        y[i] = 1.0f / (1.0f + std::exp(-x[i]));
    }
}
void NeuralNetwork::ReLU(float* __restrict x, float* __restrict y, size_t n) {
    const __m256 _zero = _mm256_setzero_ps();

    #pragma omp parallel for
    for (size_t i = 0; i <= n-8; i += 8) {
        const __m256 _x = _mm256_loadu_ps(&x[i]);
        const __m256 _res = _mm256_max_ps(_x, _zero);

        _mm256_storeu_ps(&y[i], _res);
    }

    for (size_t i = n-(n%8); i < n; i++) {
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}
void NeuralNetwork::LeakyReLU(float* __restrict x, float* __restrict y, size_t n) {
    const __m256 _cof = _mm256_set1_ps(0.1f);
    const __m256 _zero = _mm256_setzero_ps();

    #pragma omp parallel for
    for (size_t i = 0; i <= n - 8; i += 8) {
        const __m256 _x = _mm256_loadu_ps(&x[i]);
        const __m256 _x2 = _mm256_mul_ps(_x, _cof);

        const __m256 _res = _mm256_max_ps(_x2, _x);
        
        _mm256_storeu_ps(&y[i], _res);
    }

    for (size_t i = n-(n%8); i < n; i++) {
        y[i] = x[i] > 0.0f ? x[i] : (x[i] * 0.1f);
    }
}
void NeuralNetwork::ELU(float* __restrict x, float* __restrict y, size_t n) {
    const __m256 _one = _mm256_set1_ps(1.0f);
    const __m256 _zero = _mm256_setzero_ps();

    #pragma omp parallel for
    for (size_t i = 0; i <= n-8; i += 8) {
        const __m256 _x = _mm256_load_ps(&x[i]);
        const __m256 _x2 = Exp256(_x);
        const __m256 _x3 = _mm256_sub_ps(_x2, _one);
        
        const __m256 _mask = _mm256_cmp_ps(_x, _zero, _CMP_GT_OS);
        const __m256 _res = _mm256_blendv_ps(_x3, _x, _mask);

        _mm256_store_ps(&y[i], _res);
    }

    for (size_t i = n-(n%8); i < n; i++) {
        y[i] = x[i] > 0.0f ? x[i] : (std::exp(x[i]) - 1.0f);
    }
}

void NeuralNetwork::Softmax(float* __restrict x, float* __restrict y, size_t n) {
    
}