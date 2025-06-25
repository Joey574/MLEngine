#include "Activation.hpp"
#include "../MathUtils/MathUtils.hpp"


void Activation::Linear(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    return;
}
void Activation::Sigmoid(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    const __m256 _one = _mm256_set1_ps(1.0f);
    const __m256 _zero = _mm256_setzero_ps();
    const size_t n = r*c;

    #pragma omp parallel for
    for (ssize_t i = 0; i <= ((ssize_t)n)-8; i+= 8) {
        const __m256 _x = _mm256_load_ps(&x[i]);
        _mm256_store_ps(&y[i], Sigmoid(_x));
    }

    for (size_t i = n-(n%8); i < n; i++) {
        y[i] = 1.0f / (1.0f + std::exp(-x[i]));
    }
}
void Activation::ReLU(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    const __m256 _zero = _mm256_setzero_ps();
    const size_t n = r*c;

    #pragma omp parallel for
    for (ssize_t i = 0; i <= ((ssize_t)n)-8; i+= 8) {
        const __m256 _x = _mm256_load_ps(&x[i]);
        _mm256_store_ps(&y[i], ReLU(_x));
    }

    for (size_t i = n-(n%8); i < n; i++) {
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}
void Activation::LeakyReLU(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    const __m256 _cof = _mm256_set1_ps(0.1f);
    const __m256 _zero = _mm256_setzero_ps();
    const size_t n = r*c;

    #pragma omp parallel for
    for (ssize_t i = 0; i <= ((ssize_t)n)-8; i+= 8) {
        const __m256 _x = _mm256_load_ps(&x[i]);
        _mm256_store_ps(&y[i], LeakyReLU(_x));
    }

    for (size_t i = n-(n%8); i < n; i++) {
        y[i] = x[i] > 0.0f ? x[i] : (x[i] * 0.1f);
    }
}
void Activation::ELU(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    const __m256 _one = _mm256_set1_ps(1.0f);
    const __m256 _zero = _mm256_setzero_ps();
    const size_t n = r*c;

    #pragma omp parallel for
    for (ssize_t i = 0; i <= ((ssize_t)n)-8; i+= 8) {
        const __m256 _x = _mm256_load_ps(&x[i]);
        _mm256_store_ps(&y[i], ELU(_x));
    }

    for (size_t i = n-(n%8); i < n; i++) {
        y[i] = x[i] > 0.0f ? x[i] : (std::exp(x[i]) - 1.0f);
    }
}
void Activation::Softmax(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    
}

__m256 Activation::Linear(const __m256 _x) {
    return _x;
}
__m256 Activation::Sigmoid(const __m256 _x) {
    const __m256 _zero = _mm256_setzero_ps();
    const __m256 _one = _mm256_set1_ps(1.0f);

    const __m256 _nx = _mm256_sub_ps(_zero, _x);
        
    const __m256 _ex = MathUtils::Exp256(_nx);

    const __m256 _x2 = _mm256_add_ps(_one, _ex);
    const __m256 _res = _mm256_rcp_ps(_x2);
    return _res;
}
__m256 Activation::ReLU(const __m256 _x) {
    const __m256 _zero = _mm256_setzero_ps();
    const __m256 _res = _mm256_max_ps(_x, _zero);
    return _res;
}
__m256 Activation::LeakyReLU(const __m256 _x) {
    const __m256 _cof = _mm256_set1_ps(0.1f);
    const __m256 _zero = _mm256_setzero_ps();
    const __m256 _x2 = _mm256_mul_ps(_x, _cof);
    const __m256 _res = _mm256_max_ps(_x2, _x);
    return _res;
}
__m256 Activation::ELU(const __m256 _x) {
    const __m256 _one = _mm256_set1_ps(1.0f);
    const __m256 _zero = _mm256_setzero_ps();

    const __m256 _x2 = MathUtils::Exp256(_x);
    const __m256 _x3 = _mm256_sub_ps(_x2, _one);
        
    const __m256 _mask = _mm256_cmp_ps(_x, _zero, _CMP_GT_OS);
    const __m256 _res = _mm256_blendv_ps(_x3, _x, _mask);
    return _res;
}
