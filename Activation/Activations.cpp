#include "Activation.hpp"

void Activation::Linear(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    std::memcpy(y, x, r*c*sizeof(float));
}
void Activation::Softmax(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    
}

#if defined(__AVX512F__)
void Activation::Sigmoid(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    assert(__builtin_cpu_supports("avx512f"));

    const __m512 _one = _mm512_set1_ps(1.0f);
    const __m512 _zero = _mm512_setzero_ps();
    const size_t n = r*c;

    #pragma omp parallel for
    for (ssize_t i = 0; i <= ((ssize_t)n)-16; i+= 16) {
        const __m512 _x = _mm512_load_ps(&x[i]);
        
        const __m512 _nx = _mm512_sub_ps(_zero, _x);
        const __m512 _ex = MathUtils::Exp512(_nx);

        const __m512 _x2 = _mm512_add_ps(_one, _ex);
        const __m512 _res = _mm512_rcp14_ps(_x2);

        _mm512_store_ps(&y[i], _res);
    }

    for (size_t i = n-(n%16); i < n; i++) {
        y[i] = 1.0f / (1.0f + std::exp(-x[i]));
    }
}
void Activation::ReLU(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    assert(__builtin_cpu_supports("avx512f"));

    const __m512 _zero = _mm512_setzero_ps();
    const size_t n = r*c;

    #pragma omp parallel for
    for (ssize_t i = 0; i <= ((ssize_t)n)-16; i+= 16) {
        const __m512 _x = _mm512_load_ps(&x[i]);
        const __m512 _res = _mm512_max_ps(_x, _zero);

        _mm512_store_ps(&y[i], _res);
    }

    for (size_t i = n-(n%16); i < n; i++) {
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}
void Activation::LeakyReLU(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    assert(__builtin_cpu_supports("avx512f"));

    const __m512 _cof = _mm512_set1_ps(0.1f);
    const __m512 _zero = _mm512_setzero_ps();
    const size_t n = r*c;

    #pragma omp parallel for
    for (ssize_t i = 0; i <= ((ssize_t)n)-16; i+= 16) {
        const __m512 _x = _mm512_load_ps(&x[i]);

        const __m512 _x2 = _mm512_mul_ps(_x, _cof);
        const __m512 _res = _mm512_max_ps(_x2, _x);

        _mm512_store_ps(&y[i], _res);
    }

    for (size_t i = n-(n%16); i < n; i++) {
        y[i] = x[i] > 0.0f ? x[i] : (x[i] * 0.1f);
    }
}
void Activation::ELU(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    assert(__builtin_cpu_supports("avx512f"));

    const __m512 _one = _mm512_set1_ps(1.0f);
    const __m512 _zero = _mm512_setzero_ps();
    const size_t n = r*c;

    #pragma omp parallel for
    for (ssize_t i = 0; i <= ((ssize_t)n)-16; i+= 16) {
        const __m512 _x = _mm512_load_ps(&x[i]);

        const __m512 _x2 = MathUtils::Exp512(_x);
        const __m512 _x3 = _mm512_sub_ps(_x2, _one);

        const __mmask16 _mask = _mm512_cmp_ps_mask(_x, _zero, _CMP_GT_OS);
        const __m512 _res = _mm512_mask_blend_ps(_mask, _x3, _x);

        _mm512_store_ps(&y[i], _res);
    }

    for (size_t i = n-(n%16); i < n; i++) {
        y[i] = x[i] > 0.0f ? x[i] : (std::exp(x[i]) - 1.0f);
    }
}
#elif defined(__AVX2__) && defined(__FMA__)
void Activation::Sigmoid(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    assert(__builtin_cpu_supports("avx2"));
    assert(__builtin_cpu_supports("fma"));

    const __m256 _one = _mm256_set1_ps(1.0f);
    const __m256 _zero = _mm256_setzero_ps();
    const size_t n = r*c;

    #pragma omp parallel for
    for (ssize_t i = 0; i <= ((ssize_t)n)-8; i+= 8) {
        const __m256 _x = _mm256_load_ps(&x[i]);

        const __m256 _nx = _mm256_sub_ps(_zero, _x);
        const __m256 _ex = MathUtils::Exp256(_nx);

        const __m256 _x2 = _mm256_add_ps(_one, _ex);
        const __m256 _res = _mm256_rcp_ps(_x2);

        _mm256_store_ps(&y[i], _res);
    }

    for (size_t i = n-(n%8); i < n; i++) {
        y[i] = 1.0f / (1.0f + std::exp(-x[i]));
    }
}
void Activation::ReLU(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    assert(__builtin_cpu_supports("avx2"));
    assert(__builtin_cpu_supports("fma"));

    const __m256 _zero = _mm256_setzero_ps();
    const size_t n = r*c;

    #pragma omp parallel for
    for (ssize_t i = 0; i <= ((ssize_t)n)-8; i+= 8) {
        const __m256 _x = _mm256_load_ps(&x[i]);
        const __m256 _res = _mm256_max_ps(_x, _zero);

        _mm256_store_ps(&y[i], _res);
    }

    for (size_t i = n-(n%8); i < n; i++) {
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}
void Activation::LeakyReLU(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    assert(__builtin_cpu_supports("avx2"));
    assert(__builtin_cpu_supports("fma"));

    const __m256 _cof = _mm256_set1_ps(0.1f);
    const __m256 _zero = _mm256_setzero_ps();
    const size_t n = r*c;

    #pragma omp parallel for
    for (ssize_t i = 0; i <= ((ssize_t)n)-8; i+= 8) {
        const __m256 _x = _mm256_load_ps(&x[i]);

        const __m256 _x2 = _mm256_mul_ps(_x, _cof);
        const __m256 _res = _mm256_max_ps(_x2, _x);

        _mm256_store_ps(&y[i], _res);
    }

    for (size_t i = n-(n%8); i < n; i++) {
        y[i] = x[i] > 0.0f ? x[i] : (x[i] * 0.1f);
    }
}
void Activation::ELU(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    assert(__builtin_cpu_supports("avx2"));
    assert(__builtin_cpu_supports("fma"));
    
    const __m256 _one = _mm256_set1_ps(1.0f);
    const __m256 _zero = _mm256_setzero_ps();
    const size_t n = r*c;

    #pragma omp parallel for
    for (ssize_t i = 0; i <= ((ssize_t)n)-8; i+= 8) {
        const __m256 _x = _mm256_load_ps(&x[i]);

        const __m256 _x2 = MathUtils::Exp256(_x);
        const __m256 _x3 = _mm256_sub_ps(_x2, _one);
        
        const __m256 _mask = _mm256_cmp_ps(_x, _zero, _CMP_GT_OS);
        const __m256 _res = _mm256_blendv_ps(_x3, _x, _mask);

        _mm256_store_ps(&y[i], _res);
    }

    for (size_t i = n-(n%8); i < n; i++) {
        y[i] = x[i] > 0.0f ? x[i] : (std::exp(x[i]) - 1.0f);
    }
}
#else
void Activation::Sigmoid(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    const size_t n = r*c;

    #pragma omp parallel for simd
    for (size_t i = 0; i < n; i++) {
        y[i] = 1.0f / (1.0f + std::exp(-x[i]));
    }
}
void Activation::ReLU(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    const size_t n = r*c;

    #pragma omp parallel for simd
    for (size_t i = 0; i < n; i++) {
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}
void Activation::LeakyReLU(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    const size_t n = r*c;

    #pragma omp parallel for simd
    for (size_t i = 0; i < n; i++) {
        y[i] = x[i] > 0.0f ? x[i] : (x[i] * 0.1f);
    }
}
void Activation::ELU(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    const size_t n = r*c;

    #pragma omp parallel for simd
    for (size_t i = 0; i < n; i++) {
        y[i] = x[i] > 0.0f ? x[i] : (std::exp(x[i]) - 1.0f);
    }
}
#endif
