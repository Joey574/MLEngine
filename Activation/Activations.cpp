#include "Activation.hpp"

void Activation::Linear(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    std::memcpy(y, x, r*c*sizeof(float));
}

#if defined(__AVX512F__)
void Activation::Sigmoid(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    AVX512_VALID_PATH();

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
    AVX512_VALID_PATH();

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
    AVX512_VALID_PATH();

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
    AVX512_VALID_PATH();

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
void Activation::Softmax(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    AVX512_VALID_PATH();

    #pragma omp parallel for
    for (size_t i = 0; i < r; i++) {
        size_t j;

        // get max element for numerical stability
        __m512 _max = _mm512_set1_ps(x[i*c+0]);
        for (j = 1; j+16 <= c; j += 16) {
            const __m512 _x = _mm512_load_ps(&x[i*c+j]);
            _max = _mm512_max_ps(_max, _x);
        }

        float max = MathUtils::Max512(_max);
        for (; j < c; j++) {
            if (x[i*c+j] > max) { max = x[i*c+j]; }
        }


        // get row sum
        __m512 _sum = _mm512_setzero_ps();
        const __m512 _max2 = _mm512_set1_ps(max);

        for (j = 0; j+16 < c; j += 16) {
            const __m512 _x = _mm512_load_ps(&x[i*c+j]);
            const __m512 _x2 = _mm512_sub_ps(_x, _max2);

            const __m512 _res = MathUtils::Exp512(_x2);
            _sum = _mm512_add_ps(_sum, _res);

            _mm512_store_ps(&y[i*c+j], _res);
        }

        float sum = MathUtils::Sum512(_sum);
        for (; j < c; j++) {
            y[i*c+j] = std::exp(x[i*c+j]-max);
            sum += y[i*c+j];
        }


        // normalize
        float inv = 1.0f/sum;
        const __m512 _inv = _mm512_set1_ps(inv);

        for (j = 0; j+16 < c; j += 16) {
            const __m512 _y = _mm512_load_ps(&y[i*c+j]);
            const __m512 _res = _mm512_mul_ps(_y, _inv);

            _mm512_store_ps(&y[i*c+j], _res);
        }


        for (; j < c; j++) {
            y[i*c+j] *= inv;
        }
    }
}
#elif defined(__AVX2__) && defined(__FMA__)
void Activation::Sigmoid(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    AVX2_VALID_PATH();

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
    AVX2_VALID_PATH();

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
    AVX2_VALID_PATH();

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
    AVX2_VALID_PATH();
    
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
void Activation::Softmax(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    AVX2_VALID_PATH();

    #pragma omp parallel for
    for (size_t i = 0; i < r; i++) {
        size_t j;

        // get max element for numerical stability
        __m256 _max = _mm256_set1_ps(x[i*c+0]);
        for (j = 1; j+8 < c; j += 8) {
            const __m256 _x = _mm256_load_ps(&x[i*c+j]);
            _max = _mm256_max_ps(_max, _x);
        }

        float max = MathUtils::Max256(_max);
        for (; j < c; j++) {
            if (x[i*c+j] > max) { max = x[i*c+j]; }
        }


        // get row sum
        __m256 _sum = _mm256_setzero_ps();
        const __m256 _max2 = _mm256_set1_ps(max);

        for (j = 0; j+8 <= c; j += 8) {
            const __m256 _x = _mm256_load_ps(&x[i*c+j]);
            const __m256 _x2 = _mm256_sub_ps(_x, _max2);

            const __m256 _res = MathUtils::Exp256(_x2);
            _sum = _mm256_add_ps(_sum, _res);

            _mm256_store_ps(&y[i*c+j], _res);
        }

        float sum = MathUtils::Sum256(_sum);
        for (; j < c; j++) {
            y[i*c+j] = std::exp(x[i*c+j]-max);
            sum += y[i*c+j];
        }


        // normalize
        float inv = 1.0f/sum;
        const __m256 _inv = _mm256_set1_ps(inv);

        for (j = 0; j+8 < c; j += 8) {
            const __m256 _y = _mm256_load_ps(&y[i*c+j]);
            const __m256 _res = _mm256_mul_ps(_y, _inv);

            _mm256_store_ps(&y[i*c+j], _res);
        }


        for (; j < c; j++) {
            y[i*c+j] *= inv;
        }
    }
}
#else
void Activation::Sigmoid(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    SCALAR_VALID_PATH();

    const size_t n = r*c;

    #pragma omp parallel for simd
    for (size_t i = 0; i < n; i++) {
        y[i] = 1.0f / (1.0f + std::exp(-x[i]));
    }
}
void Activation::ReLU(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    SCALAR_VALID_PATH();

    const size_t n = r*c;

    #pragma omp parallel for simd
    for (size_t i = 0; i < n; i++) {
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}
void Activation::LeakyReLU(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    SCALAR_VALID_PATH();

    const size_t n = r*c;

    #pragma omp parallel for simd
    for (size_t i = 0; i < n; i++) {
        y[i] = x[i] > 0.0f ? x[i] : (x[i] * 0.1f);
    }
}
void Activation::ELU(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    SCALAR_VALID_PATH();

    const size_t n = r*c;

    #pragma omp parallel for simd
    for (size_t i = 0; i < n; i++) {
        y[i] = x[i] > 0.0f ? x[i] : (std::exp(x[i]) - 1.0f);
    }
}
void Activation::Softmax(const float* __restrict x, float* __restrict y, size_t r, size_t c) {
    SCALAR_VALID_PATH();
    
    #pragma omp parallel for
    for (size_t i = 0; i < r; i++) {

        // get max element for numerical stability
        float max = x[i*c+0];
        #pragma omp simd
        for (size_t j = 1; j < c; j++) {
            if (x[i*c+j] > max) { max = x[i*c+j]; }
        }


        // get row sum
        float sum = 0.0f;
        #pragma omp simd
        for (size_t j = 0; j < c; j++) {
            y[i*c+j] = std::exp(x[i*c+j]-max);
            sum += y[i*c+j];
        }


        // normalize
        float inv = 1.0f/sum;
        #pragma omp simd
        for (size_t j = 0; j < c; j++) {
            y[i*c+j] *= inv;
        }
    }
}
#endif
