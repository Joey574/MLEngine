#include "MathUtils.hpp"

// TODO: actually implement clear/accumulate behaviour right now it's just accumulate for MatrixColumnSum

void MathUtils::Normalize(float* __restrict a, float lower, float upper, size_t n) {
    float* pmin; float* pmax;
    std::tie(pmin, pmax) = std::minmax_element(a, a+n);
    float min = *pmin; float max = *pmax;

    #pragma omp parallel for simd
    for (size_t i = 0; i < n; i++) {
        a[i] = lower + ((a[i]-min)/(max-min)*(upper-lower));
    }
}
void MathUtils::NormalizeCol(float* __restrict a, float lower, float upper, size_t rows, size_t cols, size_t c) {
    float min, max;
    std::tie(min, max) = ColMinMax(a, rows, cols, c);

    NormalizeCol(a, lower, upper, min, max, rows, cols, c);
}

std::pair<float,float> MathUtils::ColMinMax(float* __restrict a, size_t rows, size_t cols, size_t c) {
    float min = a[c];
    float max = a[c];

    // get min/max values
    #pragma omp parallel for simd
    for (size_t i = 0; i < rows; i++) {
        const size_t idx = i*cols+c;

        if (a[idx] > max) { max = a[idx]; }
        if (a[idx] < min) { min = a[idx]; }
    }

    return { min, max };
}
void MathUtils::NormalizeCol(float* __restrict a, float lower, float upper, float min, float max, size_t rows, size_t cols, size_t c) {
    // normalize to range
    #pragma omp parallel for simd
    for (size_t i = 0; i < rows; i++) {
        const size_t idx = i*cols+c;

        a[idx] = lower + ((a[idx]-min)/(max-min)*(upper-lower));
    }
}

#if defined(__AVX512F__)
void MathUtils::Scale(float* __restrict a, float scale, size_t n) {
    AVX512_VALID_PATH();

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
void MathUtils::Scale(float* a, float scale, size_t n) {
    AVX2_VALID_PATH();

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
void MathUtils::Scale(float* __restrict a, float scale, size_t n) {
    SCALAR_VALID_PATH();

    #pragma omp simd
    for (size_t i = 0; i < n; i++) {
        a[i] *= scale;
    }
}
#endif
