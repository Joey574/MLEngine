#include "Optimizer.hpp"

#if defined(__AVX512F__)

#elif defined(__AVX2__) && defined(__FMA__)
void Optimizer::AdamCompute(float* __restrict p, float* __restrict m, float* __restrict v, const float* __restrict d, size_t size, float lr, size_t n, float b1, float b2, float epsl, size_t t) {

    // adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;
    const __m256 _factor = _mm256_set1_ps(factor);
    const __m256 _b1 = _mm256_set1_ps(b1);
    const __m256 _b2 = _mm256_set1_ps(b2);
    const __m256 _nb1 = _mm256_set1_ps(1.0f-b1);
    const __m256 _nb2 = _mm256_set1_ps(1.0f-b2);
    const __m256 _b1t = _mm256_set1_ps(1.0f-std::pow(b1, (float)t));
    const __m256 _b2t = _mm256_set1_ps(1.0f-std::pow(b2, (float)t));
    const __m256 _epsl = _mm256_set1_ps(epsl);


    // update parameters
    #pragma omp parallel for
    for (ssize_t i = 0; i <= ((ssize_t)size)-8; i += 8) {
        const __m256 _d = _mm256_load_ps(&d[i]);
	const __m256 _p = _mm256_load_ps(&p[i]);
        const __m256 _v = _mm256_load_ps(&v[i]);
        const __m256 _m = _mm256_load_ps(&m[i]);

        const __m256 _m2 = _mm256_mul_ps(_m, _b1);
        const __m256 _m3 = _mm256_fmadd_ps(_nb1, _d, _m2);

        const __m256 _d2 = _mm256_mul_ps(_d, _d);
        const __m256 _v2 = _mm256_mul_ps(_v, _b2);
        const __m256 _v3 = _mm256_fmadd_ps(_nb2, _d2, _v2);

        const __m256 _mh = _mm256_div_ps(_m3, _b1t);
        const __m256 _vh = _mm256_div_ps(_v3, _b2t);

        const __m256 _v4 = _mm256_add_ps(_vh, _epsl);
        const __m256 _v5 = _mm256_sqrt_ps(_v4);

        const __m256 _m4 = _mm256_div_ps(_mh, _v5);
        const __m256 _res = _mm256_fnmadd_ps(_factor, _m4, _p);

        _mm256_store_ps(&p[i], _res);
        _mm256_store_ps(&m[i], _m3);
        _mm256_store_ps(&v[i], _v3);
    }

    for (size_t i = size-(size%8); i < size; i++) {
        m[i] = (m[i]*b1)+(1.0f-b1)*d[i];
        v[i] = (v[i]*b2)+(1.0f-b2)*d[i]*d[i];

        float mh = m[i]/(1.0f-std::pow(b1, (float)t));
        float vh = v[i]/(1.0f-std::pow(b2, (float)t));

        p[i] -= factor*(mh/(std::sqrt(vh+epsl)));
    }
}
#else
void Optimizer::AdamCompute(float* __restrict p, float* __restrict m, float* __restrict v, const float* __restrict d, size_t size, float lr, size_t n, float b1, float b2, float epsl, size_t t) {

    // adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;

    // update parameters
    #pragma omp parallel for simd
    for (size_t i = 0; i < size; i++) {
        m[i] = (m[i]*b1)+(1.0f-b1)*d[i];
        v[i] = (v[i]*b2)+(1.0f-b2)*d[i]*d[i];

        float mh = m[i]/(1.0f-std::pow(b1, (float)t));
        float vh = v[i]/(1.0f-std::pow(b2, (float)t));

        p[i] -= factor*(mh/(std::sqrt(vh+epsl)));
    }
}
#endif
