#include "Optimizer.hpp"

__attribute__((target("avx2,fma")))
void Optimizer::SGDComputeAVX2(float* __restrict p, const float* __restrict d, size_t n, float lr) {

    // adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;
    const __m256 _factor = _mm256_set1_ps(factor);

	// update parameters
	#pragma omp parallel for
	for (ssize_t i = 0; i <= ((ssize_t)n)-8; i += 8) {
        const __m256 _d = _mm256_load_ps(d);
	    const __m256 _p = _mm256_load_ps(p);
        const __m256 _res = _mm256_fnmadd_ps(_d, _factor, _p);

	    _mm256_store_ps(p, _res);
	}

	for (size_t i = n-(n%8); i < n; i++) {
        p[i] -= d[i]*factor;		
	}
}

__attribute__((target("avx2,fma")))
void Optimizer::SGDL1ComputeAVX2(float* p, const float* d, size_t n, float lr, float lambda) {

    // adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;
    const __m256 _factor = _mm256_set1_ps(factor);
    const __m256 _lambda = _mm256_set1_ps(lambda);

    // update parameters
    #pragma omp parallel for
    for (ssize_t i = 0; i <= ((ssize_t)n)-8; i += 8) {
        const __m256 _none = _mm256_set1_ps(-1.0f);
        const __m256 _one = _mm256_set1_ps(1.0f);
        const __m256 _zero = _mm256_setzero_ps();

        const __m256 _d = _mm256_load_ps(d);
        const __m256 _p = _mm256_load_ps(p);

        const __m256 _mask = _mm256_cmp_ps(_p, _zero, _CMP_GT_OS);
        const __m256 _sign = _mm256_blendv_ps(_none, _one, _mask);

        const __m256 _d2 = _mm256_fmadd_ps(_sign, _lambda, _d);
        const __m256 _res = _mm256_fnmadd_ps(_d2, _factor, _p);

        _mm256_store_ps(p, _res);
    }

    for (size_t i = n-(n%8); i < n; i++) {
        const float sign = p[i] > 0.0f ? 1.0f : -1.0f;
        p[i] -= factor*(d[i]+(lambda*sign));	
    }
}

__attribute__((target("avx2,fma")))
void Optimizer::SGDL2ComputeAVX2(float* p, const float* d, size_t n, float lr, float lambda) {
    // adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;
    const __m256 _factor = _mm256_set1_ps(factor);
    const __m256 _lambda = _mm256_set1_ps(lambda);

    // update parameters
    #pragma omp parallel for
    for (ssize_t i = 0; i <= ((ssize_t)n)-8; i += 8) {
        const __m256 _d = _mm256_load_ps(d);
        const __m256 _p = _mm256_load_ps(p);

        const __m256 _d2 = _mm256_fmadd_ps(_p, _lambda, _d);
        const __m256 _res = _mm256_fnmadd_ps(_d2, _factor, _p);

        _mm256_store_ps(p, _res);
    }

    for (size_t i = n-(n%8); i < n; i++) {
        p[i] -= (factor*(d[i]+(lambda*p[i])));
    }
}
