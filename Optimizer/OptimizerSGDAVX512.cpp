#include "Optimizer.hpp"

__attribute__((target("avx512f")))
void Optimizer::SGDComputeAVX512(float* __restrict p, const float* __restrict d, size_t n, float lr) {
    // adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;
    const __m512 _factor = _mm512_set1_ps(factor);

	// update parameters
	#pragma omp parallel for
	for (ssize_t i = 0; i <= ((ssize_t)n)-16; i += 16) {
        const __m512 _d = _mm512_load_ps(d);
	    const __m512 _p = _mm512_load_ps(p);
        const __m512 _res = _mm512_fnmadd_ps(_d, _factor, _p);

	    _mm512_store_ps(p, _res);
	}

	for (size_t i = n-(n%16); i < n; i++) {
        p[i] -= d[i]*factor;		
	}
}

__attribute__((target("avx512f")))
void Optimizer::SGDL1ComputeAVX512(float* __restrict p, const float* __restrict d, size_t n, float lr, float lambda) {
    // adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;
    const __m512 _factor = _mm512_set1_ps(factor);
    const __m512 _lambda = _mm512_set1_ps(lambda);

    // update parameters
    #pragma omp parallel for
    for (ssize_t i = 0; i <= ((ssize_t)n)-16; i += 16) {
        const __m512 _none = _mm512_set1_ps(-1.0f);
        const __m512 _one = _mm512_set1_ps(1.0f);
        const __m512 _zero = _mm512_setzero_ps();

        const __m512 _d = _mm512_load_ps(d);
        const __m512 _p = _mm512_load_ps(p);

        const __mmask16 _mask = _mm512_cmp_ps_mask(_p, _zero, _CMP_GT_OS);
        const __m512 _sign = _mm512_mask_blend_ps(_mask, _none, _one);

        const __m512 _d2 = _mm512_fmadd_ps(_sign, _lambda, _d);
        const __m512 _res = _mm512_fnmadd_ps(_d2, _factor, _p);

        _mm512_store_ps(p, _res);
    }

    for (size_t i = n-(n%16); i < n; i++) {
        const float sign = p[i] > 0.0f ? 1.0f : -1.0f;
        p[i] -= factor*(d[i]+(lambda*sign));	
    }
}

__attribute__((target("avx512f")))
void Optimizer::SGDL2ComputeAVX2(float* p, const float* d, size_t n, float lr, float lambda) {
    // adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;
    const __m512 _factor = _mm512_set1_ps(factor);
    const __m512 _lambda = _mm512_set1_ps(lambda);

    // update parameters
    #pragma omp parallel for
    for (ssize_t i = 0; i <= ((ssize_t)n)-16; i += 16) {
        const __m512 _d = _mm512_load_ps(d);
        const __m512 _p = _mm512_load_ps(p);

        const __m512 _d2 = _mm512_fmadd_ps(_p, _lambda, _d);
        const __m512 _res = _mm512_fnmadd_ps(_d2, _factor, _p);

        _mm512_store_ps(p, _res);
    }

    for (size_t i = n-(n%16); i < n; i++) {
        p[i] -= (factor*(d[i]+(lambda*p[i])));
    }
}
