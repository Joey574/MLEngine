#include "Optimizer.hpp"

#if defined(__AVX512F__)
void Optimizer::SGDCompute(float* __restrict p, const float* __restrict d, size_t size, float lr, size_t n) {
    assert(__builtin_cpu_supports("avx512f"));

    // adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;
    const __m512 _factor = _mm512_set1_ps(factor);

	// update parameters
	#pragma omp parallel for
	for (ssize_t i = 0; i <= ((ssize_t)size)-16; i += 16) {
        const __m512 _d = _mm512_load_ps(&d[i]);
	    const __m512 _p = _mm512_load_ps(&p[i]);
        const __m512 _res = _mm512_fnmadd_ps(_d, _factor, _p);

	    _mm512_store_ps(&p[i], _res);
	}

	for (size_t i = size-(size%16); i < size; i++) {
        p[i] -= d[i]*factor;		
	}
}
void Optimizer::SGDL1Compute(float* __restrict p, const float* __restrict d, size_t size, float lr, size_t n, float lambda) {
    assert(__builtin_cpu_supports("avx512f"));

    // adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;
    const __m512 _factor = _mm512_set1_ps(factor);
    const __m512 _lambda = _mm512_set1_ps(lambda);

    // update parameters
    #pragma omp parallel for
    for (ssize_t i = 0; i <= ((ssize_t)size)-16; i += 16) {
        const __m512 _none = _mm512_set1_ps(-1.0f);
        const __m512 _one = _mm512_set1_ps(1.0f);
        const __m512 _zero = _mm512_setzero_ps();

        const __m512 _d = _mm512_load_ps(&d[i]);
        const __m512 _p = _mm512_load_ps(&p[i]);

        const __mmask16 _mask = _mm512_cmp_ps_mask(_p, _zero, _CMP_GT_OS);
        const __m512 _sign = _mm512_mask_blend_ps(_mask, _none, _one);

        const __m512 _d2 = _mm512_fmadd_ps(_sign, _lambda, _d);
        const __m512 _res = _mm512_fnmadd_ps(_d2, _factor, _p);

        _mm512_store_ps(&p[i], _res);
    }

    for (size_t i = size-(size%16); i < size; i++) {
        const float sign = (p[i] > 0.0f) - (p[i] < 0.0f);
        p[i] -= factor*(d[i]+(lambda*sign));	
    }
}
void Optimizer::SGDL2Compute(float* __restrict p, const float* __restrict d, size_t size, float lr, size_t n, float lambda) {
    assert(__builtin_cpu_supports("avx512f"));

    // adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;
    const __m512 _factor = _mm512_set1_ps(factor);
    const __m512 _lambda = _mm512_set1_ps(lambda);

    // update parameters
    #pragma omp parallel for
    for (ssize_t i = 0; i <= ((ssize_t)n)-16; i += 16) {
        const __m512 _d = _mm512_load_ps(&d[i]);
        const __m512 _p = _mm512_load_ps(&p[i]);

        const __m512 _d2 = _mm512_fmadd_ps(_p, _lambda, _d);
        const __m512 _res = _mm512_fnmadd_ps(_d2, _factor, _p);

        _mm512_store_ps(&p[i], _res);
    }

    for (size_t i = size-(size%16); i < size; i++) {
        p[i] -= (factor*(d[i]+(lambda*p[i])));
    }
}
#elif defined(__AVX2__) && defined(__FMA__)
void Optimizer::SGDCompute(float* __restrict p, const float* __restrict d, size_t size, float lr, size_t n) {
    assert(__builtin_cpu_supports("avx2"));
    assert(__builtin_cpu_supports("fma"));

    // adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;
    const __m256 _factor = _mm256_set1_ps(factor);

	// update parameters
	#pragma omp parallel for
	for (ssize_t i = 0; i <= ((ssize_t)size)-8; i += 8) {
        const __m256 _d = _mm256_load_ps(&d[i]);
	    const __m256 _p = _mm256_load_ps(&p[i]);
        const __m256 _res = _mm256_fnmadd_ps(_d, _factor, _p);

	    _mm256_store_ps(&p[i], _res);
	}

	for (size_t i = size-(size%8); i < size; i++) {
        p[i] -= d[i]*factor;		
	}
}
void Optimizer::SGDL1Compute(float* __restrict p, const float* __restrict d, size_t size, float lr, size_t n, float lambda) {
    assert(__builtin_cpu_supports("avx2"));
    assert(__builtin_cpu_supports("fma"));

    // adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;
    const __m256 _factor = _mm256_set1_ps(factor);
    const __m256 _lambda = _mm256_set1_ps(lambda);

    // update parameters
    #pragma omp parallel for
    for (ssize_t i = 0; i <= ((ssize_t)size)-8; i += 8) {
        const __m256 _none = _mm256_set1_ps(-1.0f);
        const __m256 _one = _mm256_set1_ps(1.0f);
        const __m256 _zero = _mm256_setzero_ps();

        const __m256 _d = _mm256_load_ps(&d[i]);
        const __m256 _p = _mm256_load_ps(&p[i]);

        const __m256 _mask = _mm256_cmp_ps(_p, _zero, _CMP_GT_OS);
        const __m256 _sign = _mm256_blendv_ps(_none, _one, _mask);

        const __m256 _d2 = _mm256_fmadd_ps(_sign, _lambda, _d);
        const __m256 _res = _mm256_fnmadd_ps(_d2, _factor, _p);

        _mm256_store_ps(&p[i], _res);
    }

    for (size_t i = size-(size%8); i < size; i++) {
        const float sign = (p[i] > 0.0f) - (p[i] < 0.0f);
        p[i] -= factor*(d[i]+(lambda*sign));	
    }
}
void Optimizer::SGDL2Compute(float* __restrict p, const float* __restrict d, size_t size, float lr, size_t n, float lambda) {
    assert(__builtin_cpu_supports("avx2"));
    assert(__builtin_cpu_supports("fma"));

    // adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;
    const __m256 _factor = _mm256_set1_ps(factor);
    const __m256 _lambda = _mm256_set1_ps(lambda);

    // update parameters
    #pragma omp parallel for
    for (ssize_t i = 0; i <= ((ssize_t)size)-8; i += 8) {
        const __m256 _d = _mm256_load_ps(&d[i]);
        const __m256 _p = _mm256_load_ps(&p[i]);

        const __m256 _d2 = _mm256_fmadd_ps(_p, _lambda, _d);
        const __m256 _res = _mm256_fnmadd_ps(_d2, _factor, _p);

        _mm256_store_ps(&p[i], _res);
    }

    for (size_t i = size-(size%8); i < size; i++) {
        p[i] -= (factor*(d[i]+(lambda*p[i])));
    }
}
#else
void Optimizer::SGDCompute(float* __restrict p, const float* __restrict d, size_t size, float lr, size_t n) {
    // adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;

	// update parameters
	#pragma omp parallel for simd
	for (size_t i = 0; i < size; i++) {
        p[i] -= d[i]*factor;		
	}
}
void Optimizer::SGDL1Compute(float* __restrict p, const float* __restrict d, size_t size, float lr, size_t n, float lambda) {
    // adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;

    #pragma omp parallel for simd
    for (size_t i = 0; i < size; i++) {
        const float sign = (p[i] > 0.0f) - (p[i] < 0.0f);
        p[i] -= factor*(d[i]+(lambda*sign));	
    }
}
void Optimizer::SGDL2Compute(float* __restrict p, const float* __restrict d, size_t size, float lr, size_t n, float lambda) {
    // adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;

    // update parameters
    #pragma omp parallel for simd
    for (size_t i = 0; i < size; i++) {
        p[i] -= (factor*(d[i]+(lambda*p[i])));
    }
}
#endif