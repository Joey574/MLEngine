#include "Optimizer.hpp"

#if defined(__AVX512F__)
void Optimizer::MomentumSGDCompute(float* __restrict  p, float* __restrict  v, const float* __restrict  d, size_t size, float lr, size_t n, float coef) {

    // adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;
    const __m512 _factor = _mm512_set1_ps(factor);
    const __m512 _coef = _mm512_set1_ps(coef);

	// update parameters
	#pragma omp parallel for
    for (ssize_t i = 0; i <= (ssize_t)size-16; i += 16) {
        const __m512 _d = _mm512_load_ps(&d[i]);
        const __m512 _v = _mm512_load_ps(&v[i]);
        const __m512 _p = _mm512_load_ps(&p[i]);

        const __m512 _v1 = _mm512_mul_ps(_v, _coef);
        const __m512 _v2 = _mm512_fnmadd_ps(_d, _factor, _v1);
        const __m512 _res = _mm512_sub_ps(_p, _v2);

        _mm512_store_ps(&p[i], _res);
        _mm512_store_ps(&v[i], _v2);
    }

	for (size_t i = size-(size%16); i < size; i++) {
        v[i] = (v[i]*coef)+(d[i]*factor);
        p[i] -= v[i];		
	}
}
void Optimizer::MomentumSGDL1Compute(float* __restrict p, float* __restrict  v, const float* __restrict d, size_t size, float lr, size_t n, float lambda, float coef) {

    // adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;
    const __m512 _factor = _mm512_set1_ps(factor);
    const __m512 _coef = _mm512_set1_ps(coef);
    const __m512 _lambda = _mm512_set1_ps(lambda);
    
    const __m512 _none = _mm512_set1_ps(-1.0f);
    const __m512 _one = _mm512_set1_ps(1.0f);
    const __m512 _zero = _mm512_setzero_ps();

	// update parameters
	#pragma omp parallel for
    for (ssize_t i = 0; i <= (ssize_t)size-16; i += 16) {
        const __m512 _d = _mm512_load_ps(&d[i]);
        const __m512 _v = _mm512_load_ps(&v[i]);
        const __m512 _p = _mm512_load_ps(&p[i]);

        const __mmask16 _mask = _mm512_cmp_ps_mask(_p, _zero, _CMP_GT_OS);
        const __m512 _sign = _mm512_mask_blend_ps(_mask, _none, _one);

        const __m512 _v1 = _mm512_mul_ps(_v, _coef);
        const __m512 _v2 = _mm512_fnmadd_ps(_d, _factor, _v1);

        const __m512 _v3 = _mm512_fmadd_ps(_lambda, _sign, _v2);
        const __m512 _res = _mm512_sub_ps(_p, _v3);

        _mm512_store_ps(&p[i], _res);
        _mm512_store_ps(&v[i], _v2);
    }

    for (size_t i = size-(size%16); i < size; i++) {
        const float sign = p[i] > 0.0f ? 1.0f : -1.0f;
        v[i] = (v[i]*coef)+(d[i]*factor);
        p[i] -= (v[i]+(lambda*sign));	
    }
}
void Optimizer::MomentumSGDL2Compute(float* __restrict p, float* __restrict  v, const float* __restrict d, size_t size, float lr, size_t n, float lambda, float coef) {
    
    // adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;
    const __m512 _factor = _mm512_set1_ps(factor);
    const __m512 _coef = _mm512_set1_ps(coef);
    const __m512 _lambda = _mm512_set1_ps(lambda);

	// update parameters
	#pragma omp parallel for
    for (ssize_t i = 0; i <= (ssize_t)size-16; i += 16) {
        const __m512 _d = _mm512_load_ps(&d[i]);
        const __m512 _v = _mm512_load_ps(&v[i]);
        const __m512 _p = _mm512_load_ps(&p[i]);

        const __m512 _v1 = _mm512_mul_ps(_v, _coef);
        const __m512 _v2 = _mm512_fnmadd_ps(_d, _factor, _v1);

        const __m512 _v3 = _mm512_fmadd_ps(_lambda, _p, _v2);
        const __m512 _res = _mm512_sub_ps(_p, _v3);

        _mm512_store_ps(&p[i], _res);
        _mm512_store_ps(&v[i], _v2);
    }

    for (size_t i = size-(size%16); i < size; i++) {
        v[i] = (v[i]*coef)+(d[i]*factor);
        p[i] -= v[i]+(lambda*p[i]);
    }
}

#elif defined(__AVX2__) && defined(__FMA__)
void Optimizer::MomentumSGDCompute(float* __restrict  p, float* __restrict  v, const float* __restrict  d, size_t size, float lr, size_t n, float coef) {

    // adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;
    const __m256 _factor = _mm256_set1_ps(factor);
    const __m256 _coef = _mm256_set1_ps(coef);

	// update parameters
	#pragma omp parallel for
    for (ssize_t i = 0; i <= (ssize_t)size-8; i += 8) {
        const __m256 _d = _mm256_load_ps(&d[i]);
        const __m256 _v = _mm256_load_ps(&v[i]);
        const __m256 _p = _mm256_load_ps(&p[i]);

        const __m256 _v1 = _mm256_mul_ps(_v, _coef);
        const __m256 _v2 = _mm256_fnmadd_ps(_d, _factor, _v1);
        const __m256 _res = _mm256_sub_ps(_p, _v2);

        _mm256_store_ps(&p[i], _res);
        _mm256_store_ps(&v[i], _v2);
    }

	for (size_t i = size-(size%8); i < size; i++) {
        v[i] = (v[i]*coef)+(d[i]*factor);
        p[i] -= v[i];
	}
}
void Optimizer::MomentumSGDL1Compute(float* __restrict p, float* __restrict  v, const float* __restrict d, size_t size, float lr, size_t n, float lambda, float coef) {

    // adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;
    const __m256 _factor = _mm256_set1_ps(factor);
    const __m256 _coef = _mm256_set1_ps(coef);
    const __m256 _lambda = _mm256_set1_ps(lambda);
    
    const __m256 _none = _mm256_set1_ps(-1.0f);
    const __m256 _one = _mm256_set1_ps(1.0f);
    const __m256 _zero = _mm256_setzero_ps();

	// update parameters
	#pragma omp parallel for
    for (ssize_t i = 0; i <= (ssize_t)size-8; i += 8) {
        const __m256 _d = _mm256_load_ps(&d[i]);
        const __m256 _v = _mm256_load_ps(&v[i]);
        const __m256 _p = _mm256_load_ps(&p[i]);

        const __m256 _mask = _mm256_cmp_ps(_p, _zero, _CMP_GT_OS);
        const __m256 _sign = _mm256_blendv_ps(_none, _one, _mask);

        const __m256 _v1 = _mm256_mul_ps(_v, _coef);
        const __m256 _v2 = _mm256_fnmadd_ps(_d, _factor, _v1);

        const __m256 _v3 = _mm256_fmadd_ps(_lambda, _sign, _v2);
        const __m256 _res = _mm256_sub_ps(_p, _v3);

        _mm256_store_ps(&p[i], _res);
        _mm256_store_ps(&v[i], _v2);
    }

    for (size_t i = size-(size%8); i < size; i++) {
        const float sign = p[i] > 0.0f ? 1.0f : -1.0f;
        v[i] = (v[i]*coef)+(d[i]*factor);
        p[i] -= (v[i]+(lambda*sign));	
    }
}
void Optimizer::MomentumSGDL2Compute(float* __restrict p, float* __restrict  v, const float* __restrict d, size_t size, float lr, size_t n, float lambda, float coef) {
    
    // adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;
    const __m256 _factor = _mm256_set1_ps(factor);
    const __m256 _coef = _mm256_set1_ps(coef);
    const __m256 _lambda = _mm256_set1_ps(lambda);

	// update parameters
	#pragma omp parallel for
    for (ssize_t i = 0; i <= (ssize_t)size-8; i += 8) {
        const __m256 _d = _mm256_load_ps(&d[i]);
        const __m256 _v = _mm256_load_ps(&v[i]);
        const __m256 _p = _mm256_load_ps(&p[i]);

        const __m256 _v1 = _mm256_mul_ps(_v, _coef);
        const __m256 _v2 = _mm256_fnmadd_ps(_d, _factor, _v1);

        const __m256 _v3 = _mm256_fmadd_ps(_lambda, _p, _v2);
        const __m256 _res = _mm256_sub_ps(_p, _v3);

        _mm256_store_ps(&p[i], _res);
        _mm256_store_ps(&v[i], _v2);
    }

    for (size_t i = size-(size%8); i < size; i++) {
        v[i] = (v[i]*coef)+(d[i]*factor);
        p[i] -= v[i]+(lambda*p[i]);
    }
}
#else
void Optimizer::MomentumSGDCompute(float* __restrict  p, float* __restrict  v, const float* __restrict  d, size_t size, float lr, size_t n, float coef) {

    // adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;

	// update parameters
	#pragma omp parallel for
	for (size_t i = 0; i < size; i++) {
        v[i] = (v[i]*coef)+(d[i]*factor);
        p[i] -= v[i];		
	}
}
void Optimizer::MomentumSGDL1Compute(float* __restrict p, float* __restrict  v, const float* __restrict d, size_t size, float lr, size_t n, float lambda, float coef) {

    // adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;

    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        const float sign = p[i] > 0.0f ? 1.0f : -1.0f;
        v[i] = (v[i]*coef)+(d[i]*factor);
        p[i] -= (v[i]+(lambda*sign));	
    }
}
void Optimizer::MomentumSGDL2Compute(float* __restrict p, float* __restrict  v, const float* __restrict d, size_t size, float lr, size_t n, float lambda, float coef) {
    
    // adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;

    // update parameters
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        v[i] = (v[i]*coef)+(d[i]*factor);
        p[i] -= v[i]+(lambda*p[i]);
    }
}
#endif
