#include "Optimizer.hpp"

#if defined(__AVX512F__)
void Optimizer::RMSPropCompute(float* __restrict  p, float* __restrict  g, const float* __restrict  d, size_t size, float lr, size_t n, float decay, float epsl) {
    // adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;
    const __m512 _factor = _mm512_set1_ps(factor);
    const __m512 _decay = _mm512_set1_ps(decay);
    const __m512 _ndecay = _mm512_set1_ps(1.0f-decay);
    const __m512 _epsl = _mm512_set1_ps(epsl);

	// update parameters
	#pragma omp parallel for
    for (ssize_t i = 0; i <= (ssize_t)size-16; i += 16) {
        const __m512 _d = _mm512_load_ps(&d[i]);
        const __m512 _g = _mm512_load_ps(&g[i]);
        const __m512 _p = _mm512_load_ps(&p[i]);

        const __m512 _d2 = _mm512_mul_ps(_d, _d);
        const __m512 _g2 = _mm512_mul_ps(_decay, _g);
        const __m512 _g3 = _mm512_fmadd_ps(_d2, _ndecay, _g2);

        const __m512 _ge = _mm512_add_ps(_g3, _epsl);
        const __m512 _gs = _mm512_sqrt_ps(_ge);
        const __m512 _g4 = _mm512_div_ps(_factor, _gs);

        const __m512 _res = _mm512_fnmadd_ps(_g4, _d, _p);

        _mm512_store_ps(&g[i], _g3);
        _mm512_store_ps(&p[i], _res);
    }

	for (size_t i = size-(size%16); i < size; i++) {
        g[i] = (decay*g[i])+(1.0f-decay)*d[i]*d[i];
        p[i] -= (factor / (std::sqrt(g[i]+epsl)))*d[i];
	}
}
#elif defined(__AVX2__) && defined(__FMA__)
void Optimizer::RMSPropCompute(float* __restrict  p, float* __restrict  g, const float* __restrict  d, size_t size, float lr, size_t n, float decay, float epsl) {
    // adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;
    const __m256 _factor = _mm256_set1_ps(factor);
    const __m256 _decay = _mm256_set1_ps(decay);
    const __m256 _ndecay = _mm256_set1_ps(1.0f-decay);
    const __m256 _epsl = _mm256_set1_ps(epsl);

	// update parameters
	#pragma omp parallel for
    for (ssize_t i = 0; i <= (ssize_t)size-8; i += 8) {
        const __m256 _d = _mm256_load_ps(&d[i]);
        const __m256 _g = _mm256_load_ps(&g[i]);
        const __m256 _p = _mm256_load_ps(&p[i]);

        const __m256 _d2 = _mm256_mul_ps(_d, _d);
        const __m256 _g2 = _mm256_mul_ps(_decay, _g);
        const __m256 _g3 = _mm256_fmadd_ps(_d2, _ndecay, _g2);

        const __m256 _ge = _mm256_add_ps(_g3, _epsl);
        const __m256 _gs = _mm256_sqrt_ps(_ge);
        const __m256 _g4 = _mm256_div_ps(_factor, _gs);

        const __m256 _res = _mm256_fnmadd_ps(_g4, _d, _p);

        _mm256_store_ps(&g[i], _g3);
        _mm256_store_ps(&p[i], _res);
    }

	for (size_t i = size-(size%8); i < size; i++) {
        g[i] = (decay*g[i])+(1.0f-decay)*d[i]*d[i];
        p[i] -= (factor / (std::sqrt(g[i]+epsl)))*d[i];
	}
}
#else
void Optimizer::RMSPropCompute(float* __restrict  p, float* __restrict  g, const float* __restrict  d, size_t size, float lr, size_t n, float decay, float epsl) {
    // adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;
    
    #pragma omp parallel for simd
	for (size_t i = 0; i < size; i++) {
        g[i] = (decay*g[i])+(1.0f-decay)*d[i]*d[i];
        p[i] -= (factor / (std::sqrt(g[i]+epsl)))*d[i];
	}
}
#endif


