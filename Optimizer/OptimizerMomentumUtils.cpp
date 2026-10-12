#include "Optimizer.hpp"
#include <cblas.h>

void Optimizer::MomentumSGDCompute(float* __restrict  p, float* __restrict  v, const float* __restrict  d, size_t size, float lr, size_t n, float coef) {
    const float factor = lr / (float)n;

	// update parameters
	#pragma omp parallel for simd schedule(static)
	for (size_t i = 0; i < size; i++) {
        v[i] = (v[i]*coef)+(d[i]*factor);
        p[i] -= v[i];
	}
}

void Optimizer::MomentumSGDL1Compute(float* __restrict p, float* __restrict  v, const float* __restrict d, size_t size, float lr, size_t n, float lambda, float coef) {
    const float factor = lr / (float)n;

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < size; i++) {
        const float sign = p[i] > 0.0f ? 1.0f : -1.0f;
        v[i] = (v[i]*coef)+(d[i]*factor);
        p[i] -= (v[i]+(lambda*sign));
    }
}

void Optimizer::MomentumSGDL2Compute(float* __restrict p, float* __restrict  v, const float* __restrict d, size_t size, float lr, size_t n, float lambda, float coef) {
    const float factor = lr / (float)n;

    // update parameters
    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < size; i++) {
        v[i] = (v[i]*coef)+(d[i]*factor);
        p[i] -= v[i]+(lambda*p[i]);
    }
}
