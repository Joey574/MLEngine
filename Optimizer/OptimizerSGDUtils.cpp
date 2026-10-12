#include "Optimizer.hpp"

void Optimizer::SGDCompute(float* __restrict p, const float* __restrict d, size_t size, float lr, size_t n) {
    const float factor = lr / (float)n;

	// update parameters
	#pragma omp parallel for  schedule(static)
	for (size_t i = 0; i < size; i++) {
        p[i] -= d[i]*factor;
	}
}
void Optimizer::SGDL1Compute(float* __restrict p, const float* __restrict d, size_t size, float lr, size_t n, float lambda) {
    const float factor = lr / (float)n;

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < size; i++) {
        const float sign = (p[i] > 0.0f) - (p[i] < 0.0f);
        p[i] -= factor*(d[i]+(lambda*sign));
    }
}
void Optimizer::SGDL2Compute(float* __restrict p, const float* __restrict d, size_t size, float lr, size_t n, float lambda) {
    const float factor = lr / (float)n;

    // update parameters
    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < size; i++) {
        p[i] -= (factor*(d[i]+(lambda*p[i])));
    }
}
