#include "Optimizer.hpp"

void Optimizer::RMSPropCompute(float* __restrict  p, float* __restrict  g, const float* __restrict  d, size_t size, float lr, size_t n, float decay, float epsl) {
    const float factor = lr / (float)n;

    #pragma omp parallel for simd schedule(static)
	for (size_t i = 0; i < size; i++) {
        g[i] = (decay*g[i])+(1.0f-decay)*d[i]*d[i];
        p[i] -= (factor / (std::sqrt(g[i]+epsl)))*d[i];
	}
}
