#include "Optimizer.hpp"

__attribute__((target("default")))
void Optimizer::SGDComputeAVX2(float* __restrict p, const float* __restrict d, size_t n, float lr) {

    // adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;

	// update parameters
	#pragma omp parallel for
	for (size_t i = 0; i < n; i++) {
        p[i] -= d[i]*factor;		
	}
}

__attribute__((target("default")))
void Optimizer::SGDL1ComputeAVX2(float* p, const float* d, size_t n, float lr, float lambda) {

    // adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;

    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        const float sign = p[i] > 0.0f ? 1.0f : -1.0f;
        p[i] -= factor*(d[i]+(lambda*sign));	
    }
}

__attribute__((target("default")))
void Optimizer::SGDL2ComputeAVX2(float* p, const float* d, size_t n, float lr, float lambda) {
    
    // adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;

    // update parameters
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        p[i] -= (factor*(d[i]+(lambda*p[i])));
    }
}
