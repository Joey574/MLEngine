#include "Optimizer.hpp"

void Optimizer::AdamCompute(float* __restrict p, float* __restrict m, float* __restrict v, const float* __restrict d, size_t size, float lr, size_t n, float b1, float b2, float epsl, size_t t) {
    const float factor = lr;
    const float b1Rate = 1.0f - b1;
    const float b2Rate = 1.0f - b2;

    const float b1InvDenominator = 1.0f / (1.0f - powf(b1, (float)t));
    const float b2InvDenominator = 1.0f / (1.0f - powf(b2, (float)t));

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < size; i++) {
        m[i] = (m[i] * b1) + (b1Rate * d[i]);
        v[i] = (v[i] * b2) + (b2Rate * d[i] * d[i]);

        const float mh = m[i] * b1InvDenominator;
        const float vh = v[i] * b2InvDenominator;

        p[i] -= factor * (mh / sqrtf(vh + epsl));
    }
}
