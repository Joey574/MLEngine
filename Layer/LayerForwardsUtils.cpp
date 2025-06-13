#include "Layer.hpp"

void Layer::ApplyDropoutFP(size_t n) {
    float* __restrict a = m_a;
    uint8_t* __restrict mask = m_d_dpmask;

    const float scale = 1.0f/(1.0f-m_d_rate);

    #pragma omp parallel for simd
    for (size_t i = 0; i < n*nodes; i++) {
        const bool k = m_d_dropoutdist(gen);

        const size_t byteidx = i >> 3;
        const uint8_t bitidx = i & 7;

        if (k) {
            a[i] *= scale;
            mask[byteidx] |= (1 << bitidx);
        } else {
            a[i] = 0.0f;
            mask[byteidx] &= (0 << bitidx);
        }
    }
}
