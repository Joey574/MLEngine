#include "Layer.hpp"

void Layer::ApplyDropoutFP(size_t n) {
    float* __restrict a = m_a;
    uint8_t* __restrict mask = m_d_dpmask;

    const float scale = 1.0f/(1.0f-m_d_rate);
    const size_t num_bytes = (n*nodes+7)/8;

    #pragma omp simd
    for (size_t i = 0; i < num_bytes; i++) {

        uint8_t newbyte = 0;
        newbyte |= (m_d_dropoutdist(gen) & 1) << 0;
        newbyte |= (m_d_dropoutdist(gen) & 1) << 1;
        newbyte |= (m_d_dropoutdist(gen) & 1) << 2;
        newbyte |= (m_d_dropoutdist(gen) & 1) << 3;
        newbyte |= (m_d_dropoutdist(gen) & 1) << 4;
        newbyte |= (m_d_dropoutdist(gen) & 1) << 5;
        newbyte |= (m_d_dropoutdist(gen) & 1) << 6;
        newbyte |= (m_d_dropoutdist(gen) & 1) << 7;
        
        mask[i] = newbyte;
    }

    #pragma omp parallel for simd
    for (size_t i = 0; i < n*nodes; i++) {
        const size_t byteidx = i >> 3;
        const uint8_t bitidx = i & 7;

        const bool k = (mask[byteidx] >> bitidx) & 1;

        if (k) {
            a[i] *= scale;
        } else {
            a[i] = 0.0f;
        }
    }
}
