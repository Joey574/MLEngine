#include "Layer.hpp"
#include "../MathUtils/MathUtils.hpp"

void Layer::ComputeDT(const float* __restrict truth, size_t n) {
    const float* __restrict z = m_z;
    const float* __restrict a = m_a;

    const float* __restrict nw = m_nw;
    
    float* __restrict dt = m_dt;

    // compute dt
    switch (type) {
        case LayerType::hidden:
            //MathUtils::DotProdTBDerv<true>(activation.type, truth, nw, dt, z, n, nenodes, nodes, nenodes);
            MathUtils::DotProdTB<true>(truth, nw, dt, n, nenodes, nodes, nenodes);
            (activation.derivative)(z, dt, n*nodes);
            break;
        case LayerType::output:
            // compute loss
            (*lossmetric.loss)(a, truth, dt, n, nodes);
            break;
    }
}
void Layer::ComputeDN(const float* __restrict input, size_t n) {
    float* __restrict dt = m_dt;
    float* __restrict dw = m_dw;
    float* __restrict db = m_db;

    // compute dw
    MathUtils::DotProdTA<true>(input, dt, dw, n, inodes, n, nodes);

    // prep db by copying in first values, clearing existing ones
    std::memcpy(db, dt, nodes*sizeof(float));

    // compute db
    for (size_t i = 1; i < n; i++) {

        size_t j = 0;
        for (; j+8 <= nodes; j+= 8) {
            const __m256 _a = _mm256_loadu_ps(&dt[i*nodes+j]);
            const __m256 _b = _mm256_loadu_ps(&db[j]);
            const __m256 _c = _mm256_add_ps(_a, _b);

            _mm256_storeu_ps(&db[j], _c);
        }

        for (; j < nodes; j++) {
            db[j] += dt[i*nodes+j];
        }
    }
}

void Layer::BasicBackward(const float* __restrict truth, const float* __restrict input, size_t n) {
    if (type == LayerType::input) { return; }

    ComputeDT(truth, n);
    ComputeDN(input, n);
}

void Layer::DropoutBackward(const float* __restrict truth, const float* __restrict input, size_t n) {
    if (type == LayerType::input) { return; }

    // start by computing dt
    ComputeDT(truth, n);

    // output doesn't have dropout, early out
    if (type == LayerType::output) {
        ComputeDN(input, n);
        return;
    }

    float* __restrict dt = m_dt;
    const uint8_t* __restrict mask = m_d_dpmask;

    // apply dropout
    #pragma omp parallel for simd
    for (size_t i = 0; i < n*nodes; i++) {
        size_t byteidx = i >> 3;
        uint8_t bitidx = i & 7;

        const bool k = (mask[byteidx] >> bitidx) & 1;

        if (!k) {
            dt[i] = 0.0f;
        }
    }

    // compute other derivatives
    ComputeDN(input, n);
}