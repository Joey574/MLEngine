#include "Layer.hpp"

void Layer::ComputeDT(const float* __restrict truth, size_t n) {
    const float* __restrict z = m_z;
    const float* __restrict a = m_a;
    float* __restrict dt = m_dt;

    const float* __restrict nw = m_nw;

    MathUtils::DotProdTB<true>(truth, nw, dt, n, nenodes, nodes, nenodes);
    (activation.derivative)(z, dt, n, nodes);
}
void Layer::ComputeDTOutput(const float* __restrict truth, size_t n) {
    const float* __restrict a = m_a;
    float* __restrict dt = m_dt;

    // compute loss
    (*lossmetric.loss)(a, truth, dt, n, nodes);
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
void Layer::ComputeSkipDN(const float* __restrict input, size_t n) {
    float* __restrict dt = m_dt;
    float* __restrict dw = m_dw;
    float* __restrict db = m_db;
    float* __restrict dw_skip = &dw[m_s_base*nodes];

    const float* __restrict input_skip = (*m_layers)[m_s_idx].Output<true>();

    // compute dw
    MathUtils::DotProdTA<true>(input, dt, dw, n, m_s_base, n, nodes);
    MathUtils::DotProdTA<true>(input_skip, dt, dw_skip, n, m_s_skip, n, nodes);

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

void Layer::ApplyDropoutBP(size_t n) {
    float* __restrict dt = m_dt;
    uint8_t* mask = m_d_dpmask;

    // apply dropout
    #pragma omp parallel for simd
    for (size_t i = 0; i < n*nodes; i ++) {
        size_t byteidx = i >> 3;
        uint8_t bitidx = i & 7;

        const bool k = (mask[byteidx] >> bitidx) & 1;

        if (!k) {
            dt[i] = 0.0f;
        }
    }
}
