#include "Layer.hpp"
#include "../NeuralNetwork/NeuralNetwork.hpp"

void Layer::forward(bool training, const float* __restrict x, float* __restrict z, float* __restrict a, size_t n) {
    if (type == LayerType::input) { return; }
    
    const float* __restrict const w = m_w;
    const float* __restrict const b = m_b;

    // copy bias values into total
    for (size_t i = 0; i < n; i++) {
        std::memcpy(&z[i*nodes], b, nodes*sizeof(float));
    }

    // perform dot prod with input
    NeuralNetwork::DotProd(x, w, z, n, inodes, inodes, nodes, false);

    // apply activation
    activation.activation(z, a, n*nodes);
}

void Layer::backward(const float* __restrict y, const float* __restrict pa, const float* __restrict z, const float* __restrict a, const float* __restrict nw, float* __restrict dt, float* __restrict dw, float* __restrict db, size_t nenodes, size_t n) {
    if (type == LayerType::input) { return; }

    // compute dt
    switch (type) {
        case LayerType::hidden:
            NeuralNetwork::DotProdTB(y, nw, dt, n, nenodes, nodes, nenodes, true);
            (activation.derivative)(z, dt, n*nodes);
            break;
        case LayerType::output:
            // compute loss
            (*lossmetric.loss)(a, y, dt, n, nodes);
            break;
    }

    // compute dw
    NeuralNetwork::DotProdTA(pa, dt, dw, n, inodes, n, nodes, true);
    
    // prep db by copying in first values, clearing existing ones
    std::memcpy(db, dt, nodes*sizeof(float));

    // compute db
    for (size_t i = 1; i < n; i++) {

        size_t j = 0;
        for (; j <= nodes-8; j+= 8) {
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