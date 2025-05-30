#include "Layer.hpp"

void Layer::forward(bool training, const float* __restrict const x, size_t n) {
    if (type == LayerType::input) { return; }
    
    const float* __restrict const w = m_w;
    const float* __restrict const b = m_b;

    float* __restrict const z = m_z;
    float* __restrict const a = m_a;

    // copy bias values into total
    for (size_t i = 0; i < n; i++) {
        std::memcpy(&z[i*nodes], b, nodes);
    }

    // perform dot prod with input
    NeuralNetwork::DotProd(x, w, z, n, inodes, inodes, nodes, false);

    // apply activation
    activation.activation(z, a, n*nodes);
}

void Layer::backward(const float* __restrict y, const float* __restrict pa, size_t n) {
    if (type == LayerType::input) { return; }

    const float* __restrict w = m_w;
    const float* __restrict b = m_b;

    const float* __restrict z = m_z;
    const float* __restrict a = m_a;

    float* __restrict dt = m_dt;
    float* __restrict dw = m_dw;
    float* __restrict db = m_db;
    
    // compute dt
    switch (type) {
        case LayerType::input:
            return;
        case LayerType::hidden:
            NeuralNetwork::DotProdTB(y, w, dt, n, nodes, inodes, nodes, true);
            (activation.derivative)(z, dt, n*nodes);
            break;
        case LayerType::output:
            // compute loss
            (*lossmetric.loss)(a, y, dt, n, nodes);
            break;
    }

    // compute dw
    NeuralNetwork::DotProdTA(pa, dt, dw, n, inodes, n, nodes, true);

    // compute db
    #pragma omp parallel for
		for (size_t j = 0; j < nodes; j++) {
			__m256 _sum = _mm256_setzero_ps();

			size_t k = 0;
			for (; k <= n-8; k += 8) {
				const __m256 _a = _mm256_loadu_ps(&dt[j*n+k]);
				_sum = _mm256_add_ps(_sum, _a);
			}

			db[j] = Sum256(_sum);

			for (; k < n; k++) {
				db[j] += dt[j * n + k];
			}
		}
}