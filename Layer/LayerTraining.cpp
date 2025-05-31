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

void Layer::backward(const float* __restrict y, const float* __restrict pa, const float* __restrict z, const float* __restrict a, float* __restrict dt, float* __restrict dw, float* __restrict db, size_t n) {
    if (type == LayerType::input) { return; }

    const float* __restrict w = m_w;
    const float* __restrict b = m_b;

    // compute dt
    switch (type) {
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

			db[j] = NeuralNetwork::Sum256(_sum);

			for (; k < n; k++) {
				db[j] += dt[j * n + k];
			}
		}
}