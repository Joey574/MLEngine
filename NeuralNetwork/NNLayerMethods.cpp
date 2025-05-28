#include "NeuralNetwork.hpp"

void NeuralNetwork::Layer2::forward(bool training, const float* __restrict w, const float* __restrict b, const float* __restrict x, float* __restrict z, float* __restrict a, size_t n) const {
    if (type == LayerType::input) { return; }

    // copy biases into total, clearing existing values
    for (size_t r = 0; r < n; r++) {
        FastCopy(b, &z[r*nodes], nodes);
    }

    // dot prod input and weights into total
    DotProd(x, w, z, n, pnodes, pnodes, nodes, false);


    // apply layer activation
    (*activation)(z, a, n*nodes);
}

void NeuralNetwork::Layer2::backward(float* __restrict w, const float* __restrict z, const float* __restrict a, const float* __restrict x, const float* __restrict y, size_t n) const {
    // compute dt
    switch (type) {
        case LayerType::input:
            return;
        case LayerType::hidden:
            // dotprod from next layers dt, then multiply by derivative
            DotProdTB(y, w, dt, n, nodes, pnodes, nodes, true);
            (*derivative)(z, dt, n*pnodes);
        case LayerType::output:
            // last layer applies loss method
            (*loss.loss)(a, y, dt, n, nodes);
    }

    // compute dw
    DotProdTA(x, dt, dw, n, pnodes, n, nodes, true);

    // compute db
    for (size_t j = 0; j < nodes; j++) {
		__m256 _sum = _mm256_setzero_ps();

		size_t k = 0;
		for (; k <= n-8; k += 8) {
			const __m256 _a = _mm256_loadu_ps(&dt[j*n+k]);
			_sum = _mm256_add_ps(_sum, _a);
		}

		db[j] = Sum256(_sum);

		for (; k < n; k++) {
			db[j] += dt[j*n+k];
		}
	}
}
