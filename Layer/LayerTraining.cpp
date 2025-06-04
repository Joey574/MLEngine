#include "Layer.hpp"
#include "../NeuralNetwork/NeuralNetwork.hpp"

void Layer::forward(bool training, float* __restrict x, size_t n) {
    if (type == LayerType::input) { 
        if (training) {
            m_z = x; m_a = x; 
        } else {
            m_tz = x; m_ta = x;
        }
        return; 
    }

    const float* __restrict w = m_w;
    const float* __restrict b = m_b;

    float* __restrict z;
    float* __restrict a;
    
    // change output pointers based on if we're training or not
    if (training) {
        z = m_z;
        a = m_a;
    } else {
        z = m_tz;
        a = m_ta;
    }

    // copy bias values into total
    for (size_t i = 0; i < n; i++) {
        std::memcpy(&z[i*nodes], b, nodes*sizeof(float));
    }

    // perform dot prod with input
    NeuralNetwork::DotProd(x, w, z, n, inodes, inodes, nodes, false);

    // apply activation
    activation.activation(z, a, n*nodes);
}

void Layer::backward(const float* __restrict truth, const float* __restrict input, size_t n) {
    if (type == LayerType::input) { return; }

    const float* __restrict z = m_z;
    const float* __restrict a = m_a;

    const float* __restrict nw = m_nw;
    
    float* __restrict dt = m_dt;
    float* __restrict dw = m_dw;
    float* __restrict db = m_db;

    // compute dt
    switch (type) {
        case LayerType::hidden:
            NeuralNetwork::DotProdTB(truth, nw, dt, n, nenodes, nodes, nenodes, true);
            (activation.derivative)(z, dt, n*nodes);
            break;
        case LayerType::output:
            // compute loss
            (*lossmetric.loss)(a, truth, dt, n, nodes);
            break;
    }

    // compute dw
    NeuralNetwork::DotProdTA(input, dt, dw, n, inodes, n, nodes, true);
    
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

void Layer::update(float lr, size_t n) {
    if (type == LayerType::input) { return; }

    const float* __restrict dw = m_dw;
    float* __restrict w = m_w;

    // adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;
    const __m256 _factor = _mm256_set1_ps(factor);

	// update network (bias and weights currently use same formula to update and are contiguous in memory, so both happen here)
	#pragma omp parallel for
	for (size_t i = 0; i <= layer_size-8; i += 8) {
		const __m256 _a = _mm256_loadu_ps(&dw[i]);
		const __m256 _b = _mm256_loadu_ps(&w[i]);
		const __m256 _res = _mm256_fnmadd_ps(_a, _factor, _b);

		_mm256_storeu_ps(&w[i], _res);
	}

	for (size_t i = layer_size-(layer_size%8); i < layer_size; i++) {
		w[i] -= dw[i] * factor;
	}
}