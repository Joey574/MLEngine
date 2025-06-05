#include "Layer.hpp"
#include "../NeuralNetwork/NeuralNetwork.hpp"

void Layer::forward(bool training, float* __restrict x, size_t n) {
    // calls out to the right forward prop based on passed arguments
    (this->*executeForward)(training, x, n);
}

void Layer::backward(const float* __restrict truth, const float* __restrict input, size_t n) {
    // calls out to the right back prop based on passed arguments
    (this->*executeBackward)(truth, input, n);
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
	for (size_t i = 0; i <= params-8; i += 8) {
		const __m256 _a = _mm256_loadu_ps(&dw[i]);
		const __m256 _b = _mm256_loadu_ps(&w[i]);
		const __m256 _res = _mm256_fnmadd_ps(_a, _factor, _b);

		_mm256_storeu_ps(&w[i], _res);
	}

	for (size_t i = params-(params%8); i < params; i++) {
		w[i] -= dw[i] * factor;
	}
}