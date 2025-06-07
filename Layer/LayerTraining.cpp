#include "Layer.hpp"

void Layer::backward(const float* __restrict truth, const float* __restrict input, size_t n) {
    // calls out to the right back prop based on passed arguments
    (this->*executeBackward)(truth, input, n);
}

void Layer::update(float lr, size_t n) {
    if (type == LayerType::input) { return; }

    const float* __restrict dw = m_dw;
	const float* __restrict db = m_db;
    float* __restrict w = m_w;
	float* __restrict b = m_b;

    // adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;
    const __m256 _factor = _mm256_set1_ps(factor);

	// update weights
	#pragma omp parallel for
	for (ssize_t i = 0; i <= ((ssize_t)wsize)-8; i += 8) {
		const __m256 _a = _mm256_load_ps(&dw[i]);
		const __m256 _b = _mm256_load_ps(&w[i]);
		const __m256 _res = _mm256_fnmadd_ps(_a, _factor, _b);

		_mm256_store_ps(&w[i], _res);
	}

	for (size_t i = wsize-(wsize%8); i < wsize; i++) {
		w[i] -= dw[i] * factor;
	}

	// update biases
	#pragma omp parallel for
	for (ssize_t i = 0; i <= ((ssize_t)bsize)-8; i += 8) {
		const __m256 _a = _mm256_load_ps(&db[i]);
		const __m256 _b = _mm256_load_ps(&b[i]);
		const __m256 _res = _mm256_fnmadd_ps(_a, _factor, _b);

		_mm256_store_ps(&b[i], _res);
	}

	for (size_t i = bsize-(bsize%8); i < bsize; i++) {
		b[i] -= db[i] * factor;
	}

}