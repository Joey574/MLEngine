#include "Layer.hpp"

void Layer::MomentumUpdate(float lr, size_t n) {
    const float* __restrict dw = m_dw;
	const float* __restrict db = m_db;
    float* __restrict vw = m_m_vw;
    float* __restrict vb = m_m_vb;
    float* __restrict w = m_w;
	float* __restrict b = m_b;

    // adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;
    const __m256 _factor = _mm256_set1_ps(factor);
    const __m256 _coefficient = _mm256_set1_ps(m_m_coefficient);

	// update weights
	#pragma omp parallel for
	for (ssize_t i = 0; i <= ((ssize_t)wsize)-8; i += 8) {
        const __m256 _v = _mm256_load_ps(&vw[i]);
		const __m256 _d = _mm256_load_ps(&dw[i]);
		const __m256 _w = _mm256_load_ps(&w[i]);

        const __m256 _v1 = _mm256_mul_ps(_v, _coefficient);
        const __m256 _v2 = _mm256_fmadd_ps(_d, _factor, _v1);

		const __m256 _res = _mm256_sub_ps(_w, _v2);

		_mm256_store_ps(&w[i], _res);
        _mm256_store_ps(&vw[i], _v2);
	}

	for (size_t i = wsize-(wsize%8); i < wsize; i++) {
        vw[i] = (vw[i]*m_m_coefficient)+(factor*dw[i]);
		w[i] -= vw[i];
	}

	// update biases
	#pragma omp parallel for
	for (ssize_t i = 0; i <= ((ssize_t)bsize)-8; i += 8) {
		const __m256 _v = _mm256_load_ps(&vb[i]);
		const __m256 _d = _mm256_load_ps(&db[i]);
		const __m256 _b = _mm256_load_ps(&b[i]);

        const __m256 _v1 = _mm256_mul_ps(_v, _coefficient);
        const __m256 _v2 = _mm256_fmadd_ps(_d, _factor, _v1);

		const __m256 _res = _mm256_sub_ps(_b, _v2);

		_mm256_store_ps(&b[i], _res);
        _mm256_store_ps(&vb[i], _v2);
	}

	for (size_t i = bsize-(bsize%8); i < bsize; i++) {
        vb[i] = (vb[i]*m_m_coefficient)+(factor*db[i]);
		b[i] -= vb[i];
	}
}
