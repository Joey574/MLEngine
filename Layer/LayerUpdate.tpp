#pragma once
#include "Layer.hpp"

template <bool l2>
void Layer::BasicUpdate(float lr, size_t n) {
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

		if constexpr (l2) {
			ApplyL2Update(&dw[i], &w[i], _factor, _mm256_set1_ps(m_l2_lambda));
		} else {
			ApplyBasicUpdate(&dw[i], &w[i], _factor);
		}
	}

	for (size_t i = wsize-(wsize%8); i < wsize; i++) {
		if constexpr (l2) {
			w[i] = ApplyL2Update(dw[i], w[i], factor, m_l2_lambda);
		} else {
			w[i] = ApplyBasicUpdate(dw[i], w[i], factor);
		}
	}

	// update biases
	#pragma omp parallel for
	for (ssize_t i = 0; i <= ((ssize_t)bsize)-8; i += 8) {
		if constexpr (l2) {
			ApplyL2Update(&db[i], &b[i], _factor, _mm256_set1_ps(m_l2_lambda));
		} else {
			ApplyBasicUpdate(&db[i], &b[i], _factor);
		}
	}

	for (size_t i = bsize-(bsize%8); i < bsize; i++) {
		if constexpr (l2) {
			b[i] = ApplyL2Update(db[i], b[i], factor, m_l2_lambda);
		} else {
			b[i] = ApplyBasicUpdate(db[i], b[i], factor);
		}
	}
}

template <bool l2>
void Layer::MomentumUpdate(float lr, size_t n) {
    if (type == LayerType::input) { return; }

    const float* __restrict dw = m_dw;
	const float* __restrict db = m_db;
	float* __restrict vw = m_m_vw;
	float* __restrict vb = m_m_vb;
    float* __restrict w = m_w;
	float* __restrict b = m_b;

    // adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;
	const float coef = m_m_coefficient;
    const __m256 _factor = _mm256_set1_ps(factor);
	const __m256 _coef = _mm256_set1_ps(coef);

	// update weights
	#pragma omp parallel for
	for (ssize_t i = 0; i <= ((ssize_t)wsize)-8; i += 8) {
		const __m256 _dw = _mm256_load_ps(&dw[i]);
		const __m256 _vw = _mm256_load_ps(&vw[i]);
		const __m256 _w = _mm256_load_ps(&w[i]);

		const __m256 _vw1 = _mm256_mul_ps(_vw, _coef);
		const __m256 _vw2 = _mm256_fmadd_ps(_dw, _factor, _vw1);
		const __m256 _res = _mm256_sub_ps(_w, _vw2);

		_mm256_store_ps(&w[i], _res);
		_mm256_store_ps(&vw[i], _vw2);
	}

	for (size_t i = wsize-(wsize%8); i < wsize; i++) {
		vw[i] = (vw[i]*coef)+(dw[i]*factor);
		w[i] -= vw[i];
	}

	// update biases
	#pragma omp parallel for
	for (ssize_t i = 0; i <= ((ssize_t)bsize)-8; i += 8) {
		const __m256 _db = _mm256_load_ps(&db[i]);
		const __m256 _vb = _mm256_load_ps(&vb[i]);
		const __m256 _b = _mm256_load_ps(&b[i]);

		const __m256 _vb1 = _mm256_mul_ps(_vb, _coef);
		const __m256 _vb2 = _mm256_fmadd_ps(_db, _factor, _vb1);
		const __m256 _res = _mm256_sub_ps(_b, _vb2);

		_mm256_store_ps(&b[i], _res);
		_mm256_store_ps(&vb[i], _vb2);
	}

	for (size_t i = bsize-(bsize%8); i < bsize; i++) {
		vb[i] = (vb[i]*coef)+(db[i]*factor);
		b[i] -= vb[i];
	}
}
