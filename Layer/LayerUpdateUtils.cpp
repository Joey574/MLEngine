#include "Layer.hpp"

void Layer::ApplyBasicUpdate(const float* __restrict d, float* __restrict p, const __m256 _factor) {
    const __m256 _d = _mm256_load_ps(d);
	const __m256 _p = _mm256_load_ps(p);
    const __m256 _res = _mm256_fnmadd_ps(_d, _factor, _p);
	_mm256_store_ps(p, _res);
}
float Layer::ApplyBasicUpdate(const float d, const float p, const float factor) {
    return p - (d*factor);
}

void Layer::ApplyL1Update(const float* __restrict d, float* __restrict p, const __m256 _factor, const __m256 _coef) {
    const __m256 _none = _mm256_set1_ps(-1.0f);
    const __m256 _one = _mm256_set1_ps(1.0f);
    const __m256 _zero = _mm256_setzero_ps();

    const __m256 _d = _mm256_load_ps(d);
    const __m256 _p = _mm256_load_ps(p);

    const __m256 _mask = _mm256_cmp_ps(_p, _zero, _CMP_GT_OS);
    const __m256 _sign = _mm256_blendv_ps(_none, _one, _mask);

    const __m256 _d2 = _mm256_fmadd_ps(_sign, _coef, _d);
    const __m256 _res = _mm256_fnmadd_ps(_d2, _factor, _p);
    _mm256_store_ps(p, _res);
}
float Layer::ApplyL1Update(const float d, const float p, const float factor, const float coef) {
    const float sign = p > 0.0f ? 1.0f : -1.0f;
    return p - factor*(d+(coef*sign));
}

void Layer::ApplyL2Update(const float* __restrict d, float* __restrict p, const __m256 _factor, const __m256 _coef) {
    const __m256 _d = _mm256_load_ps(d);
	const __m256 _p = _mm256_load_ps(p);

    const __m256 _d2 = _mm256_fmadd_ps(_p, _coef, _d);
    const __m256 _res = _mm256_fnmadd_ps(_d2, _factor, _p);
	_mm256_store_ps(p, _res);
}
float Layer::ApplyL2Update(const float d, const float p, const float factor, const float coef) {
    return p - (factor*(d+(coef*p)));
}
