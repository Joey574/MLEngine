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