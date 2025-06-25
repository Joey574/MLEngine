#include "MathUtils.hpp"
#include "DotProdsAVX2.tpp"

template __attribute__((target("avx2"))) void MathUtils::DotProd_AVX2<true>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
template __attribute__((target("avx2"))) void MathUtils::DotProd_AVX2<false>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
template __attribute__((target("avx2"))) void MathUtils::DotProdTA_AVX2<true>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
template __attribute__((target("avx2"))) void MathUtils::DotProdTA_AVX2<false>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
template __attribute__((target("avx2"))) void MathUtils::DotProdTB_AVX2<true>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
template __attribute__((target("avx2"))) void MathUtils::DotProdTB_AVX2<false>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
