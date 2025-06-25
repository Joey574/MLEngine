#include "MathUtils.hpp"
#include "DotProdsAVX512.tpp"

template __attribute__((target("avx512dq"))) void MathUtils::DotProd_AVX512<true>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
template __attribute__((target("avx512dq"))) void MathUtils::DotProd_AVX512<false>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
template __attribute__((target("avx512dq"))) void MathUtils::DotProdTA_AVX512<true>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
template __attribute__((target("avx512dq"))) void MathUtils::DotProdTA_AVX512<false>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
template __attribute__((target("avx512dq"))) void MathUtils::DotProdTB_AVX512<true>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
template __attribute__((target("avx512dq"))) void MathUtils::DotProdTB_AVX512<false>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
