#include "MathUtils.hpp"
#include "DotProdsScalar.tpp"
#include "DotProdsAVX.tpp"
#include "DotProdsAVX2.tpp"
#include "DotProdsAVX512.tpp"


template __attribute__((target("default"))) void MathUtils::DotProd_Scalar<true>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
template __attribute__((target("default"))) void MathUtils::DotProd_Scalar<false>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
template __attribute__((target("avx"))) void MathUtils::DotProd_AVX<true>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
template __attribute__((target("avx"))) void MathUtils::DotProd_AVX<false>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
template __attribute__((target("avx2"))) void MathUtils::DotProd_AVX2<true>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
template __attribute__((target("avx2"))) void MathUtils::DotProd_AVX2<false>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
template __attribute__((target("avx512dq"))) void MathUtils::DotProd_AVX512<true>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
template __attribute__((target("avx512dq"))) void MathUtils::DotProd_AVX512<false>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);


template __attribute__((target("default"))) void MathUtils::DotProdTA_Scalar<true>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
template __attribute__((target("default"))) void MathUtils::DotProdTA_Scalar<false>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
template __attribute__((target("avx"))) void MathUtils::DotProdTA_AVX<true>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
template __attribute__((target("avx"))) void MathUtils::DotProdTA_AVX<false>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
template __attribute__((target("avx2"))) void MathUtils::DotProdTA_AVX2<true>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
template __attribute__((target("avx2"))) void MathUtils::DotProdTA_AVX2<false>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
template __attribute__((target("avx512dq"))) void MathUtils::DotProdTA_AVX512<true>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
template __attribute__((target("avx512dq"))) void MathUtils::DotProdTA_AVX512<false>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);


template __attribute__((target("default"))) void MathUtils::DotProdTB_Scalar<true>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
template __attribute__((target("default"))) void MathUtils::DotProdTB_Scalar<false>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
template __attribute__((target("avx"))) void MathUtils::DotProdTB_AVX<true>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
template __attribute__((target("avx"))) void MathUtils::DotProdTB_AVX<false>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
template __attribute__((target("avx2"))) void MathUtils::DotProdTB_AVX2<true>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
template __attribute__((target("avx2"))) void MathUtils::DotProdTB_AVX2<false>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
template __attribute__((target("avx512dq"))) void MathUtils::DotProdTB_AVX512<true>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
template __attribute__((target("avx512dq"))) void MathUtils::DotProdTB_AVX512<false>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
