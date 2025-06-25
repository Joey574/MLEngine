#include "MathUtils.hpp"
#include "DotProdsScalar.tpp"

template __attribute__((target("default"))) void MathUtils::DotProd_Scalar<true>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
template __attribute__((target("default"))) void MathUtils::DotProd_Scalar<false>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
template __attribute__((target("default"))) void MathUtils::DotProdTA_Scalar<true>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
template __attribute__((target("default"))) void MathUtils::DotProdTA_Scalar<false>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
template __attribute__((target("default"))) void MathUtils::DotProdTB_Scalar<true>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
template __attribute__((target("default"))) void MathUtils::DotProdTB_Scalar<false>(const float* __restrict, const float* __restrict, float* __restrict, size_t, size_t, size_t, size_t);
