#include "MathUtils.hpp"
#include "DotProdsScalar.tpp"
#include "DotProdsAVX2.tpp"
#include "DotProdsAVX512.tpp"

template __attribute__((target("default"))) void MathUtils::DotProd_Scalar<true>(const float*, const float*, float*, size_t, size_t, size_t, size_t);
template __attribute__((target("default"))) void MathUtils::DotProd_Scalar<false>(const float*, const float*, float*, size_t, size_t, size_t, size_t);
template __attribute__((target("default"))) void MathUtils::DotProdTA_Scalar<true>(const float*, const float*, float*, size_t, size_t, size_t, size_t);
template __attribute__((target("default"))) void MathUtils::DotProdTA_Scalar<false>(const float*, const float*, float*, size_t, size_t, size_t, size_t);
template __attribute__((target("default"))) void MathUtils::DotProdTB_Scalar<true>(const float*, const float*, float*, size_t, size_t, size_t, size_t);
template __attribute__((target("default"))) void MathUtils::DotProdTB_Scalar<false>(const float*, const float*, float*, size_t, size_t, size_t, size_t);

template __attribute__((target("avx2,fma"))) void MathUtils::DotProd_AVX2<true>(const float*, const float*, float*, size_t, size_t, size_t, size_t);
template __attribute__((target("avx2,fma"))) void MathUtils::DotProd_AVX2<false>(const float*, const float*, float*, size_t, size_t, size_t, size_t);
template __attribute__((target("avx2,fma"))) void MathUtils::DotProdTA_AVX2<true>(const float*, const float*, float*, size_t, size_t, size_t, size_t);
template __attribute__((target("avx2,fma"))) void MathUtils::DotProdTA_AVX2<false>(const float*, const float*, float*, size_t, size_t, size_t, size_t);
template __attribute__((target("avx2,fma"))) void MathUtils::DotProdTB_AVX2<true>(const float*, const float*, float*, size_t, size_t, size_t, size_t);
template __attribute__((target("avx2,fma"))) void MathUtils::DotProdTB_AVX2<false>(const float*, const float*, float*, size_t, size_t, size_t, size_t);

template __attribute__((target("avx512f"))) void MathUtils::DotProd_AVX512<true>(const float*, const float*, float*, size_t, size_t, size_t, size_t);
template __attribute__((target("avx512f"))) void MathUtils::DotProd_AVX512<false>(const float*, const float*, float*, size_t, size_t, size_t, size_t);
template __attribute__((target("avx512f"))) void MathUtils::DotProdTA_AVX512<true>(const float*, const float*, float*, size_t, size_t, size_t, size_t);
template __attribute__((target("avx512f"))) void MathUtils::DotProdTA_AVX512<false>(const float*, const float*, float*, size_t, size_t, size_t, size_t);
template __attribute__((target("avx512f"))) void MathUtils::DotProdTB_AVX512<true>(const float*, const float*, float*, size_t, size_t, size_t, size_t);
template __attribute__((target("avx512f"))) void MathUtils::DotProdTB_AVX512<false>(const float*, const float*, float*, size_t, size_t, size_t, size_t);
