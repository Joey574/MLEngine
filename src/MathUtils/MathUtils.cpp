#include "MathUtils.tpp"

template void MathUtils::DotProd<true>(const float* a, const float* b, float* c, size_t ar, size_t ac, size_t br, size_t bc);
template void MathUtils::DotProd<false>(const float* a, const float* b, float* c, size_t ar, size_t ac, size_t br, size_t bc);

template void MathUtils::DotProdTA<true>(const float* a, const float* b, float* c, size_t ar, size_t ac, size_t br, size_t bc);
template void MathUtils::DotProdTA<false>(const float* a, const float* b, float* c, size_t ar, size_t ac, size_t br, size_t bc);

template void MathUtils::DotProdTB<true>(const float* a, const float* b, float* c, size_t ar, size_t ac, size_t br, size_t bc);
template void MathUtils::DotProdTB<false>(const float* a, const float* b, float* c, size_t ar, size_t ac, size_t br, size_t bc);