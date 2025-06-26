#include "MathUtils.hpp"
#include "DotProds.tpp"
#include "DotProdsTA.tpp"
#include "DotProdsTB.tpp"

template void MathUtils::DotProd<true>(const float*, const float*, float*, size_t, size_t, size_t, size_t);
template void MathUtils::DotProd<false>(const float*, const float*, float*, size_t, size_t, size_t, size_t);
template void MathUtils::DotProdTA<true>(const float*, const float*, float*, size_t, size_t, size_t, size_t);
template void MathUtils::DotProdTA<false>(const float*, const float*, float*, size_t, size_t, size_t, size_t);
template void MathUtils::DotProdTB<true>(const float*, const float*, float*, size_t, size_t, size_t, size_t);
template void MathUtils::DotProdTB<false>(const float*, const float*, float*, size_t, size_t, size_t, size_t);
