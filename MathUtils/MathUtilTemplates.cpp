#include "MathUtils.hpp"
#include "MathUtils.tpp"

template void MathUtils::MatrixColumnSum<true>(const float*, float*, size_t, size_t);
template void MathUtils::MatrixColumnSum<false>(const float*, float*, size_t, size_t);
